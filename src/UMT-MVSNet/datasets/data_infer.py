from torch.utils.data import Dataset

import numpy as np
import os
from preprocess import *
from colmap2mvsnet import *

from PIL import Image
from data_io import *

param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}
# Test any dataset with scale and center crop
class MVSDataset(Dataset):
    def __init__(self, datapath, input_format="COLMAP", nviews=7, listfile="", mode="infer", ndepths=192, interval_scale=1.06, inverse_depth=True,
                adaptive_scaling=True, max_h=1200, max_w=1600,sample_scale=1,base_image_size=8, 
                with_colmap_depth_map=False, with_semantic_map=False, **kwargs):
        super(MVSDataset, self).__init__()
        
        self.datapath = datapath
        self.listfile = listfile
        self.input_format = input_format # COLMAP, BLEND
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth

        self.adaptive_scaling=adaptive_scaling
        self.max_h=max_h
        self.max_w=max_w
        self.sample_scale=sample_scale
        self.base_image_size=base_image_size

        assert self.mode == "infer"
        if self.input_format == "COLMAP":
            self.with_colmap_depth_map = with_colmap_depth_map
            self.with_semantic_map = with_semantic_map

            self.metas = self.build_list("stereo/pair.txt")
            self.image_dir = os.path.join(self.datapath, 'images')
            model_dir = os.path.join(self.datapath, 'sparse')

            cameras, self.images, self.points3d = read_model(model_dir, '.bin')
            
            # intrinsic
            self.intrinsic = {}
            for camera_id, cam in cameras.items():
                params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
                if 'f' in param_type[cam.model]:
                    params_dict['fx'] = params_dict['f']
                    params_dict['fy'] = params_dict['f']
                i = np.array([
                    [params_dict['fx'], 0, params_dict['cx']],
                    [0, params_dict['fy'], params_dict['cy']],
                    [0, 0, 1]
                ])
                self.intrinsic[camera_id - 1] = i
            
            # extrinsic
            self.extrinsic = {}
            for image_id, image in self.images.items():
                #print(image.name)
                e = np.zeros((4, 4))
                e[:3, :3] = qvec2rotmat(image.qvec)
                e[:3, 3] = image.tvec
                e[3, 3] = 1
                self.extrinsic[image_id - 1] = e
        else:
            self.with_colmap_depth_map = False
            self.with_semantic_map = True
            self.metas = self.build_list("cams/pair.txt")
            

    def build_list(self, pair_file=""):
        metas = []

        # read the pair file
        with open(os.path.join(self.datapath, pair_file)) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                self.nviews = min(self.nviews, len(src_views))

                metas.append((ref_view, src_views))
        
        #print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # TODO Scale
        #intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        #img = img.resize((1000, 750),Image.ANTIALIAS)
        return self.center_img(np.array(img, dtype=np.float32))	

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def getitem_Blend(self, idx):
        meta = self.metas[idx]
        ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        
        imgs = []
        depth_values = None
        cams=[]
        extrinsics_list=[]
        
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 'blended_images/{:0>8}.jpg'.format(vid))
            proj_mat_filename = os.path.join(self.datapath, 'cams/{:0>8}_cam.txt'.format(vid))
            img = self.read_img(img_filename)
            imgs.append(img)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            cams.append(intrinsics)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics_list.append(extrinsics)
            
            if i == 0:  # reference view
                #imgs.append(self.read_img(img_filename))
                if self.inverse_depth: #slice inverse depth
                    print('Process {} inverse depth'.format(idx))
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # wether depth_end is this
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                            dtype=np.float32) # the set is [)
                    depth_end = depth_interval * self.ndepths + depth_min
                
        imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # N,C,H,W
       
        ##TO DO determine a proper scale to resize input
        resize_scale = 1
        if self.adaptive_scaling:
            h_scale = 0
            w_scale = 0       
            for view in range(self.nviews):
                height_scale = float(self.max_h) / imgs[view].shape[1]
                width_scale = float(self.max_w) / imgs[view].shape[2]
                if height_scale > h_scale:
                    h_scale = height_scale
                if width_scale > w_scale:
                    w_scale = width_scale
            if h_scale > 1 or w_scale > 1:
                print ("max_h, max_w should < W and H!")
                exit(-1)
            resize_scale = h_scale
            if w_scale > h_scale:
                resize_scale = w_scale
        
        imgs = imgs.transpose(0,2,3,1) # N H W C
        
        scaled_input_imgs, scaled_input_cams = scale_mvs_input(imgs, cams, scale=resize_scale, view_num=self.nviews)
              
        #TO DO crop to fit network
        croped_imgs, croped_cams = crop_mvs_input(scaled_input_imgs, scaled_input_cams,view_num=self.nviews,
                    max_h=self.max_h,max_w=self.max_w,base_image_size=self.base_image_size)
                    
        croped_imgs = croped_imgs.transpose(0,3,1,2) # N C H W

        new_proj_matrices = []
        for id in range(self.nviews):
            proj_mat = extrinsics_list[id]#.copy()
            # Down Scale
            proj_mat[:3, :4] = np.matmul(croped_cams[id], proj_mat[:3, :4])
            new_proj_matrices.append(proj_mat)

        new_proj_matrices = np.stack(new_proj_matrices)

        return {"imgs": croped_imgs,
                "proj_matrices": new_proj_matrices,
                "depth_values": depth_values,
                "filename": '{:0>8}'.format(view_ids[0])}
    
    def getitem_Colmap(self, idx):
        meta = self.metas[idx]
        ref_view, src_views = meta
        ref_image_filename = self.images[ref_view + 1].name
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        print(view_ids)
        sample = {}

        zs = []
        depth_values = None
        extrinsics_list=[]
        imgs = []
        cams=[]

        for p3d_id in self.images[idx+1].point3D_ids:
            if p3d_id == -1:
                continue
            transformed = np.matmul(self.extrinsic[idx], [self.points3d[p3d_id].xyz[0], self.points3d[p3d_id].xyz[1], self.points3d[p3d_id].xyz[2], 1])
            zs.append(np.asscalar(transformed[2]))
        zs_sorted = sorted(zs)
        # relaxed depth range
        depth_min = zs_sorted[int(len(zs) * .01)]
        depth_end = zs_sorted[int(len(zs) * .99)]
        depth_interval = (depth_end - depth_min) / (self.ndepths - 1)

        if self.inverse_depth: #slice inverse depth
            print('Process {} inverse depth'.format(idx))
            depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.ndepths, endpoint=False)
            depth_values = 1.0 / depth_values
            depth_values = depth_values.astype(np.float32)
        else:
            depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                    dtype=np.float32) # the set is [)
            depth_end = depth_interval * self.ndepths + depth_min

        for vid in view_ids:
            img_filename = os.path.join(self.image_dir, self.images[vid + 1].name)
            img = self.read_img(img_filename)
            imgs.append(img)
            extrinsics_list.append(self.extrinsic[vid])
            cams.append(self.intrinsic[vid])
	
        #imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # N,C,H,W
       
        ##TO DO determine a proper scale to resize input
        resize_scale = 1
        if self.adaptive_scaling:
            h_scale = 0
            w_scale = 0       
            for view in range(self.nviews):
                height_scale = float(self.max_h) / imgs[view].shape[0]
                width_scale = float(self.max_w) / imgs[view].shape[1]
                if height_scale > h_scale:
                    h_scale = height_scale
                if width_scale > w_scale:
                    w_scale = width_scale
            if h_scale > 1 or w_scale > 1:
                print ("max_h, max_w should < W and H!")
                exit(-1)
            resize_scale = h_scale
            if w_scale > h_scale:
                resize_scale = w_scale
        
        #imgs = imgs.transpose(0,2,3,1) # N H W C
        
        scaled_input_imgs, scaled_input_cams = scale_mvs_input(imgs, cams, scale=resize_scale, view_num=self.nviews)
        #TO DO crop to fit network
        croped_imgs, croped_cams = crop_mvs_input(scaled_input_imgs, scaled_input_cams,view_num=self.nviews,
                    max_h=self.max_h,max_w=self.max_w,base_image_size=self.base_image_size)
                    
        croped_imgs = croped_imgs.transpose(0,3,1,2) # N C H W
	
        new_proj_matrices = []
        for id in range(self.nviews):
            proj_mat = extrinsics_list[id]#.copy()
            # Down Scale
            proj_mat[:3, :4] = np.matmul(croped_cams[id], proj_mat[:3, :4])
            new_proj_matrices.append(proj_mat)

        new_proj_matrices = np.stack(new_proj_matrices)

        sample["imgs"] = croped_imgs
        sample["proj_matrices"] = new_proj_matrices
        sample["depth_values"] = depth_values
        sample["filename"] = (ref_image_filename)

        if self.with_colmap_depth_map:
            depth_maps = []
            for vid in view_ids:
                depth_filename = os.path.join(self.datapath, 'stereo', 'depth_maps', '{}.photometric.bin'.format(self.images[vid + 1].name))
                depth_image = read_array(depth_filename)
                depth_image = scale_image(depth_image, scale=resize_scale, interpolation='nearest')
                h, w = depth_image.shape[0:2]
                new_h = h
                new_w = w
                if new_h > self.max_h:
                    new_h = self.max_h
                else:
                    new_h = int(math.ceil(h / self.base_image_size) * self.base_image_size)
                if new_w > self.max_w:
                    new_w = self.max_w
                else:
                    new_w = int(math.ceil(w / self.base_image_size) * self.base_image_size)
                start_h = int(math.ceil((h - new_h) / 2))
                start_w = int(math.ceil((w - new_w) / 2))
                finish_h = start_h + new_h
                finish_w = start_w + new_w

                depth_image = depth_image[start_h:finish_h, start_w:finish_w]
                depth_maps.append(depth_image)

            sample["colmap_depth_maps"] = np.array(depth_maps)

        if self.with_semantic_map:
            semantic_maps = []
            for vid in view_ids:
                semantic_filename = os.path.join(self.datapath, 'semantic', format(self.images[vid + 1].name))
                semantic_image = self.read_img(semantic_filename)
                semantic_image = scale_image(semantic_image, scale=resize_scale)
                h, w = semantic_image.shape[0:2]
                new_h = h
                new_w = w
                if new_h > self.max_h:
                    new_h = self.max_h
                else:
                    new_h = int(math.ceil(h / self.base_image_size) * self.base_image_size)
                if new_w > self.max_w:
                    new_w = self.max_w
                else:
                    new_w = int(math.ceil(w / self.base_image_size) * self.base_image_size)
                start_h = int(math.ceil((h - new_h) / 2))
                start_w = int(math.ceil((w - new_w) / 2))
                finish_h = start_h + new_h
                finish_w = start_w + new_w

                semantic_image = semantic_image[start_h:finish_h, start_w:finish_w]
                semantic_image = img2semantic(semantic_image)
                semantic_maps.append(semantic_image)

            sample["semantic_maps"] = np.array(semantic_maps)

        return sample

    def __getitem__(self, idx):
        if self.input_format == "BLEND":
            return self.getitem_Blend(idx)
        else:
            return self.getitem_Colmap(idx)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Project dir.')
    args = parser.parse_args()

    dataset = MVSDataset(args.path, max_h=128, max_w=128, with_colmap_depth_map=True, with_semantic_map=True)
    
    data_loader = DataLoader(dataset, 2, shuffle=True, num_workers=0, drop_last=True)
    for batch_idx, sample in enumerate(data_loader):
        print("filename: {}".format(sample["filename"]))
        print("imgs shape:{}".format(sample["imgs"].shape))
        print("proj_matrices shape:{}".format(sample["proj_matrices"].shape))
        print("depth_values shape:{}".format(sample["depth_values"].shape))
        print("colmap_depth_maps shape:{}".format(sample["colmap_depth_maps"].shape))
        print("semantic_maps shape:{}".format(sample["semantic_maps"].shape))
