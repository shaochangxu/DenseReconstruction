import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import sys
from copy import deepcopy
import torchvision
import math

from .submodule import volumegatelight, volumegatelightgn
# Recurrent Multi-scale Module
from .rnnmodule import *
from .mask_net import *

class UMT_MVSNet_V3(nn.Module):
    def __init__(self, h=512, w=512,
                 reg_loss=False, return_depth=False, gn=True,  predict=False, with_semantic_map=False):
        super(UMT_MVSNet_V3, self).__init__() # parent init

        self.hidden_dim = 768
        self.gn = gn
        self.reg_loss = reg_loss
        self.return_depth = return_depth
        self.predict = predict
        self.with_semantic_map = with_semantic_map

        self.feature = Transformer_FeatNet(input_size=(h, w),hidden_dim=self.hidden_dim) # Transformer Net
        self.gatenet = gatenet(self.gn, 3)
        
        self.cost_transformer = Transformer_CostNet(self.hidden_dim)
        self.decoder = DecoderNet(input_size=(h, w), hidden_dim=self.hidden_dim, bias=True)
        if self.with_semantic_map:
            self.semantic_net = UNet(n_init_features = 4)
        else:
            self.semantic_net = UNet(n_init_features = 3)
        
    def forward(self, imgs, proj_matrices, depth_values, semantic_map=None):
        imgs = torch.unbind(imgs, 1) # [B, C, H, W] * N
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        batch_size, img_height, img_width = imgs[0].shape[0], imgs[0].shape[2], imgs[0].shape[3] # [B, C, H, W]
        num_depth = depth_values.shape[1] # [B, N_depth]
        num_views = len(imgs) # N
        
        ref_feature, src_features = imgs[0], imgs[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        cost_reg_list = []

        for d in range(num_depth):
            # step 2. differentiable homograph, build cost volume
            ref_volume = ref_feature
            warped_volumes = None
            for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d]) #[B, C, H, W]
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume) # [B, C, H, W]
                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume
            volume_variance = warped_volumes / len(src_features)
            cost_reg_list.append(self.feature(volume_variance)) # [B, 768] * D

        prob_volume = torch.stack(cost_reg_list, dim=0).permute(1, 0, 2) # [B, D, 768]
        prob_volume = self.cost_transformer(prob_volume) # [B, D, 768]

        prob_volume = prob_volume.reshape(batch_size * num_depth, self.hidden_dim)
        prob_volume = self.decoder(prob_volume) #[BD, 768] => [BD, H, W]
        prob_volume = prob_volume.squeeze().reshape(batch_size, num_depth, img_height, img_width) #[B, D, H, W]
        prob_volume = F.softmax(prob_volume, dim=1) # [B, H, W]
        
        if self.with_semantic_map:
            semantic_mask = self.semantic_net(torch.cat([imgs[0], semantic_map], dim=1))
        else:
            semantic_mask = self.semantic_net(imgs[0])

        if not self.reg_loss:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            if self.predict:
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            return {"depth": depth, 'prob_volume': prob_volume, "semantic_mask":semantic_mask, "photometric_confidence": photometric_confidence}
        else:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            with torch.no_grad():
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            
            return {"depth": depth, "photometric_confidence": photometric_confidence, "semantic_mask":semantic_mask}

class UMT_MVSNet_V2(nn.Module):
    def __init__(self, h=512, w=512,
                 reg_loss=False, return_depth=False, gn=True,  predict=False):
        super(UMT_MVSNet_V2, self).__init__() # parent init

        self.hidden_dim = 768
        self.gn = gn
        self.reg_loss = reg_loss
        self.return_depth = return_depth
        self.predict = predict
        self.feature = Transformer_FeatNet(input_size=(h, w),hidden_dim=self.hidden_dim) # Transformer Net
        self.gatenet = gatenet(self.gn, 3)
        
        self.cost_transformer = Transformer_CostNet(self.hidden_dim)
        self.decoder = DecoderNet(input_size=(h, w), hidden_dim=self.hidden_dim, bias=True)

        
    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1) # [B, C, H, W] * N
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        batch_size, img_height, img_width = imgs[0].shape[0], imgs[0].shape[2], imgs[0].shape[3] # [B, C, H, W]
        num_depth = depth_values.shape[1] # [B, N_depth]
        num_views = len(imgs) # N
        

        ref_feature, src_features = imgs[0], imgs[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        cost_reg_list = []

        for d in range(num_depth):
            # step 2. differentiable homograph, build cost volume
            ref_volume = ref_feature
            warped_volumes = None
            for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d]) #[B, C, H, W]
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume) # [B, C, H, W]
                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume
            volume_variance = warped_volumes / len(src_features)
            cost_reg_list.append(self.feature(volume_variance)) # [B, 768] * D

        prob_volume = torch.stack(cost_reg_list, dim=0).permute(1, 0, 2) # [B, D, 768]
        prob_volume = self.cost_transformer(prob_volume) # [B, D, 768]

        prob_volume = prob_volume.reshape(batch_size * num_depth, self.hidden_dim)
        prob_volume = self.decoder(prob_volume) #[BD, 768] => [BD, H, W]
        prob_volume = prob_volume.squeeze().reshape(batch_size, num_depth, img_height, img_width) #[B, D, H, W]
        prob_volume = F.softmax(prob_volume, dim=1) # [B, H, W]
        
        semantic_mask = torch.ones([batch_size, 2, img_height, img_width])
        if not self.reg_loss:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            if self.predict:
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            return {"depth": depth, 'prob_volume': prob_volume, "semantic_mask":semantic_mask, "photometric_confidence": photometric_confidence}
        else:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            with torch.no_grad():
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            
            return {"depth": depth, "photometric_confidence": photometric_confidence, "semantic_mask":semantic_mask}

class UMT_MVSNet_V1(nn.Module):
    def __init__(self, image_scale=0.25, max_h=512, max_w=512,
                 reg_loss=False, return_depth=False, gn=True,  predict=False):
        super(UMT_MVSNet_V1, self).__init__() # parent init
        
        self.gn = gn
        self.feature = FeatNet()

        input_size = (int(max_h*image_scale), int(max_w*image_scale)) #height, width

        #print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
        
        input_dim = [32, 16, 16, 32, 32]
        hidden_dim = [ 16, 16, 16, 16, 8]
        num_layers = 5
        kernel_size = [(3, 3) for i in range(num_layers)]
        
        self.cost_regularization = UNetConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
             bias=True, return_all_layers=False, gn=self.gn)
    
        # Cost Aggregation
        self.gatenet = gatenet(self.gn, 32)

        self.reg_loss = reg_loss
        self.return_depth = return_depth
        self.predict = predict

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1) # [B, C, H, W] * N
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        batch_size, img_height, img_width = imgs[0].shape[0], imgs[0].shape[2], imgs[0].shape[3] # [B, C, H, W]
        num_depth = depth_values.shape[1] # [B, N_depth]
        num_views = len(imgs) # N
        

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs] # [B, 32, 512, 512] * N

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        
        # Recurrent process i-th depth layer
        # initialization for drmvsnet # recurrent module
        cost_reg_list = []
        hidden_state = None
        if not self.return_depth: # Training Phase;
            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume
                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                        warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d]) #[B, C, H, W]
                        warped_volume = (warped_volume - ref_volume).pow_(2)
                        reweight = self.gatenet(warped_volume) # [B, C, H, W]
                        if warped_volumes is None:
                            warped_volumes = (reweight + 1) * warped_volume
                        else:
                            warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)
                
                # step 3. cost volume regularization
                cost_reg, hidden_state= self.cost_regularization(-1 * volume_variance, hidden_state, d) # [B, 1, H, W]
                cost_reg_list.append(cost_reg)
            
            
            prob_volume = torch.stack(cost_reg_list, dim=1).squeeze(2) # [B, D, H, W]
            prob_volume = F.softmax(prob_volume, dim=1) # get prob volume use for recurrent to decrease memory consumption
            semantic_mask = torch.ones([batch_size, 2, img_height, img_width])
            if not self.reg_loss:
                depth = depth_regression(prob_volume, depth_values=depth_values)
                if self.predict:
                    with torch.no_grad():
                        # photometric confidence
                        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                        photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                return {"depth": depth, 'prob_volume': prob_volume, "semantic_mask":semantic_mask, "photometric_confidence": photometric_confidence}
            else:
                depth = depth_regression(prob_volume, depth_values=depth_values)
                with torch.no_grad():
                    prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                
                return {"depth": depth, "photometric_confidence": photometric_confidence, "semantic_mask":semantic_mask}
        else:
            shape = ref_feature.shape
            depth_image = torch.zeros(shape[0], shape[2], shape[3]).cuda() #B X H X w
            max_prob_image = torch.zeros(shape[0], shape[2], shape[3]).cuda()
            exp_sum = torch.zeros(shape[0], shape[2], shape[3]).cuda()

            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume
               

                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                        warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d])
                        warped_volume = (warped_volume - ref_volume).pow_(2)
                        reweight = self.gatenet(warped_volume) # saliency 
                        if warped_volumes is None:
                            warped_volumes = (reweight + 1) * warped_volume
                        else:
                            warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)
                
                # step 3. cost volume regularization
                cost_reg, hidden_state= self.cost_regularization(-1 * volume_variance, hidden_state, d)

                # Start to caculate depth index
                #print('cost_reg: ', cost_reg.shape())
                prob = torch.exp(cost_reg.squeeze(1))

                d_idx = d
                depth = depth_values[:, d] # B 
                temp_depth_image = depth.view(shape[0], 1, 1).repeat(1, shape[2], shape[3])
                update_flag_image = (max_prob_image < prob).type(torch.float)
                #print('update num: ', torch.sum(update_flag_image))
                new_max_prob_image = torch.mul(update_flag_image, prob) + torch.mul(1-update_flag_image, max_prob_image)
                new_depth_image = torch.mul(update_flag_image, temp_depth_image) + torch.mul(1-update_flag_image, depth_image)
                max_prob_image = new_max_prob_image
                depth_image = new_depth_image
                exp_sum = exp_sum + prob
            
            forward_exp_sum = exp_sum
            forward_depth_map = depth_image
            
            return {"depth": forward_depth_map, "photometric_confidence": max_prob_image / forward_exp_sum}


def get_propability_map(prob_volume, depth, depth_values):
    # depth_values: B,D
    shape = prob_volume.shape
    batch_size = shape[0]
    depth_num = shape[1]
    height = shape[2]
    width = shape[3]

    depth_delta = torch.abs(depth.unsqueeze(1).repeat(1, depth_num, 1, 1) - depth_values.repeat(height, width, 1, 1).permute(2, 3, 0, 1))
    _, index = torch.min(depth_delta, dim=1) # index: B, H, W
    index = index.unsqueeze(1).repeat(1, depth_num, 1, 1)
    index_left0 = index 
    index_left1 = torch.clamp(index - 1, 0, depth_num - 1)
    index_right0 = torch.clamp(index + 1, 0, depth_num - 1)
    index_right1 = torch.clamp(index + 2, 0, depth_num - 1)

    prob_map_left0 = torch.mean(torch.gather(prob_volume, 1, index_left0), dim=1)
    prob_map_left1 = torch.mean(torch.gather(prob_volume, 1, index_left1), dim=1)
    prob_map_right0 = torch.mean(torch.gather(prob_volume, 1, index_right0), dim=1)
    prob_map_right1 = torch.mean(torch.gather(prob_volume, 1, index_right1), dim=1)
    prob_map = torch.clamp(prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1, 0, 0.9999)
    return prob_map

def mvsnet_loss(imgs, depth_est, depth_gt, mask, with_semantic_loss=False, semantic_mask=None):
    mask = mask > 0.5
    if with_semantic_loss:
        mask1 = semantic_mask[:, 0, :, :]
        step_mask = semantic_mask[:, 1, :, :]
        ref_features = torch.unbind(imgs, 1)[0]
        simplify_loss = simplifyDis(ref_features, mask1, step_mask)
        return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True) + simplify_loss
    else:
        return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

def mvsnet_cls_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False): 
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask 
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape 

    depth_num = depth_value.shape[-1]
    depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
   
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1) # round; B, H, W

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))# - mask_true # remove mask=0 pixel
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W

    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


# def gradient(pred):
#     #print(pred.shape)
#     D_dy = pred[:, :, 1:, :] - pred[:,:,:-1,:]
#     D_dx = pred[:, :, :, 1:] - pred[:,:,:,:-1]
#     return D_dx, D_dy 

def gradient_image(img):
    # img: [B, C, H, W]
    w = img.shape[3]
    h = img.shape[2]

    r = F.pad(img, (0,1,0,0))[:,:,:,1:]
    l = F.pad(img, (1,0,0,0))[:,:,:,:w]
    t = F.pad(img, (0,0,1,0))[:,:,:h,:]
    b = F.pad(img, (0,0,0,1))[:,:,1:,:]

    dx = torch.abs(r - l)
    dy = torch.abs(b - t)

    dx[:,:,:,-1] = 0
    dy[:,:,-1,:] = 0
    return dx, dy	

def compute_reconstr_loss_map(warped, ref, mask):
    alpha = 0.5

    channel = warped.shape[1]
    mask = mask.unsqueeze(1).repeat(1, channel, 1, 1) # [B, C, H, W]

    photo_loss = F.smooth_l1_loss(warped * mask, ref * mask, reduce='none')

    ref_dx, ref_dy = gradient_image(ref * mask)
    warpped_dx, warpped_dy = gradient_image(warped * mask)

    grad_loss = torch.abs(warpped_dx - ref_dx) + torch.abs(warpped_dy - ref_dy)

    return (1 - alpha)*photo_loss + alpha * grad_loss

def ssim(x, y, mask):
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)
    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
    
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    
    ssim = ssim_n / ssim_d
    channel = x.shape[1]
    mask = mask.unsqueeze(1).repeat(1,channel,1,1)
    ssim_mask = F.avg_pool2d(mask, 3, 1)
    #print(ssim.shape)

    return ssim_mask * torch.clamp((1 - ssim) / 2, 0, 1) #[B, C, H, W]

def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    img_dx, img_dy = gradient_image(img)
    depth_dx, depth_dy = gradient_image(depth)

    weights_x = torch.exp(- (lambda_wt * torch.mean(torch.abs(img_dx), 1, True)))
    weights_y = torch.exp(- (lambda_wt * torch.mean(torch.abs(img_dy), 1, True)))
    
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

def calV(src_fea, step):
    # src_fea: [B, C, H, W]
    # step: [B, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    max_R = int(min(height, width) / 20)
    step = step * max_R

    step_max = torch.exp(-step * step / 0.003)
    step_max.view(batch, height * width) # [B, H*W]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xy = torch.stack((x, y)).repeat(batch, 1, 1) # [B, 2, H*W]

    warped_src_fea = src_fea.unsqueeze(2).repeat(1, 1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1) # [B, C, N, H, W]
    W = step_max.unsqueeze(1).repeat(1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1) # [B, N, H, W]
    step_max = step_max.unsqueeze(1).repeat(1, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1) # [B, N, H, W]
    
    i = 0
    for s_y in range(-max_R, max_R + 1):
        for s_x in range(-max_R, max_R + 1):
            trans_x = xy[:, 0, :] + s_x
            trans_y = xy[:, 1, :] + s_y
            trans_x_normalized = trans_x / ((width - 1) / 2) - 1
            trans_y_normalized = trans_y / ((height - 1) / 2) - 1
            grid = torch.stack((trans_x_normalized, trans_y_normalized), dim=1).permute(0, 2, 1)  # [B, H*W, 2]
            warped_src_fea[:,:,i,:,:] = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear', padding_mode='zeros') #[B, C, N, H, W]
            W[:,i,:,:] = math.exp((-s_y*s_y - s_x * s_x) / 0.003) #[B, N, H, W]
            i = i + 1

    W = torch.where(W > step_max, torch.zeros_like(W), W)
    W = F.normalize(W, p=2, dim=1) # [B, N, H, W]
    W = W.unsqueeze(1).repeat(1, channels, (max_R * 2 + 1) * (max_R * 2 + 1), 1, 1) # [B, C, N, H, W]
    WF = W * warped_src_fea

    SWF = torch.sum(WF, dim=2) #[B, C, H, W]
    V = torch.abs(src_fea - SWF)

    return V

def simplifyDis(src_fea, mask, step):
    # src_fea: [B, C, H, W]
    channel = src_fea.shape[1]

    V_before = calV(src_fea, step)
    
    mask = mask.unsqueeze(1).repeat(1, channel, 1, 1)
    V_after = calV(src_fea * mask, step)

    loss_fn = nn.MSELoss(reduce=True, size_average=True)

    return loss_fn(V_before, V_after)

def unsup_loss(imgs, proj_matrices, depth_est, semantic_mask):
    imgs = torch.unbind(imgs, 1) # [B, C, H, W] * N
    proj_matrices = torch.unbind(proj_matrices, 1)
    assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"

    img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
    num_views = len(imgs)

    mask = semantic_mask[:, 0, :, :] # [B, H, W]
    step_mask = semantic_mask[:, 1, :, :] # [B, H, W]
    
    ref_features, src_features = imgs[0], imgs[1:]
    
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
    batch = ref_features.shape[0]

    ssim_loss = 0
    for src_fea, src_proj in zip(src_features, src_projs):
        warped_volume, vis_mask = homo_warping(src_fea, src_proj, ref_proj, depth_est) 

        recon_loss = compute_reconstr_loss_map(warped_volume, imgs[0], vis_mask) # [B, C, H, W]
        recon_loss = torch.mean(recon_loss, 1, keepdim = True) # [B, 1, H, W]

        ssim_loss += torch.mean(ssim(warped_volume, imgs[0], vis_mask))
        vis_mask = 1 - vis_mask	# [B, H, W]	
        res_map = recon_loss + 1e4 * vis_mask.unsqueeze(1) # [B, 1, H, W]

        if reproj_volume == None:
            reproj_volume =  res_map
        else:																																																										
            reproj_volume = torch.cat((reproj_volume, res_map), 1) #[B, N, H, W]
        
    top_vals, _ = torch.topk(reproj_volume, 3, dim = 1, largest = False, sorted = False) # [B, 3, H, W]
    top_masks = torch.where(top_vals < 1e4, torch.ones_like(top_vals), torch.zeros_like(top_vals))
    top_vals = top_vals * top_masks
    reconstr_loss = torch.mean(top_vals)

    smooth_loss = depth_smoothness(depth_est.view(batch, 1, img_height, img_width), imgs[0])

    simplify_loss = simplifyDis(ref_features, mask, step_mask)

    loss = 12 * reconstr_loss + 6 * ssim_loss / (num_views - 1) + 0.18 * smooth_loss + simplify_loss

    return loss
