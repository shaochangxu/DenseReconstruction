import os
import os.path

build_path = "/home/hadoop/scx/buaa/build/src/exe/colmap "
dense_dir = "/home/hadoop/scx/buaa/test_data/ws_Zone8/dense/0/"
ply_path = "/home/hadoop/scx/buaa/test_data/ws_Zone8/dense/0/Zone8_acmm.ply"

os.system("rm  " + dense_dir + "/stereo/depth_maps/*")
os.system("rm  " + dense_dir + "/stereo/normal_maps/*")
os.system(build_path + "patch_match_stereo --workspace_path " + dense_dir + " --workspace_format COLMAP --PatchMatchStereo.geom_consistency false --PatchMatchStereo.algo COLMAP --PatchMatchStereo.max_image_size 3000 --PatchMatchStereo.window_radius 15 --PatchMatchStereo.filter_min_ncc 0.005")
os.system(build_path + "stereo_fusion --workspace_path " + dense_dir + " --workspace_format COLMAP --input_type photometric --output_path " + ply_path + " --StereoFusion.max_reproj_error 2.0 --StereoFusion.max_normal_error 10.0 --StereoFusion.max_depth_error 0.1 --StereoFusion.check_num_images 5")
