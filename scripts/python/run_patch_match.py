import os
import os.path

build_path = "/home/hadoop/scx/buaa/build/src/exe/colmap "
dense_dir = "/home/hadoop/scx/buaa/test_data/ws_scan9/dense/0/"
ply_path = "/home/hadoop/scx/buaa/test_data/ws_scan9/dense/0/scan9_test.ply"

os.system("rm  " + dense_dir + "/stereo/depth_maps/*")
os.system("rm  " + dense_dir + "/stereo/normal_maps/*")
os.system("python /home/hadoop/scx/buaa/DenseReconstruction/scripts/python/modify_cfg.py --cfg_dir " + dense_dir + "/stereo"+ " --view_select_algo __auto__,11")
os.system(build_path + "patch_match_stereo --workspace_path " + dense_dir + " --workspace_format COLMAP --PatchMatchStereo.geom_consistency false --PatchMatchStereo.algo COLMAP")
os.system(build_path + "stereo_fusion --workspace_path " + dense_dir + " --workspace_format COLMAP --input_type photometric --output_path " + ply_path)
