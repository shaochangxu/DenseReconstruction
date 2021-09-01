import os
import os.path

build_path = "/home/hadoop/scx/buaa/build/src/exe/colmap "
dense_dir = "/home/hadoop/scx/buaa/test_data/ws_Zone14/dense/0/"
ply_path = "/home/hadoop/scx/buaa/test_data/ws_Zone14/dense/0/Zone14.ply"

os.system("python /home/hadoop/scx/buaa/DenseReconstruction/scripts/python/modify_cfg.py --cfg_dir " + dense_dir + "/stereo"+ " --view_select_algo __two-stage__,3,11")
os.system(build_path + "patch_match_stereo --workspace_path " + dense_dir + " --workspace_format COLMAP --PatchMatchStereo.geom_consistency false")
os.system(build_path + "stereo_fusion --workspace_path " + dense_dir + " --workspace_format COLMAP --input_type photometric --output_path " + ply_path)
