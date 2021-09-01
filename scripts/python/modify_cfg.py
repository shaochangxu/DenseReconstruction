import os
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pair_file", default=None, type=str)
parser.add_argument("--cfg_dir", required = True, type=str)
parser.add_argument("--view_select_algo", default="__auto__,7", type=str)
args = parser.parse_args()

pair_file_path = args.pair_file
cfg_dir_path = args.cfg_dir
view_select_algo = args.view_select_algo

pm_cfg_file = open(cfg_dir_path + "/patch-match.cfg", "r")
fusion_cfg_file = open(cfg_dir_path + "/fusion.cfg", "r")
modify_fusion_cfg_file = open(cfg_dir_path + "/fusion_modify.cfg", "w+")
modify_pm_cfg_file = open(cfg_dir_path + "/patch-match_modify.cfg", "w+")

filter = False
process_img_name = []
if not pair_file_path is None:
    filter = True
    pair_file = open(pair_file_path, "r")
    for rawline in pair_file:
        line = rawline.split(' ')
        file_name = line[0].replace("//", "/").split('/')[-1]
        process_img_name.append(file_name.strip())


#print process_img_name
#modify fusion config and patch_match config
for rawline in fusion_cfg_file:
    rawline = rawline.replace("//", "/").split('/')[-1]
    if filter:
        if rawline.strip() in process_img_name:
            #print rawline
            modify_fusion_cfg_file.write(rawline)
            modify_pm_cfg_file.write(rawline)
            modify_pm_cfg_file.write(view_select_algo + "\n")
    else:
        modify_fusion_cfg_file.write(rawline)
        modify_pm_cfg_file.write(rawline)
        modify_pm_cfg_file.write(view_select_algo + "\n")

os.remove(cfg_dir_path + "/fusion.cfg")
os.rename(cfg_dir_path + "/fusion_modify.cfg", cfg_dir_path + "/fusion.cfg")

os.remove(cfg_dir_path + "/patch-match.cfg")
os.rename(cfg_dir_path + "/patch-match_modify.cfg", cfg_dir_path + "/patch-match.cfg")

if filter:
    pair_file.close()
pm_cfg_file.close()
fusion_cfg_file.close()


