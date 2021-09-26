import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='rename image to 00000000.jpg')
parser.add_argument('--image_dir',type=str, help='select model')
parser.add_argument('--semantic_dir',default = "", type=str, help='select model')

args = parser.parse_args()

image_dir=args.image_dir
semantic_dir=args.semantic_dir

img_idx = 0
for filename in os.listdir(image_dir):
    before_path = os.path.join(image_dir, filename)
    after_path = os.path.join(image_dir, '{:0>8}.jpg'.format(img_idx))
    script = "mv {} {}".format(before_path, after_path)
    os.system(script)
    if not semantic_dir=="":
        before_path = os.path.join(semantic_dir, filename)
        after_path = os.path.join(semantic_dir, '{:0>8}.jpg'.format(img_idx))
        script = "mv {} {}".format(before_path, after_path)
        os.system(script)

    img_idx += 1

