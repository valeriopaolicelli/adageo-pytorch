import os
import random
import time
import numpy as np

from PIL import Image
from fda_util import FDA_source_to_target_np, parse_arguments, scale

######################################### SETUP #########################################
args = parse_arguments()
root = args.dataset_root
train_q = os.path.join(root, args.train_q)
val_q = os.path.join(root, args.val_q)
src_q_paths = [train_q, val_q]
targets = ["night", "overcast", "rain", "snow", "sun"]

assert args.target_size in ['1', '5', '20', '50', 'ALL'],\
 f"Expected target size to be in ['1', '5', '20', '50', 'ALL'], but got {args.target_size}"


######################################### FDA ###########################################
for target in targets:
    # Target queries folder
    trg_q_path = train_q + "_" + target

    target_size = lambda x: int(x) if x != 'ALL' else len(os.listdir(trg_q_path))
    target_imgs = random.sample(os.listdir(trg_q_path), target_size(args.target_size))

    for src_q_path in src_q_paths: # FDA over train or val queries
        source_images = sorted(os.listdir(src_q_path)) # sorted makes result reproducible

        # output_folder: root + "/train" or "val" + "/queries/query_pseudo_{target}"
        output_path = src_q_path + "_" + "pseudo" + "_" + target
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        # Map each source image into target domain
        # for source_img in os.listdir(src_q_path):
        for i in range(10):
            src_img = Image.open(os.path.join(src_q_path, source_images[i]))
            trg_img = Image.open(os.path.join(trg_q_path, random.choice(target_imgs)))
            src_img_np = np.asarray(src_img, np.float32)
            trg_img_np = np.asarray(trg_img, np.float32)

            src_in_trg_img = FDA_source_to_target_np(src_img_np, trg_img_np, args.beta)

            # Store the transformed image
            Image.fromarray(scale(src_in_trg_img)).save(os.path.join(output_path, source_images[i]))