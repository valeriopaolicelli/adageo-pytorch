import os
import random
import time
import numpy as np

from PIL import Image
from fda_util import FDA_source_to_target_np, parse_arguments, scale

######################################### SETUP #########################################
start = time.time()
args = parse_arguments()
root = args.dataset_root
train_q = os.path.join(root, args.train_q)
val_q = os.path.join(root, args.val_q)
src_q_paths = [train_q, val_q]
targets = args.targets
shots = args.shots

######################################### FDA ###########################################
if not args.val_beta:
    for num_shots in shots:
        for target in targets:
            # Target queries folder
            trg_q_path = train_q + "_" + target
            target_size = lambda x: int(x) if x != 'ALL' else len(os.listdir(trg_q_path))
            target_imgs = random.sample(os.listdir(trg_q_path), target_size(num_shots))

            target_start = time.time()
            for src_q_path in src_q_paths: # FDA over train or val queries
                source_images = sorted(os.listdir(src_q_path))

                # output_folder: root + "/train" or "val" + "/queries/query_{target}_pseudo_{num_shots}"
                output_path = src_q_path + "_" + target + "_" + "pseudo"  + "_" + num_shots + "_" + str(args.beta)
                if not os.path.isdir(output_path):
                    os.mkdir(output_path)

                # Map each source image into target domain
                for i in range(len(source_images)):
                    src_img = Image.open(os.path.join(src_q_path, source_images[i]))
                    trg_img = Image.open(os.path.join(trg_q_path, random.choice(target_imgs)))
                    src_img_np = np.asarray(src_img, np.float32)
                    trg_img_np = np.asarray(trg_img, np.float32)

                    src_in_trg_img = FDA_source_to_target_np(src_img_np, trg_img_np, args.beta)

                    # Store the transformed image. IMPORTANT: Scale between [0, 255] to remove numerical artifacts
                    Image.fromarray(scale(src_in_trg_img)).save(os.path.join(output_path, source_images[i]))

                print(f"Time for FDA on {num_shots} target samples, target {target}: {time.time() - target_start}s")

# Tune beta
else:
    valid_imgs_path = os.path.join(root, "beta")
    beta_src_path = os.path.join(valid_imgs_path, "source")
    if not os.path.isdir(valid_imgs_path):
        os.mkdir(valid_imgs_path)
        os.mkdir(beta_src_path)


    beta_vals = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    val_size = 5 # compare results on val_size num of source images
    src_q_path = train_q

    for beta in beta_vals:
        print(f"Processing images with beta {beta}...")
        for target in targets:
            # Target queries folder
            trg_q_path = train_q + "_" + target

            target_size = lambda x: int(x) if x != 'ALL' else len(os.listdir(trg_q_path))
            target_imgs = random.sample(os.listdir(trg_q_path), target_size(args.target_size))

            source_images = sorted(os.listdir(src_q_path)) # sorted makes result reproducible

            output_path = os.path.join(valid_imgs_path, "pseudo" + "_" + target + "_" + str(beta))
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            # Map each source image into target domain
            for i in range(val_size):
                src_img = Image.open(os.path.join(src_q_path, source_images[i]))
                trg_img = Image.open(os.path.join(trg_q_path, random.choice(target_imgs)))
                src_img_np = np.asarray(src_img, np.float32)
                trg_img_np = np.asarray(trg_img, np.float32)

                src_in_trg_img = FDA_source_to_target_np(src_img_np, trg_img_np, beta)

                # Store the transformed image. IMPORTANT: Scale between [0, 255] to remove numerical artifacts
                Image.fromarray(scale(src_in_trg_img)).save(os.path.join(output_path, source_images[i]))

                if not os.path.exists(os.path.join(beta_src_path, source_images[i])):
                    src_img.save(os.path.join(beta_src_path, source_images[i]))


print(f"Processing time: {time.time() - start}s")