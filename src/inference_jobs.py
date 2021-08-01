
import os
import sys
import time

folder = "/home/francescom/adageo-WACV2021/src"
if not os.path.abspath(os.curdir) == folder:
    sys.exit()


for out_folder in sorted(os.listdir(folder + "/runs/beta_selected/")):
    if out_folder == "readme.txt":
        continue

    output_path = folder + "/runs/beta_selected/" + out_folder
    with open(output_path + "/info.log", "r") as f:
        line = f.readline().split(", ")
        seed = line[22].split("=")[-1]
        domain = line[-1].split("_")[2]
        exp_name_seed = "inference_" + domain + "_" + seed

    filename = f"{folder}/jobs/{exp_name_seed}.job"

    content = ("" +
    "#!/bin/bash \n" +
    f"#SBATCH --job-name={exp_name_seed} \n" +
    "#SBATCH --gres=gpu:1 \n" +
    "#SBATCH --cpus-per-task=3 \n" +
    "#SBATCH --mem=30GB \n" +
    "#SBATCH --time=48:00:00 \n" +
    f"#SBATCH --output={folder}/out_job/out_{exp_name_seed}.txt \n" +
    f"#SBATCH --error={folder}/out_job/err_{exp_name_seed}.txt \n" +
    "ml purge \nml Python/3.6.6-gomkl-2018b \n" +
    "source /home/gabriele/iccv_tutto/myenv/bin/activate \n" +
    f"python {folder}/make_inference.py " +
    f"--dataset_root=/home/francescom/adageo-WACV2021/src/datasets/svox/images --test_g=test/gallery " +
    f"--model_folder={out_folder} --exp_name=beta_selected --grl --attention")

    with open(filename, "w") as file:
        _ = file.write(content)

    _ = os.system(f"sbatch {filename}")

    time.sleep(1)
