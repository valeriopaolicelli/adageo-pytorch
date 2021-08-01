import os
import sys
import time

folder = "/home/francescom/adageo-WACV2021/src"
if not os.path.abspath(os.curdir) == folder:
    sys.exit()

beta = "0.01"
for target_domain in ["snow", "rain", "overcast", "sun", "night"]:
    for shots in ["1", "50", "ALL"]:
        exp_name = f"fda_{target_domain}_{beta}_{shots}"
        filename = f"{folder}/jobs/{exp_name}.job"
        content = ("" +
        "#!/bin/bash \n" +
        f"#SBATCH --job-name={exp_name} \n" +
        "#SBATCH --gres=gpu:1 \n" +
        "#SBATCH --cpus-per-task=3 \n" +
        "#SBATCH --mem=30GB \n" +
        "#SBATCH --time=48:00:00 \n" +
        f"#SBATCH --output={folder}/out_job/out_{exp_name}.txt \n" +
        f"#SBATCH --error={folder}/out_job/err_{exp_name}.txt \n" +
        "ml purge \nml Python/3.6.6-gomkl-2018b \n" +
        "source /home/gabriele/iccv_tutto/myenv/bin/activate \n" +
        f"python /home/francescom/adageo-WACV2021/src/fda.py " +
        f"--dataset_root /home/francescom/adageo-WACV2021/src/datasets/svox/images " +
        f"--target {target_domain} --shots {shots} --beta {beta}")

        with open(filename, "w") as file:
            _ = file.write(content)

        _ = os.system(f"sbatch {filename}")
