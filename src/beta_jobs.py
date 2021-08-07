
import os
import sys
import time

folder = "/home/francescom/adageo-WACV2021/src"
if not os.path.abspath(os.curdir) == folder:
    sys.exit()

os.makedirs(f"{folder}/jobs", exist_ok=True)
os.makedirs(f"{folder}/out_job", exist_ok=True)

for seed in range(1, 3):
    for target_domain in ["snow", "rain", "overcast", "sun", "night"]:
        for shots in ["5"]:
            for beta in ["0.005", "0.001"]:
                exp_name = f"{target_domain}_{beta}_{shots}"
                exp_name_seed = f"{exp_name}_{seed}"
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
                f"python {folder}/main.py " +
                f"--seed={seed} --dataset_root=/home/francescom/adageo-WACV2021/src/datasets/svox/images " +
                f"--test_g=test/gallery --test_q=test/queries_{target_domain} --train_q=train/queries_{target_domain}_pseudo_{shots}_{beta} "
                f"--grl --attention --exp_name={exp_name} " +
                f"--grl_datasets=train/queries+test/queries_{target_domain}+train/queries_{target_domain}_pseudo_{shots}_{beta} " +
                f"--val_q=val/queries_{target_domain}_pseudo_{shots}_{beta}\n")
            
                with open(filename, "w") as file:
                    _ = file.write(content)
            
                _ = os.system(f"sbatch {filename}")

                time.sleep(1)