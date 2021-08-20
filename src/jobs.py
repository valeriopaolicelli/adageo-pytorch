
import os
import sys
import time
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Jobs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--job_type", type=str, default=None, 
        help="Job type: accepted values are (beta, fda, inference, all_domains)"
    )
    parser.add_argument("--beta", type=float, default=0.001, help = "Beta hyperparameter")
    parser.add_argument("--shots", type=int, default=5, help="Number of shots for domain adaptation")
    parser.add_argument("--city", type=str, default=None, help="City for msls_inference")

    return parser.parse_args()


def msls_inference(city):
    # Run models trained on all seeds
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    # Create jobs and ou_job folders if not present
    os.makedirs(f"{folder}/jobs", exist_ok=True)
    os.makedirs(f"{folder}/out_job", exist_ok=True)

    # Match GRL dimensionality
    grl = '+'.join(["a" for _ in range(11)])

    # Repeat inference with all trained models
    for seed in range(3):
        exp_name = f"msls/{city}_{seed}"
        filename = f"{folder}/jobs/{exp_name}.job"
        model_folder = folder + f"/runs/fda/all_{seed}_0.001_5/" + os.listdir(f"/runs/fda/all_{seed}_0.001_5/")[0]

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
        f"python {folder}/inference.py " + 
        f"--seed={seed} --dataset_root=/home/francescom/adageo-WACV2021/src/datasets/msls/{city} " +
        f"--test_g=gallery --test_q=queries " +
        f"--grl --attention --exp_name={exp_name} " +
        f"--grl_dataset={grl} --model_folder={model_folder}"
        )

        # Launch job
        with open(filename, "w") as file:
            _ = file.write(content)

        _ = os.system(f"sbatch {filename}")

        time.sleep(1)


def beta():
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    os.makedirs(f"{folder}/jobs", exist_ok=True)
    os.makedirs(f"{folder}/out_job", exist_ok=True)

    for seed in range(1, 2):
        for target_domain in ["overcast", "night"]:
            for shots in ["5"]:
                for beta in ["0.01"]:
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
                    f"--grl --attention --exp_name={exp_name_seed} " +
                    f"--grl_datasets=train/queries+test/queries_{target_domain}+train/queries_{target_domain}_pseudo_{shots}_{beta} " +
                    f"--val_q=val/queries_{target_domain}_pseudo_{shots}_{beta}\n")
                
                    with open(filename, "w") as file:
                        _ = file.write(content)
                
                    _ = os.system(f"sbatch {filename}")

                    time.sleep(1)


def fda():
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    betas = ["0.005", "0.001", "0.0005"]
    for target_domain in ["snow", "rain", "overcast", "sun", "night"]:
        for shots in ["5"]:
            for beta in betas:
                exp_name = f"fda_{target_domain}_{beta}_{shots}"
                filename = f"{folder}/jobs/{exp_name}.job"
                content = ("" +
                "#!/bin/bash \n" +
                f"#SBATCH --job-name={exp_name} \n" +
                "#SBATCH --gres=gpu:1 \n" +
                "#SBATCH --cpus-per-task=3 \n" +
                "#SBATCH --mem=30GB \n" +
                "#SBATCH --time=02:00:00 \n" +
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


        with open(filename, "w") as file:
            _ = file.write(content)

        _ = os.system(f"sbatch {filename}")

        time.sleep(1)


def all_domains(beta, shots):
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    for seed in range(3):
        exp_name_seed = f"alldomains_{beta}_{shots}_{seed}"
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
        f"--dataset_root=/home/francescom/adageo-WACV2021/src/datasets/svox/images " +
        f"--test_g=test/gallery --test_q=test/queries_night " 
        f"--train_q=train/queries_night_pseudo_{shots}_{beta}+train/queries_overcast_pseudo_{shots}_{beta}+" +
        f"train/queries_rain_pseudo_{shots}_{beta}+train/queries_snow_pseudo_{shots}_{beta}+train/queries_sun_pseudo_{shots}_{beta} " +
        f"--grl_datasets=train/queries+train/queries_night_pseudo_{shots}_{beta}+" +
        f"train/queries_overcast_pseudo_{shots}_{beta}+train/queries_rain_pseudo_{shots}_{beta}+" +
        f"train/queries_snow_pseudo_{shots}_{beta}+train/queries_sun_pseudo_{shots}_{beta}+" +
        f"test/queries_night+test/queries_overcast+test/queries_rain+test/queries_snow+test/queries_sun " +
        f"--val_q=val/queries_night_pseudo_{shots}_{beta}+val/queries_overcast_pseudo_{shots}_{beta}+"+
        f"val/queries_rain_pseudo_{shots}_{beta}+val/queries_snow_pseudo_{shots}_{beta}+val/queries_sun_pseudo_{shots}_{beta} " +
        f"--exp_name={exp_name_seed} --attention --grl --seed={seed} ")

        with open(filename, "w") as file:
            _ = file.write(content)

        _ = os.system(f"sbatch {filename}")

        time.sleep(1)


def resume_all_domains(beta, shots):
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    for seed in range(1, 3):
        old_exp_name_seed = f"alldomains_{beta}_{shots}_{seed}"
        best_model_path = old_exp_name_seed + "/" + os.listdir(f"{folder}/runs/{old_exp_name_seed}")[0] + "/best_model.pth"
        
        exp_name_seed = old_exp_name_seed + "_resume"
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
        f"--dataset_root=/home/francescom/adageo-WACV2021/src/datasets/svox/images " +
        f"--test_g=test/gallery --test_q=test/queries_night " 
        f"--train_q=train/queries_night_pseudo_{shots}_{beta}+train/queries_overcast_pseudo_{shots}_{beta}+" +
        f"train/queries_rain_pseudo_{shots}_{beta}+train/queries_snow_pseudo_{shots}_{beta}+train/queries_sun_pseudo_{shots}_{beta} " +
        f"--grl_datasets=train/queries+train/queries_night_pseudo_{shots}_{beta}+" +
        f"train/queries_overcast_pseudo_{shots}_{beta}+train/queries_rain_pseudo_{shots}_{beta}+" +
        f"train/queries_snow_pseudo_{shots}_{beta}+train/queries_sun_pseudo_{shots}_{beta}+" +
        f"test/queries_night+test/queries_overcast+test/queries_rain+test/queries_snow+test/queries_sun " +
        f"--val_q=val/queries_night_pseudo_{shots}_{beta}+val/queries_overcast_pseudo_{shots}_{beta}+"+
        f"val/queries_rain_pseudo_{shots}_{beta}+val/queries_snow_pseudo_{shots}_{beta}+val/queries_sun_pseudo_{shots}_{beta} " +
        f"--exp_name={exp_name_seed} --attention --grl --seed={seed} --resume={folder}/runs/{best_model_path}")

        with open(filename, "w") as file:
            _ = file.write(content)

        _ = os.system(f"sbatch {filename}")

        time.sleep(1)


def main():
    args = parse_arguments()
    job_type = args.job_type

    if job_type == "beta":
        beta()
    elif job_type == "fda":
        fda()
    elif job_type == "all_domains":
        all_domains(args.beta, args.shots)
    elif job_type == "resume_all_domains":
        resume_all_domains(args.beta, args.shots)
    elif job_type == "msls":
        msls_inference(args.city)
    else:
        print("Unexisting script: please provide a valid value")

if __name__ == "__main__":
    main()