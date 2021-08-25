
import os
import sys
import time
import argparse
from glob import glob


def parse_arguments():
    parser = argparse.ArgumentParser(description="Jobs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--job", type=str, default=None, 
        help="Job type: accepted values are (beta, fda, inference, all_domains)"
    )
    parser.add_argument("--beta", type=float, default=0.001, help = "Beta hyperparameter")
    parser.add_argument("--shots", type=int, default=5, help="Number of shots for domain adaptation")

    return parser.parse_args()

def scancel():
    for i in range(104814, 104979):
        _ = os.system(f"scancel {i}")


def inference_content(exp_name, exp_dir, folder, dataset_root, model_path, seed, grl = None):
    """
    Content of the .job file for msls inference
    Args:
        exp_name (str): name of the experiment (seen in squeue)
        exp_dir (str): name of the folder where experiment results are stored
        folder (str): directory with the eval.py file
        dataset_root (str): directory with the data
        model path (str): directory with best_model.pth
        seed (int): training seed
        grl (list): list of grl datasets. These are just placeholders in inference 
    """
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
            f"python {folder}/eval.py " + 
            f"--seed={seed} --dataset_root={dataset_root} " +
            f"--test_g=gallery --test_q=queries " +
            f"--attention --exp_name={exp_dir} --resume={model_path} ")

    if grl is not None:
        content += f"--grl_dataset={grl} --grl"

    return (content)


def launch(content, filename):
    """
    Lanuch a job
    """
    with open(filename, "w") as file:
        _ = file.write(content)

    _ = os.system(f"sbatch {filename}")
    time.sleep(1)


def msls_inference():
    # Run models trained on all seeds
    folder = "/home/francescom/adageo-WACV2021/src"
    if not os.path.abspath(os.curdir) == folder:
        sys.exit()

    # Create jobs and ou_job folders if not present
    os.makedirs(f"{folder}/jobs", exist_ok=True)
    os.makedirs(f"{folder}/out_job", exist_ok=True)

    # Match GRL dimensionality
    grl = '+'.join(["a" for _ in range(11)])

    cities = ["nairobi", "cph", "sf", "tokyo", "saopaulo"]
    for city in cities:
        for seed in range(3):
            # Set up arguments
            exp_name = f"{city}_{seed}"
            exp_dir= f"msls/{exp_name}"
            filename = f"{folder}/jobs/{exp_name}.job"
            logs_folder = folder +  f"/runs/fda/all_{seed}_0.001_5/"
            model_path = logs_folder + os.listdir(logs_folder)[0] + "/best_model.pth"
            dataset_root = f"/home/francescom/adageo-WACV2021/src/datasets/msls/{city}"

            # .job file content
            content = inference_content(exp_name, exp_dir, folder, dataset_root, model_path, seed, grl)
            # Launch the job
            launch(content, filename)


def st_lucia_inference():
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
        # Set up arguments
        exp_name = f"st_lucia_{seed}"
        exp_dir = f"st_lucia/{exp_name}"
        filename = f"{folder}/jobs/{exp_name}.job"
        logs_folder = folder +  f"/runs/fda/all_{seed}_0.001_5/"
        model_path = logs_folder + os.listdir(logs_folder)[0] + "/best_model.pth"
        dataset_root = "/home/francescom/adageo-WACV2021/src/datasets/st_lucia/images"

        # .job file content
        content = inference_content(exp_name, exp_dir, folder, dataset_root, model_path, seed, grl)
        # Launch the job
        launch(content, filename)



def baselines_inference():
    folder = "/home/francescom/adageo-WACV2021/src"
    base_dir = "/home/gabriele/wacv/all_complete_runs/resnet/results"
    dirs = sorted(glob(f"{base_dir}/*few*") + glob(f"{base_dir}/afn*") + 
        glob(f"{base_dir}/baseline") + glob(f"{base_dir}/coral[1-5]_w*"))

    cities = ["nairobi", "cph", "sf", "tokyo", "saopaulo"]
    for dir in dirs:
        seeds_dirs = sorted(glob(dir + "/*"))
        for seed, seed_dir in enumerate(seeds_dirs):
            model_name = seed_dir.split("/")[-2]
            try:
                model_path = glob(seed_dir + "/best_model.pth")[0]
            except IndexError:
                # There is no best_model.pth in the model folder
                continue
            os.makedirs(folder + f"/runs/baselines/msls/{model_name}", exist_ok =True) # check
            os.makedirs(folder + f"/runs/baselines/st_lucia/{model_name}", exist_ok =True) # check

            # MSLS
            for city in cities:
                exp_name = f"{city}_{model_name}_{seed}"
                exp_dir = f"baselines/msls/{model_name}/{city}_{seed}"
                filename = f"{folder}/jobs/{exp_name}.job"
                dataset_root = f"/home/francescom/adageo-WACV2021/src/datasets/msls/{city}"

                # .job file content
                content = inference_content(exp_name, exp_dir, folder,
                 dataset_root, model_path, seed)

                # Launch the job
                launch(content, filename)
            
            # ST_LUCIA
            exp_name = f"st_lucia_{model_name}_{seed}"
            exp_dir = f"baselines/st_lucia/{model_name}/st_lucia_{seed}"
            filename = f"{folder}/jobs/{exp_name}.job"
            dataset_root = f"/home/francescom/adageo-WACV2021/src/datasets/st_lucia/images/test"

            # .job file content
            content = inference_content(exp_name, exp_dir, folder,
                dataset_root, model_path, seed)

            # Launch the job
            launch(content, filename)


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
                
                    launch(content, filename)


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

                launch(content, filename)


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

        launch(content, filename)
        
        
def main():
    args = parse_arguments()
    job = args.job

    if job == "beta":
        beta()
    elif job == "fda":
        fda()
    elif job == "all_domains":
        all_domains(args.beta, args.shots)
    elif job == "msls":
        msls_inference()
    elif job == "st_lucia":
        st_lucia_inference()
    elif job == "baselines":
        baselines_inference()
    elif job == "scancel":
        scancel()
    else:
        print("Unexisting script: please provide a valid value")

if __name__ == "__main__":
    main()