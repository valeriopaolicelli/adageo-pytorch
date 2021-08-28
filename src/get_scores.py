"""
Code to extract scores from info.logs, computing average and standard deviation over all seeds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from glob import glob

BASE_FOLDER = "/home/francescom/adageo-WACV2021/src/runs"
FDA_OUTPUT_PATH = BASE_FOLDER + "/fda/beta_tuning.json"

def score_from_line(line):
    score_start_idx = line.index("R@1") + len("@R1: ")
    score_end_idx = line.index(",", score_start_idx)
    score = line[score_start_idx:score_end_idx]

    return float(score)

    
def scores_from_folder(folder):
    """
    Args:
    folder (str): path to the folder from which to extract the scores

    Return: 
    inference_scores (str): R@1 score
    """
    logs_path = os.path.join(folder, os.listdir(folder)[0], "info.log")
    inference_score = None
    with open(logs_path, "r") as f:
        for line in f:
            if "Recalls on < WholeDataset" in line:
                score = score_from_line(line)
                inference_score = score

    # If None, than there was an error
    return inference_score


def mean_std(models_folder, beta=None):
    """
    Args:
    models_folder (list): list of the folders over to which make average.
                          These folders refer to same target (e.g. city), and different seed
    beta (float): set if extracting scores for fda()

    Return: 
    (mean, std) (tuple): mean and standard deviation over seeds
    """
    seed_scores = []
    for model_folder in models_folder:
        inference_score = scores_from_folder(model_folder)
        assert inference_score is not None, f"Inference score is None for {model_folder}"
        seed_scores.append(inference_score)

    seed_scores = np.array(seed_scores)
    mean_std_scores = (np.mean(np.squeeze(seed_scores), axis=0), np.std(np.squeeze(seed_scores), axis=0))

    if beta is not None:
        results = dict.fromkeys(TARGETS, [])
        for i, target in enumerate(TARGETS):
            results[target] = (round(mean_std_scores[0][i], 2) , round(mean_std_scores[1][i], 2))
        return results
    else:
        return (round(mean_std_scores[0], 2) , round(mean_std_scores[1], 2))


def baselines_scores():
    base_folder = "/home/francescom/adageo-WACV2021/src/runs/baselines"
    output_file = base_folder + "/logger.txt"
    msls_cities = ["cph", "nairobi", "tokyo", "sf", "saopaulo"]
    
    # Initialize the logger with columns
    with open(output_file, "w+") as f:
        f.write("MODEL\t")
        for city in msls_cities:
            f.write(city + "\t")
        f.write("st_lucia\n")

        # Get scores
        architectures = sorted(os.listdir(base_folder + "/msls/"))
        for architecture in architectures: # e.g. coral1_w0.1
            if "coral" in architecture:
                continue
            arch_folder = base_folder + f"/msls/{architecture}"
            f.write(architecture + "\t")
            for city in msls_cities:
                mean, std = mean_std(sorted(glob(arch_folder + f"/*{city}*")))
                f.write(f"{mean}+-{std}" + "\t")

            # ST_LUCIA
            arch_folder = base_folder + f"/st_lucia/{architecture}"
            mean, std = mean_std(sorted(glob(arch_folder + "/st_lucia*")))
            f.write(f"{mean}+-{std}" + "\t")
            f.write("\n")


def fda():
    base_folder = "/home/francescom/adageo-WACV2021/src/runs/fda"
    output_file = base_folder + "/logger.txt"
    targets = ("all", "night", "overcast", "rain", "sun", "snow")
    betas = ("0.01", "0.001", "0.005")

    # Initialize the logger with columns
    with open(output_file, "w+") as f:
        f.write("TARGET_BETA\t")
        for target in targets[1]:
            f.write(target + "\t")
        f.write("\n")

        for beta in betas:
            f.write(beta + "\t")
            for target in targets:
                if sorted(glob(base_folder + f"/{target}*{beta}_5")) == []:
                    f.write("None\t")
                    continue
                mean, std = mean_std(sorted(glob(base_folder + f"/{target}*{beta}_5")))
                f.write(f"{mean}+-{std}" + "\t")
            f.write("\n")
        f.write("\n")


def msls():
    base_folder = "/home/francescom/adageo-WACV2021/src/runs/msls"
    output_file = base_folder + "/logger.txt"
    msls_cities = ["cph", "nairobi", "tokyo", "sf", "saopaulo"]

    # Initialize the logger with columns
    with open(output_file, "w+") as f:
        for city in msls_cities:
            f.write(city + "\t")
        f.write("\n")

        for city in msls_cities:
            mean, std = mean_std(sorted(glob(base_folder + f"/*{city}*")))
            f.write(f"{mean}+-{std}" + "\t")
        f.write("\n")


def st_lucia():
    base_folder = "/home/francescom/adageo-WACV2021/src/runs/st_lucia"
    output_file = base_folder + "/logger.txt"

    # Initialize the logger with columns
    with open(output_file, "w+") as f:
        f.write("st_lucia\n")
        mean, std = mean_std(sorted(glob(base_folder + f"/st_lucia*")))
        f.write(f"{mean}+-{std}" + "\t")
        f.write("\n")


if __name__ == "__main__":
    task = sys.argv[1]
    if task == "fda":
        BASE_FOLDER += "/fda"
        TARGETS = ("night", "overcast", "rain", "sun", "snow")
        fda()
        with open(FDA_OUTPUT_PATH, "r") as f:
            scores_dict = json.load(f)

        # pd.DataFrame.from_dict(scores_dict, orient='index').to_csv("beta_tuning.csv")

    elif task == "msls":
        msls()

    elif task == "st_lucia":
        st_lucia()

    elif task == "baselines":
        baselines_scores()

