"""
Code to extract scores from info.logs, computing average and standard deviation over all seeds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_FOLDER = "/home/francescom/adageo-WACV2021/src/runs"
FDA_OUTPUT_PATH = BASE_FOLDER + "/fda/beta_tuning.json"
MSLS_OUTPUT_PATH = BASE_FOLDER + "/msls/msls_scores.json"
LUCIA_OUTPUT_PATH = BASE_FOLDER + "/st_lucia/st_lucia_scores.json"


def score_from_line(line):
    score_start_idx = line.index("R@1") + len("@R1: ")
    score_end_idx = line.index(",", score_start_idx)
    score = line[score_start_idx:score_end_idx]

    return float(score)
    

def scores_from_folder(target, seed, beta = None):
    """
    Args:
    target (str): inference domain(fda) or city(msls)
    seed (int): seed of the run
    beta (float): beta used for FDA style transfer. Set only for fda

    Return: 
    inference_scores (dict): key value pairs where key = inference_target, value = R@1 score
    """

    inference_scores = []
    if beta is not None:
        folder = os.path.join(BASE_FOLDER, f"{target}_{seed}_{beta}_5")
    else:
        folder = os.path.join(BASE_FOLDER, f"{target}_{seed}")
    
    logs_path = os.path.join(folder, os.listdir(folder)[0], "info.log")

    with open(logs_path, "r") as f:
        for line in f:
            if "Recalls on < WholeDataset" in line:
                score = score_from_line(line)
                inference_scores.append(score)

    return inference_scores


def mean_std(target, beta=None):
    """
    Args:
    target (str): inference domain or city
    beta (float): beta used for FDA style transfer

    Return: 
    results (dict): key value pairs where key = inference_target, value = mean(R@1 scores), std(R@1 scores) over different seeds
    """
    seed_scores = []
    for seed in range(3):
        inference_scores = scores_from_folder(target, seed, beta)
        seed_scores.append(inference_scores)

    seed_scores = np.array(seed_scores)
    print(seed_scores)
    mean_std_scores = (np.mean(np.squeeze(seed_scores), axis=0), np.std(np.squeeze(seed_scores), axis=0))

    if beta is not None:
        results = dict.fromkeys(TARGETS, [])
        for i, target in enumerate(TARGETS):
            results[target] = (round(mean_std_scores[0][i], 2) , round(mean_std_scores[1][i], 2))
        return results
    else:
        return (round(mean_std_scores[0], 2) , round(mean_std_scores[1], 2))


def fda():
    scores_dict = dict()
    for target in TARGETS:
        for beta in ("0.01", "0.001", "0.005"):
            scores_dict[f"{target} - {beta}"] = mean_std(target, beta)

    json_dict = json.dumps(scores_dict, indent = 4)
    with open(FDA_OUTPUT_PATH, "w+") as f:
        f.writelines(json_dict)


def msls():
    scores_dict = dict()
    for target in TARGETS:
        print(TARGETS)
        scores_dict[target] = mean_std(target)

    json_dict = json.dumps(scores_dict, indent=4)
    with open(MSLS_OUTPUT_PATH, "w+") as f:
        f.writelines(json_dict)


def st_lucia():
    msls()


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
        BASE_FOLDER += "/msls"
        TARGETS = ("nairobi", "sf", "saopaulo", "cph", "tokyo")
        msls()

        with open(MSLS_OUTPUT_PATH, "r") as f:
            scores_dict = json.load(f)

        pd.DataFrame.from_dict(scores_dict, orient='columns').to_csv(BASE_FOLDER + "/msls_scores.csv")

    elif task == "st_lucia":
        BASE_FOLDER += "/st_lucia"
        TARGETS = ("st_lucia", )
        st_lucia()

        with open(MSLS_OUTPUT_PATH, "r") as f:
            scores_dict = json.load(f)

        pd.DataFrame.from_dict(scores_dict, orient='columns').to_csv(BASE_FOLDER + "/st_lucia.csv")

    else:
        my_list = ("arcibaldo")
        for el in my_list:
            print(el)

