import os
import numpy as np
import pandas as pd
import json


# NOTE: Code to extract scores from info.logs, computing average and standard deviation over all seeds.

BASE_FOLDER = "/home/francescom/adageo-WACV2021/src/runs"
DOMAINS = ("night", "overcast", "rain", "sun", "snow")
OUTPUT_PATH = os.path.join(BASE_FOLDER, "beta_tuning.json")

def scores_from_folder(domain, seed, beta):
    """
    Args:
    domain (str): inference domain
    seed (int): seed of the run
    beta (float): beta used for FDA style transfer

    Return: 
    inference_scores (dict): key value pairs where key = inference_domain, value = R@1 score
    """

    inference_scores = []
    folder = os.path.join(BASE_FOLDER, f"{domain}_{seed}_{beta}_5")
    logs_path = os.path.join(folder, os.listdir(folder)[0], "info.log")

    with open(logs_path, "r") as f:
        for line in f:
            if "Recalls on < WholeDataset" in line:
                domain_start_idx = line.index("test/queries_") + len("test/queries_")
                domain_end_idx = line.index(" ", domain_start_idx)
                domain = line[domain_start_idx:domain_end_idx]

                score_start_idx = line.index("R@1") + len("@R1: ")
                score_end_idx = line.index(",", score_start_idx)
                score = line[score_start_idx:score_end_idx]
                inference_scores.append(float(score))

    return inference_scores


def mean_std(domain, beta):
    """
    Args:
    domain (str): inference domain
    beta (float): beta used for FDA style transfer

    Return: 
    results (dict): key value pairs where key = inference_domain, value = mean(R@1 scores), std(R@1 scores) over different seeds
    """
    results = dict.fromkeys(DOMAINS, [])
    seed_scores = []
    for seed in range(3):
        inference_scores = scores_from_folder(domain, seed, beta)
        seed_scores.append(inference_scores)

    seed_scores = np.array(seed_scores)
    mean_std_scores = (np.mean(seed_scores, axis=0), np.std(seed_scores, axis=0))
    for i, domain in enumerate(DOMAINS):
        results[domain] = (round(mean_std_scores[0][i], 2) , round(mean_std_scores[1][i], 2))

    return results

def main():
    scores_dict = dict()
    for domain in DOMAINS:
        for beta in ("0.01", "0.001", "0.005"):
            scores_dict[f"{domain} - {beta}"] = mean_std(domain, beta)

    json_dict = json.dumps(scores_dict, indent = 4)
    with open(OUTPUT_PATH, "w+") as f:
        f.writelines(json_dict)


if __name__ == "__main__":
    main()
    with open("beta_tuning.json", "r") as f:
        scores_dict = json.load(f)

    pd.DataFrame.from_dict(scores_dict, orient='index').to_csv("beta_tuning.csv")