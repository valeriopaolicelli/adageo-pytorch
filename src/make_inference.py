import logging
import sys
import os
import torch
import torch.nn as nn

import parser
import util
import datasets
import commons
import test

######################################### SETUP #########################################
args = parser.parse_arguments()
args.output_folder = f"runs/{args.exp_name}/{args.model_folder}"

######################################### LOGGING #########################################
info_filename="info.log"
debug_filename="debug.log"
console="debug"
base_formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)

if info_filename != None:
    info_file_handler = logging.FileHandler(f"{args.output_folder}/{info_filename}")
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(base_formatter)
    logger.addHandler(info_file_handler)

if debug_filename != None:
    debug_file_handler = logging.FileHandler(f"{args.output_folder}/{debug_filename}")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(base_formatter)
    logger.addHandler(debug_file_handler)

if console != None:
    console_handler = logging.StreamHandler()
    if console == "debug": console_handler.setLevel(logging.DEBUG)
    if console == "info":  console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(base_formatter)
    logger.addHandler(console_handler)

def my_handler(type_, value, tb):
    logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
sys.excepthook = my_handler

######################################### MODEL #########################################
model = util.build_model(args)

######################################### RESUME #########################################

test_queries = sorted(os.listdir(args.dataset_root + "/test"))[2:]
for test_q in test_queries:
    # ######################################### DATASETS #########################################
    whole_test_set = datasets.WholeDataset(args.dataset_root, args.test_g, "test/" + test_q)
    logging.info(f"Test set: {whole_test_set}")

    # ######################################### TEST on TEST SET #########################################
    best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")["state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str  = test.test(args, whole_test_set, model)
    logging.info(f"Recalls on {whole_test_set}: {recalls_str}")
