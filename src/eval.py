
from datetime import datetime
import torch
import logging

import parser
import util
import datasets
import commons
import test


######################################### SETUP #########################################
args = parser.parse_arguments()
args.output_folder = f"runs/{args.exp_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(args.output_folder)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
dataset = "svox"
if args.exp_name.find("msls") != -1:
    dataset = "msls"
elif args.exp_name.find("st_lucia") != -1:
    dataset = "st_lucia"

assert args.resume != None, "resume is set to None, please set it to the path of the checkpoint to resume"

######################################### MODEL #########################################
model = util.build_model(args)

######################################### RESUME #########################################
model_state_dict = torch.load(args.resume, map_location=torch.device('cpu'))["state_dict"]
model.load_state_dict(model_state_dict, strict=False)

######################################### DATASETS #########################################
whole_test_set = datasets.WholeDataset(args.dataset_root, args.test_g, args.test_q, dataset=dataset)
logging.info(f"Test set: {whole_test_set}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str  = test.test(args, whole_test_set, model)
logging.info(f"Recalls on {whole_test_set}: {recalls_str}")

