
import os
from datetime import datetime
import torch
import logging

import parser
import util
import datasets
import commons
import test


######################################### SETUP #########################################
opt = parser.parse_arguments()
opt.output_folder = os.path.join("runs", opt.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
commons.setup_logging(opt.output_folder)
logging.info(f"Arguments: {opt}")
logging.info(f"The outputs are being saved in {opt.output_folder}")
opt.root_path = os.path.join(opt.all_datasets_path, opt.root_path)

assert opt.resume != None, "resume is set to None, please set it to the path of the checkpoint to resume"

######################################### MODEL #########################################
model = util.build_model(opt)

######################################### RESUME #########################################
model_state_dict = torch.load(opt.resume)["state_dict"]
model.load_state_dict(model_state_dict, strict=False)

######################################### DATASETS #########################################
whole_test_set = datasets.WholeDataset(opt.root_path, opt.test_g, opt.test_q)
logging.info(f"Test set: {whole_test_set.info}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str  = test.test(opt, whole_test_set, model)
logging.info(f"Recalls on {whole_test_set.info}: {recalls_str}")

