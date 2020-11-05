
import os
from os.path import join
from datetime import datetime
import argparse
import torch

import parser
import util
import datasets
import commons
import test


######################################### SETUP #########################################
opt = parser.parse_arguments()
opt.output_folder = os.path.join(opt.output_path, opt.exp_name, datetime.now().strftime("%b%d_%H-%M-%S"))
commons.setup_logging(opt.output_folder)
logging.info(f"Arguments: {opt}")
opt.root_path = os.path.join(opt.all_datasets_path, opt.root_path)

if opt.is_debug:
    logging.info("!!! Questa è solo una prova (alcuni cicli for vengono interrotti dopo 1 iterazione), i risultati non sono attendibili !!!\n")

######################################### MODEL #########################################
logging.debug(f"=> Building model")
model = util.build_model(opt)

######################################### RESUME #########################################
best_score = 0
epoch = 0
if opt.resume:
    model_state_dict = torch.load(join(opt.resume, "best_model.pth"))["state_dict"]
    model.load_state_dict(model_state_dict, strict=False)
    epoch = torch.load(join(opt.resume, "best_model.pth"))["epoch"]
    best_score = torch.load(join(opt.resume, "best_model.pth"))["best_score"]

model = model.to(opt.device)

######################################### DATASETS #########################################
logging.info(f"=> Evaluating model - Epoch: {epoch} - Best Recall@5: {best_score:.1f}")

recalls = [1, 5, 10, 20]

all_targets_recall_str = ""

if opt.scenario == 0:    
    source_test_set = datasets.WholeDataset(opt.root_path, "val/gallery", f"val/queries")
    _, _, recalls_str  = test.test(opt, source_test_set, model)
    del _
    all_targets_recall_str += recalls_str
    logging.info(f"Recalls on {source_test_set.name}: {recalls_str}")
    
    source_test_set = datasets.WholeDataset(opt.root_path, "test/gallery", f"test/queries")
    _, previous_db_features, recalls_str  = test.test(opt, source_test_set, model)
    all_targets_recall_str += recalls_str
    logging.info(f"Recalls on {source_test_set.name}: {recalls_str}")
    
    for i in range(5):
        target_test_set = datasets.WholeDataset(opt.root_path, "test/gallery", f"test/queries_{i+1}")
        _, previous_db_features, recalls_str = test.test(opt, target_test_set, model, previous_db_features)
        logging.info(f"Recalls on {target_test_set.name}: {recalls_str}")
        all_targets_recall_str += recalls_str
    
else:
    target_test_set = datasets.WholeDataset(opt.root_path, "test/gallery", f"test/queries_{opt.scenario}")
    _, _, recalls_str = test.test(opt, target_test_set, model)
    del _
    logging.info(f"Recalls on {target_test_set.name}: {recalls_str}")
    all_targets_recall_str += recalls_str

logging.info(f"Recalls all targets: {all_targets_recall_str}")

