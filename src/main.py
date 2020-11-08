
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

import parser
import util
import datasets
import commons
import test
import train
import grl_util


######################################### SETUP #########################################
opt = parser.parse_arguments()
start_time = datetime.now()
opt.output_folder = os.path.join("runs", opt.exp_name, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
commons.setup_logging(opt.output_folder)
commons.make_deterministic(opt.seed)
logging.info(f"Arguments: {opt}")
logging.info(f"The outputs are being saved in {opt.output_folder}")
opt.root_path = os.path.join(opt.all_datasets_path, opt.root_path)

######################################### MODEL #########################################
model = util.build_model(opt)

######################################### OPTIMIZER & LOSSES #########################################
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
criterion_netvlad = nn.TripletMarginLoss(margin=opt.margin ** 0.5, p=2, reduction="sum")

######################################### RESUME #########################################
if opt.resume:
    model, optimizer, best_score, start_epoch = util.resume_train(opt, model, optimizer)
else:
    start_epoch = 0

######################################### DATASETS #########################################
logging.debug(f"Loading dataset(s) {opt.root_path}")

query_train_set = datasets.QueryDataset(opt.root_path, opt.train_g, opt.train_q, opt.output_folder)
logging.info(f"Train query set: {query_train_set.info}")

whole_train_set = datasets.WholeDataset(opt.root_path, opt.train_g, opt.train_q)
logging.info(f"Train whole set: {whole_train_set.info}")

whole_val_set = datasets.WholeDataset(opt.root_path, opt.val_g, opt.val_q)
logging.info(f"Val set: {whole_val_set.info}")

whole_test_set = datasets.WholeDataset(opt.root_path, opt.test_g, opt.test_q)
logging.info(f"Test set: {whole_test_set.info}")

if opt.grl:
    grl_dataset = grl_util.GrlDataset(opt.root_path, opt.grl_datasets.split("+"))
else:
    grl_dataset = None

######################################### TRAINING #########################################
best_score = 0
not_improved = 0
for epoch in range(start_epoch, opt.n_epochs):
    
    # Start each epoch with validation
    recalls, recalls_str = test.test(opt, whole_val_set, model)
    logging.info(f"Recalls on val set {whole_val_set.info}: {recalls_str}")
    
    if recalls[5] > best_score:
        logging.info(f"Improved: previous best recall@5 = {best_score * 100:.1f}, current best recall@5 = {recalls[5] * 100:.1f}")
        is_best = True
        best_score = recalls[5]
        not_improved = 0
    else:
        is_best = False
        if not_improved >= opt.patience:
            logging.info(f"Performance did not improve for {not_improved} epochs. Stop training.")
            break
        not_improved += 1
        logging.info(f"Not improved: {not_improved} / {opt.patience}")
    
    util.save_checkpoint(opt, {"epoch": epoch, "state_dict": model.state_dict(),
        "recalls": recalls, "best_score": best_score, "optimizer": optimizer.state_dict(),
    }, is_best, filename=f"model_{epoch:02d}")
    
    logging.info(f"Start training epoch: {epoch:02d}")
    
    train.train(opt, epoch, model, optimizer, criterion_netvlad, 
                whole_train_set, query_train_set, grl_dataset)


logging.info(f"Best recall@5: {best_score * 100:.1f}")
logging.info(f"Trained for {epoch:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

######################################### TEST on TEST SET #########################################
best_model_state_dict = torch.load(os.path.join(opt.output_folder, "best_model.pth"))["state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str  = test.test(opt, whole_test_set, model)
logging.info(f"Recalls on {whole_test_set.info}: {recalls_str}")

