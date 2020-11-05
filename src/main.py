
import os
from os.path import join
from datetime import datetime
import argparse
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
opt.output_folder = os.path.join(opt.output_path, opt.exp_name, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
commons.setup_logging(opt.output_folder)
commons.make_deterministic(opt.seed)
logging.info(f"Arguments: {opt}")
opt.root_path = os.path.join(opt.all_datasets_path, opt.root_path)
DA_dict = {}


if opt.is_debug:
    logging.info("!!! Questa è solo una prova (alcuni cicli for vengono interrotti dopo 1 iterazione), i risultati non sono attendibili !!!\n")

######################################### MODEL #########################################
logging.debug(f"Building model")
model = util.build_model(opt)
model = model.to(opt.device)

######################################### OPTIM e LOSSES #########################################
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
criterion_netvlad = nn.TripletMarginLoss(margin=opt.margin ** 0.5,
                                         p=2, reduction="sum").to(opt.device)

######################################### RESUME #########################################
if opt.resume:
    opt, model, optimizer, best_score = util.resume_train(opt, model, optimizer)

######################################### DATASETS #########################################
logging.debug(f"Loading dataset(s) {opt.root_path}")

query_train_set = datasets.QueryDataset(opt.root_path, opt.train_g, opt.train_q)
logging.info(f"Train query set: {query_train_set.info}")

whole_train_set = datasets.WholeDataset(opt.root_path, opt.train_g, opt.train_q)
logging.info(f"Train whole set: {whole_train_set.info}")

whole_val_set = datasets.WholeDataset(opt.root_path, opt.val_g, opt.val_q)
logging.info(f"Val set: {whole_val_set.info}")

whole_test_set = datasets.WholeDataset(opt.root_path, opt.test_g, opt.test_q)
logging.info(f"Test set: {whole_test_set.info}")

if opt.grl:
    DA_dict["grl_dataset"] = grl_util.GrlDataset(opt.root_path, opt.grl_datasets.split("+"), opt.logger)

logging.debug(f"Training model")

logging.info(f"Eval before train")
recalls, _, recalls_str = test.test(opt, whole_val_set, model)
logging.info(f"Recalls on {whole_val_set.info}: {recalls_str}")

best_score = recalls[5]
not_improved = 0
for epoch in range(opt.start_epoch + 1, opt.n_epochs + 1):
    epoch_start_time = datetime.now()
    logging.info(f"Train epoch: {epoch:02d}")
    train_info = train.elaborate_epoch(opt, epoch, model, optimizer, criterion_netvlad, 
                                       whole_train_set, query_train_set, DA_dict)
    
    logging.debug(f"Eval NetVLAD")
    recalls, _, recalls_str = test.test(opt, whole_val_set, model)
    del _
    logging.info(f"    Recalls on {whole_val_set.info}: {recalls_str}")
    
    is_best = recalls[5] > best_score
    util.save_checkpoint(opt, {"epoch": epoch, "state_dict": model.state_dict(),
        "recalls": recalls, "best_score": best_score, "optimizer": optimizer.state_dict(),
    }, is_best, f"model_{epoch:02d}")
    train_info += f"Time epoch: {str(datetime.now() - epoch_start_time)[:-6]} - "
    if is_best:
        not_improved = 0
        best_score = recalls[5]
        train_info += "Improved"
    else:
        not_improved += 1
        train_info += f"Not Improved: {not_improved} / {opt.patience}"
    logging.info(train_info)
    if opt.patience > 0 and not_improved > (opt.patience):
        logging.info(f"Performance did not improve for {opt.patience} epochs. Stopping.")
        break

logging.info(f"Best Recall@5: {best_score:.4f}")
logging.info(f"Trained for {epoch:02d} epochs, in total in {str(datetime.now() - start_time)[:-6]}")

model_state_dict = torch.load(join(opt.output_folder, "best_model.pth"))["state_dict"]
model.load_state_dict(model_state_dict)
model = model.to(opt.device)

recalls = [1, 5, 10, 20]

_, previous_gallery_features, recalls_str  = test.test(opt, whole_test_set, model)
logging.info(f"Recalls on {whole_test_set.info}: {recalls_str}")

previous_gallery_features = None
all_targets_recall_str = ""
for i in range(5):
    target_test_set = datasets.WholeDataset(opt.root_path, "test/gallery", f"test/queries_{i+1}")
    _, previous_gallery_features, recalls_str = test.test(opt, target_test_set, model, previous_gallery_features)
    logging.info(f"Recalls on {target_test_set.info}: {recalls_str}")
    all_targets_recall_str += recalls_str

logging.info(f"Recalls all targets: {all_targets_recall_str}")

