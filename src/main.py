
import logging
from datetime import datetime
import torch
import torch.nn as nn

import parser
import util
import datasets
import commons
import test
import train
import grl_util


######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"runs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

######################################### MODEL #########################################
model = util.build_model(args)

######################################### OPTIMIZER & LOSSES #########################################
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion_netvlad = nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction="sum")

######################################### RESUME #########################################
if args.resume:
    model, optimizer, best_score, start_epoch = util.resume_train(args, model, optimizer)
else:
    start_epoch = 0

######################################### DATASETS #########################################
logging.debug(f"Loading datasets from folder {args.dataset_root}")

query_train_set = datasets.QueryDataset(args.dataset_root, args.train_g, args.train_q, args.output_folder)
logging.info(f"Train query set: {query_train_set}")

whole_train_set = datasets.WholeDataset(args.dataset_root, args.train_g, args.train_q)
logging.info(f"Train whole set: {whole_train_set}")

whole_val_set = datasets.WholeDataset(args.dataset_root, args.val_g, args.val_q)
logging.info(f"Val set: {whole_val_set}")

whole_test_set = datasets.WholeDataset(args.dataset_root, args.test_g, args.test_q)
logging.info(f"Test set: {whole_test_set}")

if args.grl:
    grl_dataset = grl_util.GrlDataset(args.dataset_root, args.grl_datasets.split("+"))
else:
    grl_dataset = None

######################################### TRAINING #########################################
best_score = 0
not_improved = 0
for epoch in range(start_epoch, args.n_epochs):
    
    # Start each epoch with validation
    recalls, recalls_str = test.test(args, whole_val_set, model)
    logging.info(f"Recalls on val set {whole_val_set}: {recalls_str}")
    
    if recalls[5] > best_score:
        logging.info(f"Improved: previous best recall@5 = {best_score * 100:.1f}, current recall@5 = {recalls[5] * 100:.1f}")
        is_best = True
        best_score = recalls[5]
        not_improved = 0
    else:
        is_best = False
        if not_improved >= args.patience:
            logging.info(f"Performance did not improve for {not_improved} epochs. Stop training.")
            break
        not_improved += 1
        logging.info(f"Not improved: {not_improved} / {args.patience}: best recall@5 = {best_score * 100:.1f}, current recall@5 = {recalls[5] * 100:.1f}")
    
    util.save_checkpoint(args, {"epoch": epoch, "state_dict": model.state_dict(),
        "recalls": recalls, "best_score": best_score, "optimizer": optimizer.state_dict(),
    }, is_best, filename=f"model_{epoch:02d}")
    
    logging.info(f"Start training epoch: {epoch:02d}")
    
    train.train(args, epoch, model, optimizer, criterion_netvlad, 
                whole_train_set, query_train_set, grl_dataset)


logging.info(f"Best recall@5: {best_score * 100:.1f}")
logging.info(f"Trained for {epoch:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

######################################### TEST on TEST SET #########################################
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")["state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str  = test.test(args, whole_test_set, model)
logging.info(f"Recalls on {whole_test_set}: {recalls_str}")

