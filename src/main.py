
import const
import util
import datasets
import commons
import test
import train
import grl_util

import os
from os.path import join
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


######################################### SETUP #########################################
parser = argparse.ArgumentParser(description='pytorch-NetVlad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
const.add_arguments(parser)
opt = parser.parse_args()
start_time = datetime.now()

opt.outputFolder = os.path.join(const.runsPath, opt.expName, datetime.now().strftime('%b%d_%H-%M-%S'))

opt.logger = commons.Logger(folder=opt.outputFolder, filename=f"logger.txt")
opt.logger.log(f'Arguments: {opt}')
opt.rootPath = os.path.join(opt.allDatasetsPath, opt.rootPath)
opt.cuda = True
opt.device = "cuda"
DA_dict = {}

commons.pause_while_running(opt.wait)
commons.make_deterministic(opt.seed)

if opt.isDebug:
    opt.logger.log("!!! Questa è solo una prova (alcuni cicli for vengono interrotti dopo 1 iterazione), i risultati non sono attendibili !!!\n")

######################################### MODEL #########################################
opt.logger.log(f"Building model", False)
model = util.build_model(opt)
model = model.to(opt.device)

######################################### OPTIM e LOSSES #########################################
optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                       model.parameters()), lr=opt.lr)
criterion_netvlad = nn.TripletMarginLoss(margin=const.margin ** 0.5,
                                         p=2, reduction='sum').to(opt.device)

######################################### RESUME #########################################
if opt.resume:
    opt, model, optimizer, best_score = util.resume_train(opt, model, optimizer)

######################################### DATASETS #########################################
opt.logger.log(f"Loading dataset(s) {opt.rootPath}", False)

query_train_set = datasets.QueryDataset(opt.rootPath, opt.trainG, opt.trainQ)
opt.logger.log(f"Train query set: {query_train_set.name}")

whole_train_set = datasets.WholeDataset(opt.rootPath, opt.trainG, opt.trainQ)
opt.logger.log(f"Train whole set: {query_train_set.name}")

whole_val_set = datasets.WholeDataset(opt.rootPath, opt.valG, opt.valQ)
opt.logger.log(f"Val set: {whole_val_set.name}")

whole_test_set = datasets.WholeDataset(opt.rootPath, opt.testG, opt.testQ)
opt.logger.log(f"Test set: {whole_test_set.name}")

if opt.grl:
    DA_dict["grl_dataset"] = grl_util.GrlDataset(opt.rootPath, opt.grlDatasets.split("+"), opt.logger)

opt.logger.log(f"Training model", False)

opt.logger.log(f"Eval before train")
_, _, recalls_str = test.test(opt, whole_test_set, model)
opt.logger.log(f"Recalls on {whole_test_set.name}: {recalls_str}")
recalls, _, recalls_str = test.test(opt, whole_val_set, model)
opt.logger.log(f"Recalls on {whole_val_set.name}: {recalls_str}")

best_score = recalls[5]
not_improved = 0
for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
    epoch_start_time = datetime.now()
    opt.logger.log(f"Train epoch: {epoch:02d}")
    train_info = train.elaborate_epoch(opt, epoch, model, optimizer, criterion_netvlad, 
                                       whole_train_set, query_train_set, DA_dict)
    
    opt.logger.log(f"Eval NetVLAD", False)
    _, _, recalls_str = test.test(opt, whole_test_set, model)
    opt.logger.log(f"    Recalls on {whole_test_set.name}: {recalls_str}")
    recalls, _, recalls_str = test.test(opt, whole_val_set, model)
    del _
    opt.logger.log(f"    Recalls on {whole_val_set.name}: {recalls_str}")
    
    is_best = recalls[5] > best_score
    util.save_checkpoint(opt, {'epoch': epoch, 'state_dict': model.state_dict(),
        'recalls': recalls, 'best_score': best_score, 'optimizer': optimizer.state_dict(),
    }, is_best, f"model_{epoch:02d}")
    train_info += f"Time epoch: {str(datetime.now() - epoch_start_time)[:-6]} - "
    if is_best:
        not_improved = 0
        best_score = recalls[5]
        train_info += "Improved"
    else:
        not_improved += 1
        train_info += f"Not Improved: {not_improved} / {opt.patience}"
    opt.logger.log(train_info)
    if opt.patience > 0 and not_improved > (opt.patience):
        opt.logger.log(f"Performance did not improve for {opt.patience} epochs. Stopping.")
        break

opt.logger.log(f"Best Recall@5: {best_score:.4f}")
opt.logger.log(f"Trained for {epoch:02d} epochs, in total in {str(datetime.now() - start_time)[:-6]}")

model_state_dict = torch.load(join(opt.outputFolder, 'best_model.pth'))['state_dict']
model.load_state_dict(model_state_dict)
model = model.to(opt.device)

final_logger = commons.Logger(folder=opt.outputFolder, filename=f"final_logger.txt")
final_logger.log(f'Arguments: {opt}\n')
recalls = [1, 5, 10, 20]

source_test_set = datasets.WholeDataset(opt.rootPath, "val/gallery", f"val/queries")
_, _, recalls_str  = test.test(opt, source_test_set, model)
final_logger.log(f"Recalls on {source_test_set.name}: {recalls_str}")

previous_db_features = None
all_targets_recall_str = ""
for i in range(5):
    target_test_set = datasets.WholeDataset(opt.rootPath, "test/gallery", f"test/queries_{i+1}")
    _, previous_db_features, recalls_str = test.test(opt, target_test_set, model, previous_db_features)
    final_logger.log(f"Recalls on {target_test_set.name}: {recalls_str}")
    all_targets_recall_str += recalls_str

final_logger.log(f"Recalls all targets: {all_targets_recall_str}")

