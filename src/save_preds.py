
import const
import util
import datasets
import commons

import numpy as np
import faiss
from tqdm import tqdm
import os
from os.path import join, exists
from os import mkdir
from shutil import copy2
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader

import random
from PIL import Image, ImageOps, ImageDraw, ImageFont


def concatenate_images(list_im, colors, name, orientation='h'):
    if orientation == 'h':
        w, h = Image.open(list_im[1]).size
        label = Image.new("RGB", (h, 30), (255, 255, 255))
        font = ImageFont.truetype('../times-ro.ttf', 37)
        draw = ImageDraw.Draw(label)
        (x, y) = (int(h/2)-75, 10)
        message = name
        color = 'rgb(0, 0, 0)' # black
        draw.text((x, y), message, fill=color, font=font)
        label = label.rotate(90, expand=1)
        imgs = [ImageOps.expand(label, border=30, fill='white')] +\
                [ImageOps.expand(ImageOps.expand(
                                    Image.open(i), border=20, fill=color), border=10, fill='white')
                                        for i, color in zip(list_im, colors)]
        imgs_comb = np.hstack([np.asarray(i) for i in imgs])
        return Image.fromarray(imgs_comb)
    else:
        imgs_comb = np.vstack([np.asarray(i) for i in list_im])
        Image.fromarray(imgs_comb).save(name)
        return None


def load_models(opt):
    opt.logger.log(f"Building models", False)
    opt.grl = False
    adageo = util.build_model(opt)
    state_dict = torch.load(f'runs/models_to_visualize/resnet_adageo/t{opt.scenario}/best_model.pth')['state_dict']
    adageo.load_state_dict(state_dict, strict=False)
    adageo = adageo.to(opt.device)
    
    opt.grl = True
    opt.grlDatasets = 'train/gallery+train/queries_all'
    grl = util.build_model(opt)
    state_dict = torch.load(f'runs/models_to_visualize/resnet_grl/t{opt.scenario}/best_model.pth')['state_dict']
    grl.load_state_dict(state_dict)
    grl = grl.to(opt.device)
    return adageo, grl


def save_preds(opt):
    adageo, grl = load_models(opt)

    eval_set = datasets.WholeDataset(opt.rootPath, "test/gallery", f"test/queries_{opt.scenario}")
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.num_workers, 
                                      batch_size=opt.cacheBatchSize, pin_memory=True)
    
    count_different = 0
    count_equal = 0
    num_queries_different_result = [2, 3, 3, 3, 3]
    num_queries_equal_result = [2, 1, 1, 1, 1]
    query_indexes, queries = [], []
    models = {'Ours':adageo, 'Best baseline':grl}
    predictions = {'Ours':[], 'Best baseline':[]}
    preds = {'Ours':[], 'Best baseline':[]}
    colors = {'Ours':[], 'Best baseline':[], 'queries': []}
        
    for model_name, model in models.items():
        opt.logger.log(f'Save preds from {model_name}...')
        model.eval()
        with torch.no_grad():
            pool_size = opt.encoder_dim * const.num_clusters
            dbFeat = np.empty((len(eval_set), pool_size), dtype='float32')
            for iteration, (input, indices) in enumerate(tqdm(test_data_loader), 1):
                input = input.to(opt.device)
                if model_name == 'Ours':
                    vlad_encoding = model(input, cache=False, mode='atten-vlad', atten_type='cam')
                else:
                    vlad_encoding = model(input, cache=True)
                del input
                dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                del vlad_encoding
            qFeat = dbFeat[eval_set.dbStruct.numDb:]
            dbFeat = dbFeat[:eval_set.dbStruct.numDb]
            faiss_index = faiss.IndexFlatL2(pool_size)
            faiss_index.add(dbFeat)
            _, predictions[model_name] = faiss_index.search(qFeat, 20)
            gt = eval_set.getPositives()


    while (count_equal < num_queries_equal_result[opt.scenario-1]) \
        or (count_different < num_queries_different_result[opt.scenario-1]):
        q_idx = random.randint(0, (eval_set.dbStruct.numQ - 1))
        pred_adageo = predictions['Ours'][q_idx][0]
        pred_grl = predictions['Best baseline'][q_idx][0]
        if np.any(np.in1d(pred_adageo, gt[q_idx])):
            if (count_equal < num_queries_equal_result[opt.scenario-1]) and opt.scenario != 4:
                if np.any(np.in1d(pred_grl, gt[q_idx])):
                    query_indexes.append(q_idx)
                    queries.append(eval_set.dbStruct.qImage[q_idx])
                    colors['queries'].append('white')
                    preds['Ours'].append(eval_set.dbStruct.dbImage[pred_adageo])
                    colors['Ours'].append('green')
                    preds['Best baseline'].append(eval_set.dbStruct.dbImage[pred_grl])
                    colors['Best baseline'].append('green')
                    count_equal += 1
            if count_different < num_queries_different_result[opt.scenario-1]:
                if not np.any(np.in1d(pred_grl, gt[q_idx])):
                    print(f'trovato count_different={count_different+1} - count_equal={count_equal}')
                    query_indexes.append(q_idx)
                    queries.append(eval_set.dbStruct.qImage[q_idx])
                    colors['queries'].append('white')
                    preds['Ours'].append(eval_set.dbStruct.dbImage[pred_adageo])
                    colors['Ours'].append('green')
                    preds['Best baseline'].append(eval_set.dbStruct.dbImage[pred_grl])
                    colors['Best baseline'].append('red')
                    count_different += 1
        else:
            if (opt.scenario == 4) and (count_equal < num_queries_equal_result[opt.scenario-1] ) and (not np.any(np.in1d(pred_grl, gt[q_idx]))):
                print(f'trovato count_different={count_different} - count_equal={count_equal+1}')
                query_indexes.append(q_idx)
                queries.append(eval_set.dbStruct.qImage[q_idx])
                colors['queries'].append('white')
                preds['Ours'].append(eval_set.dbStruct.dbImage[pred_adageo])
                colors['Ours'].append('red')
                preds['Best baseline'].append(eval_set.dbStruct.dbImage[pred_grl])
                colors['Best baseline'].append('red')
                count_equal += 1
                        
    
    concatenated = []
    concatenated.append(concatenate_images(queries, colors['queries'], 'Query'))
    for model in preds.keys():
        concatenated.append(concatenate_images(preds[model], colors[model], model))
    
    sorted_concatenated = []
    for c in concatenated:
        sorted_concatenated.append(c)
    
    concatenate_images(sorted_concatenated, None, join(opt.outputFolder, f'preds_target{opt.scenario}.jpg'), 'v')


######################################### MAIN #########################################
parser = argparse.ArgumentParser(description='pytorch-NetVlad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cacheBatchSize', type=int, default=300, help='Batch size for caching and testing')
parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
parser.add_argument('--scenario', type=int, default=0, help='Scenario su cui testare.')
parser.add_argument('--num_workers', type=int, default=6, help='Per dividere un epoca in N parti')
parser.add_argument('--allDatasetsPath', type=str, default='/home/valeriop/datasets', 
                    help='Path con tutti i dataset')
parser.add_argument('--rootPath', type=str, default='oxford60k/image', help='Root del dataset')
parser.add_argument('--testG', type=str, default='test/gallery', help='Path test gallery')
parser.add_argument('--testQ', type=str, default='test/queries', help='Path test query')
parser.add_argument('--expName', type=str, default='default',
                    help='Folder name of the current run (saved in runsPath).')

opt = parser.parse_args()

opt.outputFolder = os.path.join(const.runsPath, opt.expName, datetime.now().strftime('%b%d_%H-%M-%S'))

opt.logger = commons.Logger(folder=opt.outputFolder, filename=f"logger.txt")
opt.logger.log(f'Arguments: {opt}')
opt.rootPath = os.path.join(opt.allDatasetsPath, opt.rootPath)
opt.cuda = True
opt.device = "cuda"
opt.arch = 'resnet18'
opt.pretrain = 'places'
opt.train_part = False
opt.resume = True

if opt.scenario == 0:
    for opt.scenario in range(1, 6):
        opt.logger.log(f'---------------- Eval scenario {opt.scenario} ----------------')
        save_preds(opt)
else:
    opt.logger.log(f'---------------- Eval scenario {opt.scenario} ----------------')
    save_preds(opt)
        

