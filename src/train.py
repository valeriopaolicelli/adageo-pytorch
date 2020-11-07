
import logging
import math
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from datetime import datetime

import datasets


def elaborate_epoch(opt, epoch, model, optimizer, criterion_netvlad,
                    whole_train_set, query_train_set, grl_dataset):
    
    epoch_start_time = datetime.now()
    epoch_loss = 0
    effective_iterations = 0
    pool_size = opt.encoder_dim * opt.num_clusters
    use_cuda = opt.device == "cuda"
    
    if opt.grl:
        epoch_grl_loss = 0
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        grl_dataloader = DataLoader(dataset=grl_dataset, num_workers=opt.num_workers,
                                    batch_size=opt.grl_batch_size, shuffle=True, pin_memory=use_cuda)
    
    num_queries = len(query_train_set)
    subset_num = math.ceil(num_queries / (opt.cache_refresh_rate*opt.epoch_divider))
    subset_indexes = np.array_split(np.random.choice(np.arange(num_queries), num_queries, replace=False), subset_num)
    
    num_batches = opt.cache_refresh_rate * subset_num // opt.batch_size
    
    for sub_iter in range(subset_num):
        
        ############################################################################
        logging.debug(f"Building Cache [{sub_iter + 1}/{subset_num}] {'- with attentive features' if opt.attention else ''}")
        
        model.eval()
        
        num_galleries = whole_train_set.db_struct.num_gallery
        useful_q_indexes = list(subset_indexes[sub_iter]+num_galleries)[:(opt.cache_refresh_rate)]
        useful_g_indexes = list(range(num_galleries))
        
        subset = Subset(whole_train_set, useful_q_indexes+useful_g_indexes)
        subset_dl = DataLoader(dataset=subset, num_workers=opt.num_workers, 
                               batch_size=opt.cache_batch_size, shuffle=False,
                               pin_memory=use_cuda)
        
        cache = np.zeros((len(whole_train_set), pool_size), dtype=np.float32)
        with torch.no_grad():
            for inputs, indices in tqdm(subset_dl, ncols=100):
                inputs = inputs.to(opt.device)
                vlad_encoding = model(inputs)
                cache[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                del inputs, vlad_encoding
        
        query_train_set.cache = cache
        del cache
        
        sub_query_train_set = Subset(dataset=query_train_set, indices=subset_indexes[sub_iter][:opt.cache_refresh_rate])
        
        query_dataloader = DataLoader(dataset=sub_query_train_set, num_workers=0,
                                       batch_size=opt.batch_size, shuffle=False,
                                       collate_fn=datasets.collate_fn, pin_memory=use_cuda)
        
        # TRAIN
        model.train()
        for query, positives, negatives, neg_counts, indices in tqdm(query_dataloader, ncols=100):
            effective_iterations += 1
            if query is None:
                continue # in case we get an empty batch
            
            B, C, H, W = query.shape
            n_neg = torch.sum(neg_counts)
            
            inputs = torch.cat([query, positives, negatives])
            inputs = inputs.to(device=opt.device)
            vlad_encoding = model(inputs)
            
            vlad_q, vlad_p, vlad_n = torch.split(vlad_encoding, [B, B, n_neg])
            del query, positives, negatives, inputs, vlad_encoding
            
            optimizer.zero_grad()
            loss_triplet = 0
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    neg_index = (torch.sum(neg_counts[:i]) + n).item()
                    loss_triplet += criterion_netvlad(vlad_q[i:i + 1], vlad_p[i:i + 1], vlad_n[neg_index:neg_index + 1])
            
            loss_triplet /= n_neg.float().to(opt.device)  # normalise by actual number of negatives
            
            loss_triplet.backward()
            batch_loss = loss_triplet.item()
            epoch_loss += batch_loss
            del vlad_q, vlad_p, vlad_n, loss_triplet
            
            if opt.grl:
                images, labels = next(iter(grl_dataloader))
                images, labels = images.to(opt.device), labels.to(opt.device)
                outputs = model(images, grl=True)
                loss_grl = cross_entropy_loss(outputs, labels)
                (loss_grl * opt.grl_loss_weight).backward()
                epoch_grl_loss += loss_grl.item()
                del images, labels, outputs, loss_grl
            
            optimizer.step()
        
        logging.debug(f"Epoch[{epoch:02d}]({effective_iterations}/{num_batches}): " +
                      f"current batch triplet loss = {batch_loss:.4f}, " +
                      f"average epoch triplet loss = {epoch_loss / (effective_iterations):.4f}")
        if opt.grl: logging.debug(f"Average grl epoch loss: {epoch_grl_loss / effective_iterations:.4f}")
        
        del query_dataloader
        del query_train_set.cache
    
    logging.info(f"Finished epoch {epoch:02d} in {str(datetime.now() - epoch_start_time)[:-7]}: "
                 f"average epoch triplet loss = {epoch_loss / effective_iterations:.4f}")
    if opt.grl: logging.info(f"Average epoch grl loss: {epoch_grl_loss / effective_iterations:.4f}")

