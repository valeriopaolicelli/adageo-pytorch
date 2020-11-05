
import logging
import math
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import datasets


def elaborate_epoch(opt, epoch, model, optimizer, criterion_netvlad,
                    whole_train_set, query_train_set, DA_dict):
    
    epoch_loss = 0
    effective_iterations = 0
    pool_size = opt.encoder_dim * opt.num_clusters
    use_cuda = opt.device == "cuda"
    
    if opt.grl:
        epoch_grl_loss = 0
        grl_dataset = DA_dict["grl_dataset"]
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        grl_dataloader = DataLoader(dataset=grl_dataset, num_workers=opt.num_workers,
                                    batch_size=opt.grl_batch_size, shuffle=True, pin_memory=use_cuda)
    
    num_queries = len(query_train_set)
    subset_num = math.ceil(num_queries / (opt.cache_refresh_rate*opt.epoch_divider))
    subset_indexes = np.array_split(np.random.choice(np.arange(num_queries), num_queries, replace=False), subset_num)
    
    num_batches = opt.cache_refresh_rate * subset_num // opt.batch_size
    
    for sub_iter in range(subset_num):
        
        ############################################################################
        logging.debug(f"Building Cache [{sub_iter + 1}/{subset_num}] {'- features weighted' if opt.attention else ''}")
        
        model.eval()
        cache = np.zeros((len(whole_train_set), pool_size), dtype=np.float32)
        
        num_galleries = whole_train_set.db_struct.num_gallery
        useful_q_indexes = list(subset_indexes[sub_iter]+num_galleries)[:(opt.cache_refresh_rate)]
        useful_g_indexes = list(range(num_galleries))
        
        subset = Subset(whole_train_set, useful_q_indexes+useful_g_indexes)
        subset_dl = DataLoader(dataset=subset, num_workers=opt.num_workers, 
                               batch_size=opt.cache_batch_size, shuffle=False,
                               pin_memory=use_cuda)
        
        with torch.no_grad():
            for input, indices in tqdm(subset_dl):
                input = input.to(opt.device)
                vlad_encoding = model(input)
                cache[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                del input, vlad_encoding
                if opt.is_debug: break
        query_train_set.cache = cache
        del cache
        
        sub_query_train_set = Subset(dataset=query_train_set, indices=subset_indexes[sub_iter][:opt.cache_refresh_rate])
        
        query_dataloader = DataLoader(dataset=sub_query_train_set, num_workers=0,
                                       batch_size=opt.batch_size, shuffle=False,
                                       collate_fn=datasets.collate_fn, pin_memory=use_cuda)
        
        ########################################################################
        # TRAIN
        model.train()
        for query, positives, negatives, neg_counts, indices in tqdm(query_dataloader):
            effective_iterations += 1
            ########################################################################
            # Process NetVLAD task
            if query is None:
                continue  # in case we get an empty batch
            
            B, C, H, W = query.shape
            n_neg = torch.sum(neg_counts)
            
            input = torch.cat([query, positives, negatives])
            del query, positives, negatives
            input = input.to(device=opt.device, dtype=torch.float)
            vlad_encoding = model(input)
            del input
            
            vlad_q, vlad_p, vlad_n = torch.split(vlad_encoding, [B, B, n_neg])
            del vlad_encoding
            
            optimizer.zero_grad()
            loss_netvlad = 0
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    neg_index = (torch.sum(neg_counts[:i]) + n).item()
                    loss_netvlad += criterion_netvlad(vlad_q[i:i + 1], vlad_p[i:i + 1], vlad_n[neg_index:neg_index + 1])
            
            loss_netvlad /= n_neg.float().to(opt.device)  # normalise by actual number of negatives
            
            if not opt.is_debug: # TODO da eliminare
                loss_netvlad.backward()
            
            if opt.grl:
                images, labels = next(iter(grl_dataloader))
                images, labels = images.to(opt.device), labels.to(opt.device)
                outputs = model(images, grl=True)
                loss_da = cross_entropy_loss(outputs, labels)
                del images, labels, outputs
                (loss_da * opt.grl_loss).backward()
                epoch_grl_loss += loss_da.item()
            
            if not opt.is_debug: # TODO da eliminare
                optimizer.step()
            
            batch_loss = loss_netvlad.item()
            epoch_loss += batch_loss
            
            # End NetVLAD task
            ########################################################################
            
            del vlad_q, vlad_p, vlad_n, loss_netvlad
            if opt.is_debug: break
        
        logging.debug(f"Epoch[{epoch}]({effective_iterations}/{num_batches}): " +
                      f"batch NetVLAD Loss: {batch_loss:.4f} -" +
                      f" Avg NetVLADLoss: {epoch_loss / (effective_iterations):.4f}")
        if opt.grl: logging.debug(f"epoch_grl_loss: {epoch_grl_loss / effective_iterations:.4f}")
        
        del query_dataloader
        optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        del query_train_set.cache
        if opt.is_debug: break
    
    if opt.grl: logging.debug(f"epoch_grl_loss: {epoch_grl_loss / effective_iterations:.4f}")
    query_train_set.cache = None
    return f"Finished epoch {epoch:02d} -  Avg. Loss NetVLAD: {epoch_loss / effective_iterations:.4f} - "

