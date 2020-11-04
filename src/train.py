
import const, datasets

import math
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm


def elaborate_epoch(opt, epoch, model, optimizer, criterion_netvlad,
                    whole_train_set, query_train_set, DA_dict):
    
    epoch_loss = 0
    effective_iterations = 0
    pool_size = opt.encoder_dim * const.num_clusters
    
    if opt.grl:
        epoch_grl_loss = 0
        grl_dataset = DA_dict["grl_dataset"]
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        grl_dataloader = DataLoader(dataset=grl_dataset, num_workers=opt.num_workers,
                                    batch_size=opt.grlBatchSize, shuffle=True, pin_memory=opt.cuda)
    
    num_queries = len(query_train_set)
    subsetN = math.ceil(num_queries / (opt.cacheRefreshRate*opt.epochDivider))
#    subsetIdx = np.array_split(np.arange(num_queries), subsetN)
    subsetIdx = np.array_split(np.random.choice(np.arange(num_queries), num_queries, replace=False), subsetN)
    
    nBatches = opt.cacheRefreshRate * subsetN // opt.batchSize
    
    for subIter in range(subsetN):
        
        ############################################################################
        output = f'Building Cache [{subIter + 1}/{subsetN}] ' \
                 f'{"- features weighted" if opt.attention else ""}'
        opt.logger.log(output, False)
        
        model.eval()
        cache = np.zeros((len(whole_train_set), pool_size), dtype=np.float32)
        
        num_galleries = whole_train_set.dbStruct.numDb
        useful_q_indexes = list(subsetIdx[subIter]+num_galleries)[:(opt.cacheRefreshRate + opt.batchSize)] # + opt.batchSize è solo per sicurezza
        useful_g_indexes = list(range(num_galleries))
        
        subset = Subset(whole_train_set, useful_q_indexes+useful_g_indexes)
        subset_dl = DataLoader(dataset=subset, num_workers=opt.num_workers, 
                               batch_size=opt.cacheBatchSize, shuffle=False,
                               pin_memory=opt.cuda)
        
        with torch.no_grad():
            for input, indices in tqdm(subset_dl):
                input = input.to(opt.device)
                vlad_encoding = model(input)
                cache[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                del input, vlad_encoding
                if opt.isDebug: break
        query_train_set.cache = cache
        del cache
        
#        sub_query_train_set = Subset(dataset=query_train_set, indices=subsetIdx[subIter])
        sub_query_train_set = Subset(dataset=query_train_set, indices=subsetIdx[subIter][:opt.cacheRefreshRate])
        
        query_data_loader = DataLoader(dataset=sub_query_train_set, num_workers=0,
                                       batch_size=opt.batchSize, shuffle=False,
                                       collate_fn=datasets.collate_fn, pin_memory=opt.cuda)
        
        ########################################################################
        # TRAIN
        model.train()
        for query, positives, negatives, negCounts, indices in tqdm(query_data_loader):
            effective_iterations += 1
            ########################################################################
            # Process NetVLAD task
            if query is None:
                continue  # in case we get an empty batch
            
            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            
            input = torch.cat([query, positives, negatives])
            del query, positives, negatives
            input = input.to(device=opt.device, dtype=torch.float)
            vlad_encoding = model(input)
            del input
            
            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])
            del vlad_encoding
            
            optimizer.zero_grad()
            loss_netvlad = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss_netvlad += criterion_netvlad(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])
            
            loss_netvlad /= nNeg.float().to(opt.device)  # normalise by actual number of negatives
            
            if not opt.isDebug: # TODO da eliminare
                loss_netvlad.backward()
            
            if opt.grl:
                images, labels = next(iter(grl_dataloader))
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images, grl=True)
                loss_da = cross_entropy_loss(outputs, labels)
                del images, labels, outputs
                (loss_da * opt.grlLoss).backward()
                epoch_grl_loss += loss_da.item()
            
            if not opt.isDebug: # TODO da eliminare
                optimizer.step()
            
            batch_loss = loss_netvlad.item()
            epoch_loss += batch_loss
            
            # End NetVLAD task
            ########################################################################
            
            del vladQ, vladP, vladN, loss_netvlad
            if opt.isDebug: break
        
        output = f"Epoch[{epoch}]({effective_iterations}/{nBatches}): " \
                 f"batch NetVLAD Loss: {batch_loss:.4f} -" \
                 f" Avg NetVLADLoss: {epoch_loss / (effective_iterations):.4f}"
        opt.logger.log(output, False)
        if opt.grl: opt.logger.log(f"epoch_grl_loss: {epoch_grl_loss / effective_iterations:.4f}", False)
        
        del query_data_loader
        optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        del query_train_set.cache
        if opt.isDebug: break
    
    if opt.grl: opt.logger.log(f"epoch_grl_loss: {epoch_grl_loss / effective_iterations:.4f}", False)
    query_train_set.cache = None
    return f"Finished epoch {epoch:02d} -  Avg. Loss NetVLAD: {epoch_loss / effective_iterations:.4f} - "

