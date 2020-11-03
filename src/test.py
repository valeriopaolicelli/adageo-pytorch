
import const

import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def test(opt, eval_set, model, previous_dbFeat=None):
    if not opt.load_pos:
        if previous_dbFeat is not None: # se le features del database sono già state calcolate, non le ricalcolo!
            subset = Subset(eval_set, list(range(eval_set.dbStruct.numDb, len(eval_set))))
            test_data_loader = DataLoader(dataset=subset, num_workers=opt.num_workers, 
                                          batch_size=opt.cacheBatchSize, pin_memory=True)
        else:
            test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.num_workers, 
                                          batch_size=opt.cacheBatchSize, pin_memory=True)
        
        model.eval()
        with torch.no_grad():
            opt.logger.log(f"Extracting Features {'weighted' if opt.attention else ''}", False)
            pool_size = opt.encoder_dim * const.num_clusters
            dbFeat = np.empty((len(eval_set), pool_size), dtype='float32')
            
            for iteration, (input, indices) in enumerate(tqdm(test_data_loader), 1):
                input = input.to(opt.device)
                if opt.attention:
                    vlad_encoding = model(input, cache=False, mode='atten-vlad', atten_type=opt.atten_type)
                else:
                    vlad_encoding = model(input, cache=True)
                del input
                dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                del vlad_encoding
                if opt.isDebug: break
        
        qFeat = dbFeat[eval_set.dbStruct.numDb:]
        if previous_dbFeat is None:
            dbFeat = dbFeat[:eval_set.dbStruct.numDb]
        else:
            dbFeat = previous_dbFeat
        
        opt.logger.log(f'Building faiss index', False)
        
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(dbFeat)
        
        opt.logger.log(f'Calculating recall @ N', False)
        
        if opt.isDebug: # TODO da eliminare
            predictions = np.ones((eval_set.dbStruct.numQ, 20), dtype=np.uint16)
        else:
            _, predictions = faiss_index.search(qFeat, 20)
            
        # for each query get those within threshold distance
        gt = eval_set.getPositives()
        
        if opt.save_pos:
            np.save("predictions", predictions)
            opt.logger.log('Predictions saved', False)
            
            np.save("positives", gt)
            opt.logger.log('Positives saved', False)
    
    else:
        predictions = np.load("predictions.npy", allow_pickle=True)
        opt.logger.log(f'Predictions loaded | shape: {predictions.shape}', False)
        gt = np.load("positives.npy", allow_pickle=True)
        opt.logger.log(f'Positives loaded | shape: {gt.shape}', False)
    
    n_values = [1, 5, 10, 20]
    
    correct_at_n = np.zeros(len(n_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[query_index])):
                correct_at_n[i:] += 1
                break
    
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ
    
    recalls = {}  # make dict for output
    recalls_str = ""
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        recalls_str += f"{recall_at_n[i] * 100:.1f} \t"
    
    return recalls, dbFeat, recalls_str.replace(".", ",")

