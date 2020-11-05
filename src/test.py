
import logging
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def test(opt, eval_set, model, previous_gallery_features=None):
    if previous_gallery_features is not None: # se le features del database sono già state calcolate, non le ricalcolo!
        subset = Subset(eval_set, list(range(eval_set.db_struct.num_gallery, len(eval_set))))
        test_dataloader = DataLoader(dataset=subset, num_workers=opt.num_workers, 
                                      batch_size=opt.cache_batch_size, pin_memory=True)
    else:
        test_dataloader = DataLoader(dataset=eval_set, num_workers=opt.num_workers, 
                                      batch_size=opt.cache_batch_size, pin_memory=True)
    
    model.eval()
    with torch.no_grad():
        logging.debug(f"Extracting Features {'weighted' if opt.attention else ''}")
        pool_size = opt.encoder_dim * opt.num_clusters
        gallery_features = np.empty((len(eval_set), pool_size), dtype="float32")
        
        for iteration, (input, indices) in enumerate(tqdm(test_dataloader), 1):
            input = input.to(opt.device)
            vlad_encoding = model(input)
            del input
            gallery_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del vlad_encoding
            if opt.is_debug: break
    
    query_features = gallery_features[eval_set.db_struct.num_gallery:]
    if previous_gallery_features is None:
        gallery_features = gallery_features[:eval_set.db_struct.num_gallery]
    else:
        gallery_features = previous_gallery_features
    
    logging.debug("Building faiss index")
    
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(gallery_features)
    
    logging.debug("Calculating recall @ N")
    
    if opt.is_debug: # TODO da eliminare
        predictions = np.ones((eval_set.db_struct.num_queries, 20), dtype=np.uint16)
    else:
        _, predictions = faiss_index.search(query_features, 20)
    
    # for each query get those within threshold distance
    ground_truths = eval_set.getPositives()
    
    n_values = [1, 5, 10, 20]
    
    correct_at_n = np.zeros(len(n_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], ground_truths[query_index])):
                correct_at_n[i:] += 1
                break
    
    recall_at_n = correct_at_n / eval_set.db_struct.num_queries
    
    recalls = {}  # make dict for output
    recalls_str = ""
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        recalls_str += f"{recall_at_n[i] * 100:.1f} \t"
    
    return recalls, gallery_features, recalls_str.replace(".", ",")

