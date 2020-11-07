
import logging
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def test(opt, eval_set, model):
    test_dataloader = DataLoader(dataset=eval_set, num_workers=opt.num_workers, 
                                  batch_size=opt.cache_batch_size, pin_memory=True)
    
    model.eval()
    with torch.no_grad():
        logging.debug(f"Extracting Features {'weighted' if opt.attention else ''}")
        pool_size = opt.encoder_dim * opt.num_clusters
        gallery_features = np.empty((len(eval_set), pool_size), dtype="float32")
        
        for inputs, indices in tqdm(test_dataloader, ncols=100):
            inputs = inputs.to(opt.device)
            vlad_encoding = model(inputs)
            gallery_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del inputs, vlad_encoding
    
    query_features = gallery_features[eval_set.db_struct.num_gallery:]
    gallery_features = gallery_features[:eval_set.db_struct.num_gallery]
    
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(gallery_features)
    
    logging.debug("Calculating recalls")
    
    _, predictions = faiss_index.search(query_features, 20)
    
    ground_truths = eval_set.getPositives()
    
    n_values = [1, 5, 10, 20]
    
    correct_at_n = np.zeros(len(n_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], ground_truths[query_index])):
                correct_at_n[i:] += 1
                break
    
    recall_at_n = correct_at_n / eval_set.db_struct.num_queries
    
    recalls = {} # make dict for output
    recalls_str = ""
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        recalls_str += f"{recall_at_n[i] * 100:.1f} \t"
    
    return recalls, recalls_str.replace(".", ",")

