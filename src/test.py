
import logging
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def test(args, eval_set, model):
    test_dataloader = DataLoader(dataset=eval_set, num_workers=args.num_workers, 
                                 batch_size=args.cache_batch_size, pin_memory=True)
    
    model.eval()
    with torch.no_grad():
        logging.debug(f"Extracting {'attentive ' if args.attention else ''}features ")
        features_dim = args.encoder_dim * args.num_clusters
        gallery_features = np.empty((len(eval_set), features_dim), dtype="float32")
        
        for inputs, indices in tqdm(test_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            vlad_encoding = model(inputs)
            gallery_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del inputs, vlad_encoding
    
    query_features = gallery_features[eval_set.db_struct.num_gallery:]
    gallery_features = gallery_features[:eval_set.db_struct.num_gallery]
    
    faiss_index = faiss.IndexFlatL2(features_dim)
    faiss_index.add(gallery_features)
    
    del gallery_features
    if args.faiss_gpu:
        faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)
     
    logging.debug("Calculating recalls")
    
    _, predictions = faiss_index.search(query_features, 20)
    
    ground_truths = eval_set.get_positives()
    
    n_values = [1, 5, 10, 20]
    
    recalls = np.zeros(len(n_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], ground_truths[query_index])):
                recalls[i:] += 1
                break
    
    recalls = recalls / eval_set.db_struct.num_queries * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(n_values, recalls)])
    return recalls, recalls_str

