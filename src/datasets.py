
import h5py
import itertools
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import faiss
from glob import glob
import os

def transform(dataset):
    if dataset == "svox" or dataset == "st_lucia":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif dataset == "msls":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((360, 640)),
        ])

    raise Exception(f"Dataset must be a value between [svox, msls, st_lucia], instead got {dataset}")
    

db_struct = namedtuple("db_struct", ["gallery_images", "gallery_utms", "query_images", "query_utms", "num_gallery", "num_queries",
                                     "val_pos_dist_threshold", "train_pos_dist_threshold"])


def parse_db_struct(dataset_root, gallery_path, query_paths):
    gallery_path = f"{dataset_root}/{gallery_path}"
    if not os.path.exists(gallery_path): raise Exception(f"{gallery_path} does not exist")
    db_images = sorted(glob(f"{gallery_path}/**/*.jpg", recursive=True))
    db_utms =  np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in db_images])
    query_paths = query_paths.split("+")
    q_images = [] 
    for query_path in query_paths:
        query_path = f"{dataset_root}/{query_path}"
        if not os.path.exists(query_path): raise Exception(f"{query_path} does not exist")
        q_images.extend(sorted(glob(f"{query_path}/**/*.jpg", recursive=True))) 
    q_utms =   np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in q_images])
    num_gallery = len(db_images)
    num_queries = len(q_images)
    val_pos_dist_threshold = 25
    train_pos_dist_threshold = 10
    return db_struct(db_images, db_utms, q_images, q_utms, num_gallery, num_queries,
                     val_pos_dist_threshold, train_pos_dist_threshold)


class WholeDataset(data.Dataset):
    # Dataset with both gallery and query images, used for inference (testing and building cache)
    def __init__(self, dataset_root, gallery_path, query_paths, dataset=None):
        super().__init__()
        self.transform = transform(dataset)
        self.db_struct = parse_db_struct(dataset_root, gallery_path, query_paths)
        self.images = [dbIm for dbIm in self.db_struct.gallery_images]
        self.images += [qIm for qIm in self.db_struct.query_images]
        # Find positives within val_pos_dist_threshold (25 meters), in self.positives_per_query
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.positives_per_query = knn.radius_neighbors(self.db_struct.query_utms, 
                                              radius=self.db_struct.val_pos_dist_threshold,
                                              return_distance=False)
        self.info = f"< WholeDataset queries: {query_paths} ({self.db_struct.num_queries}); gallery: {gallery_path} ({self.db_struct.num_gallery}) >"
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transform(img)
        return img, index
    def __len__(self):
        return len(self.images)
    def __str__(self):
        return self.info
    def get_positives(self):
        return self.positives_per_query


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None
    query, positive, negatives, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    neg_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, neg_counts, indices


class QueryDataset(data.Dataset):
    # Dataset class used for training.
    def __init__(self, dataset_root, gallery_path, query_paths, output_folder):
        super().__init__()
        self.output_folder = output_folder
        self.transform = transform()
        self.margin = 0.1
        self.db_struct = parse_db_struct(dataset_root, gallery_path, query_paths)
        self.n_neg_samples = 1000 # Number of negatives to randomly sample
        self.n_neg = 10 # Number of negatives per query in each batch
        # Find positives within train_pos_dist_threshold (10 meters), in self.positives_per_query
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.positives_per_query = list(knn.radius_neighbors(self.db_struct.query_utms,
                                                              radius=self.db_struct.train_pos_dist_threshold,
                                                              return_distance=False))
        self.positives_per_query = [np.sort(positives) for positives in self.positives_per_query]
        
        # Make sure that each query in the train set has at least one positive, otherwise it should be deleted
        queries_without_any_positive = np.where(np.array([len(p) for p in self.positives_per_query]) == 0)[0]
        if len(queries_without_any_positive) > 0:
            with open(f"{output_folder}/queries_without_any_positive.txt", "w") as file:
                for query_index in queries_without_any_positive:
                    file.write(f"{db_struct.query_images[query_index]}\n")
            raise Exception(f"There are {len(queries_without_any_positive)} queries in the training " + 
                            f"set without any positive (within {db_struct.train_pos_dist_threshold} meters) in " +
                            "the gallery! Please remove these images, as they're not used for training. " +
                            "The paths of these images have been saved in " + 
                            f"{output_folder}/queries_without_any_positive.txt")
        
        # Find negatives further than val_pos_dist_threshold (25 meters), in self.negatives_per_query
        positives = knn.radius_neighbors(self.db_struct.query_utms,
                                         radius=self.db_struct.val_pos_dist_threshold,
                                         return_distance=False)
        self.negatives_per_query = []
        for pos in positives:
            self.negatives_per_query.append(np.setdiff1d(np.arange(self.db_struct.num_gallery),
                                                         pos, assume_unique=True))
        self.neg_cache = [np.empty((0,)) for _ in range(self.db_struct.num_queries)]
        self.info = f"< QueryDataset queries: {query_paths} ({self.db_struct.num_queries}); gallery: {gallery_path} ({self.db_struct.num_gallery}) >"
    def __getitem__(self, index):
        with h5py.File(f"{self.output_folder}/cache.hdf5", mode="r") as h5: 
            cache = h5.get("cache")
            features_dim = cache.shape[1]
            queries_offset = self.db_struct.num_gallery
            query_features = cache[index + queries_offset].astype(np.float32)
            if np.all(query_features==0):
                raise Exception(f"For query {self.db_struct.query_images[index]} with index {index} features have not been computed!!!")
            positives_features = cache[self.positives_per_query[index].tolist()].astype(np.float32)
            faiss_index = faiss.IndexFlatL2(features_dim)
            faiss_index.add(positives_features)
            # Search the best positive (within 10 meters AND nearest in features space)
            best_pos_dist, best_pos_num = faiss_index.search(query_features.reshape(1, -1), 1)
            best_pos_dist = best_pos_dist.item()
            best_pos_index = self.positives_per_query[index][best_pos_num[0]].item()
            # Sample 1000 negatives randomly and concatenate them with the previous top 10 negatives (neg_cache)
            neg_samples = np.random.choice(self.negatives_per_query[index], self.n_neg_samples)
            neg_samples = np.unique(np.concatenate([self.neg_cache[index], neg_samples]))
            neg_features = np.array([cache[int(neg_sample)] for neg_sample in neg_samples]).astype(np.float32)
        
        faiss_index = faiss.IndexFlatL2(features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        neg_dist, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.n_neg)
        neg_dist = neg_dist.reshape(-1)
        neg_nums = neg_nums.reshape(-1)
        violating_neg = (neg_dist ** 0.5) < (best_pos_dist ** 0.5) + (self.margin ** 0.5)
        if np.sum(violating_neg) < 1:
            return None # If none are violating then skip this query
        neg_nums = neg_nums[violating_neg][:self.n_neg]
        neg_indices = neg_samples[neg_nums].astype(np.int32)
        self.neg_cache[index] = neg_indices # Update nearest negatives in neg_cache
        query = self.transform(Image.open(self.db_struct.query_images[index]))
        positive = self.transform(Image.open(self.db_struct.gallery_images[best_pos_index]))
        negatives = [self.transform(Image.open(self.db_struct.gallery_images[i])) for i in neg_indices]
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [index, best_pos_index] + neg_indices.tolist()
    def __len__(self):
        return self.db_struct.num_queries
    def __str__(self):
        return self.info

