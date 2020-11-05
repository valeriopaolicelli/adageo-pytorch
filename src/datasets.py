
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import faiss
import glob
import os

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

db_struct = namedtuple("db_struct", ["gallery_images", "gallery_utms", "query_images", "query_utms", "num_gallery", "num_queries",
                                     "val_pos_dist_threshold", "train_pos_dist_threshold"])


def parse_db_struct(root_path, gallery_path, query_path):
    gallery_path = os.path.join(root_path, gallery_path)
    query_path = os.path.join(root_path, query_path)
    if not os.path.exists(gallery_path): raise Exception(f"{gallery_path} non esiste")
    if not os.path.exists(query_path): raise Exception(f"{query_path} non esiste")
    db_images = sorted(glob.glob(f"{gallery_path}/**/*.jpg", recursive=True))
    q_images =  sorted(glob.glob(f"{query_path}/**/*.jpg", recursive=True))
    db_utms =  np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in db_images])
    q_utms =   np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in q_images])
    num_gallery = len(db_images)
    num_queries = len(q_images)
    val_pos_dist_threshold = 25
    train_pos_dist_threshold = 10
    return db_struct(db_images, db_utms, q_images, q_utms, num_gallery, num_queries,
                     val_pos_dist_threshold, train_pos_dist_threshold)


class WholeDataset(data.Dataset):
    """
    __getitem__(index) ritorna:
        img, index
    dove img può essere sia una gallery che una query
    """
    def __init__(self, root_path, gallery_path, query_path):
        super().__init__()
        self.input_transform = input_transform()
        self.db_struct = parse_db_struct(root_path, gallery_path, query_path)
        self.images = [dbIm for dbIm in self.db_struct.gallery_images]
        self.images += [qIm for qIm in self.db_struct.query_images]
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.positives = knn.radius_neighbors(self.db_struct.query_utms, 
                                              radius=self.db_struct.val_pos_dist_threshold,
                                              return_distance=False)
        self.info = f"< queries: {query_path} ({self.db_struct.num_queries}); gallery: {gallery_path} ({self.db_struct.num_gallery}) >"
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.input_transform(img)
        return img, index
    def __len__(self):
        return len(self.images)
    def getPositives(self):
        return self.positives


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
    # batch è costruito come lista di elementi che non sono None
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None
    query, positive, negatives, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    # per ogni elemento in negatives vede quanti elementi contiene, quindi li salva
    # in una lista
    neg_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    # unisce le sottoliste della lista negatives, i confini sono delineati da neg_counts
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, neg_counts, indices


class QueryDataset(data.Dataset):
    """
    __getitem__(index) ritorna:
        query, positive, negatives, [index, pos_index] + neg_indices.tolist()
    mentre con collate_fn ritorna:
        query, positive, negatives, neg_counts, indices
    """
    def __init__(self, root_path, gallery_path, query_path):
        super().__init__()
        self.input_transform = input_transform()
        self.margin = 0.1
        self.db_struct = parse_db_struct(root_path, gallery_path, query_path)
        self.n_neg_sample = 1000 # number of negatives to randomly sample
        self.n_neg = 10  # number of negatives used for training
        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.nontrivial_positives = list(knn.radius_neighbors(self.db_struct.query_utms, # 10 metri
                                                              radius=self.db_struct.train_pos_dist_threshold,
                                                              return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives, lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]
        # potential negatives are those outside of val_pos_dist_threshold range
        potential_positives = knn.radius_neighbors(self.db_struct.query_utms, # 25 metri
                                                   radius=self.db_struct.val_pos_dist_threshold,
                                                   return_distance=False)
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.db_struct.num_gallery),
                                                         pos, assume_unique=True))
        self.cache = None  # structure that contains the features cached from the whole train set
        self.neg_cache = [np.empty((0,)) for _ in range(self.db_struct.num_queries)]
        self.info = f"< queries: {query_path} ({self.db_struct.num_queries}); gallery: {gallery_path} ({self.db_struct.num_gallery}) >"
    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset
        ###############################################################
        # Start reading features whole train set (cache)
        q_offset = self.db_struct.num_gallery
        query_features = self.cache[index + q_offset]
        if np.all(query_features==0):
            raise Exception(f"Per la query {index} con shape {query_features.shape} le features sono tutte 0!!! " +
                            f"Non le hai calcolate! {self.db_struct.query_images[index]} | {self.nontrivial_positives[index].tolist()}" + 
                            "NOTA BENE: questo errore potrebbe essere dovuto al fatto che le queries considerate nel QueryDataset sono "+
                            "meno rispetto a quelle considerate nel WholeDataset, quindi gli indici sono sfasati!!!" + 
                            "Puoi trovare il codice per eliminare le queries di train inutili in fondo a questo file")
        pos_features = self.cache[self.nontrivial_positives[index].tolist()]
        pool_size = self.cache.shape[1]
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(pos_features)
        dist_pos, pos_nums = faiss_index.search(query_features.reshape(1, -1), 1)
        dist_pos = dist_pos.item()
        pos_index = self.nontrivial_positives[index][pos_nums[0]].item()
        neg_sample = np.random.choice(self.potential_negatives[index], self.n_neg_sample)
        neg_sample = np.unique(np.concatenate([self.neg_cache[index], neg_sample]))
        feat_len = self.cache.shape[1]
        neg_features = np.zeros((len(neg_sample.tolist()), feat_len), dtype=np.float32)
        for idx, neg_feature in enumerate(neg_sample.tolist()):
            neg_features[idx] = self.cache[int(neg_feature)]
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(neg_features)
        dist_neg, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.n_neg * 10)
        dist_neg = dist_neg.reshape(-1)
        neg_nums = neg_nums.reshape(-1)
        violating_neg = (dist_neg ** 0.5) < (dist_pos ** 0.5) + (self.margin ** 0.5)
        if np.sum(violating_neg) < 1:
            # if none are violating then skip this query
            return None
        neg_nums = neg_nums[violating_neg][:self.n_neg]
        neg_indices = neg_sample[neg_nums].astype(np.int32)
        self.neg_cache[index] = neg_indices
        # End reading features whole train set (cache)
        ###############################################################
        query = Image.open(self.db_struct.query_images[index])
        positive = Image.open(self.db_struct.gallery_images[pos_index])
        query = self.input_transform(query)
        positive = self.input_transform(positive)
        negatives = []
        for neg_index in neg_indices:
            negative = Image.open(self.db_struct.gallery_images[neg_index])
            negative = self.input_transform(negative)
            negatives.append(negative)
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [index, pos_index] + neg_indices.tolist()
    def __len__(self):
        return len(self.queries)


"""
# questo codice trova le immagini query che non hanno una positiva entro 10 metri
# si consiglia di eseguirlo sui dataset di train, visto che le query di train senza
# gallery entro 10 metri non vengono usate. 
struct = parse_db_struct("/home/valerio/datasets/pitts30k/image/train", "gallery", "queries")
knn = NearestNeighbors(n_jobs=-1)
knn.fit(struct.gallery_utms)
nontrivial_positives = list(knn.radius_neighbors(struct.query_utms, # 10 metri
                                                 radius=struct.train_pos_dist_threshold,
                                                 return_distance=False))

for i, posi in enumerate(nontrivial_positives):
    nontrivial_positives[i] = np.sort(posi)

queries = np.where(np.array([len(x) for x in nontrivial_positives]) == 0)[0]
queries_paths = [struct.query_images[q_index] for q_index in queries]
print(f"sto per eliminare {len(queries)} immagini query, sei sicuro di voler continuare?")
#for path in queries_paths: os.remove(path) # decommenta questo per eliminare le immagini
"""

