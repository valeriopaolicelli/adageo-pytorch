import torch
import torchvision.transforms as transforms
import torch.utils.data as data

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

dbStruct = namedtuple('dbStruct', ['dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'valPosDistThr', 'trainPosDistThr'])


def parse_dbStruct(rootPath, galleryPath, queryPath):
    galleryPath = os.path.join(rootPath, galleryPath)
    queryPath = os.path.join(rootPath, queryPath)
    if not os.path.exists(galleryPath): raise Exception(f"{galleryPath} non esiste")
    if not os.path.exists(queryPath): raise Exception(f"{queryPath} non esiste")
    db_images = sorted(glob.glob(f"{galleryPath}/**/*.jpg", recursive=True))
    q_images =  sorted(glob.glob(f"{queryPath}/**/*.jpg", recursive=True))
    db_utms =  np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in db_images])
    q_utms =   np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in q_images])
    numDb = len(db_images)
    numQ = len(q_images)
    valPosDistThr = 25
    trainPosDistThr = 10
    return dbStruct(db_images, db_utms, q_images, q_utms, numDb, numQ,
                    valPosDistThr, trainPosDistThr)


class WholeDataset(data.Dataset):
    """
    __getitem__(index) ritorna:
        img, index
    dove img può essere sia una gallery che una query
    """
    def __init__(self, rootPath, galleryPath, queryPath):
        super().__init__()
        self.input_transform = input_transform()
        self.l = []
        self.dbStruct = parse_dbStruct(rootPath, galleryPath, queryPath)
        self.images = [dbIm for dbIm in self.dbStruct.dbImage]
        self.images += [qIm for qIm in self.dbStruct.qImage]
        self.positives = None
        self.name = f"< queries: {queryPath} ({self.dbStruct.numQ}); gallery: {galleryPath} ({self.dbStruct.numDb}) >"
    def __getitem__(self, index):
        self.l.append(index)
        img = Image.open(self.images[index])
        img = self.input_transform(img)
        return img, index
    def __len__(self):
        return len(self.images)
    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.valPosDistThr,
                                                  return_distance=False)
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
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    # unisce le sottoliste della lista negatives, i confini sono delineati da negCounts
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))
    return query, positive, negatives, negCounts, indices


class QueryDataset(data.Dataset):
    """
    __getitem__(index) ritorna:
        query, positive, negatives, [index, posIndex] + negIndices.tolist()
    mentre con collate_fn ritorna:
        query, positive, negatives, negCounts, indices
    """
    def __init__(self, rootPath, galleryPath, queryPath):
        super().__init__()
        self.input_transform = input_transform()
        self.margin = 0.1
        self.dbStruct = parse_dbStruct(rootPath, galleryPath, queryPath)
        self.nNegSample = 1000 # number of negatives to randomly sample
        self.nNeg = 10  # number of negatives used for training
        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, # 10 metri
                                                              radius=self.dbStruct.trainPosDistThr,
                                                              return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives, lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]
        # potential negatives are those outside of valPosDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ, # 25 metri
                                                   radius=self.dbStruct.valPosDistThr,
                                                   return_distance=False)
        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                                                         pos, assume_unique=True))
        self.cache = None  # structure that contains the features cached from the whole train set
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]
        self.name = f"< queries: {queryPath} ({self.dbStruct.numQ}); gallery: {galleryPath} ({self.dbStruct.numDb}) >"
    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset
        ###############################################################
        # Start reading features whole train set (cache)
        qOffset = self.dbStruct.numDb
        qFeat = self.cache[index + qOffset]
        if np.all(qFeat==0):
            raise Exception(f"Per la query {index} con shape {qFeat.shape} le features sono tutte 0!!! Non le hai calcolate! {self.dbStruct.qImage[index]} | {self.nontrivial_positives[index].tolist()}")
        posFeat = self.cache[self.nontrivial_positives[index].tolist()]
        pool_size = self.cache.shape[1]
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(posFeat)
        dPos, posNN = faiss_index.search(qFeat.reshape(1, -1), 1)
        dPos = dPos.item()
        posIndex = self.nontrivial_positives[index][posNN[0]].item()
        negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
        negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
        feat_len = self.cache.shape[1]
        negFeat = np.zeros((len(negSample.tolist()), feat_len), dtype=np.float32)
        for idx, negativeFeature in enumerate(negSample.tolist()):
            negFeat[idx] = self.cache[int(negativeFeature)]
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(negFeat)
        dNeg, negNN = faiss_index.search(qFeat.reshape(1, -1), self.nNeg * 10)
        dNeg = dNeg.reshape(-1)
        negNN = negNN.reshape(-1)
        violatingNeg = (dNeg ** 0.5) < (dPos ** 0.5) + (self.margin ** 0.5)
        if np.sum(violatingNeg) < 1:
            # if none are violating then skip this query
            return None
        negNN = negNN[violatingNeg][:self.nNeg]
        negIndices = negSample[negNN].astype(np.int32)
        self.negCache[index] = negIndices
        # End reading features whole train set (cache)
        ###############################################################
        query = Image.open(self.dbStruct.qImage[index])
        positive = Image.open(self.dbStruct.dbImage[posIndex])
        query = self.input_transform(query)
        positive = self.input_transform(positive)
        negatives = []
        for negIndex in negIndices:
            negative = Image.open(self.dbStruct.dbImage[negIndex])
            negative = self.input_transform(negative)
            negatives.append(negative)
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()
    def __len__(self):
        return len(self.queries)


