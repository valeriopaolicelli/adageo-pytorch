
import os
import logging
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np
import faiss
from os.path import join
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm

import network
import resnet
import grl_util
import datasets


def save_checkpoint(opt, state, is_best, filename):
    os.makedirs(join(opt.output_folder, "models"), exist_ok=True)
    model_path = join(opt.output_folder, "models", filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(opt.output_folder, "best_model.pth"))


def resume_train(opt, model, optimizer):
    logging.debug(f"Loading checkpoint: {opt.resume}")
    checkpoint = torch.load(opt.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    best_score = checkpoint["best_score"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    logging.debug(f"Loaded checkpoint: start_epoch = {start_epoch}, " \
                  f"current_best_recall@5 = {best_score}")
    return opt, model, optimizer, best_score, start_epoch


def get_clusters(opt, cluster_set, model):
    num_descriptors = 50000
    desc_per_image = 40
    num_images = ceil(num_descriptors / desc_per_image)
    if not "biost" in opt.train_q: # TODO set a parameter
        cluster_set = Subset(cluster_set, list(range(cluster_set.db_struct.num_gallery)))
    
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), num_images, replace=True))
    dataloader = DataLoader(dataset=cluster_set,
                            num_workers=opt.num_workers, batch_size=opt.cache_batch_size, 
                            shuffle=False, sampler=sampler)
    with torch.no_grad():
        model = model.eval().to(opt.device)
        logging.debug(f"Extracting {'attentive' if opt.attention else ''} descriptors ")
        descriptors = np.zeros(shape=(num_descriptors, opt.encoder_dim), dtype=np.float32)
        for iteration, (inputs, indices) in enumerate(tqdm(dataloader, ncols=100), 1):
            inputs = inputs.to(opt.device)
            encoder_out = model(inputs, mode="feat")
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), opt.encoder_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * opt.cache_batch_size * desc_per_image
            for ix in range(image_descriptors.size(0)):
                sample = np.random.choice(image_descriptors.size(1), desc_per_image, replace=False)
                startix = batchix + ix * desc_per_image
                descriptors[startix:startix + desc_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
    niter = 100
    kmeans = faiss.Kmeans(opt.encoder_dim, opt.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
    return kmeans.centroids, descriptors


def build_model(opt):
    logging.debug(f"Building {'attentive ' if opt.attention else ''}NetVLAD {'with GRL' if opt.grl else ''}")
    opt.encoder_dim = 512
    backbone = resnet.resnet18(pretrain=opt.pretrain)
    logging.debug("Train only layer3 and layer4 of the backbone, freeze the previous ones")
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for name2, params in child.named_parameters():
            params.requires_grad = False
    
    netvlad_layer = network.NetVLAD(num_clusters=opt.num_clusters, dim=opt.encoder_dim)
    
    if opt.grl:
        grl_discriminator = grl_util.get_discriminator(opt.encoder_dim, len(opt.grl_datasets.split("+")))
    else:
        grl_discriminator = None
    
    model = network.AttenNetVLAD(backbone, netvlad_layer, grl_discriminator, attention=opt.attention)
    
    if not opt.resume:
        cluster_set = datasets.WholeDataset(opt.root_path, opt.train_g, opt.train_q)
        logging.debug(f"Compute clustering and initialize NetVLAD layer based on {cluster_set.info}")
        centroids, descriptors = get_clusters(opt, cluster_set, model)
        model.netvlad_layer.init_params(centroids, descriptors)
    
    return model

