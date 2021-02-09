
import os
import logging
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np
import faiss
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm

import network
import resnet
import grl_util
import datasets


def save_checkpoint(args, state, is_best, filename):
    os.makedirs(f"{args.output_folder}/models", exist_ok=True)
    model_path = f"{args.output_folder}/models/{filename}"
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, f"{args.output_folder}/best_model.pth")


def resume_train(args, model, optimizer):
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    best_score = checkpoint["best_score"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    logging.debug(f"Loaded checkpoint: start_epoch = {start_epoch}, " \
                  f"current_best_R@5 = {best_score:.1f}")
    return model, optimizer, best_score, start_epoch


def get_clusters(args, cluster_set, model):
    num_descriptors = 50000
    desc_per_image = 40
    num_images = ceil(num_descriptors / desc_per_image)
    if not "biost" in args.train_q: # TODO set a parameter
        cluster_set = Subset(cluster_set, list(range(cluster_set.db_struct.num_gallery)))
    
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), num_images, replace=True))
    dataloader = DataLoader(dataset=cluster_set,
                            num_workers=args.num_workers, batch_size=args.cache_batch_size, 
                            shuffle=False, sampler=sampler)
    with torch.no_grad():
        model = model.eval().to(args.device)
        logging.debug(f"Extracting {'attentive' if args.attention else ''} descriptors ")
        descriptors = np.zeros(shape=(num_descriptors, args.encoder_dim), dtype=np.float32)
        for iteration, (inputs, indices) in enumerate(tqdm(dataloader, ncols=100), 1):
            inputs = inputs.to(args.device)
            encoder_out = model(inputs, mode="feat")
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), args.encoder_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * args.cache_batch_size * desc_per_image
            for ix in range(image_descriptors.size(0)):
                sample = np.random.choice(image_descriptors.size(1), desc_per_image, replace=False)
                startix = batchix + ix * desc_per_image
                descriptors[startix:startix + desc_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
    niter = 100
    kmeans = faiss.Kmeans(args.encoder_dim, args.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
    return kmeans.centroids, descriptors


def build_model(args):
    logging.debug(f"Building {'attentive ' if args.attention else ''}NetVLAD {'with GRL' if args.grl else ''}")
    args.encoder_dim = 512
    backbone = resnet.resnet18(pretrain=args.pretrain)
    logging.debug("Train only layer3 and layer4 of the backbone, freeze the previous ones")
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    
    netvlad_layer = network.NetVLAD(num_clusters=args.num_clusters, dim=args.encoder_dim)
    
    if args.grl:
        grl_discriminator = grl_util.get_discriminator(args.encoder_dim, len(args.grl_datasets.split("+")))
    else:
        grl_discriminator = None
    
    model = network.AttenNetVLAD(backbone, netvlad_layer, grl_discriminator, attention=args.attention)
    
    if not args.resume:
        cluster_set = datasets.WholeDataset(args.dataset_root, args.train_g, args.train_q)
        logging.debug(f"Compute clustering and initialize NetVLAD layer based on {cluster_set}")
        centroids, descriptors = get_clusters(args, cluster_set, model)
        model.netvlad_layer.init_params(centroids, descriptors)
    
    return model.to(args.device)

