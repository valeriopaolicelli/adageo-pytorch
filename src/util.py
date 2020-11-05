
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

import netvlad
import network
import resnet
import grl_util
import datasets


def get_ckpt_path(resume_path, ckpt):
    if ckpt == "latest":
        resume_ckpt = join(resume_path, "checkpoint.pth")
    elif ckpt == "best":
        resume_ckpt = join(resume_path, "best_model.pth")
    return resume_ckpt


def get_state_dict(old, new):
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in old.items() if k in new}
    # 2. overwrite entries in the existing state dict
    new.update(pretrained_dict)
    # 3. load the new state dict
    return new


def save_checkpoint(opt, state, is_best, filename):
    os.makedirs(join(opt.output_folder, "models"), exist_ok=True)
    model_out_path = join(opt.output_folder, "models", filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.output_folder, "best_model.pth"))


def resume_train(opt, model, optimizer):
    resume_ckpt = get_ckpt_path(opt.resume, opt.ckpt)
    if os.path.isfile(resume_ckpt):
        output = f"Loading checkpoint: {resume_ckpt}"
        logging.debug(output)
        
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        
        if "epoch" in checkpoint.keys():
            opt.start_epoch = checkpoint["epoch"]
        
        if "state_dict" in checkpoint:
            model.load_state_dict(get_state_dict(checkpoint["state_dict"], model.state_dict()))
        elif "model" in checkpoint:
            model.load_state_dict(get_state_dict(checkpoint["model"], model.state_dict()))
        model = model.to(opt.device)
        
        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]
        else:
            best_score = 0
        
        optimizer.load_state_dict(get_state_dict(checkpoint["optimizer"], optimizer.state_dict()))
        current_lr = optimizer.param_groups[0]["lr"]
        output = f"Train parameters: lr= {current_lr}, start_epoch= {opt.start_epoch}, " \
                 f"n_epochs= {opt.n_epochs}, current_best_recall@5= {best_score}"
        logging.debug(output)
        
        output = f"Loaded checkpoint: {resume_ckpt} (epoch {checkpoint['epoch']})"
        logging.debug(output)
        
    else:
        output = f"No checkpoint found at: {resume_ckpt}"
        logging.debug(output)
        raise Exception(f"No checkpoint found at: {resume_ckpt}")
    return opt, model, optimizer, best_score


def get_clusters(opt, cluster_set, model):
    num_descriptors = 50000
    desc_per_image = 40
    num_images = ceil(num_descriptors / desc_per_image)

    if not "biost" in opt.train_q: # TODO mettere un parametro per togliere questo controllo
        cluster_set = Subset(cluster_set, list(range(cluster_set.db_struct.num_gallery)))
    
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), num_images, replace=True))
    dataloader = DataLoader(dataset=cluster_set,
                            num_workers=opt.num_workers, batch_size=opt.cache_batch_size, 
                            shuffle=False, sampler=sampler)
    with torch.no_grad():
        model = model.eval().to(opt.device)
        output = f"Extracting Descriptors {'weighted' if opt.attention else ''}"
        logging.debug(output)
        
        descriptors = np.zeros(shape=(num_descriptors, opt.encoder_dim), dtype=np.float32)
        for iteration, (input, indices) in enumerate(tqdm(dataloader), 1):
            input = input.to(opt.device)
            encoder_out = model(input, mode="feat")
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), opt.encoder_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * opt.cache_batch_size * desc_per_image
            for ix in range(image_descriptors.size(0)):
                # sample different location for each image in batch
                sample = np.random.choice(image_descriptors.size(1), desc_per_image, replace=False)
                startix = batchix + ix * desc_per_image
                descriptors[startix:startix + desc_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
            
            del input, image_descriptors
            if opt.is_debug: break
    
    logging.debug("Clustering..")
    
    niter = 100
    kmeans = faiss.Kmeans(opt.encoder_dim, opt.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    
    logging.debug("Storing centroids {kmeans.centroids.shape}")
    
    centroids = kmeans.centroids
    logging.debug("Done!")
    
    return centroids, descriptors


def build_model(opt):
    ######## Build the encoder branch
    opt.encoder_dim = 512
    backbone = resnet.resnet18(pretrained=True, noBN=True, pretrain=opt.pretrain)
    if opt.train_part:
        # Train only the last 4th and 5th convblock of the backbone
        logging.debug("Train partial network!")
        for name, child in backbone.named_children():
            if name == "layer3":
                break
            for name2, params in child.named_parameters():
                params.requires_grad = False
    else:
        logging.debug("Train whole network!")

    ######## Build the netvlad branch
    netvlad_layer = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=opt.encoder_dim)
    
    if opt.grl:
        grl_discriminator = grl_util.get_discriminator(opt.encoder_dim, len(opt.grl_datasets.split("+")))
    else:
        grl_discriminator = None
    
    model = network.AttenNetVLAD(backbone, netvlad_layer, grl_discriminator, attention=opt.attention)
    logging.debug("Built AttenNetVLAD!")
    
    if not opt.resume:
        logging.debug("Compute clustering and init VLAD layer")
        cluster_set = datasets.WholeDataset(opt.root_path, opt.train_g, opt.train_q)
        centroids, descriptors = get_clusters(opt, cluster_set, model)
        model.netvlad_layer.init_params(centroids, descriptors)
    
    return model

