
import const
import netvlad
import network
import resnet
import alexnet
import grl_util
import commons
import datasets

from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np
import faiss
from os.path import join, isfile
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def get_ckpt_path(resume_path, ckpt):
    if ckpt == 'latest':
        resume_ckpt = join(resume_path, 'checkpoint.pth')
    elif ckpt == 'best':
        resume_ckpt = join(resume_path, 'best_model.pth')
    return resume_ckpt


def get_state_dict(old, new):
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in old.items() if k in new}
    # 2. overwrite entries in the existing state dict
    new.update(pretrained_dict)
    # 3. load the new state dict
    return new


def save_checkpoint(opt, state, is_best, filename):
    commons.create_dir_if_not_exists(join(opt.outputFolder, "models"))
    model_out_path = join(opt.outputFolder, "models", filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.outputFolder, 'best_model.pth'))


def resume_train(opt, model, optimizer):
    resume_ckpt = get_ckpt_path(opt.resume, opt.ckpt.lower())
    if isfile(resume_ckpt):
        output = f"Loading checkpoint: {resume_ckpt}"
        opt.logger.log(output, False)
        
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        
        if 'epoch' in checkpoint.keys():
            opt.start_epoch = checkpoint['epoch']
        
        if "state_dict" in checkpoint:
            model.load_state_dict(get_state_dict(checkpoint["state_dict"], model.state_dict()))
        elif "model" in checkpoint:
            model.load_state_dict(get_state_dict(checkpoint["model"], model.state_dict()))
        model = model.to(opt.device)
        
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
        else:
            best_score = 0
        
        optimizer.load_state_dict(get_state_dict(checkpoint['optimizer'], optimizer.state_dict()))
        current_lr = optimizer.param_groups[0]['lr']
        output = f"Train parameters: lr= {current_lr}, start_epoch= {opt.start_epoch}, " \
                 f"nEpochs= {opt.nEpochs}, current_best_recall@5= {best_score}"
        opt.logger.log(output, False)
        
        output = f"Loaded checkpoint: {resume_ckpt} (epoch {checkpoint['epoch']})"
        opt.logger.log(output, False)
        
    else:
        output = f"No checkpoint found at: {resume_ckpt}"
        opt.logger.log(output, False)
        raise Exception(f"No checkpoint found at: {resume_ckpt}")
    return opt, model, optimizer, best_score


def get_clusters(opt, cluster_set, model):
    nDescriptors = 50000
    nPerImage = 40
    nIm = ceil(nDescriptors / nPerImage)

    if not "biost" in opt.trainQ: # TODO mettere un parametro per togliere questo controllo
        cluster_set = Subset(cluster_set, list(range(cluster_set.dbStruct.numDb)))

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=True))
    data_loader = DataLoader(dataset=cluster_set,
                             num_workers=opt.num_workers, batch_size=opt.cacheBatchSize, 
                             shuffle=False, sampler=sampler)
    with torch.no_grad():
        model.eval()
        model.cuda()
        output = f'Extracting Descriptors {"weighted" if opt.attention else ""}'
        opt.logger.log(output, False)
        
        descriptors = np.zeros(shape=(nDescriptors, opt.encoder_dim), dtype=np.float32)
        for iteration, (input, indices) in enumerate(tqdm(data_loader), 1):
            input = input.to(opt.device)
            encoder_out = model(input, mode='atten-feat')
            l2_out = F.normalize(encoder_out, p=2, dim=1)
            image_descriptors = l2_out.view(l2_out.size(0), opt.encoder_dim, -1).permute(0, 2, 1)
            batchix = (iteration - 1) * opt.cacheBatchSize * nPerImage
            for ix in range(image_descriptors.size(0)):
                # sample different location for each image in batch
                sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                startix = batchix + ix * nPerImage
                descriptors[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
            
            del input, image_descriptors
            if opt.isDebug: break
    
    opt.logger.log(f'Clustering..', False)
    
    niter = 100
    kmeans = faiss.Kmeans(opt.encoder_dim, const.num_clusters, niter=niter, verbose=False)
    kmeans.train(descriptors)
    
    opt.logger.log(f'Storing centroids {kmeans.centroids.shape}', False)
    
    centroids = kmeans.centroids
    opt.logger.log(f'Done!', False)
    
    return centroids, descriptors


def build_model(opt):
    ######## Build the encoder branch
    if opt.arch == 'resnet18':
        opt.encoder_dim = 512
        backbone = resnet.resnet18(pretrained=True, noBN=True, pretrain=opt.pretrain.lower())
        if opt.train_part:
            # Train only the last 4th and 5th convblock of the backbone
            opt.logger.log(f"Train partial network!", False)
    
            for name, child in backbone.named_children():
                if name == 'layer3':
                    break
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        else:
            opt.logger.log(f"Train whole network!", False)

    ######## Build the netvlad branch
    netvlad_layer = netvlad.NetVLAD(num_clusters=const.num_clusters, dim=opt.encoder_dim)
    
    if opt.grl:
        grl_discriminator = grl_util.get_discriminator(opt.encoder_dim, len(opt.grlDatasets.split("+")))
    else:
        grl_discriminator = None
    
    model = network.AttenNetVLAD(backbone, netvlad_layer, grl_discriminator, attention=opt.attention)
    opt.logger.log(f'Built AttenNetVLAD!', False)
    
    if not opt.resume:
        opt.logger.log(f"Compute clustering and init VLAD layer", False)
        cluster_set = datasets.WholeDataset(opt.rootPath, opt.trainG, opt.trainQ)
        centroids, descriptors = get_clusters(opt, cluster_set, model)
        model.netvlad_layer.init_params(centroids, descriptors)
        del centroids, descriptors, cluster_set
    
    return model

