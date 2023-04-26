import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import argparse

import resnet


def get_cam(feature_conv, weight_softmax, class_idx, raw_img_size):
    # generate the class activation maps
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, raw_img_size))
    return output_cam


def extract_cam(net, img_pil, img_variable, features_blobs, weight_softmax):
    raw_img_size = (img_pil.size[0], img_pil.size[1])
    output = net(img_variable)  # returns outputs of fc, layer4 with BN, layer4 without BN layers
    feature = output[2].cpu().detach().numpy()
    probs, idx = output[0].data.squeeze().sort(0, True)
    idx = idx.cpu().numpy()
    # generate class activation mapping for the top1 prediction
    top1_class = [idx[0]]
    CAMs = get_cam(feature, weight_softmax, top1_class, raw_img_size)
    # render the CAM and output
    img = np.asarray(img_pil)
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], img_pil.size), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    return result


def hook_feature_places(module, input, output):
    output = output[0]
    features_blobs_places.append(output.data.cpu().numpy())


######################################### MAIN #########################################
parser = argparse.ArgumentParser('CAM getter')
parser.add_argument('-D', '--dataset_root', required=True, help='Dataset path')
parser.add_argument('-O', '--output_path', default=None, help='Output path')
parser.add_argument('-R', '--resume', required=True, help='Checkpoint path')
args = parser.parse_args()
print(f"Arguments: {args}")

# Build and resume model
backbone = resnet.resnet18(pretrain=args.pretrain).cuda()
model_state_dict = torch.load(args.resume)["state_dict"]
new_dict = {k.replace('backbone.', ''): v for k, v in model_state_dict.items() if 'backbone' in k}
backbone.load_state_dict(new_dict, strict=True)

# Hook the feature extractor
features_blobs_places = []
backbone._modules.get('layer4').register_forward_hook(hook_feature_places)
weight_softmax = np.squeeze(list(backbone.parameters())[-2].cpu().data.numpy())

# Search recursively all the images in dataset_root path
images = sorted(glob(os.path.join(args.dataset_root, '**', '*.jpg'), recursive=True) +
                glob(os.path.join(args.dataset_root, '**', '*.png'), recursive=True))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

for image_fn in tqdm(images, ncols=100):
    img_pil = Image.open(image_fn)
    img_variable = Variable(preprocess(img_pil).unsqueeze(0)).cuda()
    result = extract_cam(backbone, img_pil, img_variable, features_blobs_places, weight_softmax)

    path_rgb = args.output_path if args.output_path is not None else os.path.dirname(image_fn) + '_color_cam'
    if not os.path.exists(path_rgb):
        os.makedirs(path_rgb)

    name = os.path.basename(image_fn)
    cv2.imwrite(os.path.join(path_rgb, name), result)

