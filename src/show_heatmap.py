from os import listdir, mkdir
from os.path import join, exists

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from tqdm import tqdm
import resnet_places


netPlaces = resnet_places.resnet18(pretrained=True, noBN=True)
netImagenet = resnet_places.resnet18(pretrained=True, noBN=True)
finalconv_name = 'layer4'

resume_places = 'runs/cam-places/checkpoints/model_best.pth.tar'
dict_places = torch.load(resume_places)
netPlaces.load_state_dict(dict_places, strict=False)
netPlaces.eval()

resume_Imagenet = 'runs/cam-imagenet/checkpoints/model_best.pth.tar'
dict_Imagenet = torch.load(resume_Imagenet)
netImagenet.load_state_dict(dict_Imagenet, strict=False)
netImagenet.eval()


# hook the feature extractor
features_blobsplaces = []
def hook_featureplaces(module, input, output):
    output = output[0]
    features_blobsplaces.append(output.data.cpu().numpy())

features_blobsImagenet = []
def hook_featureImagenet(module, input, output):
    output = output[0]
    features_blobsImagenet.append(output.data.cpu().numpy())

netPlaces._modules.get(finalconv_name).register_forward_hook(hook_featureplaces)
netImagenet._modules.get(finalconv_name).register_forward_hook(hook_featureImagenet)

# get the softmax weight
paramsplaces = list(netPlaces.parameters())
weight_softmaxplaces = np.squeeze(paramsplaces[-2].data.numpy())

paramsImagenet = list(netImagenet.parameters())
weight_softmaxImagenet = np.squeeze(paramsImagenet[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


def extractCAM(net, img_pil, img_variable, features_blobs, weight_softmax):
    logit = net(img_variable)
    feature = logit[2].cpu().detach().numpy()
    logit = logit[0]
    probs, idx = logit.data.squeeze().sort(0, True)
    idx = idx.numpy()
    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(feature, weight_softmax, [idx[0]], size_upsample)
    # render the CAM and output
    img = np.asarray(img_pil)
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], img_size), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    return result

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess1 = transforms.Compose([
    # transforms.CenterCrop(224),
    # transforms.Resize(480),
])

preprocess3 = transforms.Compose([
    transforms.CenterCrop(224),
    # transforms.Resize(480),
])

preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    normalize])


num_samples = 50
sets = ['val/queries/00', 'test/queries_1', 'test/queries_2', 'test/queries_3', 'test/queries_4', 'test/queries_5']
in_path = f'/home/valerio/datasets/oxford60k/image'
out_path = f'/home/valerio/datasets/examples/CAM'
if not exists(out_path):
    mkdir(out_path)
for i, folder in enumerate(sets, 1):
    path = join(in_path, folder)
    if not exists(path):
        raise Exception(f'Image path not found: {path}')
    o_path = join(out_path, folder.split('/')[1])
    if not exists(o_path):
        mkdir(o_path)
    images = sorted(list(listdir(path)))[:num_samples]
    for index, image in tqdm(enumerate(images, 1)):
        img_pil = Image.open(join(path, image))
        # if i > 1:
        #     img = np.asarray(img_pil)
        #     img = img[:-70, 20:-20, :]
        #     img = Image.fromarray(img)
        #     img_pil1 = preprocess3(img_pil)
        #
        # else:
        #     img_pil1 = preprocess1(img_pil)
        img_pil1 = preprocess1(img_pil)
        del img_pil
        if i == 1:
            img_pil1.save(join(o_path, f'{index}_sourceval_original.jpg'))
        else:
            img_pil1.save(join(o_path, f'{index}_target{i-2}_original.jpg'))

        img_size = img_pil1.size
        size_upsample = (img_size[0], img_size[1])
        img_tensor = preprocess2(img_pil1)
        img_variable = Variable(img_tensor.unsqueeze(0))

        img = np.asarray(img_pil1)

        result = extractCAM(netPlaces, img_pil1, img_variable, features_blobsplaces, weight_softmaxplaces)
        if i == 1:
            cv2.imwrite(join(o_path, f'{index}_sourceval_places_c.jpg'), result)
        else:
            cv2.imwrite(join(o_path, f'{index}_target{i-2}_places_c.jpg'), result)

        result = extractCAM(netImagenet, img_pil1, img_variable, features_blobsImagenet, weight_softmaxImagenet)
        if i == 1:
            cv2.imwrite(join(o_path, f'{index}_sourceval_imagenet_c.jpg'), result)
        else:
            cv2.imwrite(join(o_path, f'{index}_target{i-2}_imagenet_c.jpg'), result)
