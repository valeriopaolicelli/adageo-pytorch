
import glob
import random
import logging
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        dx = -grads.new_tensor(1) * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return GradientReversalFunction.apply(x)


def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator

grl_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class GrlDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, datasets_paths):
        """
        datasets_paths è una lista con le cartelle che contengono gli N datasets.
        dall'esterno GrlDataset ha 1000000 elementi, e quando ne richiedo uno
        me ne torna uno a caso, del dataset index % N, per assicurarsi che ogni
        dataset abbia la stessa probabilità di uscire
        """
        super().__init__()
        self.num_classes = len(datasets_paths)
        logging.info(f"GrlDataset ha {self.num_classes} classi")
        self.images_paths = []
        for dataset_path in datasets_paths:
            self.images_paths.append(sorted(glob.glob(f"{root_path}/{dataset_path}/**/*.jpg", recursive=True)))
            logging.info(f"    La classe {dataset_path} ha {len(self.images_paths[-1])} immagini")
            if len(self.images_paths[-1]) == 0:
                raise Exception(f"Ha 0 immagini, c'è qualche problema, lancio un'eccezione !!!")
        # suppongo che tutte le immagini abbiano la stessa dimensione
        self.transform = grl_transform
    def __getitem__(self, index):
        num_class = index % self.num_classes
        images_of_class = self.images_paths[num_class]
        # ne prendo una a caso
        image_path = random.choice(images_of_class)
        tensor = self.transform(Image.open(image_path).convert("RGB"))
        return tensor, num_class
    def __len__(self):
        return 1000000

