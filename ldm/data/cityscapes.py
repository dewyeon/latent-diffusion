from collections import defaultdict
import math, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class CityscapesDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size, # final resolution = (img_size) x (2 * img_size)
        mask_folder = None,
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.mask_folder = mask_folder
        self.image_size = image_size[0]
        self.cities = defaultdict(list)
        self.images = []
        self.masks = []
        
        cities = [c for c in os.listdir(folder)]
        for c in cities:
            c_items = [name.split('_') for name in os.listdir(os.path.join(folder, c))]
            for it in c_items:
                name = it[1] + '_' + it[2]
                self.images.append(c + '/' + c + '_' + name + '_leftImg8bit.png')
                self.masks.append(c + '/' + c + '_' + name + '_gtFine_labelIds.png')

        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(image_size)
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        mask_name = self.masks[index]
        image_path = Path(self.folder, image_name) 
        mask_path = Path(self.mask_folder, mask_name) if self.mask_folder is not None else None
        img = Image.open(image_path)
        mask = Image.open(mask_path) if mask_path is not None else None
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return self.transform(img), mask