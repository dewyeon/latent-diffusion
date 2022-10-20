from collections import defaultdict
import math, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class ADE20KDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        mask_folder = None,
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.mask_folder = mask_folder
        self.image_size = image_size[0]
        self.images = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]

        # maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        path = self.images[index]
        _, image = os.path.split(path)
        image = image.replace('jpg', 'png')
        mask_path = Path(self.mask_folder, image) if self.mask_folder is not None else None
        img = Image.open(path).convert('RGB')
        mask = Image.open(mask_path) if mask_path is not None else None
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return self.transform(img), mask
