from collections import defaultdict
import math, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
from einops import rearrange, reduce
from PIL import Image

# category encoding functions 

def decimal_to_bits(x, bits):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    x = (x * 255).int().clamp(0, 255)


    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b c h w -> b c 1 h w')

    bits = ((x & mask) != 0).float()
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits * 2 - 1
    return bits

def bits_to_decimal(x, bits):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)

    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d = bits)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0., 1.)

def decimal_to_onehot(x, num_classes, log=False):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    device = x.device
    x = x.squeeze(0)
    x_onehot = F.one_hot(x.to(torch.int64), num_classes).to(device)
    x_onehot = rearrange(x_onehot, 'h w d -> d h w').float()
    
    if log:
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
        log_x = torch.log(x_onehot.float().clamp(min=1e-30), device=device)
        
        return log_x

    return x_onehot

def onehot_to_decimal(x):
    """ output s image tensor from 0 to 1 """
    decimal = x.argmax(1)
    decimal = decimal.unsqueeze(1)
    return (decimal / 255).clamp(0., 1.)

class CityscapesBase(Dataset):
    def __init__(
        self,
        folder,
        image_size, # final resolution = (img_size) x (2 * img_size)
        class_encoding,
        num_classes,
        bits=5,
        bit_scale=1., 
        mask_folder = None,
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.mask_folder = mask_folder
        self.cities = defaultdict(list)
        self.images = []
        self.masks = []
        self.class_encoding = class_encoding
        self.bits = bits
        self.bit_scale = bit_scale
        self.num_classes = num_classes
        
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
        example = dict()
        image_name = self.images[index]
        mask_name = self.masks[index]
        image_path = Path(self.folder, image_name) 
        mask_path = Path(self.mask_folder, mask_name) if self.mask_folder is not None else None
        img = Image.open(image_path)
        mask = Image.open(mask_path) if mask_path is not None else None
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
        
        if self.class_encoding == 'analog_bits':
            mask = decimal_to_bits(mask, self.bits) * self.bit_scale
        elif self.class_encoding == 'onehot':
            mask = decimal_to_onehot(mask, self.num_classes)
        else:
            raise ValueError(f'invalid category encoding {self.class_encoding}')

        example["image"] = self.transform(img)
        example["segmentation"] = mask
        return example
    
    class CityscapesTrain(CityscapesBase):
        def __init__(self, **kwargs):
            super().__init__(folder="/home/juyeon/data/cityscapes/leftImg8bit/train", mask_folder="/home/juyeon/data/cityscapes/gtFine/train", **kwargs)
            
    class CityscapesValidation(CityscapesBase):
        def __init__(self, **kwargs):
            super().__init__(folder="/home/juyeon/data/cityscapes/leftImg8bit/val", mask_folder="/home/juyeon/data/cityscapes/gtFine/val", **kwargs)