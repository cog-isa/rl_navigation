import torch
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2,3})

class ToTensor:
    def __init__(self, is_test=False):
        self.is_test = is_test
    
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)
#         depth = depth.resize((320, 240))
        
        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000
            
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)
        
        return {'image': image, 'depth': depth}
    
    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            
            return img.float().div(255)
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img
        
class RandomHorizontalFlip:
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        if not _is_pil_image(image):
            raise TypeError(
                    'img should be PIL Image. Gor {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                    'img should be PIL Image. Gor {}'.format(type(depth)))
        
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        
        return {'image': image, 'depth': depth}

class RandomChannelSwap:
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))
    
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError(
                    'img should be PIL Image. Gor {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                    'img should be PIL Image. Gor {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) -1)])])
        
        return {'image': image, 'depth': depth}
    
class depthDatasetMemory(torch.utils.data.Dataset):
    def __init__(self, path_to_data, dataset, transform=None):
        self.path_to_data, self.dataset = path_to_data, dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = Image.open('{}/rgb/{}'.format(self.path_to_data, sample[0]))
        depth = Image.open('{}/depth/{}'.format(self.path_to_data, sample[1]))
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample    
    
    def __len__(self):
        return len(self.dataset)
    
def loadToMem(path_to_csv):
    print('Loading dataset ', end='')
    
    train = []
    with open(path_to_csv, mode='r') as csv_file:
        for row in csv_file:
            row = row.strip()
            train.append(row.split(','))
              
    from sklearn.utils import shuffle
    train = shuffle(train, random_state=7)
    
    print('Loaded ({0})'.format(len(train)))
#     print('Loaded ({0})'.format(len(nyu2_test)))
    return train


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])

def getDefaultTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainTestData(batch_size, path_to_data):
    transformed_training = depthDatasetMemory(path_to_data=path_to_data+'/full_images_folder',
                                             dataset=loadToMem(path_to_data+'/full_images_folder/train.csv'),
                                             transform=getDefaultTransform())
    transformed_testing = depthDatasetMemory(path_to_data=path_to_data+'/validation',
                                             dataset=loadToMem(path_to_data+'/validation/validation.csv'),
                                             transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=True)