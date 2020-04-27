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
        rgb0, rgb1, depth0, depth1 = sample['rgb0'], sample['rgb1'], sample['depth0'], sample['depth1']
        
        rgb0 = self.to_tensor(rgb0)
        rgb1 = self.to_tensor(rgb1)
        depth0 = self.to_tensor(depth0)
        depth1 = self.to_tensor(depth1)
        
#         print(sample['d_gps'], sample['d_compas'], torch.FloatTensor(sample['d_gps']), torch.FloatTensor([sample['d_compas']]))
        return {'rgb0': rgb0,'rgb1': rgb1, 'depth0': depth0, 'depth1': depth1,
                'd_gps': torch.FloatTensor(sample['d_gps']), 'd_compas': torch.FloatTensor([sample['d_compas']]),
                'compas': torch.FloatTensor([sample['compas']])}
    
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
        
        
class depthDatasetMemory(torch.utils.data.Dataset):
    def __init__(self, path_to_data, dataset, transform=None):
        self.path_to_data, self.dataset = path_to_data, dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rgb0 = Image.open('{}/{}'.format(self.path_to_data, sample[0]))
        rgb1 = Image.open('{}/{}'.format(self.path_to_data, sample[1]))
        depth0 = Image.open('{}/{}'.format(self.path_to_data, sample[2]))
        depth1 = Image.open('{}/{}'.format(self.path_to_data, sample[3]))
        d_gps = sample[4]
        d_compas = sample[5]
        compas = sample[6]
        sample = {'rgb0': rgb0,'rgb1': rgb1, 'depth0': depth0, 'depth1': depth1, 
                  'd_gps': d_gps, 'd_compas': d_compas, 'compas': compas}
        if self.transform: sample = self.transform(sample)
        return sample    
    
    def __len__(self):
        return len(self.dataset)
    
def loadToMem(path_to_csv):
    print('Loading dataset ', end='')
    
    train = []
    with open(path_to_csv, mode='r') as csv_file:
        #reduce data
        max_data = 5000
        for i, row in enumerate(csv_file):
            row = row.strip()
            row = row.split(',')
            rgb0, rgb1, depth0, depth1 = row[0], row[1], row[2], row[3]
            d_gps = np.array(list(map(np.float, row[4].strip('[]').split())))
            d_compas = np.float(row[5].strip('[]'))
            compas = np.float(row[6].strip('[]'))
            train.append((rgb0, rgb1, depth0, depth1, d_gps, d_compas, compas))
            if i >= max_data:
                break
    from sklearn.utils import shuffle
    train = shuffle(train, random_state=7)
    
    print('Loaded ({0})'.format(len(train)))
#     print('Loaded ({0})'.format(len(nyu2_test)))
    return train


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])

def getTrainTestData(batch_size, path_to_data):
    transformed_training = depthDatasetMemory(path_to_data=path_to_data+'/Train',
                                             dataset=loadToMem(path_to_data+'/Train/train.csv'),
                                             transform=getNoTransform())
    transformed_testing = depthDatasetMemory(path_to_data=path_to_data+'/Val',
                                             dataset=loadToMem(path_to_data+'/Val/val.csv'),
                                             transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=True)