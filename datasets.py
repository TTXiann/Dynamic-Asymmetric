import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, args, split):

        data_folder = args.data_folder
        data_name = args.data_name

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = torch.LongTensor(json.load(j))[:, :args.max_len]


        if 'LEVIR_CC' in data_name:
            img_path = '/project/datasets/RSICC/Levir-CC-dataset/images'

            if self.split == 'TRAIN':
                self.cpi = 5         
                self.dataset_size = int(len(self.captions) // 1)
                self.img_path = os.path.join(img_path, 'train')
                self.ids = json.load(open(os.path.join(data_folder, 'train_ids.json'), 'r')) # 1-to-5
            else:
                self.cpi = 1   
                self.dataset_size = int(len(self.captions) // 5)
                self.img_path = os.path.join(img_path, 'test')
                self.ids = json.load(open(os.path.join(data_folder, 'test_ids.json'), 'r'))

            self.pathA = 'A'
            self.pathB = 'B'


        elif 'DUBAI' in data_name:
            self.img_path = '/project/datasets/RSICC/DUBAI'

            if self.split == 'TRAIN':
                self.cpi = 5         
                self.dataset_size = int(len(self.captions) // 1)
                self.ids = json.load(open(os.path.join(data_folder, 'train_ids.json'), 'r'))
            else:
                self.cpi = 1   
                self.dataset_size = int(len(self.captions) // 5)
                self.ids = json.load(open(os.path.join(data_folder, 'test_ids.json'), 'r'))

            self.pathA = '500_2000'
            self.pathB = '500_2010'

        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        ids = self.ids[index // self.cpi]

        if self.transform is not None:
            img1 = Image.open(os.path.join(self.img_path, self.pathA, ids)).convert('RGB')
            img2 = Image.open(os.path.join(self.img_path, self.pathB, ids)).convert('RGB')
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = torch.stack([img1, img2], dim=0)

        if self.split == 'TRAIN': # 1-to-1
            caption = self.captions[index]
            return imgs, caption
        else: # 1-to-5
            caption = self.captions[(index*5):(index+1)*5] # 1-to-5
            return imgs, caption

    def __len__(self):
        return self.dataset_size
