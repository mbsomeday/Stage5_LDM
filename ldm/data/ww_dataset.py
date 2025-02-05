import torch, os, torchvision, random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class my_dataset(Dataset):
    def __init__(self, ds_dir, txt_name, get_data_start, get_data_end):
        # # 先判断是否是D4，是的话需要单独处理
        # ds_name = ds_dir.split(os.sep)[-1][:2]
        # self.D4_flag = True if ds_name == 'D4' else False

        self.ds_dir = ds_dir.replace('\\', os.sep)
        self.txt_name = txt_name
        self.get_data_start = get_data_start
        self.get_data_end = get_data_end
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.init_ImagesLabels()

    def init_ImagesLabels(self):
        images, labels = [], []

        txt_path = os.path.join(self.ds_dir, 'dataset_txt', self.txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()
        for line in data:
            line = line.replace('\\', os.sep)
            line = line.strip()
            contents = line.split()

            image_path = os.path.join(self.ds_dir, contents[0])
            images.append(image_path)
            labels.append(contents[-1])

        # # 只从数据集中取100个
        # random.seed(13)
        # random.shuffle(images)
        # random.shuffle(labels)
        #
        # images = images[:100]
        # labels = labels[:100]

        if self.get_data_end > len(images):
            self.get_data_end = len(images)

        images = images[self.get_data_start, self.get_data_end]
        labels = labels[self.get_data_start, self.get_data_end]

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        image = self.img_transforms(image)
        label = np.array(label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'file_path': image_path,
            'image_name': image_name
        }

        return image_dict











