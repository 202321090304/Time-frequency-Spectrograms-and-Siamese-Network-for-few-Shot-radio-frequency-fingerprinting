import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F




class SPDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='train', split_ratio=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.data_type = data_type
        self.split_ratio = split_ratio

        for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
            print("name:{}, index:{}".format(class_name, idx))

        if data_type == 'train':
            for class_name in self.class_to_idx.keys():
                class_path = os.path.join(data_dir, class_name)
                path_a = os.path.join(class_path, 'a')
                path_b = os.path.join(class_path, 'b')

                images_a = self._collect_images(path_a) if os.path.exists(path_a) else []
                images_b = self._collect_images(path_b) if os.path.exists(path_b) else []
                print(f"Class: {class_name}, Images in 'a': {len(images_a)}, Images in 'b': {len(images_b)}")
                for img_a, img_b in zip(sorted(images_a), sorted(images_b)):
                    self.samples.append((img_a, img_b, self.class_to_idx[class_name]))

        elif data_type == 'test':
            for class_name in self.class_to_idx.keys():
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    for root, _, files in os.walk(class_path):
                        for file in sorted(files):
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                self.samples.append((file_path, self.class_to_idx[class_name]))

        random.shuffle(self.samples)
        self.num_samples = int(len(self.samples) * self.split_ratio)
        self.samples = self.samples[:self.num_samples]
        print(f"Loaded {self.num_samples} images out of {len(self.samples)} total images.")
        print(f"Number of classes: {len(self.class_to_idx)}")


    def _collect_images(self, folder_path):
        images = []
        for root, _, files in os.walk(folder_path):
            images.extend(
                [os.path.join(root, f) for f in sorted(files) if os.path.isfile(os.path.join(root, f))]
            )
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            img_path_a, img_path_b, label = self.samples[idx]
            image_a = Image.open(img_path_a).convert('L')
            image_b = Image.open(img_path_b).convert('L')

            if self.transform:
                image_a = self.transform(image_a)
                image_b = self.transform(image_b)

            return [image_a, image_b], label

        elif self.data_type == 'test':
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label
       


class InferenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        for root, dirs, files in os.walk(data_dir):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    self.samples.append((file_path, -1))  # -1 表示未知标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # 转为灰度图

        if self.transform:
            image = self.transform(image)

        return image, label
