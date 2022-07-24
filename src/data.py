# Library imports
import os
import numpy as np
from PIL import Image
import config
import torch
from itertools import combinations


def generate_pair_file_for_test_images(dataDict, root_path, class_dict):
    image_paths = []

    for k, v in dataDict.items():
        test_set = dataDict[k][1]    
        for p in test_set:
            if k+"/"+p not in image_paths:
                image_paths.append(k+"/"+p)    

    paired_list = list(combinations(image_paths, 2))

    ones = []
    zeros = []
    for tuple_ in paired_list:
        if tuple_[0].split('/')[0] == tuple_[1].split('/')[0]:
            ones.append(f"{tuple_[0]} {tuple_[1]} 1")
        else:
            zeros.append(f"{tuple_[0]} {tuple_[1]} 0")
            
    zeros = zeros[:len(ones)]            
    with open(os.path.join(root_path, "src", "test_pair.txt"), "w") as f:
        for o, z in zip(ones, zeros):
            f.write(o)
            f.write("\n")
            f.write(z)
            f.write("\n")


def get_train_test_images_data_dict(data_path):
    temp_dict = {}
    for class_name in os.listdir(data_path):
        curr_path = os.path.join(data_path, class_name)
        images = []
        for image in os.listdir(curr_path):
            images.append(image)
        indices = np.random.permutation(len(images))
        train_indices = indices[:int(indices.shape[0] * (1-config.TEST_SIZE))]
        test_indices = indices[int(indices.shape[0] * (1-config.TEST_SIZE)):]

        images = np.array(images)
        train_images_name = images[train_indices]
        test_images_name = images[test_indices]
        temp_dict[class_name] = (train_images_name, test_images_name)
    return temp_dict 


class ArcFaceDataset(torch.utils.data.dataset.Dataset):
    """
        data_path : path to the folder containing images
        train : to specifiy to load training or testing data 
        transform : Pytorch transforms [required - ToTensor(), optional - rotate, flip]
    """
    def __init__(self, data_path, class_dict, train_test_image_data_dict, train=True, transform=None):
        
        self.data_path = data_path
        self.train = train
        self.class_dict = class_dict
        self.train_test_image_data_dict = train_test_image_data_dict
        self.data, self.targets = self.load(self.data_path, train)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = Image.open(self.data[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.targets[idx]
            

    def load(self, data_path, train):
        images = []
        targets = []
        for class_name in os.listdir(data_path):
            target = self.class_dict[class_name]
            curr_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(curr_path):
                if image_name in self.train_test_image_data_dict[class_name][0] and train:
                    images.append(os.path.join(curr_path, image_name))
                    targets.append(target)
                elif image_name in self.train_test_image_data_dict[class_name][1] and not train:
                    images.append(os.path.join(curr_path, image_name))
                    targets.append(target)
        
        indices = np.random.permutation(len(images))
        images = np.array(images)[indices]
        targets = np.array(targets, dtype=np.int64)[indices]
        return images, targets