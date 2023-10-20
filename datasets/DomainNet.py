from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from tqdm import tqdm
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224



def read_domainnet_data_test(dataset_path, domain_name, split="train",shotnum=999999999,cate=None):
    data_paths = []
    data_labels = []
    shot = [0 for _ in range(345)]
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label)>=90: continue
            #if int(label)<245 or int(label)>249: continue
            if shot[int(label)]<shotnum:
                if cate == None:
                    shot[int(label)]+=1
                    data_path = path.join(dataset_path, data_path)
                    data_paths.append(data_path)
                    data_labels.append(int(label))
                elif label == cate:
                    shot[int(label)]+=1
                    data_path = path.join(dataset_path, data_path)
                    data_paths.append(data_path)
                    data_labels.append(int(label))
                
    return data_paths, data_labels

def read_domainnet_data_train(dataset_path, domain_name, split="train",shotnum=999999999,cate=None):
    data_paths_client = []
    data_labels_client = []
    shot = [0 for _ in range(345)]
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            if int(label)>=90: continue
            #if int(label)<245 or int(label)>249: continue
            if shot[int(label)]<shotnum:
                if cate == None:
                    shot[int(label)]+=1
                    data_path = path.join(dataset_path, data_path)
                    data_paths_client.append(data_path)
                    data_labels_client.append(int(label))
                elif int(label) == cate:
                    shot[int(label)]+=1
                    data_path = path.join(dataset_path, data_path)
                    data_paths_client.append(data_path)
                    data_labels_client.append(int(label))
                    
    return data_paths_client, data_labels_client


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(DomainNet, self).__init__()
        self.data = data_paths
        self.target = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.target[index] 
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data)


def get_domainnet_dloader(base_path,domain_name, batch_size, preprocess,num_workers=16):
    dataset_path = path.join(base_path)
    train_data_paths_client,train_data_labels_client = read_domainnet_data_train(dataset_path, domain_name, split="train",shotnum=30)
    test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain_name, split="test",shotnum=99999999)
    train_dataset_client = DomainNet(train_data_paths_client, train_data_labels_client, preprocess)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess)
    return train_dataset_client,test_dataset

def get_domainnet_dataset_single(domain_name,preprocess,cate=None):
    dataset_path = '/home/share/DomainNet'
    
    train_data_paths, train_data_labels = read_domainnet_data_train(dataset_path, domain_name, split="train",shotnum=9999999,cate = cate)
    test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain_name, split="test",shotnum=9999999,cate = cate)
    train_dataset = DomainNet(train_data_paths, train_data_labels, preprocess)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess)
    
    return train_dataset, test_dataset

def get_all_domainnet_dloader(base_path, batch_size, preprocess,num_workers=16):
    dataset_path = path.join(base_path)
    domains = ['clipart', 'infograph', 'painting', 'real', 'sketch']
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    
    for domain in domains:
        train_data_paths_client,train_data_labels_client = read_domainnet_data_train(dataset_path, domain, split="train",shotnum=30)
        test_data_paths, test_data_labels = read_domainnet_data_test(dataset_path, domain, split="test",shotnum=99999999)
        train_paths.extend(train_data_paths_client)
        train_labels.extend(train_data_labels_client)
        test_paths.extend(test_data_paths)
        test_labels.extend(test_data_labels)
    train_dataset_client = DomainNet(train_paths, train_labels, preprocess)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess)
    
    return train_dataset_client,test_dataset