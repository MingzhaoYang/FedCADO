from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

imgsize = 224

nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
f = os.listdir(nicopp_path)
for i in range(len(f)):
    f[i] = f[i].lower()
nicopp_class_prompts = sorted(f)  
        
def read_nicopp_data(dataset_path, domain_name, split="train",shotnum=999999999):
    data_paths = []
    data_labels = []
    shot = [0 for _ in range(60)]
    split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
    
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            b = line.split('/')
            b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
            data_path =f"{'/'.join(b)}"
            data_path = path.join(dataset_path, data_path)
            label = nicopp_class_prompts.index(b[-2])
            if shot[int(label)]<shotnum:
                data_paths.append(data_path)
                data_labels.append(label)
                shot[int(label)]+=1
            #print(shot)
            #print(aa)
                
    return np.array(data_paths), np.array(data_labels)

class Nicopp(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(Nicopp, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index] 
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)



def get_nicopp_dataset(transform,divide):
    dataset_path = '/home/share/NICOpp'
    train_data_paths, train_data_labels = read_nicopp_data(dataset_path, divide, split="train",shotnum=30)
    test_data_paths, test_data_labels = read_nicopp_data(dataset_path, divide, split="test",shotnum=999999999)
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset

def get_all_nicopp_dataset(transform):
    dataset_path = '/home/share/NICOpp'
    nico_domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock','water']
    train_data_paths = []
    train_data_labels = []
    test_data_paths = []
    test_data_labels = []
    for i in range(6):
        train_data_path, train_data_label = read_nicopp_data(dataset_path, nico_domains[i], split="train",shotnum=30)
        test_data_path, test_data_label = read_nicopp_data(dataset_path, nico_domains[i], split="test",shotnum=999999999)
        train_data_paths.append(train_data_path)
        train_data_labels.append(train_data_label)
        test_data_paths.append(test_data_path)
        test_data_labels.append(test_data_label)
    
    train_paths = np.concatenate(train_data_paths)
    train_labels = np.concatenate(train_data_labels)
    test_paths = np.concatenate(test_data_paths)
    test_labels = np.concatenate(test_data_labels)
    
    train_dataset = Nicopp(train_paths, train_labels, transform)
    test_dataset = Nicopp(test_paths, test_labels, transform)
    
    return train_dataset, test_dataset



def read_nicou_data(dataset_path, domain_name, split="train",shotnum=999999999,cate = None):
    data_paths = []
    data_labels = []
    shot = [0 for _ in range(60)]
    class_style = {}
    for i in os.listdir('/home/share/NICOpp/txtlist/NICO_unique_official'):
        if '.DS_Store' in i: continue
        c,s = i.split('_')[0],i.split('_')[1]
        if c in class_style.keys():# and i.split('_')[2]=='test.txt':
            if s not in class_style[c]:
                class_style[c].append(s)
        else:
            class_style[c] = [s,]
    for cla in class_style.keys():
        class_style[cla] = sorted(class_style[cla])
    class_style = sorted(class_style.items(), key=lambda x: x)
    files = []
    for cla in class_style:
        c,s = cla[0],cla[1][domain_name]
        file = '/home/share/NICOpp/txtlist/NICO_unique_official/'+'_'.join([c,s])+f'_{split}.txt'
        files.append(file)

    # split_file = path.join(dataset_path, "txtlist/NICO_unique_official", "{}_{}.txt".format(domain_name, split))
    for split_file in files:
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if '.DS_Store' in line: continue
                line = line.strip()
                b = line.split('/')
                nicopp_class_prompts.index(b[-3])
                b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
                data_path =f"{'/'.join(b[4:])}"
                # data_path, label = line.split(' ')
                data_path = path.join('/home/share/NICOpp', data_path)
                #label = int(label)
                label = nicopp_class_prompts.index(b[-3])
                if shot[int(label)]<shotnum:
                    if cate == None:
                        data_paths.append(data_path)
                        data_labels.append(label)
                        shot[int(label)]+=1
                    elif label == cate:
                        data_paths.append(data_path)
                        data_labels.append(label)
                        shot[int(label)]+=1
                        
    return np.array(data_paths), np.array(data_labels)


def get_nicou_dataset(transform,divide):
    dataset_path = '/home/share/NICOpp'
    
    train_data_paths, train_data_labels = read_nicou_data(dataset_path, divide, split="train",shotnum=30)
    test_data_paths, test_data_labels = read_nicou_data(dataset_path, divide, split="test",shotnum=9999999)
    
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset


def get_all_nicou_dataset(transform):
    dataset_path = '/home/share/NICOpp'
    nico_domains = [0,1,2,3,4,5]
    train_data_paths = []
    train_data_labels = []
    test_data_paths = []
    test_data_labels = []
    for i in range(6):
        train_data_path, train_data_label = read_nicou_data(dataset_path, nico_domains[i], split="train",shotnum=30)
        test_data_path, test_data_label = read_nicou_data(dataset_path, nico_domains[i], split="test",shotnum=999999999)
        train_data_paths.append(train_data_path)
        train_data_labels.append(train_data_label)
        test_data_paths.append(test_data_path)
        test_data_labels.append(test_data_label)
    
    train_paths = np.concatenate(train_data_paths)
    train_labels = np.concatenate(train_data_labels)
    test_paths = np.concatenate(test_data_paths)
    test_labels = np.concatenate(test_data_labels)
    
    train_dataset = Nicopp(train_paths, train_labels, transform)
    test_dataset = Nicopp(test_paths, test_labels, transform)
    
    return train_dataset, test_dataset


def get_nicou_dataset_single(transform,divide,cate= None):
    dataset_path = '/home/share/NICOpp'
    
    train_data_paths, train_data_labels = read_nicou_data(dataset_path, divide, split="train",shotnum=9999999,cate = cate)
    test_data_paths, test_data_labels = read_nicou_data(dataset_path, divide, split="test",shotnum=9999999,cate = cate)
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset


def get_nicopp_dataset_classes(transform,classes=None):
    dataset_path = '/home/share/NICOpp'
    train_data_paths, train_data_labels = read_nicopp_data_classes(dataset_path,classes = classes, split="train",shotnum=30)
    test_data_paths, test_data_labels = read_nicopp_data_classes(dataset_path,classes = classes, split="test",shotnum=999999999)
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset


def read_nicopp_data_classes(dataset_path, classes, split="train",shotnum=999999999):
    data_paths = []
    data_labels = []
    domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock','water']
    min_cls = classes*10
    max_cls = 9 + classes*10
    for domain_name in domains:
        shot = [0 for _ in range(60)]
        split_file = path.join(dataset_path, "NICO_DG_official", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                b = line.split('/')
                b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
                data_path =f"{'/'.join(b)}"
                data_path = path.join(dataset_path, data_path)
                label = nicopp_class_prompts.index(b[-2])
                if int(label)>=min_cls and int(label) <= max_cls:
                    if shot[int(label)]<shotnum :
                        data_paths.append(data_path)
                        data_labels.append(label)
                        shot[int(label)]+=1
                #print(shot)
                #print(aa)

    return np.array(data_paths), np.array(data_labels)
    
def get_nicou_dataset_classes(transform,classes=None):
    dataset_path = '/home/share/NICOpp'
    train_data_paths, train_data_labels = read_nicou_data_classes(dataset_path,classes = classes, split="train",shotnum=30)
    test_data_paths, test_data_labels = read_nicou_data_classes(dataset_path,classes = classes, split="test",shotnum=999999999)
    train_dataset = Nicopp(train_data_paths, train_data_labels, transform)
    test_dataset = Nicopp(test_data_paths, test_data_labels, transform)
    
    return train_dataset, test_dataset


def read_nicou_data_classes(dataset_path, classes, split="train",shotnum=999999999):
    data_paths = []
    data_labels = []
    class_style = {}
    for i in os.listdir('/home/share/NICOpp/txtlist/NICO_unique_official'):
        if '.DS_Store' in i: continue
        c,s = i.split('_')[0],i.split('_')[1]
        if c in class_style.keys():# and i.split('_')[2]=='test.txt':
            if s not in class_style[c]:
                class_style[c].append(s)
        else:
            class_style[c] = [s,]
    for cla in class_style.keys():
        class_style[cla] = sorted(class_style[cla])
    class_style = sorted(class_style.items(), key=lambda x: x)
    files = []
    for cla in class_style[classes*10:10+classes*10]:
        c= cla[0]
        for s in cla[1]:
            file = '/home/share/NICOpp/txtlist/NICO_unique_official/'+'_'.join([c,s])+f'_{split}.txt'
            files.append(file)
    # split_file = path.join(dataset_path, "txtlist/NICO_unique_official", "{}_{}.txt".format(domain_name, split))
    for split_file in files:
        shot = [0 for _ in range(60)]
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if '.DS_Store' in line: continue
                line = line.strip()
                b = line.split('/')
                nicopp_class_prompts.index(b[-3])
                b[-1],label = b[-1].split(' ')[0],b[-1].split(' ')[1]
                data_path =f"{'/'.join(b[4:])}"
                # data_path, label = line.split(' ')
                data_path = path.join('/home/share/NICOpp', data_path)
                #label = int(label)
                label = nicopp_class_prompts.index(b[-3])
                if shot[int(label)]<shotnum:
                    data_paths.append(data_path)
                    data_labels.append(label)
                    shot[int(label)]+=1
                        
    return np.array(data_paths), np.array(data_labels)