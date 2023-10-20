import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
from datasets.DomainNet import get_domainnet_dloader,get_all_domainnet_dloader
from datasets.TingImagenet import TinyImageNet_load
from datasets.openimage import get_openimage_dataset
from datasets.NICOPP import get_nicopp_dataset,get_nicou_dataset,get_all_nicopp_dataset,get_all_nicou_dataset,get_nicou_dataset_single,get_nicopp_dataset_classes,get_nicou_dataset_classes
import os
import logging
import copy
from collections import OrderedDict
from utils import partition,Truncated,evaluation
from client import Client
from accelerate import Accelerator
from server import Server
from PIL import Image
# logging.basicConfig()
from torch.utils.data import DataLoader, Dataset
from datasets.openimage import get_openimage_classes

# os.environ['CUDA_VISIBLE_DEVICES'] ='2'
class ServerData_read(Dataset):
    def __init__(self, root_dir,transforms=None):
        super(ServerData_read, self).__init__()
        self.root_dir = root_dir
        
        path = r"/home/share/DomainNet/clipart"
        f = os.listdir(path)
        for i in range(len(f)):
            f[i] = f[i].lower()
        self.class_prompts = sorted(f) 
        self.classes = {c:i for i,c in enumerate(self.class_prompts) if i<90}
        
        #nicopp_path = "/home/share/NICOpp/NICO_DG/autumn"
        #f = os.listdir(nicopp_path)
        #for i in range(len(f)):
        #    f[i] = 'an image of '+f[i].lower()
        #self.class_prompts = sorted(f) 
        #self.classes = {c:i for i,c in enumerate(self.class_prompts) if i<60}
        
        #open_image_class_prompts,open_image_rough_classes = get_openimage_classes()
        #self.class_prompts = open_image_rough_classes
        #self.classes = {c:i for i,c in enumerate(self.class_prompts) if i<20}      
        
        self.images = []
        self.targets = []
        self.transforms = transforms
        for c in self.classes:
            class_dir = os.path.join(self.root_dir,str(c))
            #class_dir = os.path.join(self.root_dir, str(c))
            for image_name in os.listdir(class_dir):
                if '.ipynb_checkpoints' in image_name: continue
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.targets.append(self.classes[c])
        print(len(self.images))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = self.targets[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
            
            #img = self.transforms(img, return_tensors="pt").pixel_values
            #img = img.squeeze(0)
            
        return img, target
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/home/share/DomainNet")#/home/share/DomainNet/home/share/tiny-imagenet-200
    parser.add_argument('--alpha', default=1,type=float,help='degree of non-iid, only used for tinyimagenet')
    parser.add_argument('--beta', default=0,type=float,help='degree of noise')
    parser.add_argument('--data', default='nicopp',help='tinyimagenet or domainnet or openimage or nicopp or nicou')
    parser.add_argument('--seed', default=0,type=int,)
    parser.add_argument('--batch_size', default=32,type=int,)
    parser.add_argument('--serverbs', default=256,type=int,)
    parser.add_argument('--serverepoch', default=10,type=int,)
    parser.add_argument('--clientepoch', default=20,type=int,)
    parser.add_argument('--learningrate', default=0.005,type=float,)
    parser.add_argument('--num_clients', default=6,type=int,help='number of clinets, only used for tinyimagenet')
    parser.add_argument('--split-type', default='shard',help='dirichlet or shard')
    return parser

drop_last = False

########################################################################################################################
parser = get_parser()
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = False

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision=None)

if args.data  == 'domainnet': 
    num_classes = 90
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224,224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    #domains = ['painting', 'quickdraw', 'real', 'sketch']
    domains = ['clipart', 'infograph', 'painting','quickdraw', 'real', 'sketch']#['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    clients,test_data = [],[]
    server_labeled = {ddd:None for ddd in domains}
    server_dataset,test_dataset = get_all_domainnet_dloader(args.base_path,256,transform)
    server_loader = torch.utils.data.DataLoader(server_dataset, batch_size=256, num_workers=8,shuffle=True, pin_memory=True)
    server_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=8,shuffle=False, pin_memory=True)
    print(f'server train data num:',len(server_dataset),len(test_dataset))
    for domain in domains:
        # print(domain)
        train_dataset_client,testdataset = get_domainnet_dloader(args.base_path,domain,args.batch_size,transform)
        print(f'client {domain} data num:',len(train_dataset_client),len(testdataset))
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=256, num_workers=8,shuffle=False, pin_memory=True)
        trainloader = torch.utils.data.DataLoader(train_dataset_client, batch_size=args.batch_size, num_workers=8,shuffle=True,drop_last = True , pin_memory=True)
        trainloader = accelerator.prepare(trainloader)
        test_loader = accelerator.prepare(test_loader)
        client = Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name = domain)
        client.model = accelerator.prepare(client.model)
        clients.append(client)
        test_data.append(test_loader)
        
elif args.data =='nicopp':
    num_classes = 60
    nico_domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock','water']
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_all_nicopp_dataset(transform)
    print('server data num:',len(server_train_data),len(server_test_data))
    
    num_clients = 6
    clients,test_data = [],[]
    
    #test_data.append(torch.utils.data.DataLoader(server_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    server_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    for i in range(num_clients):
        
        ##feature skew
        print(f'getting client {i} data')
        client_train_data,client_test_data = get_nicopp_dataset(transform,divide =nico_domains[i])
        print(f'client {i} data num:',len(client_train_data),len(client_test_data))
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True,drop_last = True , pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name = i))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False,drop_last = True , pin_memory=True))
        
        ##label skew
        #print(f'getting client {i} data')
        #client_train_data,client_test_data = get_nicopp_dataset_classes(transform,classes=i)
        #print(f'client {i} data num:',len(client_train_data),len(client_test_data))
        #trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True,drop_last = True , pin_memory=True)
        #clients.append(Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name='{}_{}'.format(10*i,9+10*i)))
       # test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False,drop_last = True , pin_memory=True))
        
elif args.data =='openimage':
    num_classes = 20
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_openimage_dataset(transform,divide = [0,1,2,3,4,5],max_num = 30)
    print('server data num:',len(server_train_data),len(server_test_data))
    num_clients = 6
    clients,test_data = [],[]
    server_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    
    for i in range(num_clients):
        print(f'getting client {i} data')
        client_train_data,client_test_data = get_openimage_dataset(transform,divide = i,max_num = 30)
        
        print(f'client {i} data num:',len(client_train_data),len(client_test_data))
        
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name = i))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
        
elif args.data =='nicou':
    num_classes = 60
    nico_domains = [0,1,2,3,4,5]
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    print('getting server data')
    server_train_data,server_test_data = get_all_nicou_dataset(transform)
    print('server data num:',len(server_train_data),len(server_test_data))
    
    num_clients = 6
    clients,test_data = [],[]
    
    #test_data.append(torch.utils.data.DataLoader(server_test_data, batch_size=256, num_workers=8,shuffle=False, pin_memory=True))
    
    server_loader = torch.utils.data.DataLoader(server_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True, pin_memory=True)
    for i in range(num_clients):
        ##feature skew
        print(f'getting client {i} data')
        client_train_data,client_test_data = get_nicou_dataset(transform,divide =nico_domains[i])
        print(f'client {i} data num:',len(client_train_data),len(client_test_data))
        trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True,drop_last = True , pin_memory=True)
        clients.append(Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name = nico_domains[i]))
        test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False,drop_last = True , pin_memory=True))
        ##label skew
        #print(f'getting client {i} data')
        #client_train_data,client_test_data = get_nicou_dataset_classes(transform,classes=i)
        #print(f'client {i} data num:',len(client_train_data),len(client_test_data))
        #trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=args.batch_size, num_workers=8,shuffle=True,drop_last = True , pin_memory=True)
        #clients.append(Client(trainloader,num_classes,beta=args.beta,accelerator=accelerator,domain_name='{}_{}'.format(10*i,9+10*i)))
        #test_data.append(torch.utils.data.DataLoader(client_test_data, batch_size=256, num_workers=8,shuffle=False,drop_last = True , pin_memory=True))
        
#train classifiers

#for i,client in enumerate(clients):
#    client.train(client= i,lr=args.learningrate,epochs = args.clientepoch,test_data = test_data[i],change_backbone=True)    

#load synthetic dataset
dataset = ServerData_read(f'/home/share/gen_data/semisup_cipqr_5img_10repeat_crossglo_classglo_scale10_0_90',transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=256,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
server = Server(transform,args.serverbs,num_classes)

##synthetic dataset dataloader
#server.update_features(dataloader = dataloader)

##centralized ceiling dataloader
server.update_features(dataloader = server_loader)

server.train(lr=args.learningrate,epochs = args.clientepoch,test_data = test_data)

#server.multi_tea_kd_train(lr=args.learningrate,epochs = args.clientepoch,test_data = test_data)

#server.sp_tea_kd_train(lr=args.learningrate,epochs = args.clientepoch,test_data = test_data)



