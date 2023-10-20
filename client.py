import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torchvision
from utils import evaluation
from diffusers import DDPMScheduler
from diffusers.utils import randn_tensor
from diffusers.models.embeddings import Timesteps
import os
import copy
from datasets.NICOPP import get_nicopp_dataset,get_nicou_dataset,get_all_nicopp_dataset,get_all_nicou_dataset,get_nicou_dataset_single
from datasets.DomainNet import get_domainnet_dloader,get_all_domainnet_dloader,get_domainnet_dataset_single
import torchvision.transforms as transforms
class Client: # as a user
    def __init__(self, dataloader, classes , beta=0,accelerator=None,domain_name = None):
        self.dataloader = dataloader
        self.model = ClientTune(classes).cuda()
        self.ori_features = None
        self.beta = beta
        self.K = 5
        self.classes = classes
        self.accelerator = accelerator
        self.domain_name = domain_name
        
    def train(self,client,lr,epochs,test_data,change_backbone=None):
        if change_backbone==True:
            backbones = ['mobilenetv3','resnet18','resnet34','mobilenetv2','vgg16','shufflenet']
            backbone = backbones[client]

            if backbone=='mobilenetv3':
                self.model.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)#MobileNetV2()
                self.model.encoder.classifier[3] = torch.nn.Identity()
                self.model.final_proj = nn.Linear(1024,self.classes)
            elif backbone=='resnet18':
                self.model.encoder = torchvision.models.resnet18(pretrained=True)#MobileNetV2()
                self.model.encoder.fc = torch.nn.Identity()
                self.model.final_proj = nn.Linear(512,self.classes)
            elif backbone=='resnet34':
                self.model.encoder = torchvision.models.resnet34(pretrained=True)#MobileNetV2()
                self.model.encoder.fc = torch.nn.Identity()
                self.model.final_proj = nn.Linear(512,self.classes)
            elif backbone=='resnet50':
                self.model.encoder = torchvision.models.resnet50(pretrained=True)#MobileNetV2()
                self.model.encoder.fc = torch.nn.Identity()
                self.model.final_proj = nn.Linear(2048,self.classes)
            elif backbone=='mobilenetv2':
                self.model.encoder = torchvision.models.mobilenet_v2(pretrained=True)#MobileNetV2()
                self.model.encoder.classifier[1] = torch.nn.Identity()
                self.model.final_proj = nn.Linear(1280,self.classes)
            elif backbone=='vgg16':
                self.model.encoder = torchvision.models.vgg16(pretrained=True)#MobileNetV2()
                self.model.encoder.classifier[6] = torch.nn.Identity()
                self.model.final_proj = nn.Linear(4096,self.classes)
            elif backbone=='shufflenet':
                self.model.encoder = torchvision.models.shufflenet_v2_x1_0(pretrained=True)#MobileNetV2()
                self.model.encoder.fc = torch.nn.Identity()
                self.model.final_proj = nn.Linear(1024,self.classes)
                #self.model.encoder = torchvision.models.vgg11(pretrained=True)#MobileNetV2()
                #self.model.encoder.classifier[6] = torch.nn.Identity()
                #self.model.final_proj = nn.Linear(4096,self.classes)
            self.model = self.model.cuda()
            print(self.model.encoder)
            print(backbone)
        
        noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler", torch_dtype=torch.float16)
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)#weight_decay=0.001
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        noise_scheduler,optimizer,scheduler = self.accelerator.prepare(noise_scheduler,optimizer,scheduler)
        generator = torch.Generator("cuda")
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for i, (image, label) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                #noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
                #timestep = torch.randint(low=0,high=99, size=[image.shape[0],], generator=generator, device=image.device)
                #noised_image = noise_scheduler.add_noise(image, noise, timestep)
                with self.accelerator.accumulate(self.model):
                    #output = self.model(noised_image,timestep)
                    output = self.model(image)

                    
                loss = task_criterion(output,label)
                self.accelerator.backward(loss)
                optimizer.step()
            scheduler.step()
            print("loss:",loss.detach().item(),'lr:',scheduler.get_last_lr()[0])
            if epoch%5 ==0:
                #self.model.eval()
                #save_path = os.path.join("output/nicou_img30_{}_epoch_{}.tar".format(self.domain_name,epoch))
    
                #if isinstance(self.model, torch.nn.DataParallel):
                #    torch.save(self.model.module.state_dict(), save_path)
                #else:
                #    torch.save(self.model.state_dict(), save_path)
                top1, topk = evaluation(self.model,test_data)
                print(f'final server model: top1 {top1}, top5 {topk}')    
        self.model.eval()
        save_path = os.path.join("output/nicopp_img10_{}_epoch_{}_{}.tar".format(self.domain_name,epoch,backbone)) 
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        top1, topk = evaluation(self.model,test_data)
        print(f'final server model: top1 {top1}, top5 {topk}')  
        
        
    def bn(self,client):
        save_path = os.path.join("output/nicou_y_{}_epoch_{}.tar".format(self.domain_name,99))
        state_dict = torch.load(save_path)
        self.model.load_state_dict(state_dict)
        
        
        #print(self.model.state_dict()['encoder.layer4.1.bn2.running_var'])
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
        
        
        for classes in range(self.classes):  
            #print('self',self.model.state_dict()['encoder.layer4.1.bn2.running_var'])
            model = copy.deepcopy(self.model)
            model.train()
            #client_train_data,_ = get_nicou_dataset_single(transform,divide =client,cate = classes)
            client_train_data,_ = get_domainnet_dataset_single(client,transform,cate=classes)
            print(f'client {client} cls {classes} data num:',len(client_train_data))
            trainloader = torch.utils.data.DataLoader(client_train_data, batch_size=32, num_workers=8,shuffle=True,drop_last = False , pin_memory=True)   
            for i, (image, label) in enumerate((trainloader)):
                image = image.cuda()
                label = label.cuda()
                output = model(image)
            save_path = os.path.join("output/nicou_y_{}_epoch_{}_class_{}.tar".format(self.domain_name,99,classes)) 
            torch.save(model.state_dict(), save_path)
            #print('model',model.state_dict()['encoder.layer4.1.bn2.running_var'])
        
        
    
class ClientTune(nn.Module):
    def __init__(self, classes=345):
        super(ClientTune, self).__init__()
        
        #self.encoder = ClientImageEncoder()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder.fc = torch.nn.Identity()
        self.final_proj = nn.Sequential(
            nn.Linear(512,classes)
        )
#         self.generator = torch.Generator("cuda")
        
#         self.noise_level = noise_level
        
        #self.final_proj = nn.Sequential(
        #    nn.Linear(2048,classes,dtype = torch.float16),
            #nn.ReLU(),
            #nn.Linear(1024,classes,dtype = torch.float16),
            # nn.Linear(768,512),
            # nn.ReLU(),
            # nn.Linear(512,classes)
       # )
    
    def forward(self, x, get_fea=False,input_image=True):
        
        if input_image:
            #with torch.no_grad():
            x =  self.encoder(x)
            
        if get_fea:
            return x.view(x.shape[0],-1)
        
        # out = self.final_proj(torch.abs(torch.fft.fft(fea.view(fea.shape[0],-1))))
        out = self.final_proj(x.view(x.shape[0],-1))
        
        return out
    