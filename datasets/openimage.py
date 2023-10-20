from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from tqdm import tqdm
import codecs
import csv
import json

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

class_level = {'Baked goods':['Pretzel','Bagel','Muffin','Cookie','Bread','Croissant'],\
               'Bird':['Woodpecker','Parrot','Magpie','Eagle','Falcon','Sparrow'],\
               'Building':['Convenience store','House','Tower','Office building','Castle','Skyscraper'],\
               'Carnivore':['Bear','Leopard','Fox','Tiger','Lion','Otter'],\
               'Clothing':['Shorts','Dress','Swimwear','Brassiere','Tiara','Shirt'],\
               'Drink':['Beer','Cocktail','Coffee','Juice','Tea','Wine'],\
               'Fruit':['Apple','Lemon','Banana','Strawberry','Peach','Pineapple'],\
               'Furniture':['Chair','Desk','Couch','Wardrobe','Bed','Shelf'],\
               'Home appliance':['Washing machine','Toaster','Oven','Blender','Gas stove','Mechanical fan'],\
               'Human body':['Human eye','Skull','Human mouth','Human ear','Human nose','Human foot'],\
               'Kitchen utensil':['Spatula','Spoon','Fork','Knife','Whisk','Cutting board'],\
               'Land vehicle':['Ambulance','Cart','Bus','Van','Truck','Car'],\
               'Musical instrument':['Drum','Guitar','Harp','Piano','Violin','Accordion'],\
               'Office supplies':['Pen','Poster','Calculator','Whiteboard','Box','Envelope'],\
               'Plant':['Maple','Willow','Rose','Lily','Common sunflower','Houseplant'],\
               'Reptile':['Dinosaur','Lizard','Snake','Tortoise','Crocodile','Sea turtle'],\
               'Sports equipment (Ball)':['Football','Tennis ball','Baseball bat','Golf ball','Rugby ball','Volleyball (Ball)'],\
               'Toy':['Doll','Balloon','Dice','Flying disc','Kite','Teddy bear'],\
               'Vegetable':['Potato','Carrot','Broccoli','Cabbage','Bell pepper','Pumpkin'],\
               'Weapon':['Knife','Axe','Sword','Handgun','Shotgun','Dagger']}
rough_classes=[]
chosen_classes = []
divided_classes = {}
divided_classes[0] = []
divided_classes[1] = []
divided_classes[2] = []
divided_classes[3] = [] 
divided_classes[4] = []
divided_classes[5] = []
for rough_class in class_level:
    rough_classes.append(rough_class)
    divided_classes[0].append(class_level[rough_class][0])
    divided_classes[1].append(class_level[rough_class][1])
    divided_classes[2].append(class_level[rough_class][2])
    divided_classes[3].append(class_level[rough_class][3])
    divided_classes[4].append(class_level[rough_class][4])
    divided_classes[5].append(class_level[rough_class][5])
    
    for det_class in class_level[rough_class]:
        chosen_classes.append(det_class)
def get_openimage_classes():
    return divided_classes,rough_classes

class openimage(Dataset):
    def __init__(self, split,transforms,divide,max_num):
        super(openimage, self).__init__()
        
        base_path = os.path.join('/home/share/openimage/',split)
        
        labels_path = os.path.join(base_path ,'labels','classifications.csv')
        classes_path = os.path.join(base_path ,'metadata','classes.csv')
        
       # clases_idx('/m/011k07': 0) 
        self.classes_idx = {}
        self.classes = []
        i=0
        with codecs.open(classes_path) as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                self.classes.append(row['label'])
                self.classes_idx[row['id']] = i
                i=i+1
        
        self.data = []
        self.target = []
        if isinstance(divide,list):
            divide = divide
        else:
            divide = [divide]
            
        for i in divide:
            shot = [0 for _ in range(20)]
            if max_num != None:
                total_max_num = max_num * 21
            len_data = 0
            with codecs.open(labels_path) as f:
                for row in csv.DictReader(f, skipinitialspace=True):
                    if self.classes[self.classes_idx[row['LabelName']]] in divided_classes[i] :
                        class_idx = divided_classes[i].index(self.classes[self.classes_idx[row['LabelName']]])
                        if max_num != None :
                            if shot[class_idx] > max_num:
                                continue

                        self.data.append(os.path.join(base_path,'data',row['ImageID']+'.jpg'))
                        self.target.append(class_idx)

                        len_data = len_data + 1
                        shot[class_idx] = shot[class_idx] + 1
                    if max_num != None:
                        if len_data > total_max_num and min(shot)!=0:
                            break
                        #self.target.append(self.classes_idx[row['LabelName']])
        
            print('image num each class:',shot)
                    
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

def get_openimage_dataset(transforms,divide,max_num):
    
    train_dataset = openimage('train',transforms,divide,max_num)
    test_dataset = openimage('test',transforms,divide,max_num = 1000)
    
    return train_dataset,test_dataset
