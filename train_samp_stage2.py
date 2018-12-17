import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from block_layers import *
from gan_model import *
import os
import time
from functions import *
from losses import *
import matplotlib.pyplot as plt
from sample import sampler
from dataset_generator import *
from trainer import train_and_sample
import torchvision.utils as utils
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))


trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(params['inp_size']),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_data_gen = dataset_generator(os.path.join(params['base_path'], 'data'), os.path.join(params['base_path'], 'data', 'flowers'), transform = trans)

sample_data_gen = dataset_generator(os.path.join(params['base_path'], 'data'), os.path.join(params['base_path'], 'data', 'flowers'), transform = trans, sampling = True)

loader = dataloader(train_data_gen)


gen = gen_1()

gen.apply(initialize_weights)

gen = gen.to(device)

gen2 = gen_2(gen)

gen2.apply(initialize_weights)

gen2 = gen2.to(device)

dis = dis_2()

dis.apply(initialize_weights)

dis = dis.to(device)

trained_model1_name = ""

model1_trained_path = os.path.join(params['save_path'], 1, 'model', trained_model1_name)

train_and_sample(gen2, dis, loader, sample_data_gen, params, stage = 2, pre_gen= model1_trained_path , save_path = params['save_path'], sample_every = 10, save_every = params['save_every'])
