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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))



def train_and_sample(gen, dis, dataset, sample_dataset, config, stage = 1, pre_gen = None, pri_img_every = 100, save_path = '', sample_every = 10, save_every = 100):

    """
    This function is used to train the generator and discriminator of both stage 1 and 2.
    gen: Generator object.
    dis: Discriminator object.
    dataset: The dataloader which is used to obtain images for training.
    sample_dataset: The dataset from which pre trained sentence embeddings for sampling is obtained.
    config: Dictionary containing the hyperparameters.
    stage: Indicates which stage so that stage specific operations can be done.
    pre_gen: If stage 2 this gives the path to the trained stage 1 generator.
    pri_img_every: Tells the iteration frequency for which loss and fake images generated has to be displayed.
    save_path: Containes the path to which the models and the images has to be stored.
    sample_every: Indicates the epoch frequency for sampling images from the trained generator.
    save_every: Indicates the epoch frequency for which the model has to be saved.
    """


    if (stage == 2 and pre_gen == None):
        return 'Give the path to the trained stage 1 generator'
    elif(stage == 2 and pre_gen != None):
        gen.gen_1.load_state_dict(torch.load(pre_gen)['generator'])

    if (stage == 1):
        save_path = os.path.join(save_path, '1')
    else:
        save_path = os.path.join(save_path, '2')

    params = config
    noise_dim = params['noise_dim']
    batch_size = params['batch_size']
    gen_lr = params['gen_lr']
    dis_lr = params['dis_lr']

    noise = torch.FloatTensor(batch_size, noise_dim)
    noise = noise.to(device)

    imgen_noise = torch.FloatTensor(batch_size, noise_dim).normal_(0,1)
    imgen_noise = imgen_noise.to(device)

    real_labels = torch.FloatTensor(batch_size).fill_(1)
    fake_labels = torch.FloatTensor(batch_size).fill_(0)
    real_labels = real_labels.to(device)
    fake_labels = fake_labels.to(device)

    optimizer_dis = optim.Adam(dis.parameters(), lr = dis_lr, betas = (0.5,0.999))
    gen_layers = []
    for layer in gen.parameters(): #ommiting the stage 1 generator layers in stage 2 for optimizing.
        if layer.requires_grad:
            gen_layers.append(layer)
    optimizer_gen = optim.Adam(gen_layers, lr = gen_lr, betas = (0.5, 0.999))

    for epoch in range(params['epoch']):
        er_d = []
        er_g = []
        kl = []
        start = time.time()
        print('Epoch {}'.format(epoch+1))
        if (epoch > 0 and ((epoch +1) % params['lr_decay_epoch'] == 0)): #decaying the learning rate after every specified interval
            gen_lr *= 0.5
            for par in optimizer_gen.param_groups:
                par['lr'] = gen_lr
            dis_lr *= 0.5
            for par in optimizer_dis.param_groups:
                par['lr'] = dis_lr

        for i, data in enumerate(dataset,0):
            real_image, embedding = data
            real_image = real_image.to(device)
            embedding = embedding.to(device)


            noise.data.normal_(0,1)
            gen.train()
            _, fake_image, mean, variance = gen(embedding, noise)   #genrate fake image

            dis.zero_grad() #updating discriminator
            error_d, real_error, wrong_error, fake_error = discriminator_loss(dis, fake_image, real_image,fake_labels, real_labels, mean, stage)
            er_d.append(error_d.item())
            error_d.backward()
            optimizer_dis.step()

            gen.zero_grad() #updating generator
            error_g = generator_loss(dis, fake_image, real_labels, mean)
            er_g.append(error_g.item())
            kl_los = kl_loss(mean, variance)
            kl.append(kl_los.item())
            total_error = error_g + kl_los * params['kl_coeff']
            total_error.backward()
            optimizer_gen.step()

            if (((i+1)%pri_img_every) == 0):
                print('Discriminator_error: {}'.format(error_d.item()))
                print('Generator_error:{}'.format(error_g.item()))
                print('KL loss:{}'.format(kl_los.item()))

                print('Running discriminator loss: {}'.format(sum(er_d)/len(er_d)))
                print('Running generator loss: {}'.format(sum(er_g)/len(er_g)))
                print('Running KL loss: {}'.format(sum(kl)/len(kl)))

                previous, current, _, _ = gen(embedding, imgen_noise)
                save_image(real_image, current, epoch+1, os.path.join(save_path, 'images'))
                show = utils.make_grid(real_image[0:16])
                image_show(show)
                show = utils.make_grid(current[0:16])
                image_show(show)
                if previous is not None:
                    save_image(None, previous, epoch+1, os.path.join(save_path, 'images'))

        elapsed_time = time.time() - start
        print('Epoch {} completed in {:.0f}minutes {:.0f}seconds'.format(epoch+1, elapsed_time//60, elapsed_time%60))
        print('Discriminator loss for this epoch: {}'.format(sum(er_d)/len(er_d)))
        print('Generator loss for this epoch: {}'.format(sum(er_g)/len(er_g)))
        print('KL loss for this epoch: {}'.format(sum(kl)/len(kl)))

        if ((epoch+1) % save_every == 0):
            save_model(gen, dis,optimizer_gen, optimizer_dis, epoch + 1, os.path.join(save_path, 'model'), stage=stage)
        if ((epoch+1) % sample_every == 0):
            sampler(gen, sample_dataset, epoch+1, noise = imgen_noise, save_path = os.path.join(save_path, 'images'))


    save_model(gen, dis, optimizer_gen, optimizer_dis, params['epoch'], os.path.join(save_path, 'model') , stage = stage)
