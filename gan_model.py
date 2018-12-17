import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from block_layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))




class gen_1(nn.Module):

    """
    Generator module for stage 1. It performs the following.
    Embedding->Conditional_augmentation->Input latent vector->Upsampling->64 x 64 x 3 image.
    """

    def __init__(self):
        super(gen_1, self).__init__()
        self.gen_dim = params['gen_fea_dim'] * 8
        self.con_dim = params['con_aug_dim']
        self.noise_dim = params['noise_dim']
        self.initialize()


    def initialize(self):

        """
        This function initializes the layer required for performing generator functions.
        self.aug_net module converts the embedding into the input latent vector.
        self.fc layer converts the input latent vector into small feature map.
        self.upsample blocks performs a series of upsampling steps.
        self.generated takes the upsampled feature map and outputs a 64 x 64 x 3 image.
        """

        input_size = self.noise_dim + self.con_dim
        gen_fea = self.gen_dim
        self.aug_net = conditional_aug()

        self.fc = nn.Sequential(
                      nn.Linear(input_size, gen_fea * 4 * 4, bias = False),
                      nn.BatchNorm1d(gen_fea* 4 * 4),
                      nn.ReLU(inplace = True)
        )

        self.upsample1 = up_sampling(gen_fea, gen_fea//2)
        self.upsample2 = up_sampling(gen_fea//2, gen_fea//4)
        self.upsample3 = up_sampling(gen_fea//4, gen_fea//8)
        self.upsample4 = up_sampling(gen_fea//8, gen_fea//16)

        self.generated = nn.Sequential(
                        conv3(gen_fea//16, 3),
                        nn.Tanh()
        )


    def forward(self, embedding, noise):
        c_aug, mean, variance = self.aug_net(embedding)
        inp_vec = torch.cat((noise, c_aug), 1)
        down_image = self.fc(inp_vec)
        down_image = down_image.view(-1, self.gen_dim, 4, 4)
        down_image = self.upsample1(down_image)
        down_image = self.upsample2(down_image)
        down_image = self.upsample3(down_image)
        down_image = self.upsample4(down_image)

        fake_image = self.generated(down_image)
        return None, fake_image, mean, variance






class dis_1(nn.Module):

    """
    Discriminator module for state 1. It performs the following steps.
    Real of Fake image->Down sampling->text and image feature combining->Output score.
    """

    def __init__(self):
        super(dis_1, self).__init__()
        self.dis_dim = params['dis_fea_dim']
        self.con_dim = params['con_aug_dim']
        self.initialize()

    def initialize(self):

        """
        This function initializes the modules required for state 1 discriminator.
        self.down_sampler take the input image and down samples it into 4 x 4 x 512 feature map
        self.conditioned_result containes the decision maker block which outputs the score.
        self.unconditioned_result is not applied to stage 1 discriminator.
        """

        dis_dim = self.dis_dim
        con_dim = self.con_dim

        self.down_sampler = nn.Sequential(
                        nn.Conv2d(3, dis_dim, 4, stride = 2, padding = 1, bias = False),
                        nn.LeakyReLU(0.2, inplace = True),
                        nn.Conv2d(dis_dim, dis_dim * 2, 4, stride = 2, padding = 1, bias = False),
                        nn.BatchNorm2d(dis_dim * 2),
                        nn.LeakyReLU(0.2, inplace = True),
                        nn.Conv2d(dis_dim * 2, dis_dim * 4, 4, stride = 2, padding = 1, bias = False),
                        nn.BatchNorm2d(dis_dim*4),
                        nn.LeakyReLU(0.2, inplace = True),
                        nn.Conv2d(dis_dim * 4, dis_dim * 8, 4, stride = 2, padding = 1, bias = False),
                        nn.BatchNorm2d(dis_dim * 8),
                        nn.LeakyReLU(0.2, inplace = True)
        )

        self.conditioned_result = decision_maker(dis_dim, con_dim)
        self.unconditioned_result = None


    def forward(self, image):
        down_image = self.down_sampler(image)
        return down_image





class gen_2(nn.Module):

    """
    Module for state 2 generator. It performs the following
    Stage 1 image and embedding->conditional aug.->Down sampling->image and text feature combining->
    residual blocks-> Up sampling -> 256 x 256 x 3 image.

    gen_1: It takes a stage 1 trained generator as input to obtain the stage 1 image.
           The stage 1 generator is disabled for training.
    """

    def __init__(self, gen_1):
        super(gen_2, self).__init__()
        self.gen_dim = params['gen_fea_dim']
        self.con_dim = params['con_aug_dim']
        self.noise_dim = params['noise_dim']
        self.gen_1 = gen_1
        for par in self.gen_1.parameters():
            par.requires_grad = False
        self.initialize()


    def build_res_block(self, obj, feature_no):

        """
        This function build a series of residual blocks
        and then return it as nn.Sequential layer.
        """

        layers = []
        for i in range(params['res_blocks']):
            layers.append(obj(feature_no))
        return nn.Sequential(*layers)


    def initialize(self):
        """
        This function initializes the modules required for the generator.
        self.aug_net gives the condition vector given the text embedding.
        sel.down_sampler down samples the stage 1 image into a 16 x 16 x 512 feature map.
        self.combiner processes the concatenated text feature map and down sampled image feature map.
        self.residual contains the series of residual blocks.
        self.upsample performs a series of up sampling steps
        self.generated converts the upsampled feature map into a 256 x 256 x 3 image.
        """

        gen_dim = self.gen_dim
        self.aug_net = conditional_aug()

        self.down_sampler = nn.Sequential(
                         conv3(3, gen_dim),
                         nn.ReLU(inplace = True),
                         nn.Conv2d(gen_dim, gen_dim * 2, 4, stride = 2, padding = 1, bias = False),
                         nn.BatchNorm2d(gen_dim*2),
                         nn.ReLU(inplace = True),
                         nn.Conv2d(gen_dim*2, gen_dim * 4, 4, stride = 2, padding = 1, bias = False),
                         nn.BatchNorm2d(gen_dim * 4),
                         nn.ReLU(inplace = True)
        )

        self.combiner = nn.Sequential(
                        conv3(self.con_dim + gen_dim * 4, gen_dim * 4),
                        nn.BatchNorm2d(gen_dim * 4),
                        nn.ReLU(inplace = True)
        )

        self.residual = self.build_res_block(Res_block, gen_dim * 4)
        self.upsample1 = up_sampling(gen_dim * 4, gen_dim * 2)
        self.upsample2 = up_sampling(gen_dim * 2, gen_dim)
        """
        ***************************************************************
        The stage 2 processing is on 256 x 256 images which is not possible
        without powerful GPU. So I modified the network from original version
        by working and outputting 64 x 64 images. Use the commented section
        to use 64 x 64 images in stage 2.
        ****************************************************************
        """
        self.upsample3 = up_sampling(gen_dim, gen_dim//2) #self.upsample3 = up_sampling(gen_dim, gen_dim//2, scale = 1)
        self.upsample4 = up_sampling(gen_dim//2, gen_dim//4) #self.upsample4 = up_sampling(gen_dim, gen_dim//2, scale = 1)

        self.generated = nn.Sequential(
                         conv3(gen_dim//4, 3),
                         nn.Tanh()
        )


    def forward(self, embedding, noise):
        _,fake_image, _, _ = self.gen_1(embedding, noise)
        fake_image1 = fake_image.detach()
        down_image = self.down_sampler(fake_image1)
        c_aug, mean, variance = self.aug_net(embedding)
        c_aug = c_aug.view(-1, self.con_dim, 1, 1)
        c_aug = c_aug.repeat(1,1,16,16)
        code = torch.cat((down_image, c_aug), 1)
        combined = self.combiner(code)
        combined = self.residual(combined)
        image = self.upsample1(combined)
        image = self.upsample2(image)
        image = self.upsample3(image)
        image = self.upsample4(image)

        fake_image2 = self.generated(image)
        return fake_image1, fake_image2, mean, variance





class dis_2(nn.Module):

    """
    Module for stage 2 discriminator. It performs the following.
    Generated or fake image-> Down sampling -> Combinng text and image feature maps -> Output score.
    """

    def __init__(self):
        super(dis_2, self).__init__()
        self.dis_dim = params['dis_fea_dim']
        self.con_dim = params['con_aug_dim']
        self.initialize()

    def initialize(self):

        """
        This function initializes the modules required for the discriminator.
        self.down_sampler down samples the input image into a 4 x 4 x 512 feature map.
        self.conditioned_result containes a decision maker object that makes decision based on text and image.
        self.unconditioned_result containes a decision maker object that makes decision based on image only.
        """

        dis_dim = self.dis_dim
        con_dim = self.con_dim
        """
        ************************************************************************************
        Change the kernel size to 3 and the stride to 1 for the last two nn.Conv2d layers in the
        sequential layer if 256 x 256 images cannot be trained in the GPU.
        ************************************************************************************
        """
        self.down_sampler = nn.Sequential(
                      nn.Conv2d(3, dis_dim, 4, stride = 2, padding = 1, bias = False),
                      nn.LeakyReLU(0.2, inplace = True),
                      nn.Conv2d(dis_dim, dis_dim * 2, 4, stride = 2, padding = 1, bias = False),
                      nn.BatchNorm2d(dis_dim * 2),
                      nn.LeakyReLU(0.2, inplace = True),
                      nn.Conv2d(dis_dim * 2, dis_dim * 4, 4, stride = 2, padding = 1, bias = False),
                      nn.BatchNorm2d(dis_dim * 4),
                      nn.LeakyReLU(0.2, inplace = True),
                      nn.Conv2d(dis_dim * 4, dis_dim * 8, 4, stride = 2, padding = 1, bias = False),
                      nn.BatchNorm2d(dis_dim * 8),
                      nn.LeakyReLU(0.2, inplace = True),
                      nn.Conv2d(dis_dim * 8, dis_dim * 16, 4, stride = 2, padding = 1, bias = False),
                      nn.BatchNorm2d(dis_dim * 16),
                      nn.LeakyReLU(0.2, inplace = True),
                      nn.Conv2d(dis_dim * 16, dis_dim * 32, 4, stride = 2, padding = 1, bias = False),
                      nn.BatchNorm2d(dis_dim * 32),
                      nn.LeakyReLU(0.2, inplace = True),
                      conv3(dis_dim * 32, dis_dim * 16),
                      nn.BatchNorm2d(dis_dim * 16),
                      nn.LeakyReLU(0.2, inplace = True),
                      conv3(dis_dim * 16, dis_dim * 8),
                      nn.BatchNorm2d(dis_dim * 8),
                      nn.LeakyReLU(0.2, inplace = True),
        )

        self.conditioned_result = decision_maker(dis_dim, con_dim, condition = True)
        self.unconditioned_result = decision_maker(dis_dim, con_dim, condition = False)


    def forward(self, image):
        down_image = self.down_sampler(image)
        return down_image
