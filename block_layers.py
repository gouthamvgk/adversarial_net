"""
This file contains several small blocks required for constructing
the generator and discriminator.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))  #dictionary containing the parameters



def conv3(in_maps, out_maps, stride = 1):

    """
    Return a convolution layer with specified number of output feature maps.
    It is padded with value of 1 and the bias is turned off.
    in_maps: Number of feature maps in input
    out_maps: Number of feature maps required in output.
    stride: Stride step for convolution.
    """

    return nn.Conv2d(in_maps, out_maps, 3, stride, padding = 1, bias = False)



def up_sampling(in_maps, out_maps, scale = 2):

    """
    It is the upsampling block which increases the dimension of the given tensor.
    It upsamples by a scale of 2 and passes it through convolutional and batch norm layers.
    in_maps: Number of feature maps in input.
    output_maps: Number of feature maps required in output.
    scale: Amount by which the image has to be scaled.
    """

    up_sampler = nn.Sequential(
               nn.Upsample(scale_factor = scale),
               conv3(in_maps, out_maps),
               nn.BatchNorm2d(out_maps),
               nn.ReLU(inplace = True)
               )
    return up_sampler





class Res_block(nn.Module):

    """
    This is basic residual block used in stage 2 generator.
    It takes the input vector and performs convolution over it and
    passes it through batch norm and relu activation layer.
    It has skip connection for ease of gradient flow.
    feature_no : Number of required feature maps.
    """

    def __init__(self, feature_no):
        super(Res_block, self).__init__()
        self.layer = nn.Sequential(
                   conv3(feature_no, feature_no),
                   nn.BatchNorm2d(feature_no),
                   nn.ReLU(inplace = True),
                   conv3(feature_no, feature_no),
                   nn.BatchNorm2d(feature_no)
        )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, inp):
        temp = inp
        output = self.layer(temp)
        output += temp
        output = self.relu(output)
        return output





class conditional_aug(nn.Module):

    """
    This block perform the conditional augmentation on text embedding to
    produce a conditional vector which along with the noise vector is
    used as input to the generator.
    self.fc takes the embedding vector and outputs a combined vector of mean and variance.
    """

    def __init__(self):
        super(conditional_aug, self).__init__()
        self.embed_dim = params['embed_dim']
        self.aug_dim = params['con_aug_dim']
        self.fc = nn.Linear(self.embed_dim, 2*self.aug_dim)
        self.relu = nn.ReLU(inplace = True)


    def params_to_aug(self, embedding):

        """
        embedding: text-embedding for the sentence.
        mean: mean of the embedding found by passing throuh fully connected layer
        variance: variance of the embedding found by passing though fc layer
        std_dev : calculated from variance
        c_aug = epsilon * std_dev + mean
        """

        output = self.fc(embedding)
        output = self.relu(output)
        mean = output[:, :self.aug_dim]
        variance = output[:, self.aug_dim:]
        std_dev = torch.mul(variance, 0.5)
        std_dev = torch.exp(std_dev)
        epsilon = nn.init.normal_(torch.empty(std_dev.size()))
        epsilon = epsilon.to(device)
        c_aug = torch.mul(epsilon, std_dev)
        c_aug = torch.add(c_aug, mean)

        return c_aug, mean, variance

    def forward(self, embedding):
        c_aug, mean, variance = self.params_to_aug(embedding)

        return c_aug, mean, variance




class decision_maker(nn.Module):

    """
    This class is used to find the discriminator output from the
    encoding obtained from generated image and text embedding.
    dis_fea: Down sampled version of the generated image
    text_fea: augmented form of the text embedding using which the
              image was generated.
    condition: Indicates whether decision has to made for stage 1 or 2 discriminator.
    self.classifier takes the combined feature map of image and text and outputs a score.
    """

    def __init__(self, dis_fea, text_fea, condition = True):
        super(decision_maker, self).__init__()
        self.dis_fea = dis_fea
        self.text_fea = text_fea
        self.condition = condition
        #passes the combined version of image and text through various layers
        if condition:
            self.classifier = nn.Sequential(
                                conv3(dis_fea * 8 + text_fea, dis_fea * 8),
                                nn.BatchNorm2d(dis_fea * 8),
                                nn.LeakyReLU(0.2, inplace = True),
                                nn.Conv2d(dis_fea * 8, 1, 4, stride = 4),
                                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                                nn.Conv2d(dis_fea * 8, 1, 4, stride = 4),
                                nn.Sigmoid()
            )


    def forward(self, down_image, text_vec = None):

        """
        Generated image is down sampled and the text vector is repeated in width and height
        dimensions to be concatenated with the down sampled image tensor.It is passed through
        the classifier to obtain the discriminator score.
        """

        if self.condition and text_vec is not None:
            text_vec = text_vec.view(-1, self.text_fea,1,1)
            text_vec = text_vec.repeat(1,1,4,4)
            text_image_vec = torch.cat((text_vec, down_image), 1)
        else:
            text_image_vec = down_image

        output = self.classifier(text_image_vec)
        return output.view(-1)
