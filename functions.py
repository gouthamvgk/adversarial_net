import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.utils as utils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))



def initialize_weights(layer):

    """
    This function is applied to every layer of the module when called like module.apply(initialize_weights).
    It initializes the weight of every layer according to its type.
    The initialization range is obtained from the orginal hyperparameters used by the author.
    """

    layer_name = layer.__class__.__name__
    if layer_name.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif layer_name.find('Linear') != -1:
        layer.weight.data.normal_(0.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)




def save_image(real_image, fake_image, epoch_no, save_path):

    """
    Given a Tensor of real images and fake images generated from the
    generator this function saves it on the disk by making them as a
    grid.
    real_image: images from original dataset
    fake_image: image generated from the generator.
    epoch_no: epoch number during which image is generated.
    save_path: path to save the image.
    """

    fake_images = fake_image[0:params['save_no']]
    if real_image is not None:
        real_images = real_image[0:params['save_no']]
        utils.save_image(real_images, save_path + '/real_image.png', normalize = True)
        utils.save_image(fake_images, save_path + '/img_fake_epo_' + str(epoch_no) +'.png', normalize = True)
    else:
        utils.save_image(fake_images, save_path + '/pre_stage1_fake_epo_' + str(epoch_no) + '.png' , normalize = True)






def save_model(gen, dis,gen_opt, dis_opt, epoch,save_path, stage=1):

    """
    This function checkpoints the model given the generator, discriminator
    and their corresponding optimizers.
    """

    torch.save({
        'stage': stage,
        'generator':gen.state_dict(),
        'discriminator':dis.state_dict(),
        'gen_optimizer':gen_opt.state_dict(),
        'dis_optimizer':dis_opt.state_dict()
    }, save_path + '/model_stg'+str(stage)+'_epoch' + str(epoch) +'.pth')




def image_show(image):

    """
    This function is used to visualize the images generated by the generator
    during training.  The images are made into a grid and then given to this
    function. The image is converted to a numpy tensor and the mean and
    standard deviation are added and then displayed.
    """
    
    inp = image.detach()
    inp = inp.to('cpu')
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated
