import pickle
import os
"""
This file containes all the hyperparameters for model construction and
training process. Befor training modify for the required hyperparameter
values and then run it so that a pickle object is generated which is used
by all the other files.
"""
param = dict()

param['noise_dim'] = 100
param['inp_size'] = 64

param['batch_size'] = 64
param['save_every'] = 100
param['epoch'] = 600

param['gen_lr'] = 2e-4
param['dis_lr'] = 2e-4
param['lr_decay_epoch'] = 100
param['save_no'] = 64

param['kl_coeff'] = 2.0

param['embed_dim'] = 1024
param['res_blocks'] = 4
param['con_aug_dim'] = 128
param['dis_fea_dim'] = 64
param['gen_fea_dim'] = 128

param['base_path'] = os.getcwd()
param['save_path'] = os.path.join(param['base_path'], 'save')

pickle.dump(param, open('param_dict.pkl', 'wb'))
