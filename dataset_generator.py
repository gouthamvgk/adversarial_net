import torchvision.utils as utils
from PIL import Image
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))




class dataset_generator():

    """
    This class is used to construct the dataset for the generator. It loads the images
    and the corresponding pre trained embeddings for sentences describing thoses images.
    image_path: path to the image files.
    file_path: path to the pickled objects which contain information about file names,
               class and pre trained embeddings
    image_size: size of the image to be returned by the class
    transform: torchvision.transforms.Compose object which contains the transformations
               to be applied to the image.
    sampling: If true the dataset is constructed for the test set so that sampling of
              images from pre trained generator can be done.
    """

    def __init__(self, image_path, file_path, image_size = 64, transform = None, sampling = False):
        self.transform = transform
        self.image_size = image_size
        self.image_path = image_path
        self.sample = sampling
        if sampling == False:
            self.train_path = file_path + '/train/'
        else:
            self.train_path = file_path + '/test/'
        self.file_names = []
        self.img_class = []
        self.embeddings = []

        self.load_filename()
        self.load_class()
        self.load_embeddings()



    def __len__(self):
        """
        Returns number of files in the dataset.
        """
        return len(self.file_names)



    def load_filename(self):

        """
        Loads the filenames of the images from the pickled object stored in
        file_path.
        """

        path = os.path.join(self.train_path, 'filenames.pickle')
        file = open(path, 'rb')
        self.file_names = pickle.load(file)
        print('File names loaded for {} files'.format(len(self.file_names)))
        file.close()



    def load_class(self):

        """
        Loads the class information about the images from the pickled object
        stored in file_path.
        """

        path = os.path.join(self.train_path, 'class_info.pickle')
        file = open(path, 'rb')
        self.img_class = pickle.load(file)
        print('Class labels loaded for {} files'.format(len(self.img_class)))
        file.close()



    def load_embeddings(self):

        """
        Loads the pre trained embeddings for the sentences from the pickled object
        and then converts it into torch Tensor for processing.
        """

        path = os.path.join(self.train_path, 'char-CNN-RNN-embeddings.pickle')
        file = open(path, 'rb')
        embeddings = pickle.load(file, encoding = 'iso-8859-1')
        embeddings = np.array(embeddings)
        #embeddings = torch.from_numpy(embeddings)
        #embeddings = embeddings.to(device)
        self.embeddings = embeddings
        print('Embeddings load for {} files'.format(embeddings.shape[0]))
        print('Each file consists of {} embeddings of size {}'.format(embeddings.shape[1], embeddings.shape[2]))
        file.close()



    def load_image(self, image_name):

        """
        Given the image path, it loads the image, resizes it and then applies
        the given transformations to it.
        """

        path = os.path.join(self.image_path, image_name)
        image = Image.open(path).convert('RGB')
        temp_size = int(self.image_size * 76 /64)
        image = image.resize((temp_size, temp_size), Image.BILINEAR)
        if self.transform is not None:
            image = self.transform(image)
            #image = image.to(device)
        return image



    def __getitem__(self, index):

        """
        Magic function that enables iterable access of the dataset.
        index: index of the image and the corresponding sentence
              embedding to be returned.
        """

        img_name = self.file_names[index]
        img_name += '.jpg'

        embedding = self.embeddings[index, :, :]
        choosen = random.randint(0, embedding.shape[0]-1)
        embedding = embedding[choosen, :]
        embedding = torch.from_numpy(embedding)
        #embedding = embedding.to(device)
        image = self.load_image(img_name)

        if (self.sample):
            return embedding, choosen
        else:
            return image, embedding




def dataloader(data_gen, num_workers = 6):
    """
    Return a dataloader that constructs the batches from the dataset.
    num_workers: number of sub process to be used for batch construction.
    """
    return torch.utils.data.DataLoader(data_gen, batch_size = params['batch_size'], shuffle = True, num_workers= num_workers, drop_last = True)
