import numpy as np
import torch
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = pickle.load(open('param_dict.pkl', 'rb'))





def sampler(gen, dataset, epoch, default = True, noise = None, no_image = 16, save_path = ''):


    """
    This function is used to sample images from the trained generator.
    gen: Trained generator used as an sampler.
    dataset: The dataset from which the pre-defined embeddings is obtained.
    epoch: The epoch at which sampling is called.
    default: If True Noise input is provided, else should be constructed within the function.
    noise: The noise input.
    no_image: Number of images to be sampled.
    save_path: Path to save the sampled images.
    """


    l = len(dataset)
    choosen = []
    gen.eval()
    embedding = torch.zeros(no_image, 1024)
    index = random.sample(range(0,l), no_image) #choose randomly from the provided dataset.

    for s in range(no_image):
        embedd, chose = dataset[index[s]]
        choosen.append(chose)
        embedding[s] = embedd

    embedding = embedding.to(device)
    
    if (default):
        Noise = noise[0:no_image, :]
    else:
        Noise = torch.FloatTensor(no_image, 100).normal_(0,1)
        Noise = Noise.to(device)

    _, generated, _, _ = gen(embedding, Noise)
    for j in range(no_image):
        name = 'SaImg_epo' + str(epoch) + '_' + str(j) + '.png'
        path = os.path.join(save_path, name)
        image = generated[j].data
        image = image.to("cpu")
        image = image.numpy()
        image = (image + 1.0) * 127.5
        image = image.astype(np.uint8)
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image)
        image.save(path)
