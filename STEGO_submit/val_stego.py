import os
from os.path import join
import wget
from train_segmentation import LitUnsupervisedSegmenter
from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from utils import get_transform
import torch.nn.functional as F
from crf import dense_crf
import torch
import matplotlib.pyplot as plt
from utils import unnorm, remove_axes
import cv2
import numpy as np

#manage directory
saved_models_dir = join("..", "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)

#load the pre-trained model
#model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, saved_model_name)).cuda()
#load the trained model
save_path = '/mnt/data/training_weights/base_corp5_3000.ckpt'
model = LitUnsupervisedSegmenter.load_from_checkpoint(save_path).cuda()

#the query image
root = "/mnt/data/input_dataset"
def load_images_from_folder(folder):
    images = {}
    img_gt = {}
    filename_r = []
    for filename in os.listdir(folder):
      if filename[-3:] == 'jpg':
        filename_r.append(filename[:-3])
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        transform = get_transform(448, False, "center")
        img = transform(im_pil).unsqueeze(0).cuda()
        if img is not None:
          images[filename[:-3]] = img
      elif filename[-3:] == 'png':
        img = cv2.imread(os.path.join(folder,filename))
        #print('imgshape',img.shape)
        if img is not None:
          img_gt[filename[:-3]] = img
    return images, img_gt, filename_r

load_image, img_gt, filename_r = load_images_from_folder(root)
count = 0
#validation

with torch.no_grad():
  for i in filename_r:
    count += 1
    img = load_image[i]
    code1 = model(img)
    code2 = model(img.flip(dims=[3]))
    code  = (code1 + code2.flip(dims=[3])) / 2
    code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)


    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
    cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

    single_img = img[0].cpu()
    linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
    cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)


    fig, ax = plt.subplots(1,4, figsize=(5*4,5))
    ax[0].imshow(unnorm(img)[0].permute(1,2,0).cpu())
    ax[0].set_title("Image")
    ax[1].imshow(model.label_cmap[cluster_pred])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(model.label_cmap[linear_pred])
    ax[2].set_title("linear Predictions")
    ax[3].imshow(model.label_cmap[img_gt[i][:,:,0]])
    ax[3].set_title("groundtruth")
    remove_axes(ax)
    fig.savefig('./output/'+save_path[:-5]+'_'+str(count)+'.png') 
    break

torch.cuda.empty_cache()