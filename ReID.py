# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
import os
import scipy.io
from PCB_model import *

######################################################################
# Load Data

data_transforms = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "static/Market-1501-v15.09.15/pytorch"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery_test', 'query_images']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=False, num_workers=8) for x in ['gallery_test', 'query_images']}

######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = "models/net_best_combine.pth"
    network.load_state_dict(torch.load(save_path), strict=False)
    return network

######################################################################
# Extract feature
# Extract feature from  a trained model.

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_features(model, input):
    return model(input)

def features(model, imgs):
    features = torch.FloatTensor()
    list_name_image = []
    features_total = []
    for img in imgs:
        list_name_image.append(img)
        directory = img
        img = Image.open(directory)
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = data_transforms(img)
        img = torch.unsqueeze(img, 0)
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 512, 6).zero_()
        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            output = extract_features(model, img)

            f = output.data.cpu()
            ff = ff+f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        features = torch.cat((features, ff), 0)

    features_total.append(features)

    return features_total, list_name_image

#######################################################################################################
# model_structure = PCB(2220)
#
# model_structure = model_structure.convert_to_rpp()
#
# model = load_network(model_structure)
#
# model = PCB_test(model, True)
#
# # Change to test mode
# model = model.eval()

# query_feature, list_name_query_image = features(model, list_images_query)
#
# lst_query_features = []
# for i in range(0, len(query_feature[0])):
#     query_feature_i = query_feature[0][i]
#     lst_query_features.append(query_feature_i)
#######################################################################################################


query_path = image_datasets['query_images'].imgs

def extract_feature_unknown_image():
    list_images_query = []
    for i in range(len(query_path)):
        list_images_query.append(query_path[i][0])

    model_structure = PCB(2220)

    model_structure = model_structure.convert_to_rpp()

    model = load_network(model_structure)

    model = PCB_test(model, True)

    # Change to test mode
    model = model.eval()

    query_feature, list_name_query_image = features(model, list_images_query)

    lst_query_features = []
    for i in range(0, len(query_feature[0])):
        query_feature_i = query_feature[0][i].numpy()
        lst_query_features.append(query_feature_i)

    return lst_query_features[0]

#######################################################################################################################################

# extract features gallery images

# gallery_path = image_datasets['gallery_test'].imgs

# list_images_gallery = []
# for i in range(len(gallery_path)):
#     list_images_gallery.append(gallery_path[i][0])

# gallery_feature, list_name_gallery_image = features(model, list_images_gallery)
#
# lst_gallery_features = []
# for i in range(0, len(gallery_feature[0])):
#     gallery_feature_i = gallery_feature[0][i].numpy()
#     lst_gallery_features.append(gallery_feature_i)
#
#
# res = {}
# for key in list_name_gallery_image:
#     for value in lst_gallery_features:
#         arr_value = value.tolist()
#         res[key] = arr_value
#         lst_gallery_features.remove(value)
#         break
# #
# f = open("dict_image_feature.txt", "w")
# f.write("{\n")
# for k in res.keys():
#     f.write("r'{}':{},\n".format(k, res[k]))
# f.write("}")
# f.close()





