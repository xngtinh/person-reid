import argparse
import scipy.io
import torch
import numpy as np
import time
import os
from torchvision import datasets
import matplotlib.pyplot as plt

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.005)  # pause a bit so that plots are updated

######################################################################
def getFeature():
    result = scipy.io.loadmat('result/RPP_H_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    return query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)

    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

#######################################################################
def visualize(index_list, query_label, query_path, image_datasets):
    try:
        numgocam = int(index_list[0].split('_')[1])
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path = index_list[i]
            # label = gallery_label[index[i]]
            label = int(index_list[0].split('_')[0][-1])
            imshow(img_path)
            if label == query_label:
                ax.set_title('Cam: %d'%(numgocam), color='green')
            else:
                ax.set_title('Cam: %d'%(numgocam), color='red')
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index_list[i]]
    fig.savefig("static/show"+str(numgocam)+".png")

#####################################################################################
def demo(query_path= r"static/Market-1501-v15.09.15/pytorch\query\0003\3_2_292.jpg"):
    data_dir = 'static/Market-1501-v15.09.15/pytorch'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['gallery', 'query']}
    i=0

    numidquery = query_path.split("_")[0][-1]

    for j in range(0, len(image_datasets['query'].imgs)):
        path = image_datasets['query'].imgs[j][0]
        if path == query_path:
            i = j
            break

    query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label = getFeature()
    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

    index_A = []
    index_B = []
    index_C = []
    index_D = []

    for ii in range(0, 100):
        iter = index[ii]
        path_gallery = image_datasets['gallery'].imgs[iter][0]

        numcam = path_gallery.split("_")[1]

        numidgallery = path_gallery.split("_")[0][-1]

        if numcam == '1' and numidquery == numidgallery:
            index_A.append(path_gallery)
        elif numcam == '2' and numidquery == numidgallery:
            index_B.append(path_gallery)
        elif numcam == '3' and numidquery == numidgallery:
            index_C.append(path_gallery)
        elif numcam == '4' and numidquery == numidgallery:
            index_D.append(path_gallery)

    #######################################################################

    query_path, _ = image_datasets['query'].imgs[i]
    query_label = query_label[i]

    if len(index_A) != 0:
        visualize(index_A, query_label, query_path, image_datasets)

    if len(index_B) != 0:
        visualize(index_B, query_label, query_path, image_datasets)

    if len(index_C) != 0:
        visualize(index_C, query_label, query_path, image_datasets)

    if len(index_D) != 0:
        visualize(index_D, query_label, query_path, image_datasets)

#
if __name__ == "__main__":
    demo()
