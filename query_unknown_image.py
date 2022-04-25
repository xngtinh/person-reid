from ReID import *
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob


feature_unknown = extract_feature_unknown_image()
xx = feature_unknown.reshape(2, -1)

file = open("dict_image_feature.txt", "r")

contents = file.read()
dictionary = ast.literal_eval(contents)
file.close()

def sort_img(feature_unknown, dictionary):
    lst_image = []
    lst_dist = []
    for img_name, feature in dictionary.items():
        y = np.array(feature)
        yy = y.reshape(2, -1)
        dist = np.linalg.norm(feature_unknown - yy)
        lst_image.append(img_name)
        lst_dist.append(dist)

    return lst_image, lst_dist

lst_image, lst_dist = sort_img(xx, dictionary)

res = {}
for key in lst_image:
    for value in lst_dist:
        arr_value = value.tolist()
        res[key] = arr_value
        lst_dist.remove(value)
        break


sorted_res = dict(sorted(res.items(), key=lambda kv: kv[1]))

lst_show = []
for k in sorted_res.keys():
    if len(lst_show) <= 3:
        lst_show.append(k)

def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.005)

query_path = glob.glob("static/Market-1501-v15.09.15/pytorch/query_images/0000/*.jpg")

fig = plt.figure(figsize=(16,4))
ax = plt.subplot(1,5,1)
ax.axis('off')
imshow(query_path[0],'unknown image')
for i in range(4):
    ax = plt.subplot(1,5,i+2)
    ax.axis('off')
    img_path = lst_show[i]
    # label = gallery_label[index[i]]
    label = int(lst_show[i].split('_')[1][-1])
    imshow(img_path)
    ax.set_title('ID: %d' % (label), color='green')

fig.savefig("static/show_unknown.png")

