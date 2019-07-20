"""
General purpose utilities script
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob


def load_imgs_from_folder(folder_name):
    """
    Loads a set of images. Loads all the images in a folder
    """
    imgs_fns = [img_fn for img_fn in glob.glob('SoilSamples/{}/*.jpg'.format(folder_name))]
    imgs = [cv2.imread(img_fn,cv2.IMREAD_COLOR) for img_fn in imgs_fns]
    return imgs

def copy_imgs(imgs):
    return [img.copy() for img in imgs]

def get_cmap(img):
    if len(img.shape) == 2:
        return 'gray'
    else:
        return None

def display_imgs(imgs, width=2, height=2, title=''):
    """
    - Displays width*height number of imgs
    """
    fig = plt.figure(figsize=(width*6,height*4))
    for i in range(width*height):
        fig.add_subplot(height, width, i+1)
        plt.imshow(imgs[i],cmap = get_cmap(imgs[i]))
        plt.axis('off')
    plt.suptitle(title, fontsize=36)
    plt.show()

def plt_show(img, size='big', title=''):
    """
    - Use matplotlib to show a image (large)
    """
    if size == 'big':
        plt.figure(figsize=(14,14))
    elif size == 'med':
        plt.figure(figsize=(8,8))
    elif size == 'small':
        plt.figure(figsize=(4,4))
    else:
        print('Warning: Size {} doens\'t exist, defaulting to medium'.format(size))
        plt.figure(figsize=(8,8))

    plt.imshow(img, cmap=get_cmap(img))
    plt.title(title)
    plt.show()

def calc_area(M1,M2):
    return np.pi*M1*M2/4.

def calc_eccentricity(M1,M2):
    return np.sqrt(1.-M1/M2)

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def plot_hist(l, max_val, num_bins=40, x_label='', y_label='', title=''):
    figure, ax = plt.subplots(figsize=(12,12))
    bins = np.linspace(0,max_val,num_bins+1)
    n, bins, patches = ax.hist(np.clip(l, bins[0], bins[-1]), bins=bins, density=False)
    
    xticklabels = ax.get_xticks()
    xticklabels = xticklabels[:-1]
    xticklabels = ['{:.1f}'.format(xticklabel) for xticklabel in xticklabels]
    xticklabels[-1] += '+'
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel(x_label,size=16)
    ax.set_ylabel(y_label,size=16)
    ax.set_title(title,size=24)
    plt.show()

def area_histogram(all_ellipses_props):
    all_ellipses_props = flatten_list(all_ellipses_props)
    all_areas = [calc_area(*M1M2) for _,M1M2,_ in all_ellipses_props]
    plot_hist(all_areas,
              max_val=20000,
              num_bins=40,
              x_label='Pebble Area (Pixels)',
              y_label='Pixel Area (Frequency)',
              title='Pebble Area Histogram')

def shape_histogram(all_ellipses_props):
    all_ellipses_props = flatten_list(all_ellipses_props)
    all_eccentricitys = [calc_eccentricity(*M1M2) for _,M1M2,_ in all_ellipses_props]
    plot_hist(all_eccentricitys,
              max_val=1.0,
              num_bins=100,
              x_label='Pebble Eccentricity (Pixels)',
              y_label='Pixel Eccentricity (Frequency)',
              title='Pebble Eccentricity Histogram')