'''
Purpose: to segment and categorize pebbles in an image
'''


from __future__ import print_function
import cv2 as cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from gen_utils import *
from ip_utils import *
import os


def segmentPebbles(im):
    # save a copy of the original image for drawing ellipses later
    orig = im.copy()
    
    # hsv colorspace is better than RGB fight me
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    
    channelThresh = []
   
    hsvThresh = hsv.copy()
    
    # calculate and apply threshold masks for each HSV channel
    for n, c in [('Hue', 0), ('Sat', 1), ('Val', 2)]:
        # use Otsu's thresholding to perform binary threshold
        # outputs threshold value (0-255) and binarized image
        threshVal, thresh = cv2.threshold(hsv[:, :, c], 0, 255, cv2.THRESH_OTSU)
        
        if c == 0:
            thresh = 255 - thresh
            
        # Use the binarized image as a mask
        hsvThresh[:, :, c] = cv2.bitwise_and(hsvThresh[:, :, c], hsvThresh[:, :, c], mask=thresh)      
        
        # uncomment to see the masked version of each channel
        cv2.imshow(n, hsvThresh[:, :, c])
        cv2.waitKey()
        channelThresh.append(thresh)

    finalThresh = np.ones(channelThresh[0].shape, dtype=np.uint8)
    
    # apply each channel threshold
    for i in range(len(channelThresh)):
        finalThresh = finalThresh & channelThresh[i]

    hsvThresh[finalThresh == 0] = 0
    kernel = np.ones((3,3))
    finalThresh = cv2.morphologyEx(finalThresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    frame = cv2.cvtColor(hsvThresh, cv2.COLOR_HSV2BGR)
    
    # Perform watershed segmentation
    wsMarkers, wsImgs = watershed([frame], [finalThresh], thresh=0.12)
    
    # fit ellipses    
    ellipses_imgs, all_ellipses_props = fit_ellipses([orig], wsMarkers)    
    return ellipses_imgs, all_ellipses_props, finalThresh


if __name__ == '__main__':
    # folder of source images, only images should be in this folder. All common image types
    # should work
    folder = 'SoilSamples\Set4'
    files = sorted(os.listdir(folder))
        
    for foregroundFile in files:
        filePath = os.path.join(folder, foregroundFile)
        im = cv2.imread(filePath)
        
        if im is None:
            continue
        
        ellipses_imgs, all_ellipses_props, finalThresh = segmentPebbles(im)
        
        # place images side by side for quality of life
        sbs = np.hstack((255 * cv2.cvtColor(finalThresh, cv2.COLOR_GRAY2BGR), \
                         ellipses_imgs[0]))    
        
        cv2.imshow('Output', cv2.resize(sbs, (0, 0), fx=1, fy=1))
        cv2.imwrite('./science_{0}'.format(foregroundFile), sbs)
        area_histogram(all_ellipses_props)
        shape_histogram(all_ellipses_props)
        cv2.waitKey()
        cv2.destroyAllWindows()
