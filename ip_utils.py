"""
Image processing utilities script
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def cvt_color(imgs, cvt):
    """
    - Converts an entire list of images from one color space to another
    """
    cvt_imgs = [cv2.cvtColor(img, cvt) for img in imgs]
    return cvt_imgs



def blur(imgs, method, **param):
    """
    Applies a certain bluring method to a list of images.
    Params:
        - Average: ksize
        - Gaussian: ksize, sigma
        - Median: ksize
        - Bilateral: d, sigmaClr, sigmaSpc
    """
    if method == 'average':
        ksize = (param['ksize'], param['ksize'])
        blur_imgs = [cv2.blur(img, ksize) for img in imgs]
    elif method == 'gaussian':   
        ksize = (param['ksize'], param['ksize'])
        blur_imgs = [cv2.GaussianBlur(img, ksize, param['sigma']) for img in imgs]
    elif method == 'median':   
        blur_imgs = [cv2.medianBlur(img, param['ksize']) for img in imgs]
    elif method == 'bilateral':
        blur_imgs = [cv2.bilateralFilter(img, param['d'], param['sigmaClr', param['sigmaSpc']]) for img in imgs]
    else:
        raise Exception('{} Not Valid Blurring Method'.format(method))
    return blur_imgs


def simple_thresh(imgs, lower, upper):
    """
    Thresholds and list of images
    """
    return [cv2.inRange(img, lower, upper) for img in imgs]


def otsu_thresh(imgs):
    return [cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for img in imgs]

def morph_tf(masks, operation, kernel_size, iterations):
    """
    Applied a morphological transform on a list of masks
    """
    if kernel_size == 0 or iterations == 0:
        return masks

    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if operation == 'close':
        return [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations) for mask in masks]
    elif operation == 'open':
        return [cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations) for mask in masks]
    elif operation == 'erode':
        return [cv2.erode(mask, kernel, iterations=iterations) for mask in masks]
    elif operation == 'dilate':
        return [cv2.dilate(mask, kernel, iterations=iterations) for mask in masks]
    else:
        raise Exception('{} not a valid morphological operator'.format(operation))


def apply_mask(imgs, masks):
    """
    Applies a list of masks to a list of images
    """

    return [cv2.bitwise_and(img,img,mask=mask) for img, mask in zip(imgs, masks)]


def find_contours(masks):
    """
    Finds the contours of an entire list of masks
    """
    return [cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] for mask in masks]


def draw_contours(imgs, cnts_list):
    """
    Draw contours on a list of images
    """
    [cv2.drawContours(img, cnts, -1, (0,255,0), 5) for img, cnts in zip(imgs, cnts_list)]


def auto_canny(image, sigma=0.33):
    """
    - Does canny edge detection. Automatically finds appropriate thresholds for you.
    - Source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


def watershed(imgs, masks, thresh):
    """
    Applies watershed algorithm to a list of images
    """
    kernel = np.ones((3,3), np.uint8)
    # sure background area
    sure_bgs = [cv2.dilate(mask,kernel,iterations=2) for mask in masks]
    # Finding sure foreground area
    dist_transforms = [cv2.distanceTransform(mask,cv2.DIST_L2,5) for mask in masks]
    sure_fgs = [cv2.threshold(dist_transform,thresh*dist_transform.max(),255,0)[1]\
                for dist_transform in dist_transforms]

    # Finding unknown region
    sure_fgs = [np.uint8(sure_fg) for sure_fg in sure_fgs]
    unknowns = [cv2.subtract(sure_bg,sure_fg) for sure_bg,sure_fg in zip(sure_bgs,sure_fgs)]
    
    # Marker labelling
    markers = [cv2.connectedComponents(sure_fg)[1] for sure_fg in sure_fgs]
    # Add one to all labels so that sure background is not 0, but 1
    markers = [marker+1 for marker in markers]
    # Now, mark the region of unknown with zero
    for marker, unknown in zip(markers,unknowns):
        marker[unknown==255] = 0
    markers = [cv2.watershed(img, marker) for img, marker in zip(imgs, markers)]
    
    ws_imgs = []
    for img, marker in zip(imgs,markers):
        ws_img = img.copy()
        ws_img[marker == -1] = [0,255,0]
        ws_imgs.append(ws_img)
        
    return markers, ws_imgs


def fit_ellipses(imgs, all_markers):
    ellipses_imgs = []
    all_ellipses_props = []
    for img, img_marker in zip(imgs, all_markers):
        ellipses_img = img.copy()
        ellipses_prop = []
        for marker in np.unique(img_marker):
            # Skip the first (background) marker
            if marker == 1: continue

            # Draw contours arround Watershed markers
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[img_marker == marker] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
            max_cnt = max(cnts, key=cv2.contourArea)

            # Draw ellipse arround object of interest
            # Save properties of ellipse for later analysis
            try:
                ellipse = cv2.fitEllipse(max_cnt)
                ma, MA = ellipse[1]
                #print('ellipse data: {0}'.format(ellipse))
                ellipseArea = np.pi * MA * ma
                if ellipseArea > 2000 and MA / ma < 3:
                    #print('ellipse area: {0}'.format(ellipseArea))
                    cv2.ellipse(ellipses_img,ellipse,(0,255,0),5)
                    ellipses_prop.append(ellipse)
            except:
                pass
        ellipses_imgs.append(ellipses_img)
        all_ellipses_props.append(ellipses_prop)
    return ellipses_imgs, all_ellipses_props