import cv2
import numpy as np
from ssd import *
from sad import *
from ncc import *


def findCorners(image, window_size, k, thresh):
    # Find x and y derivatives
    dy, dx = np.gradient(image)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = image.shape[0]
    width = image.shape[1]

    cornerList = []
    copiedImage = image.copy()
    outputImage = cv2.cvtColor(copiedImage, cv2.COLOR_GRAY2RGB)
    offset = window_size/2

    # Loop through the images and detect the corners
    print "Finding Corners of the image"
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculating corner response using determinant and trace
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            # If corner response crosses threshold, point is marked
            if r > thresh:
                # print x, y, r
                cornerList.append([x, y, r])
                outputImage.itemset((y, x, 0), 0)
                outputImage.itemset((y, x, 1), 0)
                outputImage.itemset((y, x, 2), 255)
    return outputImage, cornerList


# Set the desired number of levels here (level = 1 is the original image)
levels = 3


def getOriginalImages():
    # Change the images here to test
    left = cv2.imread('left1.png')
    right = cv2.imread('right1.png')

    originalLeft = left.copy()
    originalRight = right.copy()

    return originalLeft, originalRight


def resolution(image, levels):
    if(levels < 1):
        print ("Enter valid number of levels")

    h, w, c = image.shape

    outputImage = image

    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]

    return outputImage


def initImage():
    left,right = getOriginalImages()
    left = resolution(left, levels)
    right = resolution(right, levels)
    cv2.imshow('Input left image', left)
    cv2.imshow('Input right image', right)

    # Set the template size here
    templateSize = 7

    # Set the matching window here
    window = 100

    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    window_size = 3
    k = 0.15
    thresh = 100000

    finalLeft, cornerList = findCorners(left, int(window_size), float(k), int(thresh))
    finalRight, cornerList = findCorners(right, int(window_size), float(k), int(thresh))

    return finalLeft, finalRight, templateSize, window


# Stereo matching using SSD
def ssd():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    cv2.imshow('Corner Response Left Image', left)
    cv2.imshow('Corner Response Right Image', right)
    left = cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_ssd(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_ssd(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return leftDisparity, rightDisparity


# Stereo matching using SAD
def sad():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    cv2.imshow('Corner Response Left Image', left)
    cv2.imshow('Corner Response Right Image', right)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_sad(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_sad(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return leftDisparity, rightDisparity


# Stereo matching using normalized correlation
def ncc():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    cv2.imshow('Corner Response Left Image', left)
    cv2.imshow('Corner Response Right Image', right)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    leftDisparity = np.abs(disparity_ncorr(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rightDisparity = np.abs(disparity_ncorr(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

    return leftDisparity, rightDisparity


# Validity check of the two images
def validity(left, right):
    r1, c1 = left.shape
    r2, c2 = right.shape

    # Validate left image by calculating left - right image disparities
    for i in range(0, r1, 1):
        for j in range(0, c1, 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0

    # Validate left image by calculating right - left image disparities
    for i in range(0, r2, 1):
        for j in range(0, c2, 1):
            if right[i, j] != left[i, j]:
                right[i, j] = 0

    cv2.imshow('Validated Left Image', left)
    cv2.imshow('Validated Right Image', right)


# Averaging is performed in the neighborhood to fill the gaps (zeroes)
def averaging(left, right):
    kernel = np.ones((5, 5), np.float32) / 25
    left = cv2.filter2D(left, -1, kernel)
    right = cv2.filter2D(right, -1, kernel)
    cv2.imshow('Averaged Left Image', left)
    cv2.imshow('Averaged Right Image', right)


# Propogating disparity to the lower level of the pyramid and updating the disparities
def propogate(left, right):
    h, w = left.shape

    for k in range(levels-1, 0, -1):
        outputLeft = left.copy()
        for i in range(0, h, 2 ** (k)):
            for j in range(0, w, 2 ** (k)):
                outputLeft[i:i + 2 ** (k), j:j + 2 ** (k)] = left[i, j]
        cv2.imshow('Propogated Left Disparity ' + str(k), outputLeft)

    for k in range(levels-1, 0, -1):
        outputRight = right.copy()
        for i in range(0, h, 2 ** (k)):
            for j in range(0, w, 2 ** (k)):
                outputRight[i:i + 2 ** (k), j:j + 2 ** (k)] = left[i, j]
        cv2.imshow('Propogated Right Disparity ' + str(k), outputRight)

    newLeft, newRight = getOriginalImages()
    newLeft = cv2.cvtColor(newLeft, cv2.COLOR_BGR2GRAY)
    newRight = cv2.cvtColor(newRight, cv2.COLOR_BGR2GRAY)
    newLeftDisparity = np.abs(disparity_ssd(newLeft, right, templateSize=7, window=100, lambdaValue=0.0))
    newRightDisparity = np.abs(disparity_ssd(newRight, left, templateSize=7, window=100, lambdaValue=0.0))

    # Scale disparity maps
    newLeftDisparity = cv2.normalize(newLeftDisparity, newLeftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)
    newRightDisparity = cv2.normalize(newRightDisparity, newRightDisparity, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)

    updatedLeft = newLeftDisparity + outputLeft
    updatedRight = newRightDisparity + outputRight
    cv2.imshow('Updated left', updatedLeft)
    cv2.imshow('Updated right', updatedRight)


def selectScore():
    score = raw_input("Select a matching score: 1.SSD   2.SAD   3.NCC ")

    if score == '1':
        left, right = ssd()
        cv2.imshow('Left Disparity', left)
        cv2.imshow('Right Disparity', right)
        validity(left, right)
        averaging(left,right)
        propogate(left, right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif score == '2':
        left, right = sad()
        cv2.imshow('Left Disparity', left)
        cv2.imshow('Right Disparity', right)
        validity(left, right)
        averaging(left, right)
        propogate(left, right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif score == '3':
        left, right = ncc()
        cv2.imshow('Left Disparity', left)
        cv2.imshow('Right Disparity', right)
        validity(left, right)
        averaging(left, right)
        propogate(left, right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print ("Select a valid matching score")


selectScore()
