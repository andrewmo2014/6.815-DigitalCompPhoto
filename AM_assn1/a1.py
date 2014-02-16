#a1.py
#Andrew Moran
#6.815
#Assignment 1 (or 0, depending on who asks)
#by Abe Davis
#
#You can change the 'as' calls below if you want to call functions differently (i.e. numpy.function() instead of np.function())

import imageIO as io
import numpy as np
import scipy as sp

def brightness(im, factor):
    imb = im.copy() #always work on a copy of the image!
    #modify brightness of image by factor
    imb = imb*factor
    np.clip(imb, 0, 1, out=imb) #Extra Credit
    return imb


def contrast(im, factor, midpoint=0.3):
    imc = im.copy()
    #modify contrast of image
    imc = (imc-midpoint)*factor + midpoint
    np.clip(imc, 0, 1, out=imc) #Extra Credit
    return imc


def frame(im):
    imf = im.copy()
    #apply frame to image
    height = np.shape(imf)[0]
    width = np.shape(imf)[1]
    black = [0,0,0]
    imf[:, 0] = black
    imf[0, :] = black
    imf[:, width-1] = black
    imf[height-1, :] = black
    return imf


def BW(im, weights=[0.3,0.6,0.1]):
    img = im.copy()
    #create black and white image
    grey = np.dot(img[:,:], weights)
    img[:,:,0] = grey
    img[:,:,1] = grey
    img[:,:,2] = grey
    return img


def lumiChromi(im):
    #create lumi and chromi images. Remember to work on copies of im!
    imO = im.copy()
    imL = imO
    imL = BW(imL)
    imC = imO/imL
    return (imL, imC)


def brightnessContrastLumi(im, brightF, contrastF, midpoint=0.3):
    outim = im.copy()
    #modify brightness and contrast without changing the chrominance
    (imL, imC) = lumiChromi(outim)
    imL = brightness(imL, brightF)
    imL = contrast(imL, contrastF, midpoint)
    outim = imL*imC
    return outim


def rgb2yuv(im):
    imyuv = im.copy()
    #convert to yuv using the matrix given in the assignment handout
    A = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
    imyuv[:,:] = np.dot(imyuv[:,:], sp.transpose(A))
    return imyuv


def yuv2rgb(im):
    imrgb = im.copy()
    #convert to rgb using the matrix given in the assignment handout
    A = [[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]]
    imrgb[:,:] = np.dot(imrgb[:,:], sp.transpose(A))
    return imrgb


def saturate(im, k):
    imS = im.copy()
    #create and return saturated image
    imyuv = rgb2yuv(imS)
    imyuv[:,:,1] *= k
    imyuv[:,:,2] *= k
    imS = yuv2rgb(imyuv)
    return imS
    

#spanish castle URL: http://www.johnsadowski.com/big_spanish_castle.php
#HINT: to invert color for a YUV image, negate U and V
def spanish(im):
    #Create a pair of images for a spanish castle illusion
    imI = im.copy()
    height = np.shape(imI)[0]
    width = np.shape(imI)[1]
    Y_constant = 0.5

    imL = lumiChromi(imI)[0]
    imL[height/2, width/2] = [0,0,0]
    
    imC = imI.copy()
    imC = rgb2yuv(imC)
    imC[:,:,0] = Y_constant
    imC[:,:,1] *= -1.0
    imC[:,:,2] *= -1.0
    imC = yuv2rgb(imC)
    imC[height/2, width/2] = [0,0,0]
    
    return (imL, imC)


def histogram(im, N):
    #create the histogram

    height = np.shape(im)[0]
    width = np.shape(im)[1]
    
    h = np.zeros(N, dtype=np.int32)
    imL = im.copy()
    imL = BW(im)

    for x in range(width):
        for y in range(height):
            Y = imL[y,x,0]
            k = int(Y*N)
            h[k] += 1
            
    h=h/float(sum(h))
    return h

def printHisto(im, N, scale):
    #print the histogram
    h = histogram(im, N)
    h = h*N*scale
    for binNum in range(len(h)):
        row = ""
        for val in range(int(h[binNum])):
            row+="X"
        print row
    return
        
        
