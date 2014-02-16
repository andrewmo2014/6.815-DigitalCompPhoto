# Assignment 5 for 6.815/865
# Submission: 1
# Deadline: 10/09/2013
# Your name: Andrew Moran
# Reminder: 
# - Don't hand in data
# - Don't forget README.txt

import numpy as np
import scipy as sp
import bilagrid as bl

def computeWeight(im, epsilonMini=0.002, epsilonMaxi=0.99):
	out = np.ones( im.shape )
	out[ im < epsilonMini ] = 0.0
	out[ im > epsilonMaxi ] = 0.0
	return out  


def computeFactor(im1, w1, im2, w2):
	img1 = im1 * w1
	img2 = im2 * w2
	img1 = img1.flatten()
	img2 = img2.flatten()	
	slice1 = np.arange(len(img1))[img1>0]
	slice2 = np.arange(len(img2))[img2>0]
	common = np.intersect1d( slice1, slice2 )
	out = np.median( img2[common]/img1[common] )
	return out


def makeHDR(imageList, epsilonMini=0.002, epsilonMaxi=0.99):

	#reverse list to have last image have ki = 1.0
	imageList = imageList[::-1]

	out = np.zeros( imageList[0].shape )
	sumW = np.zeros( out.shape )

	#initialization
	im1 = imageList[0]
	w1 = computeWeight( im1, 0.0, epsilonMaxi )

	out += (w1 * im1)
	sumW += w1
	
	kj = 1.0
	for i in range( len(imageList)-1 ):

		im1 = imageList[i]
		im2 = imageList[i+1]

		if( i == 0):
			w1 = computeWeight( im1, 0.0, epsilonMaxi )
		else:
			w1 = computeWeight( im1, epsilonMini, epsilonMaxi )
		if( i+1 == len(imageList)-1 ):
			w2 = computeWeight( im2, epsilonMini, 1.0 )
		else:
			w2 = computeWeight( im2, epsilonMini, epsilonMaxi )

		k = computeFactor( im1, w1, im2, w2) #k = ki/kj
		ki = k*kj
		kj = ki

		out += (w2 / ki * im2)
		sumW += w2
	
	#clip zeros to smallest non-zero val	
	sumW[ sumW==0 ] = np.inf
	epsilon = np.min( sumW )
	sumW[ sumW == np.inf ] = epsilon

	out /= sumW
	return out

    
def toneMap(im, targetBase=100, detailAmp=1, useBila=False):

	img = im.copy()

	#clip zeros to smallest val
	img[img == 0] = np.inf
	small = np.min(img)
	img[img == np.inf] = small

	#lumi-chromi
	(imL, imC) = lumiChromi(im)
	imL_orig = imL

	#Determine smallest non-zero val
	imL[imL == 0] = np.inf
	epsilon = np.min(imL)
	
	#Compute log
	imL_orig += epsilon
	imL_log = np.log10( imL_orig )

	#Determine standard dev
	dim = np.max([ im.shape[0], im.shape[1] ])
	sdev = dim/float(50)

	#Base (Large) Log
	if( useBila ):
		base_log = bl.bilateral_grid( imL_log, sdev, 0.4 )
	else:
		base_log = sp.ndimage.filters.gaussian_filter( imL_log, [sdev, sdev, 0] )
	#Detail Log
	detail_log = imL_log - base_log

	#Factors
	baseRange = np.max( base_log ) - np.min( base_log )
	k = np.log10(targetBase)/baseRange

	#Output
	outLog = detailAmp*detail_log + k*(base_log - np.max( base_log ))
	outIntensity = np.power(10, outLog)
	out = outIntensity * imC
	return out


def BW(im, weights=[0.3,0.6,0.1]):
    img = im.copy()
    #create black and white image
    grey = np.dot(img[:,:], weights)
    img[:,:,0] = grey
    img[:,:,1] = grey
    img[:,:,2] = grey

    #clip zeros to smallest val
    img[ img == 0] = np.inf
    epsilon = np.min( img )
    img[ img == np.inf ] = epsilon

    return img


def lumiChromi(im):
    #create lumi and chromi images. Remember to work on copies of im!
    imO = im.copy()
    imL = BW(imO)
    imC = imO/imL
    return (imL, imC)


