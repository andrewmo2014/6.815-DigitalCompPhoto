#assignment 6 starter code
#by Abe Davis
#
# Student Name: Andrew Moran
# MIT Email: andrewmo@mit.edu

import numpy as np
import scipy
from scipy import linalg
from scipy import ndimage
import imageIO as io


def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)


def pix(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns a pixel.
        If y,x is outside the image and repeatEdge==True , you should return the nearest pixel in the image
        If y,x is outside the image and repeatEdge==False , you should return a black pixel
    '''
    height = im.shape[0]
    width = im.shape[1]
   
    if repeatEdge:
       if (x<0) or (x>=width) or (y<0) or (y>=height):
          return np.array([0,0,0])
       else:
          return im[y,x]
    else:
       clipX = min(width-1, max(x,0))
       clipY = min(height-1, max(y,0))
       return im[clipY, clipX]


def interpolateLin(im, y, x, repeatEdge=0):
    '''same as from previous assignment'''
    x_int = int(x)
    y_int = int(y)

    dx = x - x_int
    dy = y - y_int
    
    n00 = pix(im, y_int  , x_int  , repeatEdge)
    n01 = pix(im, y_int  , x_int+1, repeatEdge)
    n10 = pix(im, y_int+1, x_int  , repeatEdge)
    n11 = pix(im, y_int+1, x_int+1, repeatEdge)
    
    a = n00*(1-dx) + n01*dx
    b = n10*(1-dx) + n11*dx

    pixel = a*(1-dy) + b*dy
    return pixel


def applyHomography(source, out, H, bilinear=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out. 
    If bilinear=True use bilinear interpolation, otherwise use NN. Keep in mind that we are iterating 
    through the output image, and the transformation from output pixels to source pixels is the inverse of 
    the one from source pixels to the output. Does not return anything.'''

    for y,x in imIter(out):
    	(y1, x1, w1) = np.dot( np.linalg.inv( H ), np.array([[y],[x],[1]]))
    	y2 = y1[0]/w1[0]
    	x2 = x1[0]/w1[0]
    	if ((y2>=0) and (y2<source.shape[0])) and ((x2>=0) and (x2<source.shape[1])):
	    	if (bilinear):
	    		pixel = interpolateLin(source, y2, x2)
	    	else:
	    		pixel = pix(source, int(y2), int(x2))
    		out[y,x] = pixel


def addConstraint(systm, i, constr):
    '''Adds the constraint constr to the system of equations ststm. constr is simply listOfPairs[i] from the 
    argument to computeHomography. This function should fill in 2 rows of systm. We want the solution to our 
    system to give us the elements of a homography that maps constr[0] to constr[1]. Does not return anything'''

    y = constr[0][0]
    x = constr[0][1]
    y1 = constr[1][0]
    x1 = constr[1][1]

    row1 = np.array([y, x, 1, 0, 0, 0, -y*y1, -x*y1, -y1])
    row2 = np.array([0, 0, 0, y, x, 1, -y*x1, -x*x1, -x1])
    systm[i] = row1
    systm[i+1] = row2


def computeHomography(listOfPairs):
    '''Computes and returns the homography that warps points listOfPairs[-][0] to listOfPairs[-][1]'''
    A = np.zeros((len(listOfPairs)*2, 9))
    for i in range(len(listOfPairs)):
    	addConstraint( A, i*2, listOfPairs[i])

    u, s, v = np.linalg.svd(A)
    x = v[-1,:]
    H = x.reshape((3,3))
    return H


def computeTransformedBBox(imShape, H):
    '''computes and returns [[ymin, xmin],[ymax,xmax]] for the transformed version of the rectangle described in 
    imShape. Keep in mind that when you usually compute H you want the homography that maps output pixels into 
    source pixels, whereas here we want to transform the corners of our source image into our output coordinate 
    system.'''
    b00 = np.array([0, 0, 1])
    b10 = np.array([imShape[0]-1, 0, 1])
    b01 = np.array([0, imShape[1]-1, 1])
    b11 = np.array([imShape[0]-1, imShape[1]-1, 1])

    Hinv = np.linalg.inv(H)

    b00H = np.dot( Hinv, b00 )
    b10H = np.dot( Hinv, b10 )
    b01H = np.dot( Hinv, b01 )
    b11H = np.dot( Hinv, b11 )

    b00_y = b00H[0]/b00H[2]
    b10_y = b10H[0]/b10H[2]
    b01_y = b01H[0]/b01H[2]
    b11_y = b11H[0]/b11H[2]

    b00_x = b00H[1]/b00H[2]
    b10_x = b10H[1]/b10H[2]
    b01_x = b01H[1]/b01H[2]
    b11_x = b11H[1]/b11H[2]

    y = np.array([ b00_y, b10_y, b01_y, b11_y])
    x = np.array([ b00_x, b10_x, b01_x, b11_x])

    ymin = min(y)
    ymax = max(y)
    xmin = min(x)
    xmax = max(x)

    return [[ymin, xmin], [ymax, xmax]]


def bboxUnion(B1, B2):
    '''No, this is not a professional union for beat boxers. Though that would be awesome. Rather, you should take 
    two bounding boxes of the form [[ymin, xmin,],[ymax, xmax]] and compute their union. Return a new bounding box 
    of the same form. Beat boxing optional...'''
    ymin = np.floor( min(B1[0][0], B2[0][0]) )
    ymax = np.ceil( max(B1[1][0], B2[1][0]) )
    xmin = np.floor( min(B1[0][1], B2[0][1]) )
    xmax = np.ceil( max(B1[1][1], B2[1][1]) )

    return [[ymin, xmin], [ymax, xmax]]


def translate(bbox):
    '''Takes a bounding box, returns a translation matrix that translates the top left corner of that bounding box 
    to the origin. This is a very short function.'''
    ty = bbox[0][0]
    tx = bbox[0][1]
    out = np.array([ [1,0,ty], [0,1,tx], [0,0,1] ])
    return out


def stitch(im1, im2, listOfPairs):
    '''Stitch im1 and im2 into a panorama. The resulting panorama should be in the coordinate system of im2, though 
    possibly extended to a larger image. That is, im2 should never appear distorted in the resulting panorama, only 
    possibly translated. Returns the stitched output (which may be larger than either input image).'''
    H1 = computeHomography( listOfPairs )
    H = np.linalg.inv(H1)

    bbox1 = [[0,0], [im2.shape[0]-1, im2.shape[1]-1]]
    bbox2 = computeTransformedBBox(im1.shape, H)

    bbox = bboxUnion( bbox1, bbox2 )
    trans = translate( bbox )

    height = bbox[1][0] - bbox[0][0] + 1 
    width = bbox[1][1] - bbox[0][1] + 1

    out = io.constantIm( height, width, 0.0)

    ty = trans[0,2]
    tx = trans[1,2]
    out[ -ty:-ty+im2.shape[0], -tx:-tx+im2.shape[1] ] = im2

    Htrans = np.dot(H, trans)
    applyHomographyFast( im1, out, Htrans, True )

    return out


#######6.865 Only###############

def stitchH(im1, im2, H1):
    '''Stitch im1 and im2 into a panorama. The resulting panorama should be in the coordinate system of im2, though 
    possibly extended to a larger image. That is, im2 should never appear distorted in the resulting panorama, only 
    possibly translated. Returns the stitched output (which may be larger than either input image).'''
    H = np.linalg.inv(H1)

    bbox1 = [[0,0], [im2.shape[0]-1, im2.shape[1]-1]]
    bbox2 = computeTransformedBBox(im1.shape, H)

    bbox = bboxUnion( bbox1, bbox2 )
    trans = translate( bbox )

    height = bbox[1][0] - bbox[0][0] + 1 
    width = bbox[1][1] - bbox[0][1] + 1

    out = io.constantIm( height, width, 0.0)

    ty = trans[0,2]
    tx = trans[1,2]
    out[ -ty:-ty+im2.shape[0], -tx:-tx+im2.shape[1] ] = im2

    Htrans = np.dot(H, trans)
    applyHomographyFast( im1, out, Htrans, True )

    return out


def applyHomographyFast(source, out, H, bilinear=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out. This version should 
    only iterate over the pixels inside the bounding box of source's image in out.'''
    bbox = computeTransformedBBox(source.shape, H)

    for y in range( int(np.floor(bbox[0][0])), int(np.ceil(bbox[1][0] + 1)) ):
    	for x in range( int(np.floor(bbox[0][1])), int(np.ceil(bbox[1][1] + 1)) ):
	    	(y1, x1, w1) = np.dot( H, np.array([[y],[x],[1]]))
	    	y2 = y1[0]/w1[0]
	    	x2 = x1[0]/w1[0]
	    	if ((y2>=0) and (y2<source.shape[0])) and ((x2>=0) and (x2<source.shape[1])):
		    	if (bilinear):
		    		pixel = interpolateLin(source, y2, x2)
		    	else:
		    		pixel = pix(source, int(y2), int(x2))
	    		out[y,x] = pixel


def maxPixAndWeight(sourceW, outW, sourceP, outP):
    maxWeight = outW
    maxPix = outP
    if sourceW[0] >= outW[0]:
        maxWeight = sourceW
        maxPix = sourceP
    return (maxPix, maxWeight)


def blendedPix(source, out, sourceWeights, outWeights, y, x, y2, x2, highF=False):
    outWeight = outWeights[y,x]
    sourceWeight = interpolateLin(sourceWeights, y2, x2)

    sumWeight = outWeight + sourceWeight

    outPix = interpolateLin(out, y, x)
    sourcePix = interpolateLin(source, y2, x2)

    if sumWeight[0]<1e-8:
        return (sourcePix, sourceWeight)
    else:
        if highF:
            (maxP, maxW) = maxPixAndWeight(sourceWeight, outWeight, sourcePix, outPix)
            return ((1.0/sumWeight) * (outWeight*outPix + sourceWeight*sourcePix), maxW)
        else:
            return ((1.0/sumWeight) * (outWeight*outPix + sourceWeight*sourcePix), sumWeight)


def applyHomographyFastBlended(source, out, H, bilinear=False, sourceWeights=None, outWeights=None, highF=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out. This version should 
    only iterate over the pixels inside the bounding box of source's image in out.'''
    bbox = computeTransformedBBox(source.shape, H)

    for y in range( int(np.floor(bbox[0][0])), int(np.ceil(bbox[1][0] + 1)) ):
        for x in range( int(np.floor(bbox[0][1])), int(np.ceil(bbox[1][1] + 1)) ):
            (y1, x1, w1) = np.dot( H, np.array([[y],[x],[1]]))
            y2 = y1[0]/w1[0]
            x2 = x1[0]/w1[0]
            if ((y2>=0) and (y2<source.shape[0])) and ((x2>=0) and (x2<source.shape[1])):
                if (bilinear):
                    (out[y,x], outWeights[y,x]) = blendedPix(source, out, sourceWeights, outWeights, y, x, y2, x2, highF)
                else:
                    (out[y,x], outWeights[y,x]) = blendedPix(source, out, sourceWeights, outWeights, y, x, int(y2), int(x2), highF)


def computeNHomographies(listOfListOfPairs, refIndex):
    '''This function takes a list of N-1 listOfPairs and an index. It returns a list of N homographies corresponding 
    to your N images. The input N-1 listOfPairs describes all of the correspondences between images I(i) and I(i+1). 
    The index tells you which of the images should be used as a reference. The homography returned for the reference 
    image should be the identity.'''

    #####################################################################################
    ## 4 Image Example  ##
    #####################################################################################
    #   ---------      ---------      ---------      ---------      ---------              
    #   |  im0  |      |  im1  |      |  im2  |      |  im3  |      |  im4  |  
    #   |  H_0  |  --> |  H_1  |  --> |  H_2  |  <-- |  H_3  |  <-- |  H_4  |
    #   | ref-2 |      | ref-1 |      |  ref  |      | ref+1 |      | ref+2 |       
    #   ---------      ---------      ---------      ---------      ---------      
    #  (H_0)(H_1)I      (H_1)I            I         ((H_3)^-1)I   ((H_4)^-1)((H_3)^-1)I
    #   
    #####################################################################################
    
    identity = np.identity(3)
    Hlist = [ computeHomography( listOfListOfPairs[i] ) for i in range(len(listOfListOfPairs)) ]
    Hlist.insert(refIndex, identity)

    finalH = Hlist
    H = Hlist[refIndex]

    for i in range( len(Hlist) ):
        if i < refIndex:
            ind = len(Hlist[:refIndex])-(i+1)
            H = np.dot( np.linalg.inv( Hlist[ind]), H )
        elif i > refIndex:
            ind = i
            H = np.dot( Hlist[ind], H )
        else:
            ind = i
            H = Hlist[ind]
        finalH[ind] = H

    return finalH


def lowFreqIm(im, sigmaG):
    return ndimage.filters.gaussian_filter( im, [sigmaG, sigmaG, 0])


def highFreqIm(im, sigmaG):
    return im - ndimage.filters.gaussian_filter( im, [sigmaG, sigmaG, 0])


def compositeNImages(listOfImages, listOfH, listOfWeights=None, twoScale=False):
    '''Computes the composite image. listOfH is of the form returned by computeNHomographies. Hint: You will need to 
    deal with bounding boxes and translations again in this function.'''
    bbox = getOuterBBox( listOfImages, listOfH )
    trans = translate(bbox)

    height = bbox[1][0] - bbox[0][0] + 1 
    width = bbox[1][1] - bbox[0][1] + 1

    out = io.constantIm( height, width, 0.0)

    if (listOfWeights is not None):
        weightSum = io.constantIm( height, width, 0.0)

        if (twoScale):
            sigmaG = 2.0
            L_low = map(lowFreqIm, listOfImages, [sigmaG] * len(listOfImages))
            L_high = map(highFreqIm, listOfImages, [sigmaG] * len(listOfImages))
            weightSumL = io.constantIm( height, width, 0.0)
            weightSumH = io.constantIm( height, width, 0.0)
            outL = io.constantIm( height, width, 0.0)
            outH = io.constantIm( height, width, 0.0)

    for i in range(len(listOfH)):

        H = np.dot( listOfH[i], trans)

        if (listOfWeights is not None):
            w = listOfWeights[i]
            imW = listOfImages[i].copy()
            imW[:,:,0] = w
            imW[:,:,1] = w
            imW[:,:,2] = w

            if( twoScale ):
                imL = L_low[i]
                applyHomographyFastBlended(imL, outL, H, True, imW, weightSumL)
                imH = L_high[i]
                applyHomographyFastBlended(imH, outH, H, True, imW, weightSumH, True)

            else:
                im = listOfImages[i]
                applyHomographyFastBlended(im, out, H, True, imW, weightSum)
        else:
            applyHomographyFast(listOfImages[i], out, H, True)

    if (listOfWeights is not None):
        if( twoScale ):

            #io.imwrite(weightSumL, 'debugWeightsLow.png', 1.0)
            #io.imwrite(weightSumH, 'debugWeightsHigh.png', 1.0)

            out = outL + outH
  
    return out


def stitchN(listOfImages, listOfListOfPairs, refIndex):
    '''Takes a list of N images, a list of N-1 listOfPairs, and the index of a reference image. The listOfListOfPairs 
    contains correspondences between each image Ii and image I(i+1). The function should return a completed panorama'''
    return compositeNImages( listOfImages, computeNHomographies(listOfListOfPairs, refIndex) )


#####Helpers#####

def getOuterBBox(listOfImages, listOfH):
    bbox = computeTransformedBBox(listOfImages[0].shape, listOfH[0])

    for i in range(len(listOfH)):
        im1 = listOfImages[i]
        bbox1 = computeTransformedBBox(im1.shape, listOfH[i])
        bbox = bboxUnion( bbox, bbox1 )

    return bbox


def bbTrans( im1, im2, H):
    bbox1 = [[0,0], [im2.shape[0]-1, im2.shape[1]-1]]
    bbox2 = computeTransformedBBox(im1.shape, H)

    bbox = bboxUnion( bbox2, bbox1 )
    trans = translate( bbox )
    return trans

