#npr.py
import imageIO as io
#import a2
import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
import random as rnd
import nprHelper as helper
import math

def brush(out, y, x, color, texture):
    '''out: the image to draw to. y,x: where to draw in out. color: the color of the stroke. texture: the texture of the stroke.'''
    outH = out.shape[0]
    outW = out.shape[1]
    texH = texture.shape[0]
    texW = texture.shape[1]

    colorIm = texture.copy()
    colorIm[:,:] = color

    xOffsetT = xOffsetB = texW/2
    yOffsetT = yOffsetB = texH/2

    if (yOffsetB*2 != texH): yOffsetB += 1
    if (xOffsetB*2 != texW): xOffsetB += 1

    if ((x>texW/2 and x<outW-texW/2) and (y>texH/2 and y<outH-texH/2)):
    	out[y-yOffsetT:y+yOffsetB, x-xOffsetT:x+xOffsetB] = out[y-yOffsetT:y+yOffsetB, x-xOffsetT:x+xOffsetB]*(1-texture) + colorIm*(texture)


def singleScalePaint(im, out, importance, texture, size=10, N=1000, noise=0.3):
    '''Paints with all brushed at the same scale using importance sampling.'''
    k = float(size)/max(texture.shape)
    texScaled = helper.scaleImage(texture, k)
    outH = out.shape[0]
    outW = out.shape[1]

    strokes = 0
    while (strokes < N):
    	(y, x) = (rnd.randrange(0, outH), rnd.randrange(0, outW))
    	u = np.random.rand()

    	if importance[y,x][0] > u:

            col = im[y, x]
            modNoise = 1-noise/2+noise*np.random.rand(3)
            col *= modNoise
            brush(out, y, x, col, texScaled)
            strokes += 1


def painterly(im, texture, N=10000, size=50, noise=0.3):
    '''First paints at a coarse scale using all 1's for importance sampling, then paints again at size/4 scale using the sharpness map for importance sampling.'''
    importanceLow = np.ones_like(im)
    outLow = io.constantIm( im.shape[0], im.shape[1], 0.0 )
    singleScalePaint(im, outLow, importanceLow, texture, size, N, noise)

    importanceHigh = helper.sharpnessMap(im)
    singleScalePaint(im, outLow, importanceHigh, texture, size/4, N, noise)

    return outLow


def computeAngles(im):
    '''Return an image that holds the angle of the smallest eigenvector of the structure tensor at each pixel. If you have a 3 channel image as input, just set all three channels to be the same value theta.'''
    out = im.copy()
    tensor = helper.computeTensor(im, 3, 5)

    for y,x in helper.imIter(tensor):

    	M = tensor[y,x]
    	w, v = np.linalg.eigh( M )
    	vLarge = v[:, np.argmax(np.abs(w))]
    	angle = np.arctan2( vLarge[1], vLarge[0]) + math.pi
    	out[y,x] = np.array([angle, angle, angle])

    return out


def singleScaleOrientedPaint(im, out, thetas, importance, texture, size, N, noise, nAngles=36):
    '''same as single scale paint but now the brush strokes will be oriented according to the angles in thetas.'''
    k = float(size)/max(texture.shape)
    texScaled = helper.scaleImage(texture, k)
    outH = out.shape[0]
    outW = out.shape[1]

    brushes = helper.rotateBrushes(texScaled, nAngles)

    strokes = 0
    while (strokes < N):
    	(y, x) = (rnd.randrange(0, outH), rnd.randrange(0, outW))
    	u = np.random.rand()

    	if importance[y,x][0] > u:

            col = im[y, x]
            modNoise = 1-noise/2+noise*np.random.rand(3)
            col *= modNoise

            i = int((thetas[y,x][0]*nAngles)/(2*np.pi)) % nAngles
            brush(out, y, x, col, brushes[i])
            strokes += 1


def orientedPaint(im, texture, N=7000, size=50, noise=0.3):
    '''same as painterly but computes and uses the local orientation information to orient strokes.'''
    importanceLow = np.ones_like(im)
    outLow = io.constantIm( im.shape[0], im.shape[1], 0.0 )
    thetas = computeAngles(im)
    singleScaleOrientedPaint(im, outLow, thetas, importanceLow, texture, size, N, noise)

    importanceHigh = helper.sharpnessMap(im)
    singleScaleOrientedPaint(im, outLow, thetas, importanceHigh, texture, size/4, N, noise)

    return outLow

