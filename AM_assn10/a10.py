#Light field Assignment
#By Abe Davis

import numpy as np
import scipy as sp
from scipy import ndimage
import imageIO as io

#   ###  *NOTES!*  ###
# Our light fields are to be indexed
# LF[v,u,y,x,color]
#
# Our focal stacks are to be indexed
# FS[image, y, x, color]


def apertureView(LF):
    '''Takes a light field, returns 'out,' an image with nx*ny sub-pictures representing the value of each pixel in each of the nu*nv views.'''
    nv = LF.shape[0]
    nu = LF.shape[1]
    ny = LF.shape[2]
    nx = LF.shape[3]
  
    out = io.constantIm( nv*ny, nu*nx, 0.0)
    for y in range(ny):
        for x in range(nx):
            for v in range(nv):
                for u in range(nu):
                    out[nv*y+v,nu*x+u] = LF[v,u,y,x]
    return out


def epiSlice(LF, y):
    '''Takes a light field. Returns the epipolar slice with constant v=(nv/2) and constant y (input argument).'''
    nv = LF.shape[0]
    nu = LF.shape[1]
    ny = LF.shape[2]
    nx = LF.shape[3]
    v=(nv/2)

    out = io.constantIm( nu, nx, 0.0)
    for x in range(nx):
        for u in range(nu):
            out[u, x] = LF[v,u,y,x]
    return out


def shiftFloat(im, dy, dx):
    '''Returns a copy of im shifted by the floating point values dy in y and dx in x. We want this to be fast so use either scipy.ndimage.map_coordinates or scipy.ndimage.interpolation.affine_transform.'''
    #return sp.ndimage.interpolation.shift(im, (dy,dx,0), mode='nearest')
    grid = sp.mgrid[0.:im.shape[0], 0.:im.shape[1], 0.:im.shape[2]]
    grid[0] += dy
    grid[1] += dx
    out = sp.ndimage.map_coordinates(im, grid, order=3)
    return out


def refocusLF(LF, maxParallax=0.0, aperture=17):
    '''Takes a light field as input and outputs a focused image by summing over u and v with the correct shifts applied to each image. Use aperture*aperture views, centered at the center of views in LF.
        A view at the center should not be shifted. Views at opposite ends of the aperture should have shifts that differ by maxParallax. See handout for more details.'''
    Nv = LF.shape[0]
    Nu = LF.shape[1]
    Ny = LF.shape[2]
    Nx = LF.shape[3]
    centerv = Nv/2 #v coordinate of center view
    centeru = Nu/2 #u coordinate of center view

    out = io.constantIm(Ny, Nx, 0.0)
    sumW = 0

    for v in range(aperture):
        for u in range(aperture):
            vInd = centerv - aperture/2 + v 
            uInd = centeru - aperture/2 + u 
            im = LF[vInd,uInd]
            dv = (vInd-centerv)*maxParallax/float(aperture)
            du = (uInd-centeru)*maxParallax/float(aperture)
            im = shiftFloat(im, dv, -du)
            sumW += 1
            out += im
    return out/sumW


def rackFocus(LF, aperture=8, nIms = 15, minmaxPara=-7.0, maxmaxPara=2.0):
    '''Takes a light field, returns a focal stack. See handout for more details '''
    paras = np.linspace(minmaxPara, maxmaxPara, nIms)
    ims = map(lambda maxPara : refocusLF( LF, maxPara, aperture ), paras)
    return np.array( ims )


def sharpnessMap(im, exponent=1.0, sigma=1.0):
    '''Computes the sharpness map of one image. This will be used when we compute all-focus images. See handout.'''
    L = BW(im)
    blur=ndimage.filters.gaussian_filter(L, sigma)
    high=L-blur
    energy=high*high
    sharpness=ndimage.filters.gaussian_filter(energy, 4*sigma)
    sharpness=np.power(sharpness, exponent)
    return imageFrom1Channel(sharpness)

    
def sharpnessStack(FS, exponent=1.0, sigma=1.0):
    '''This should take a focal stack and return a stack of sharpness maps. We provide this function for you.'''
    SS = np.zeros_like(FS)
    for i in xrange(FS.shape[0]):
        SS[i]=sharpnessMap(FS[i], exponent, sigma)
    return SS


def depthStack(FS):
    '''This should take a focal stack and return a stack of depth maps. We provide this function for you.'''
    depths = np.linspace(0.0, 1.0, len(FS))
    ZS = map(lambda d : io.constantIm(FS.shape[1],FS.shape[2],d), depths)
    return np.array( ZS )


def fullFocusLinear(stack, exponent=1.0, sigma=1.0):
    '''takes a numpy array stack[image, y, x, color] and returns an all-focus image and a depth map. See handout.'''
    s0 = stack[0]
    out = np.zeros_like(s0)
    zmap = np.zeros_like(s0)
    sumW = np.zeros_like(s0)
    SS = sharpnessStack(stack, exponent, sigma)
    ZS = depthStack(stack)

    for i in range(len(stack)):
        imF = stack[i] * SS[i]
        imZ = ZS[i] * SS[i]
        out += imF
        zmap += imZ
        sumW += SS[i]

    out/=nonZero(sumW)
    zmap/=nonZero(sumW)
    zmap=np.power(zmap,2)
    return out, zmap

#Helpers
def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)

def nonZero(im):
    im[ im==0 ] = np.inf
    epsilon = np.min( im )
    im[ im == np.inf ] = epsilon
    return im

def BW(im, weights=[0.3,0.6,0.1]):
    lum = np.zeros((im.shape[0], im.shape[1]))
    lum = np.dot(im, weights)
    return lum

def imageFrom1Channel(a):
    out=np.empty([a.shape[0], a.shape[1], 3])
    for i in xrange(3):
        out[:, :,i]=a
    return out

