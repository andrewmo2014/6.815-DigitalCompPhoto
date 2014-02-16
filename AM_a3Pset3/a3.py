import numpy as np
from scipy import ndimage
import scipy
from scipy import signal
from numpy import exp
import scipy as sp

def check_module():
    return 'Andrew Moran'

      
def boxBlur(im, k):
    ''' Return a  blured image filtered by box filter'''
    im_out = im.copy()
    offset = int(k/2)
    n = float(k*k)
    
    for y,x in imIter(im_out):
        pixel = np.array([0,0,0], dtype=np.float64)
        for dy in range(k):
            for dx in range(k):
                pixel += pix(im, y+dy-offset, x+dx-offset, False)
        im_out[y,x] = pixel/n
                  
    return im_out


def convolve(im, kernel):
    ''' Return an image filtered by kernel'''

    im_out = im.copy()
    offsety = 0
    offsetx = 0
    if (kernel.shape[0]%2==1):
        offsety = int(kernel.shape[0]/2)
    if (kernel.shape[1]%2==1):
        offsetx = int(kernel.shape[1]/2)

    for y,x in imIter(im_out):
        im_out[y,x] = np.array([0,0,0], dtype=np.float64)
        for ky,kx in imIter(kernel):
            im_out[y,x] += pix(im, y+ky-offsety, x+kx-offsetx, False) * kernel[ky, kx]        
  
    return im_out

def gradientMagnitude(im):
    ''' Return the sum of the absolute value of the graident  
    The gradient is the filtered image by Sobel filter '''

    im_out = im.copy()

    Sobel=np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    G_x = convolve(im, Sobel)
    G_y = convolve(im, np.transpose(Sobel))
    im_out = np.sqrt( G_x**2 + G_y**2 )
    
    return im_out

def gaussCalc(x, sigma, truncate=3):
    a = 1/float(np.sqrt(2*np.pi*sigma**2))
    r = (x**2)/float(2*sigma**2)
    val = a*np.exp(-1.0*r)
    return val


def gauss2DCalc(y, x, sigma, truncate=3):
    a = 1/float(2*np.pi*sigma**2)
    r = ((x**2)+(y**2))/float(2*sigma**2)
    val = a*np.exp(-1.0*r)
    return val

def horiGaussKernel(sigma, truncate=3):
    '''Return an one d kernel'''
    L = int(2*sigma*truncate+1)
    output = np.zeros(L)
    
    #return some_array
    a = 1/float(np.sqrt(2*np.pi*sigma**2))

    offset = int(L/2)
    for r in range(L):
        x = ((r-offset)**2)/float(2*sigma**2)
        G = a*np.exp(-1.0*x)
        output[r] = G
    
    return np.array([output])

def gaussianBlur(im, sigma, truncate=3):

    #return gaussian_blured_image

    horizontal = horiGaussKernel(sigma, truncate)
    vertical = np.transpose(horizontal)
    G_h = convolve(im, horizontal)
    out = convolve(G_h, vertical)
    
    return out


def gauss2D(sigma=2, truncate=3):
    '''Return an 2-D array of gaussian kernel'''
    L = int(2*sigma*truncate+1)
    output = np.zeros((L,L))
    
    #return some_array
    a = 1/float(2*np.pi*sigma**2)

    offset = int(L/2)
    for x in range(L):
        for y in range(L):
            r = ((x-offset)**2 + (y-offset)**2)/float(2*sigma**2)
            G = a*np.exp(-1.0*r)
            output[y,x] = G
        
    return output

def unsharpenMask(im, sigma, truncate, strength):

    out = im + strength*(im - convolve(im, gauss2D(sigma, truncate)))
    return out

def bilateral(im, sigmaRange, sigmaDomain):

    out = im.copy()
    
    G_d = gauss2D(sigmaDomain)    
    D = G_d.shape[0]
    offsetD = int(D/2)

    for y,x in imIter(out):
        tot = np.array([0,0,0], dtype=np.float64)
        k = 0.0
        for dy in range(D):
            for dx in range(D):
                r = im[y,x][0] - pix(im, y+dy-offsetD, x+dx-offsetD, False)[0]
                g = im[y,x][1] - pix(im, y+dy-offsetD, x+dx-offsetD, False)[1]
                b = im[y,x][2] - pix(im, y+dy-offsetD, x+dx-offsetD, False)[2]
                imDist = np.sqrt(r**2 + g**2 + b**2)

                cVal = gaussCalc( imDist, sigmaRange, truncate=2)
                gVal = G_d[dy, dx]

                k += gVal*cVal
                tot += gVal*cVal*pix(im, y+dy-offsetD, x+dx-offsetD, False)
                
        tot *= 1/float(k)
                
        out[y,x] = tot   
        
    return out


def bilaYUV(im, sigmaRange, sigmaY, sigmaUV):
    '''6.865 only: filter YUV differently'''

    im2 = rgb2yuv(im)
    out = im.copy()
    
    G_uv = gauss2D(sigmaUV)
    U = G_uv.shape[1]
    offsetU = int(U/2)
    
    G_y = gauss2D(sigmaY)
    Y = G_y.shape[1]
    offsetY = int(Y/2)

    for k,x in imIter(im2):
        tot = np.array([0,0,0], dtype=np.float64)
        kY = 0.0
        kU = 0.0
        kV = 0.0
        for dy in range(U):
            for dx in range(U):
                y = im2[k,x][0] - pix(im2, k+dy-offsetU, x+dx-offsetU, False)[0]
                u = im2[k,x][1] - pix(im2, k+dy-offsetU, x+dx-offsetU, False)[1]
                v = im2[k,x][2] - pix(im2, k+dy-offsetU, x+dx-offsetU, False)[2]
                imDist = np.sqrt(y**2 + u**2 + v**2)
                cVal = gaussCalc( imDist, sigmaRange, truncate=2)

                gValU = G_uv[dy, dx]
                gValV = G_uv[dy, dx]
                gValY = pixNew(G_y, dy-offsetY, dx-offsetY, True)

                kY += gValY*cVal
                kU += gValU*cVal
                kV += gValV*cVal
                tot[0] += gValY*cVal*pix(im2, k+dy-offsetU, x+dx-offsetU, False)[0]
                tot[1] += gValU*cVal*pix(im2, k+dy-offsetU, x+dx-offsetU, False)[1]
                tot[2] += gValV*cVal*pix(im2, k+dy-offsetU, x+dx-offsetU, False)[2]
                
        tot[0] *= 1/float(kY)
        tot[1] *= 1/float(kU)
        tot[2] *= 1/float(kV)
 
        out[k,x] = tot   
        
    return yuv2rgb(out)


# Helpers

#From Pset2
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

#Variation of Edge Padding from Pset2
def pixNew(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns a pixel.
        If y,x is outside the image and repeatEdge==True , you should return the nearest pixel in the image
        If y,x is outside the image and repeatEdge==False , you should return a black pixel
    '''
    height = im.shape[0]
    width = im.shape[1]
   
    if repeatEdge:
       if (x<0) or (x>=width) or (y<0) or (y>=height):
          return 0.0
       else:
          return im[y,x]
    else:
       clipX = min(width-1, max(x,0))
       clipY = min(height-1, max(y,0))
       return im[clipY, clipX]

#From Pset1
def lumiChromi(im):
    #create lumi and chromi images. Remember to work on copies of im!
    imO = im.copy()
    imL = imO
    imL = BW(imL)
    imC = imO/imL
    return (imL, imC)

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
