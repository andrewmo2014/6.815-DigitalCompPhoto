#assignment 4 starter code
#by Abe Davis
#
# Student Name: Andrew Moran
# MIT Email: andrewmo@mit.edu

import numpy as np
import scipy as sp
import imageIO as io
import math

def denoiseSeq(imageList):
    '''Takes a list of images, returns a denoised image
    '''
    height = imageList[0].shape[0]
    width = imageList[0].shape[1]
    mean = imageList[0].copy()

    for i in range(1, len(imageList)):
        mean += imageList[i]
    mean /= float(len(imageList))
    return mean


def logSNR(imageList, scale=1.0/20.0):
    '''takes a list of images and a scale. Returns an image showing log10(snr)*scale'''
    mean = denoiseSeq(imageList)
    meanSquared = np.square(mean)
    
    height = mean.shape[0]
    width = mean.shape[1]
    variance = np.zeros( imageList[0].shape ) 

    for i in range(len(imageList)):
        diff = np.clip(imageList[i] - mean, .00001, np.inf)
        variance += diff*diff
    variance /= float(len(imageList)-1)

    snr = meanSquared/variance
    return np.log10(snr)*scale


def align(im1, im2, maxOffset=20):
    '''takes two images and a maxOffset. Returns the y, x offset that best aligns im2 to im1.'''

    width = im1.shape[1]
    height = im1.shape[0]

    error = np.inf
    y = 0
    x = 0

    for dy in range(-maxOffset, maxOffset+1):
        for dx in range(-maxOffset, maxOffset+1):
            temp = np.roll(im2,dx,axis=1)
            temp = np.roll(temp,dy,axis=0)
            e = np.linalg.norm(temp[maxOffset:-maxOffset,maxOffset:-maxOffset] - im1[maxOffset:-maxOffset,maxOffset:-maxOffset])**2
            if( e <= error ):
                error = e
                y = dy
                x = dx

    return y, x


def alignAndDenoise(imageList, maxOffset=20):
    '''takes a list of images and a max offset. Aligns all of the images to the first image in the list, and averages to denoise. Returns the denoised image.'''
    im1 = imageList[0]
    alignList = list()
    alignList.append( im1.copy() )
    for i in range(1, len(imageList)):
        y,x = align(im1, imageList[i], maxOffset)
        temp = np.roll(imageList[i], x, axis=1)
        im2 = np.roll(temp, y, axis=0)
        alignList.append(im2)
    return denoiseSeq(alignList)
    

def basicGreen(raw, offset=1):
    '''takes a raw image and an offset. Returns the interpolated green channel of your image using the basic technique.'''
    out=raw.copy()
    for y,x in imIter(raw):
        if( y>0 and y<raw.shape[0]-1 and x>0 and x<raw.shape[1]-1):
            if( (x+y+offset)%2 == 1 ):
                out[y,x] = .25*(raw[y-1,x] + raw[y,x-1] + raw[y,x+1] + raw[y+1,x])
    return out
    

def basicRorB(raw, offsetY, offsetX):
    '''takes a raw image and an offset in x and y. Returns the interpolated red or blue channel of your image using the basic technique.'''
    out =raw.copy()

    for y,x in imIter(raw):
        if( y>0 and y<raw.shape[0]-1 and x>0 and x<raw.shape[1]-1):

            if((y+offsetY)%2 == 0):
                if((x+offsetX)%2 == 1):
                    out[y,x] = .5*(raw[y,x-1] + raw[y,x+1])
            else:
                if((x+offsetX)%2 == 0):
                    out[y,x] = .5*(raw[y-1,x] + raw[y+1,x])
                else:
                    out[y,x] = .25*(raw[y-1,x-1] + raw[y+1,x-1] + raw[y-1,x+1] + raw[y+1,x+1])
    return out


def basicDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''takes a raw image and a bunch of offsets. Returns an rgb image computed with our basic techniche.'''

    height = raw.shape[0]
    width = raw.shape[1]
    out = io.constantIm(height, width, [0,0,0])

    out[:,:,0] = basicRorB(raw, offsetRedY, offsetRedX)
    out[:,:,1] = basicGreen(raw) #use default offsetGreen
    out[:,:,2] = basicRorB(raw, offsetBlueY, offsetBlueX)

    return out
    

def edgeBasedGreenDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''same as basicDemosaic except it uses the edge based technique to produce the green channel.'''

    height = raw.shape[0]
    width = raw.shape[1]
    out = io.constantIm(height, width, [0,0,0])

    out[:,:,0] = basicRorB(raw, offsetRedY, offsetRedX)
    out[:,:,1] = edgeBasedGreen(raw) #use default offsetGreen
    out[:,:,2] = basicRorB(raw, offsetBlueY, offsetBlueX)

    return out


def edgeBasedGreen(raw, offset=1):
    '''same as basicGreen, but uses the edge based technique.'''

    out=raw.copy()
    for y,x in imIter(raw):
        if( y>0 and y<raw.shape[0]-1 and x>0 and x<raw.shape[1]-1):
            if( (x+y+offset)%2 == 1 ):
                vertical = abs( raw[y-1,x] - raw[y+1,x] )
                horizontal = abs( raw[y,x-1] - raw[y,x+1] )

                if(vertical < horizontal):
                    out[y,x] = 0.5*(raw[y-1,x] + raw[y+1,x])
                elif(horizontal < vertical):
                    out[y,x] = 0.5*(raw[y,x-1] + raw[y,x+1])
                else:
                    out[y,x] = .25*(raw[y-1,x] + raw[y,x-1] + raw[y,x+1] + raw[y+1,x])
    return out


def greenBasedRorB(raw, green, offsetY, offsetX):
    '''Same as basicRorB but also takes an interpolated green channel and uses this channel to implement the green based technique.'''
    
    out =raw.copy()
    for y,x in imIter(raw):
        if( y>0 and y<raw.shape[0]-1 and x>0 and x<raw.shape[1]-1):

            if((y+offsetY)%2 == 0):
                if((x+offsetX)%2 == 1):
                    out[y,x] = green[y,x] + .5*(raw[y,x-1]-green[y,x-1]) + .5*(raw[y,x+1]-green[y,x+1])
            else:
                if((x+offsetX)%2 == 0):
                    out[y,x] = green[y,x] + .5*(raw[y-1,x]-green[y-1,x]) + .5*(raw[y+1,x]-green[y+1,x])
                else:
                    out[y,x] = green[y,x] + .25*(raw[y-1,x-1]-green[y-1,x-1]) + .25*(raw[y+1,x-1]-green[y+1,x-1]) + .25*(raw[y-1,x+1]-green[y-1,x+1]) + .25*(raw[y+1,x+1]-green[y+1,x+1])
    return out


def improvedDemosaic(raw, offsetGreen=0, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    '''Same as basicDemosaic but uses edgeBasedGreen and greenBasedRorB.'''

    height = raw.shape[0]
    width = raw.shape[1]
    out = io.constantIm(height, width, [0,0,0])

    green = edgeBasedGreen(raw) #use default offsetGreen
    out[:,:,0] = greenBasedRorB(raw, green, offsetRedY, offsetRedX)
    out[:,:,1] = green
    out[:,:,2] = greenBasedRorB(raw, green, offsetBlueY, offsetBlueX)

    return out


def split(raw):
    '''splits one of Sergei's images into a 3-channel image with height that is floor(height_of_raw/3.0). Returns the 3-channel image.'''

    width = raw.shape[1]
    height = raw.shape[0]
    crop = math.floor(height/3.0)

    out = io.constantIm(crop, width, [0,0,0])

    out[:,:,0] = raw[2*crop:3*crop,:]
    out[:,:,1] = raw[crop:2*crop,:]
    out[:,:,2] = raw[:crop,:]

    return out


def sergeiRGB(raw, alignTo=1):
    '''Splits the raw image, then aligns two of the channels to the third. Returns the aligned color image.'''

    mid = split(raw)
    out = mid

    r = mid[:,:,0]
    g = mid[:,:,1]
    b = mid[:,:,2]

    channels = [r,g,b]
    last = channels.pop(alignTo)
    
    y1,x1 = align(last, channels[0])
    temp1 = np.roll(channels[0], x1, axis=1)
    ch1 = np.roll(temp1, y1, axis=0)
    
    y2,x2 = align(last, channels[1])
    temp2 = np.roll(channels[1], x2, axis=1)
    ch2 = np.roll(temp2, y2, axis=0)

    newIm = [ch1, ch2]
    newIm.insert(alignTo, last)

    out[:,:,0] = newIm[0]
    out[:,:,1] = newIm[1]
    out[:,:,2] = newIm[2]

    return out


def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)

