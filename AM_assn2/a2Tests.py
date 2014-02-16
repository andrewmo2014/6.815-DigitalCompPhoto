#a2.test
from a2 import *
import imageIO as io
import numpy as np
import math

#Check Signature
print check_my_module()

imIn = io.imread('panda.png')

#Scale with nearest neighbor
imOut = scaleNN( imIn, 3.5 )
io.imwrite( imOut, 'pandaScaledNN.png')

imIn2 = io.imread('panda2.png')

#Scale with Bilinear Interpolation
imOut1 = scaleLin( imIn2, 3.5 )
io.imwrite( imOut1, 'pandaScaledLin.png')

#Scale with Bicubic Interpolation
imOut2 = scaleBicubic( imIn2, 3.5 )
io.imwrite( imOut2, 'pandaScaledBicubic.png')

#Scale with Biquadratic Interpolation
imOut2_1 = scaleBiquadratic( imIn2, 3.5 )
io.imwrite( imOut2_1, 'pandaScaledBiquad.png')

#Rotate
imOut3 = rotate(imIn, math.pi/4)
io.imwrite( imOut3, 'pandaRot45.png')

#Warp single segment
imIn3 = io.imread('bear.png')
imOut4 = warpBy1( imIn3, segment(0,0,10,0), segment(10,10,30,15))
io.imwrite( imOut4, 'bearWarped.png')

#Morphing
im1 = io.imread('fredo2.png')
im2 = io.imread('werewolf.png')
segmentsBefore=np.array([segment(83, 131, 112, 129), segment(143, 126, 163, 131), segment(103, 200, 127, 189), segment(121, 221, 134, 201), segment(79, 251, 45, 207), segment(126, 249, 164, 171), segment(8, 54, 49, 19), segment(173, 67, 147, 35)])
segmentsAfter=np.array([segment(84, 113, 105, 109), segment(137, 105, 157, 103), segment(102, 172, 133, 152), segment(129, 197, 145, 177), segment(122, 222, 82, 190), segment(137, 214, 160, 136), segment(14, 51, 35, 15), segment(171, 41, 146, 12)])

imageList = morph(im1, im2, segmentsBefore, segmentsAfter, N=3)
for i in range(len(imageList)):
    k=i+1
    io.imwrite( imageList[i], 'morph' + str('%03d' %k) + '.png')
