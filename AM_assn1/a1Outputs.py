import a1 as a1
from imageIO import *
import numpy as np
import scipy as sp

#Original image
im = imread('in.png')
imwrite(im, 'out.png')

#Brightness
out1 = a1.brightness(im, 1.5)
imwrite(out1, 'out1.png')

#Contrast
out2 = a1.contrast(im, 0.5, 0.6)
imwrite(out2, 'out2.png')

#Frame
out3 = a1.frame(im)
imwrite(out3, 'out3.png')

#Black and White
out4 = a1.BW(im, weights=[0.3,0.6,0.1])
imwrite(out4, 'out4.png')

#Luminance & Chrominance
out5_1 = a1.lumiChromi(im)[0]
imwrite(out5_1, 'out5_1.png')
out5_2 = a1.lumiChromi(im)[1]
imwrite(out5_2, 'out5_2.png')

#Brighness Contrast Luminance
out6 = a1.brightnessContrastLumi(im, 0.6, 1.5, 0.3)
imwrite(out6, 'out6.png')

#Saturate
out7 = a1.saturate(im, 2)
imwrite(out7, 'out7.png')

#Spanish Castle
out8_1a = a1.spanish(im)[0]
out8_1b = a1.spanish(im)[1]
imwrite(out8_1a, 'out8_1a.png')
imwrite(out8_1b, 'out8_1b.png')

imCastle = imread('castle_small.png')
out8_2a = a1.spanish(imCastle)[0]
out8_2b = a1.spanish(imCastle)[1]
imwrite(out8_2a, 'out8_2a.png')
imwrite(out8_2b, 'out8_2b.png')
