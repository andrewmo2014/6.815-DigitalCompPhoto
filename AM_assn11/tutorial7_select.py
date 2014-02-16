# Halide tutorial lesson 7.

# This (trivial) lesson demonstrates how to use conditionals in Halide. 
# We will write an image thresholding filter that makes everything under 
# 0.5 black and everything above white

import os, sys
from halide import *

import imageIO

def main():
 
    im=imageIO.imread('rgb.png')

    input = Image(Float(32), im)

    thresholding=Func()
    x, y, c = Var(), Var(), Var()

    # We will threshold based on the green channel. 
    # select(predicate, value1, value2) returns value1 if predicate 
    # is true, and value2 otherwise
    thresholding[x, y, c] = select(input[x,y,1]<0.5, 0.0, 1.0)


    output = thresholding.realize(input.width(), input.height(), input.channels());

    outputNP=numpy.array(Image(output))
    imageIO.imwrite(outputNP, 'threshold.png')
    
    print "Success!\n"
    return 0;




if __name__ == '__main__':
    main()

# Exercise

# write a Halide Func that takes a 1-channel image as input and tests if 
# each pixel is a local maximum. If it is bigger than its 4 neighbors, 
# set the output to 1.0, and 0.0 otherwise. 
# Hint: the logical keyword 'and' works in Halide as well

# Yes, it's not the best local maximum detection and bad things could happen in the diagonal. Oh well. 