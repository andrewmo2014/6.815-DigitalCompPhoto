# Halide tutorial lesson 8.

# This lesson illustrates reductions

# reductions are also known as fold, accumulate or aggregate

# We will compute the average RGB value of an image
# that is, we will return a 1D array with only 3 values

# One important thing to remember from this lesson is that there is never 
# any explicit 'for' loop in Halide, whether is't the loop over output pixels
# or the reduction loop over the input pixels for the reduction. It's all implicit

import os, sys
from halide import *

import imageIO

def main():
    
    # As usual, let's load an input
    im=imageIO.imread('rgb.png')    
    # and create a Halide representation of this image
    input = Image(Float(32), im)

    # Next we declaure the Vars and Func
    x, y, c = Var(), Var(), Var() 
    mySum = Func()

    # The central tool to express a reduction is a reduction domain, called RDom
    # it corresponds to the bounds of the reduction loop you would write in an 
    # imperative language. Here we want to iterate over a 2D domain corresponding 
    # to the whole image. 
    # Note however that we have decided to compute a different sum for each channel, 
    # So we will not reduce over channels. 
    r = RDom(0,    input.width(), 0,    input.height())                
    # careful, RDom are defined as base, extent, not min, max.
    # This means that RDom(a, b) goes from a to a+b (obviously not an issue here)

    # Given a reduction domain, we define the Expr that we will sum over, in this
    # case the pixel values. By construction, the first and second dimension of a 
    # reduction domain are called x and y. In this case they happen to correspond 
    # to the image x and y coordinates but they don't have to. 
    # Note that x & y are the reduction variables but c is a normal Var.
    # this is because our sum is over x,y but not over c. There will be a different 
    # sum for each channel. 
    val=input[r.x, r.y, c]

    # A reduction Func first needs to be initialized. Here, our sum gets initialized to 0
    # Note that the function domain here is only the channel. 
    mySum[c]=0.0

    # Finally, we define what the reduction should do for each reduction value. 
    # In this case, we eant to add each reduction value to the output
    # This is called the update function, and it's going to be called for each 
    # location in the RDom. 
    # You never write an explicit loop over the RDom, Halide does it for you. 
    mySum[c] +=val

    # We now call realize() to compile and execute. 
    output = mySum.realize(input.channels());

    outputNP=numpy.array(Image(output))
    print outputNP

    # equivalent Python code

    out = numpy.empty((3));
    # first loop to initialize the reduction
    for c in xrange(input.channels()):
        out[c]=0.0
    # VERY IMPORTANT : 
    # Note that loops for the reduction variables are outermost. 
    # whereas c in this case is innermost. 

    #Reduction loops
    for ry in xrange(0, input.height()):
        for rx in xrange(0, input.width()):

            # free variable loop
            for c in xrange(input.channels()):
                #update function
                print ry, rx, c
                out[c] += input[rx, ry, c]


#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

