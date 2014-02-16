# Halide tutorial lesson 8.

# This lesson illustrates convolution using reduction

import os, sys
from halide import *
import time
import numpy

import imageIO

def boxBlur(im, indexOfBlur=0):

    # we will compute brute-force box filters of size 5x5
    kernel_width=5
    input = Image(Float(32), im)

    #First declare the Func and Vars
    blur=Func('blur')  
    x, y, c = Var('x'), Var('y'), Var('c') 

    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1), c]


    if indexOfBlur==0:
        print 'Box blur as a simple reduction'
        # first version of our box blur. Simple but not efficient. 

        # Our reduction domain will cover the footprint of the kernel
        # It corresponds to the bounds of the inner loop you would write
        # in Python for the output of each pixel 
        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')                            

        # next we initialize the convolution sum at each pixel to zero
        blur[x,y,c] = 0.0

        # Finally, the update equation describes what happens at each reduction location
        # It will be called for each value in the reduction domain r.x, r.y and for each 
        # free variable triplet x, y, c
        # in this case, the reduction domain r.x and r.y get added to the free variables
        # x and y but it doesn't need to be the case in general for reductions. 
        blur[x,y,c] += clamped[x+r.x-kernel_width/2, 
                               y+r.y-kernel_width/2, 
                               c] / kernel_width**2

        # equivalent Python code
        if False: 
            out=numpy.empty([input.width(), input.height(), input.channels()])
            # initialization 
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        out[x,y,c]=0
            # Again, the reduction variable loops must be first
            for ry in xrange(kernel_width):
                for rx in xrange(kernel_width):
                    # then the free variable loops
                    for y in xrange(input.height()):
                        for x in xrange(input.width()):
                            for c in xrange(input.channels):
                                out[x,y,c]+=clampedInput[x+rx-kernel_width/2,
                                                         y+ry-kernel_width/2,
                                                         c]  / kernel_width**2

            # The reduction loops are always outside the free variable loops to 
            # ensure unambiguous semantic. 
            # This can be a problem for convolutions, because the above order 
            # has very poor locality. For each iteration of rx, ry, we need to access 
            # data all over the image. 
            # Unfortunately, Halide's .reorder command does not allow you to swap the
            # order of reduction and free variables because it could change the semantic
            # (for example if we use reductions over time to perform a simulation such 
            #    as a game of life. If we put the loop over x y outside that of time, 
            # the results would be very different)
            # Fortunately there is a trick that fixes things in cases like convolution
            # where the calculations at different free variable locations are independent. 

    if indexOfBlur==1:
        print 'Box blur with the helper/inline trick'

        # We now introduce the helper / inline trick to get a better loop order
        # We will add one pointiwise stage after the reduction. The reduction will get 
        # be scheduled with the default: inline. As a result, Halide will inline the 
        # whole reduction for each triplet x, y, c. We will see that it all simplifies 
        # into what we want: the outer loops will be on x,y, c and the reduction variable 
        # will, in practice, end up as inner loops

        # the first stage is the same reduction as above 
        # with the same reduction domain
        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')  
        # same initialization                          
        blur[x,y,c] = 0.0
        # same update equation
        blur[x,y,c] += clamped[x+r.x-kernel_width/2, y+r.y-kernel_width/2, c] / kernel_width**2

        # But we now add an extra stage that consumes the output of our reduction
        superBlur = Func('superBlur')
        superBlur[x,y,c]=blur[x,y,c]

        blur=superBlur # just so that the code below use the last stage

        # That's it. Simply adding this stage and letting it scheduled as inline (the default)
        # wil trick Halide into putting free variable loops outside of teh reduction loops. 
        # Let's see why. 
        # As usual, everything starts with the consumer. Halide schedules superBlur with loops
        # for the free variables x, y, c.
        # Then it inlines blur inside the innermost loop of superBlur. At this point, x, y, c 
        # are known and have a single value each. Inlining means that we dump the code needed to
        # compute the producer for all the values needed by the consumer. Here we just need value(s) 
        # for a single triplet x,y,c.  Halide generates the blur code for domain that covers only 
        # this triplet. In a sense, from the perspective of blur, teh reduction variables are still 
        # the outer loop and there are inner loops for x, y, and c but they are restricted to a single 
        # iteration each. Let's loop at teh equivalent Python code to make this clearer

        #equivalent Python code before simplification
        if False: 
            superBlur=numpy.empty([input.width(), input.height(), input.channels()])
            # loops for superBlur
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        # Now we inline blur
                        # by now, x, y and c are fixed
                        # We need to produce the blur values needed by superBlur[x,y,c]
                        # that is, we need to produce a single value of blur for fixe x,y,c
                        # essentially, the width and height and number of channels we need are just 1

                        # In this version of the code we'll be verbose and keep these 1-iteration loops
                        # We allocate a buffer or the required size (here 1*1*1)
                        tmp=numpy.empty([1,1,1])
                        # we perform the initialization
                        for yi in xrange(1):
                            for xi in xrange(1):
                                for ci in xrange(1):
                                    tmp[xi,yi,ci]=0
                        # Then we write the same set of 5 nested loops as before
                        # We start with the reduction variables. 
                        # From the perspective of blur, they are always the outer loops
                        for ry in xrange(kernel_width):
                            for rx in xrange(kernel_width):
                                # Then we write the inner loops over the required free-variable domain
                                # In our case, we just need one single value and the loops have size 1
                                for yi in xrange(1):
                                    for xi in xrange(1):
                                        for ci in xrange(1):
                                            tmp[xi,yi,ci]+=clampedInput[x+xi+rx-kernel_width/2,
                                                                        y+yi+ry-kernel_width/2,
                                                                        c+ci]  / kernel_width**2
                                            #where xi=0, yi=0, ci=0
                        superBlur[x,y,c]=tmp[0,0,0]


       #equivalent Python code with 1-iteration loops removed
       # We have succesfully achieved the order we wanted

        if False: 
            superBlur=numpy.empty([input.width(), input.height(), input.channels()])
            #loops for superBlur
            for y in xrange(input.height()):
                for x in xrange(input.width()):
                    for c in xrange(input.channels):
                        tmp=0
                        for ry in xrange(kernel_width):
                            for rx in xrange(kernel_width):
                                tmp+=clampedInput[x+rx-kernel_width/2,
                                                  y+ry-kernel_width/2,
                                                  c]  / kernel_width**2
                        superBlur[x,y,c]=tmp




    if indexOfBlur==2:
        print 'Box blur with the *sum* sugar'

        # The helper/inline trick and convolutions are so common that there is a sugar called 'sum'
        # sum takes care of teh initialization (to 0), the extra inilined stage and the update. 
        # The syntax then becomes
        
        # define the reduction domain as before
        r = RDom(0,    kernel_width, 0,    kernel_width, 'r')                            

        #define the Func directly as a sum over Expr that involve the reduction domain
        blur[x,y,c] = sum(clamped[x+r.x-kernel_width/2, 
                                  y+r.y-kernel_width/2, 
                                  c])

        # the equivalent Pyhton code is the same as above. 
        # This formulation is both concise and efficient. 


    # The speed difference between these versions is not huge in the absence of parallelism. 
    # Modern machines offer a bandwidth taht is often enough for sequential code. 

    blur.compile_jit()
    t=time.time()
    numTimes=5
    for i in xrange(numTimes):
        output = blur.realize(input.width(), input.height(), input.channels())
    dt=time.time()-t
    print '           took ', dt/numTimes, 'seconds\n'

    return output, dt

def main():    
    im=imageIO.imread('hk.png')
    #print 'loading input file'
    #im=numpy.load('Input/hk.npy')
    #print '    done loading input file, size: ', im.shape

    output=None
    for i in xrange(3):
        output, dt=boxBlur(im, i)
        
    #outputNP=numpy.array(Image(output))
    #imageIO.imwrite(outputNP)

    numpy.save('Input/hk.npy', im)

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises

# Write a Separable Gaussian blur in Halide. 

