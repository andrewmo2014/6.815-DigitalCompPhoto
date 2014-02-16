# Halide tutorial lesson 10.

# This lesson illustrates the scheduling of a two-stage separable convolution 
# It is similar to tutorial 6 but uses reductions instead of a direct expression 
# of the sum 

# It also illustartes a slightly different use of reductions, 
# since each stage is only a 1D sum. 

import os, sys
from halide import *
import time
import numpy

import imageIO

def boxBlur(im, indexOfSchedule, tileX=128, tileY=128):

    # we will compute a 5x5 box blur
    kernel_width=5

    input = Image(Float(32), im)

    # Declaration of Vars and Funcs
    x, y, c = Var('x'), Var('y'), Var('c') #declare domain variables
    blur_x = Func('blur_x') 
    blur_y = Func('blur_y')
    blur   = Func('blur')

    # declare and define a clamping function that restricts x,y to the image size
    # boring but necessary
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                          clamp(y, 0, input.height()-1), c]

    # Since we're using separable convolution, each stage is a 1D sum. 
    # First, we'll define the horizontal blur. 

    # Note how the reduction domain below is 1D
    rx = RDom(0,    kernel_width, 'rx')                            
    # the update equation only sums over rx.x values
    # We use the sum shorthand to make sure that the reduction 
    # will have a reasonable loop ordering. 
    blur_x[x,y,c] = sum(clamped[x+rx.x-kernel_width/2, y, c]) 

    # Next we define the vertical blur stage. 
    # Here too, we use a 1D reduction domain. 
    ry = RDom(0,    kernel_width, 'ry')                

    # the first coordinate of a reduction domain is always called x. 
    # This might be a little ocnfusing here, because we want to use the 1D 
    # reduction domain for a sumamtion over y. 
    # Not the ry.x below in the y coordinate calculation. 
    blur_y[x,y,c] = sum(blur_x[x, y+ry.x-kernel_width/2, c])

    # Finally we normalize    
    blur[x,y,c] = blur_y[x,y,c]/(kernel_width**2)

    if indexOfSchedule==0: 
        print '\n ', 'default schedule'
        # our first schedule is the default schedule. 
        # Everything gets inlined. 
        # This reduces to a brute force 2D non-spearable blur. 

    if indexOfSchedule==1: 
        print '\n', 'root first stage'
        # To get a more reasonable schedule for non-tiny filter sizes and 
        # avoid redudant calculation, we schedule as root the stages that are 
        # consumed by a stencil consumer, i.e. a consumer that will use each 
        # produced value multiple times. 
        # In our case, this means scheduling teh first stage as root. 
        # We will compute the values of blur_x for the whole image before moving on 
        # to the next stage
        blur_x.compute_root()

    if indexOfSchedule==2: 
        print '\n', 'tile ', tileX,'x', tileY, ' + interleave'
        # We next seek to strike a balance between the lack of redundancy 
        # of root and the locality of inline. For this, we schedule the last stage as tile
        # and compute a tile of teh first stage as needed just before computing 
        # the corresponding tiel of teh consumer. If the tiles are small enough, 
        # they will stay in cache. 

        # delcare the extra Vars needed for tiles
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        # schedule teh final stage as tiled
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY)
        # tile is a shorthand for split+reorder
        # it will generate teh following nested loops:
        # for yo:
        #    for xo:
        #        for yi:
        #            for xi:

        # specify that the prducer stage should be scheduled at the xo granularity 
        # of the consumer. It will produce a tile of blur_x before the cmputation 
        # of a tile of the final blur

        blur_x.compute_at(blur, xo)

        # On my machine, this schedule by itself doesn't yield a huge speedup. 
        # This is because we have optimized locality and the bottleneck is now
        # computation. We need parallelism to do better 

    if indexOfSchedule==3: 
        print '\n', 'tile ', tileX,'x', tileY, '+ parallel'
        # Same tiled and interleaved schedule, but computed in parallel over rows of tiles
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        # simply add the .parallel command
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        # blur_x 'scheduling commands are unchanged, but it inherits blur's parallelism 
        # because it is computed at the level of blur's xo, which is inside the yo
        # parallel loop
        blur_x.compute_at(blur, xo)

    if indexOfSchedule==4: 
        print '\n', 'tile ', tileX,'x', tileY, ' + parallel+vector without interleaving'

        # FInally we check that parallelism without locality is not enough. 
        # We still perform computation in tiles but do not interleave blur_x and blur 
        # anymore. This means that we compute all the tiles of blur_x before computing 
        # any final tile. 
        # Tiling is still useful to extract parallelism and also it allows us to keep 
        # everything but interleaving equal compared to schedule 3. 
        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')
        blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        # we now need to specify blur_x's tiling and paralelism because it is
        # not schedule at the granularity of blur's tiles anymore. 
        blur_x.compute_root().tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)


    blur.compile_jit()
    t=time.time()
    numTimes=5
    for i in xrange(numTimes):
        output = blur.realize(input.width(), input.height(), input.channels())
    dt=time.time()-t
    print '           took ', dt/numTimes, 'seconds'

    return output, dt

def main():    
    #im=imageIO.imread('hk.png')
    path='Input/hk.npy'
    print 'loading file ', path
    im=numpy.load(path)
    print '         done. size ', im.shape

    output=None

    #first explore the different schedules
    for i in xrange(5):
        output, dt=boxBlur(im, i, 256, 256)

    # then explore tile sizes
    for tile in [64, 128, 256, 512, 1024]: 
        output, dt=boxBlur(im, 3,  tile)
    
    #outputNP=numpy.array(Image(output))
    #imageIO.imwrite(outputNP)
    #numpy.save('Input/hk.npy', im)

#usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()

# exercises

# For your machine, report the timings for the 35MPixel hk image
# What is the best tile size for you ? 

