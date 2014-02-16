

import os, sys
from halide import *
# The only Halide module  you need is halide. It includes all of Halide


def smoothGradientNormalized():
    '''use Halide to compute a 512x512 smooth gradient equal to x+y divided by 1024
    Do not worry about the schedule. 
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    gradient = Func()
    x, y = Var(), Var()
    gradient[x,y] = cast(Float(32), x+y)
    output = gradient.realize(512, 512)
    outputNP = numpy.array(Image(output))/1024

    return (outputNP, gradient)


def wavyRGB():
    '''Use a Halide Func to compute a wavy RGB image like that obtained by the following 
    Python formula below. output[y, x, c]=(1-c)*cos(x)*cos(y)
    Do not worry about the schedule. 
    Hint : you need one more domain dimension than above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    gradient = Func()
    x, y, z = Var(), Var(), Var()
    gradient[x,y,z] = cast(Float(32), (1-z)*cos(x)*cos(y))
    output = gradient.realize(400, 400, 3)
    outputNP = numpy.array(Image(output))

    return (outputNP, gradient)


def luminance(im):
    '''input is assumed to be our usual numpy image representation with 3 channels. 
    Use Halide to compute a 1-channel image representing 0.3R+0.6G+0.1B
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''
    input = Image(Float(32), im)
    lumi = Func()
    x, y = Var(), Var()
    value = 0.3*input[x, y, 0]+0.6*input[x, y, 1]+0.1*input[x, y, 2]
    lumi[x, y] = value
    output = lumi.realize(input.width(), input.height())
    outputNP = numpy.array(Image(output))

    return (outputNP, lumi)


def  sobel(lumi):
    ''' lumi is assumed to be a 1-channel numpy array. 
    Use Halide to apply a SObel filter and return the gradient magnitude. 
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func''' 
    input = Image(Float(32), lumi)
    x, y= Var('x'), Var('y')
    
    gx = Func('gx') 
    gy = Func('gy') 
    gradientMagnitude=Func('gradientMagnitude') 

    clamped = Func('clamped') 
    clamped[x, y] = input[clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)]

    gx[x,y]= (-clamped[x-1,y-1]+clamped[x+1,y-1]
                -2*clamped[x-1,y]+2*clamped[x+1,y]
                -clamped[x-1,y+1]+clamped[x+1,y+1])/4.0

    gy[x,y]= (-clamped[x-1,y-1]+clamped[x-1,y+1]
                -2*clamped[x,y-1]+2*clamped[x,y+1]
                -clamped[x+1,y-1]+clamped[x+1,y+1])/4.0

    gradientMagnitude[x,y]= sqrt(gx[x,y]**2 + gy[x,y]**2)
    output = gradientMagnitude.realize(input.width(), input.height());
    outputNP=numpy.array(Image(output))

    return (outputNP, gradientMagnitude)


def pythonCodeForBoxSchedule5(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. 
        e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
        
        # schedule 5:
        # blur_y.compute_root() 
        # blur_x.compute_at(blur_y, x)

        listedOrder = []

        inputP = numpy.transpose(lumi)
        width, height = inputP.shape[0]-2, inputP.shape[1]-2
        blur_y=numpy.empty((width, height))
        blur_x=numpy.empty((width, height+2))

        for y in xrange(height):
            for x in xrange(width):
                blur_x[x,y]=(inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3

                listedOrder.append( ('blur_x', x, y) )

            #compute blur_y
            for x in xrange(width):
                blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3

                listedOrder.append( ('blur_y', x, y) )

        return listedOrder


def pythonCodeForBoxSchedule6(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. 
        e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
        
        # schedule 6:
        # blur_y.tile(x, y, xo, yo, xi, yi, 2, 2)
        # blur_x.compute_at(blur_y, yo)

        listedOrder = []

        inputP = numpy.transpose(lumi)
        width, height = inputP.shape[0]-2, inputP.shape[1]-2
        blur_y=numpy.empty((width, height))
        for xo in xrange((width+1)/2): 
            for yo in xrange((height+1)/2):
                # first compute blur_x
                # allocate a temporary buffer
                blur_x=numpy.empty((2, 2+2)) 
                for xi in xrange(2):
                    x=xo*2+xi
                    if x>=width: x=width-1
                    for yi in xrange(2+2):
                        y=yo*2+yi
                        if y>=height: y=height-1
                        blur_x[xi,yi]=(inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3

                        listedOrder.append( ('blur_x', x, y) )

                #compute blur_y
                for xi in xrange(2):
                    x=xo*2+xi
                    if x>=width: x=width-1
                    for yi in xrange(2):
                        y=yo*2+yi
                        if y>=height: y=height-1
                        blur_y[x,y] = (blur_x[xi,yi]+blur_x[xi,yi+1]+blur_x[xi,yi+2])/3

                        listedOrder.append( ('blur_y', x, y) )

        return listedOrder


def pythonCodeForBoxSchedule7(lumi):    
        ''' lumi is assumed to be a 1-channel numpy array. 
        Write the python nested loops corresponding to the 3x3 box schedule 5
        and return a list representing the order of evaluation. 
        Each time you perform a computation of blur_x or blur_y, put a triplet with the name 
        of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y. 
        e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''
        
        # schedule 7
        # blur_y.split(x, xo, xi, 2)
        # blur_x.compute_at(blur_y, y)

        listedOrder = []

        inputP = numpy.transpose(lumi)
        width, height = lumi.shape[0]-2, lumi.shape[1]-2
        blur_y=numpy.empty((width, height))
        blur_x=numpy.empty((width, height+2))

        for x in xrange(width):
            for y in xrange(height):
                blur_x[x,y]=(inputP[x,y]+inputP[x+1,y]+inputP[x+2,y])/3

                listedOrder.append( ('blur_x', x, y) )

        for y in xrange(height):
            for xo in xrange((width+1)/2):
                for xi in xrange((width+1)/2):
                    x = xo*2 + xi
                    if x>=width: x=width-1
                    blur_y[x,y] = (blur_x[x,y]+blur_x[x,y+1]+blur_x[x,y+2])/3

                    listedOrder.append( ('blur_y', x, y) )

        return listedOrder


########### PART 2 ##################

def localMax(lumi):
    ''' the input is assumed to be a 1-channel image
    for each pixel, return 1.0 if it's a local maximum and 0.0 otherwise
    Don't forget to handle pixels at the boundary.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    input = Image(Float(32), lumi)
    localMaximum = Func('localMaximum')
    x, y = Var('x'), Var('y')

    clamped = Func('clamped') 
    clamped[x, y] = input[clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1)]

    localMaximum[x,y] = select(((clamped[x+1, y] < clamped[x,y]) &
                                (clamped[x-1, y] < clamped[x,y]) &
                                (clamped[x, y+1] < clamped[x,y]) &
                                (clamped[x, y-1] < clamped[x,y])), 1.0, 0.0 )

    output = localMaximum.realize(input.width(), input.height())
    outputNP=numpy.array(Image(output))

    return (outputNP, localMaximum)


def GaussianSingleChannel(input, sigma, trunc=3):
    '''takes a single-channel image or Func IN HALIDE FORMAT as input 
        and returns a Gaussian blurred Func with standard 
        deviation sigma, truncated at trunc*sigma on both sides
        return two Funcs corresponding to the two stages blurX, blurY. This will be
        useful later for scheduling. 
        We advise you use the sum() sugar
        We also advise that you first generate the kernel as a Halide Func
        You can assume that input is a clamped image and you don't need to worry about
        boundary conditions here. See calling example in test file. '''

    horiGaussKernel = Func('horiGaussKernel')
    blur_x = Func('blur_x')
    blur_y = Func('blur_y')
    blur = Func('blur')
    x, y = Var('x'), Var('y')

    #horiGaussKernel
    length = 2*sigma*trunc+1
    center = sigma*trunc
    a = 1.0/sqrt(2*numpy.pi*sigma**2)
    horiGaussKernel[x] = a*exp( -1.0*(((x-center)**2)/(2*sigma**2)) )

    #convolution along x
    rx = RDom(0, length, 'rx')
    blur_x[x,y] = sum(horiGaussKernel[rx.x]*input[x+rx.x-length/2, y]) 
    #convolution along y
    ry = RDom(0, length, 'ry')                
    blur_y[x,y] = sum(horiGaussKernel[ry.x]*blur_x[x, y+ry.x-length/2])
    #final blur sum 
    blur[x,y] = blur_y[x,y]/(length**2)    

    return (blur_x, blur)    


def harris(im, scheduleIndex):
    ''' im is a numpy RGB array. 
    return the location of Harris corners like the reference Python code, but computed
    using Halide. 
    when scheduleIndex is zero, just schedule all the producers of non-local consumers as root.
    when scheduleIndex is 1, use a smart schedule that makes use of parallelism and 
    has decent locality (tiles are often a good option). Do not worry about vectorization. 
    Note that the local maximum criterion is simplified compared to our original Harris
    You might want to reuse or copy-paste some of the code you wrote above        
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    #Read input
    input = Image(Float(32), im)

    #Constants
    Sobel= Image(Float(32), numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]))
    SobelT= Image(Float(32), numpy.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]))

    sigma = 1
    factor = 4
    trunc = 3
    k = 0.15
    thr = 0.0

    #Declare
    x,y,c = Var('x'), Var('y'), Var('c')

    #clamp
    clamped = Func('clamped') 
    clamped[x, y, c] = input[clamp(x, 0, input.width()-1),
                       clamp(y, 0, input.height()-1), c]
    
    #Compute Tensor
    BW = Func('BW')
    BW[x, y] = 0.3*clamped[x, y, 0]+0.6*clamped[x, y, 1]+0.1*clamped[x, y, 2]
    L1, L = GaussianSingleChannel(BW, sigma, trunc)

    gx = Func('gx')
    gy = Func('gy')

    #Sobel
    rs = RDom(0, Sobel.width(), 0, Sobel.height(), 'rs')
    gx[x,y] = sum(Sobel[rs.x, rs.y]*L[x+rs.x-Sobel.width()/2, y+rs.y-Sobel.height()/2])
    gy[x,y] = sum(SobelT[rs.x, rs.y]*L[x+rs.x-Sobel.width()/2, y+rs.y-Sobel.height()/2])

    gxx = Func('gxx')
    gyy = Func('gyy')
    gxy = Func('gxy')
    gxx[x,y] = gx[x,y]**2
    gyy[x,y] = gy[x,y]**2
    gxy[x,y] = gx[x,y]*gy[x,y]

    gxx1, gxx = GaussianSingleChannel(gxx, sigma*factor)
    gyy1, gyy = GaussianSingleChannel(gyy, sigma*factor)
    gxy1, gxy = GaussianSingleChannel(gxy, sigma*factor)

    #Corners
    det = Func('det')
    det[x,y] = gxx[x,y]*gyy[x,y]-gxy[x,y]**2  
    trace = Func('trace')
    trace[x,y] = gxx[x,y]+gyy[x,y]
    response = Func('response')
    response[x,y] = det[x,y]-k*trace[x,y]**2

    thresholding = Func('thresholding')
    thresholding[x, y] = select((response[x,y]>thr), 1.0, 0.0)

    localMaximum = Func('localMaximum')
    localMaximum[x,y] = select(((response[x+1, y] < response[x,y]) &
                                (response[x-1, y] < response[x,y]) &
                                (response[x, y+1] < response[x,y]) &
                                (response[x, y-1] < response[x,y])), 1.0, 0.0)

    #AND thresholding with loclMaximum to get harrisCorners
    harrisCorners = Func('harrisCorners')
    harrisCorners[x,y] = select(((thresholding[x,y] > 0) & (localMaximum[x,y] > 0)), 1.0, 0.0)

    if scheduleIndex == 1:

        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')

        clamped.compute_root()
        BW.compute_root()

        L.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        L1.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)

        gx.compute_root()
        gy.compute_root()

        gxx.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        gxx1.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        gyy.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        gyy1.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        gxy.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)
        gxy1.compute_root().tile(x, y, xo, yo, xi, yi, 256, 128).parallel(yo)

        det.compute_root()
        trace.compute_root()
        response.compute_root()
        thresholding.compute_root()
        localMaximum.compute_root()


    else:
        clamped.compute_root()
        BW.compute_root()
        L1.compute_root()
        L.compute_root()
        gx.compute_root()
        gy.compute_root()
        gxx1.compute_root()
        gxx.compute_root()
        gyy1.compute_root()
        gyy.compute_root()
        gxy1.compute_root()
        gxy.compute_root()
        det.compute_root()
        trace.compute_root()
        response.compute_root()
        thresholding.compute_root()
        localMaximum.compute_root()


    output = harrisCorners.realize(input.width(), input.height())
    outputNP = numpy.array(Image(output))

    return (outputNP, harrisCorners)


