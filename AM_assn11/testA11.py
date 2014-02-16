import imageIO
import numpy 
import a11
from halide import *

def main():
    im=imageIO.imread('rgb.png')
    lumi=im[:,:,1] #I'm lazy, I'll just use green
    smallLumi=numpy.transpose(lumi[0:5, 0:5])

    imTest1Channel = numpy.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                                  [1.0, 0.0, 0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0, 0.0, 1.0],
                                  [1.0, 1.0, 1.0, 1.0, 1.0]])

    imTest3Channel = numpy.array([[[1.0,1.0,1.0], [1.0,1.0,1.0]  ,[1.0,1.0,1.0]  ,[1.0,1.0,1.0]  ,[1.0,1.0,1.0]],
                                  [[1.0,1.0,1.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0,1.0,1.0]],
                                  [[0.0,0.0,0.0], [1.0, 1.0, 1.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0,1.0,1.0]],
                                  [[1.0,1.0,1.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0,1.0,1.0]],
                                  [[1.0,1.0,1.0], [1.0,1.0,1.0]  ,[1.0,1.0,1.0]  ,[1.0,1.0,1.0]  ,[1.0,1.0,1.0]]])

    # Replace if False: by if True: once you have implement the required functions. 
    # Exercises:
    if True:
        outputNP, myFunc=a11.smoothGradientNormalized()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'normalizedGradient.png')
    if True:
        outputNP, myFunc=a11.wavyRGB()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'rgbWave.png')
    if True: 
        outputNP, myFunc=a11.sobel(lumi)
        imageIO.imwrite(outputNP, 'sobelMag.png')
        print ' Dimensionality of Halide Func:', myFunc.dimensions()

    if True: 
        L=a11.pythonCodeForBoxSchedule5(smallLumi)
        print L
    if True: 
        L=a11.pythonCodeForBoxSchedule6(smallLumi)
        print L
    if True: 
        L=a11.pythonCodeForBoxSchedule7(smallLumi)
        print L
    if True: 
        outputNP, myFunc=a11.localMax(lumi)
        #outputNP, myFunc=a11.localMax(imTest)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'maxi.png')
    if True: 
        input=Image(Float(32), lumi)
        x, y = Var('x'), Var('y')
        clamped = Func('clamped') 
        clamped[x, y] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1)]
        blurX, finalBlur= a11.GaussianSingleChannel(clamped , sigma=2, trunc=3)
        out = finalBlur.realize(input.width(), input.height())
        outNP = numpy.array(Image(out))
        imageIO.imwrite(outNP, 'gaussianSingle.png')

    if True:
        #im=imageIO.imread('hk.png')
        im=numpy.load('Input/hk.npy')
        scheduleIndex=0

        L=[]
        t=time.time()
        outputNP, myFunc=a11.harris(im, scheduleIndex)
        t0 = time.time()-t
        L.append(t0)

        print "scheduleIndex",scheduleIndex, " harris took ", t0, " seconds"
        print ' Dimensionality of Halide Func:', myFunc.dimensions()

        hIm=Image(outputNP)
        mpix=hIm.width()*hIm.height()/1e6
        print 'best: ', numpy.min(L), 'average: ', numpy.mean(L)
        print  '%.5f ms per megapixel (%.7f ms for %d megapixels)' % (numpy.mean(L)/mpix*1e3, numpy.mean(L)*1e3, mpix)

        imageIO.imwrite(outputNP, 'harrisFinal.png')
  

 #usual python business to declare main function in module. 
if __name__ == '__main__': 
    main()
