#nprScript.py
import imageIO as io
import npr
import glob
import random as rnd
import numpy as np
import nprHelper as nprh
import math


def testBrush(out, brushTex, nStrokes = 10, col=np.array([1.0,1.0,1.0])):
    h = out.shape[0]
    w = out.shape[1]
    for i in xrange(nStrokes):
        y=int(rnd.random()*h*0.9999)
        x=int(rnd.random()*w*0.9999)
        npr.brush(out, int(y), int(x), col, brushTex)


def testAngle(im):
    thetas = npr.computeAngles(im)
    out = np.zeros_like(thetas)
    for y in xrange(thetas.shape[0]):
        for x in xrange(thetas.shape[1]):
            theta = thetas[y,x,0]
            if theta<0: theta+=2*math.pi
            out[y,x]=theta/2.0/math.pi
    io.imwrite(out, 'testangle.png')


def testSingleScale(im, texture, outputName, N=10000, size=50, noise=0.3):
    out = np.zeros_like(im)
    npr.singleScalePaint(im, out, np.ones_like(im), texture, size, N, noise)
    io.imwrite(out, str(outputName+"SingleScale"+".png"))
    
def testPainterly(im, texture, outputName, N=10000, size=50, noise=0.3):
    io.imwrite(npr.painterly(im, texture, N, size, noise), str(outputName+"Painterly"+".png"))

def testSingleScaleOrientedPaint(im, texture, outputName, N=10000, size=50, noise=0.3, nAngles=36):
    out = np.zeros_like(im)
    thetas = npr.computeAngles(im)
    npr.singleScaleOrientedPaint(im, out, thetas, np.ones_like(im), texture, size, N, noise, nAngles)
    io.imwrite(out, str(outputName+"SingleScaleOriented"+".png"))

def testOrientedPaint(im, texture, outputName, N=10000, size=50, noise=0.3):
    io.imwrite(npr.orientedPaint(im, texture, N, size, noise), str(outputName+"OrientedPaint"+".png"))


def runTests(im, texture, imname):
    testSingleScale(im, texture, imname)
    testPainterly(im, texture, imname)
    testSingleScaleOrientedPaint(im, texture, imname)
    testOrientedPaint(im, texture, imname)


brush1 = io.imread('brush.png')
longBrush = io.imread('longBrush.png')
bigBrush = io.imread('longBrush2.png')
roundIm = io.imread('round.png')


liz = io.imread('liz.png')
china = io.imread('china.png')
vpd = io.imread('villeperdue.png')
hotair = io.imread('HotAirBalloons.png')


#brushtest = np.zeros([200,200,3])
#testBrush(brushtest, brush1)
#io.imwrite(brushtest, "brushtest1.png")
#testAngle(roundIm)
#runTests(liz, brush1, "Liz")
#runTests(china, brush1, "China")
#runTests(vpd, brush1, "VPD")

#runTests(liz, bigBrush, "LizBig")
#runTests(china, bigBrush, "ChinaBig")
#runTests(vpd, bigBrush, "VPDBig")
#runTests(hotair, bigBrush, "HotAir")

#runTests(liz, longBrush, "LizTestLongBrush")
#runTests(china, longBrush, "ChinaTestLongBrush")
#runTests(roundIm, longBrush, "RoundImLong")
