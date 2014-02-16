#script for running/testing assignment 6
#Starter code by Abe Davis
#
#
# Student Name: Andrew Moran
# MIT Email: andrewmo@mit.edu

import a6
import numpy as np
import glob
import imageIO as io
from scipy import linalg

def getPNGsInDir(path):
    fnames = glob.glob(path+"*.png")
    pngs = list()
    for f in fnames:
        #print f
        imi = io.getImage(f)
        pngs.append(imi)
    return pngs

def getRawPNGsInDir(path):
    fnames = glob.glob(path+"*.png")
    pngs = list()
    pngnames = list()
    print path
    for f in fnames:
        print f
        imi = io.imreadGrey(f)
        pngs.append(imi)
        pngnames.append(f)
    return pngs, pngnames

def testApplyHomographyPoster():
    signH = np.array([[1.12265192e+00, 1.44940136e-01, 1.70000000e+02], [8.65164180e-03, 1.19897030e+00, 9.50000000e+01],[  2.55704864e-04, 8.06420365e-04, 1.00000000e+00]])
    green = io.getImage("green.png")
    poster = io.getImage("poster.png")
    a6.applyHomography(poster, green, signH, True)
    io.imwrite(green, "HWDueAt9pm_applyHomography.png")


def testComputeAndApplyHomographyPoster():
    green = io.getImage("green.png")
    poster = io.getImage("poster.png")

    h, w = poster.shape[0]-1, poster.shape[1]-1
    pointListPoster=[np.array([0, 0, 1]), np.array([0, w, 1]), np.array([h, w, 1]), np.array([h, 0, 1])]
    pointListT=[np.array([170, 95, 1]), np.array([171, 238, 1]), np.array([233, 235, 1]), np.array([239, 94, 1])]

    listOfPairs=zip(pointListPoster, pointListT)
    
    H = a6.computeHomography(listOfPairs)
    #print H
    a6.applyHomography(poster, green, H, True)
    io.imwrite(green, "HWDueAt9pm_computeHomography.png")

def testComputeAndApplyHomographyPrudential():
    pru = io.getImage("pruSkyline.png")
    poster = io.getImage("andrewTag.png")

    h, w = poster.shape[0]-1, poster.shape[1]-1
    pointListPoster=[np.array([0, 0, 1]), np.array([0, w, 1]), np.array([h, w, 1]), np.array([h, 0, 1])]
    #pointListPru=[np.array([78, 120, 1]), np.array([99, 160, 1]), np.array([150, 159, 1]), np.array([132, 118, 1])]
    pointListPru=[np.array([78, 120, 1], dtype=np.float64), np.array([99, 160, 1], dtype=np.float64), np.array([207, 156, 1], dtype=np.float64), np.array([194, 115, 1], dtype=np.float64)]

    listOfPairs=zip(pointListPoster, pointListPru)
    
    H = a6.computeHomography(listOfPairs)
    #print H
    a6.applyHomography(poster, pru, H, True)
    io.imwrite(pru, "Fun.png")

########

def testComputeAndApplyHomographyStata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairsS=zip(pointList1, pointList2)
    HS=a6.computeHomography(listOfPairsS)
    #multiply by 0.2 to better show the transition
    out=im2*0.5
    
    a6.applyHomography(im1, out, HS, True)
    io.imwrite(out, "stata_computeAndApplyHomography.png")

def testStitchStata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "stata_stitch.png")

def testStitchScience():
    im1=io.imread('science/science-1.png')
    im2=io.imread('science/science-2.png')
    pointList1=[np.array([307, 15, 1], dtype=np.float64), np.array([309, 106, 1], dtype=np.float64), np.array([191, 102, 1], dtype=np.float64), np.array([189, 47, 1], dtype=np.float64)]
    pointList2=[np.array([299, 214, 1], dtype=np.float64), np.array([299, 304, 1], dtype=np.float64), np.array([182, 292, 1], dtype=np.float64), np.array([183, 236, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "science_stitch.png")

def testStitchMIT():
    im1=io.imread('mit1.png')
    im2=io.imread('mit0.png')
    pointList1=[np.array([196, 245, 1], dtype=np.float64), np.array([250, 320, 1], dtype=np.float64), np.array([138, 306, 1], dtype=np.float64), np.array([113, 260, 1], dtype=np.float64)]
    pointList2=[np.array([200, 48, 1], dtype=np.float64), np.array([255, 115, 1], dtype=np.float64), np.array([150, 109, 1], dtype=np.float64), np.array([119, 65, 1], dtype=np.float64)]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitch(im1, im2, listOfPairs)
    io.imwrite(out, "MyPano.png")

def testStitchN0Stata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitchN([im1, im2], [listOfPairs], 0)
    io.imwrite(out, "stata_stitchN0.png")

def testStitchN1Stata():
    im1=io.imread('stata/stata-1.png')
    im2=io.imread('stata/stata-2.png')
    pointList1=[np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])]
    pointList2=[np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])]
    listOfPairs=zip(pointList1, pointList2)
    out = a6.stitchN([im1, im2], [listOfPairs], 1)
    io.imwrite(out, "stata_stitchN1.png")

def testStitchNVancouver():
    #im1=io.imread('vancouverPan/vancouver4.png')
    #im2=io.imread('vancouverPan/vancouver3.png')
    im1=io.imread('vancouverPan/vancouver2.png')
    im2=io.imread('vancouverPan/vancouver1.png')
    im3=io.imread('vancouverPan/vancouver0.png')
    imList = [im1, im2, im3]#, im4, im5]


    pointList1a=[np.array([99, 326, 1], dtype=np.float64), np.array([271, 247, 1], dtype=np.float64), np.array([180, 178, 1], dtype=np.float64), np.array([179, 276, 1], dtype=np.float64)]
    pointList2a=[np.array([124, 169, 1], dtype=np.float64), np.array([284, 98, 1], dtype=np.float64), np.array([189, 25, 1], dtype=np.float64), np.array([194, 125, 1], dtype=np.float64)]
    listOfPairs2=zip(pointList1a, pointList2a)

    pointList1b=[np.array([176, 300, 1], dtype=np.float64), np.array([318, 204, 1], dtype=np.float64), np.array([258, 203, 1], dtype=np.float64), np.array([181, 138, 1], dtype=np.float64)]
    pointList2b=[np.array([179, 180, 1], dtype=np.float64), np.array([317, 86, 1], dtype=np.float64), np.array([256, 87, 1], dtype=np.float64), np.array([173, 15, 1], dtype=np.float64)]
    listOfPairs1=zip(pointList1b, pointList2b)

    #pointList1_3=[np.array([165, 186, 1], dtype=np.float64), np.array([173, 146, 1], dtype=np.float64), np.array([188, 80, 1], dtype=np.float64), np.array([164, 40, 1], dtype=np.float64)]
    #pointList2_3=[np.array([153, 298, 1], dtype=np.float64), np.array([162, 253, 1], dtype=np.float64), np.array([178, 188, 1], dtype=np.float64), np.array([156, 151, 1], dtype=np.float64)]
    #listOfPairs3=zip(pointList1_3, pointList2_3)

    #pointList1_4=[np.array([156, 151, 1], dtype=np.float64), np.array([220, 34, 1], dtype=np.float64), np.array([184, 160, 1], dtype=np.float64), np.array([186, 89, 1], dtype=np.float64)]
    #pointList2_4=[np.array([151, 304, 1], dtype=np.float64), np.array([226, 189, 1], dtype=np.float64), np.array([180, 316, 1], dtype=np.float64), np.array([190, 239, 1], dtype=np.float64)]
    #listOfPairs4=zip(pointList1_4, pointList2_4)

    listOfListOfPairs = [listOfPairs1, listOfPairs2]# listOfPairs4]
    #listOfListOfPairs = [listOfPairs]#, listOfPairs2]#, listOfPairs2]#, listOfPairs3, listOfPairs4]

    out = a6.stitchN(imList, listOfListOfPairs, 0)
    io.imwrite(out, "vancouver_stitchN.png")

#testApplyHomographyPoster()
#testComputeAndApplyHomographyPoster()
#testComputeAndApplyHomographyPrudential()
#testComputeAndApplyHomographyStata()
#testStitchStata()
#testStitchScience()
#testStitchMIT()

#testStitchN0Stata()
#testStitchN1Stata()

#testStitchNVancouver()
#testStitchN0Vancouver()
#testStitchN1Vancouver()
#testStitchN2Vancouver()

#***You can test on the first N images of a list by feeding im[:N] as the argument instead of im***

