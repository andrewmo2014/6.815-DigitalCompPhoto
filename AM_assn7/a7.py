import numpy as np
import scipy
from scipy import ndimage
import imageIO as io
import random as rnd
import a6 as a6

class point():
  def __init__(self, x, y):
    self.x=x
    self.y=y

class feature():
  def __init__(self, pt, descriptor):
    self.pt=pt
    self.descriptor=descriptor

class correspondence():
  def __init__(self, pt1, pt2):
    self.pt1=pt1
    self.pt2=pt2

def BW(im, weights=[0.3,0.6,0.1]):
  lum = np.zeros((im.shape[0], im.shape[1]))
  lum = np.dot(im, weights)
  return lum


def computeTensor(im, sigmaG=1, factorSigma=4):
  '''im_out: 3-channel-2D array. The three channels are Ixx, Ixy, Iyy'''
  lum = BW(im)
  blurLum = ndimage.filters.gaussian_filter( lum, sigmaG )

  Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

  I_x = ndimage.filters.convolve( blurLum, Sobel, mode='reflect' )
  I_y = ndimage.filters.convolve( blurLum, np.transpose( Sobel ), mode='reflect' )

  I_xx = I_x*I_x
  I_yy = I_y*I_y
  I_xy = I_x*I_y

  I_r = ndimage.filters.gaussian_filter( I_xx, sigmaG*factorSigma )
  I_g = ndimage.filters.gaussian_filter( I_xy, sigmaG*factorSigma )
  I_b = ndimage.filters.gaussian_filter( I_yy, sigmaG*factorSigma )

  im_out = io.constantIm( im.shape[0], im.shape[1], 0.0)
  im_out[:,:,0] = I_r
  im_out[:,:,1] = I_g
  im_out[:,:,2] = I_b

  return im_out


def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
  '''resp: 2D array charactering the response'''

  im = computeTensor(im, sigmaG, factorSigma)

  I_xx = im[:,:,0]
  I_xy = im[:,:,1]
  I_yy = im[:,:,2]

  detM = I_xx*I_yy - I_xy*I_xy
  traceM = I_xx*I_yy

  R = detM - k*traceM*traceM
  return R


def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
  '''result: a list of points that locate the images' corners'''

  R = cornerResponse( im, k, sigmaG, factor)
  maxData = ndimage.filters.maximum_filter( R, maxiDiam )
  isMax = (R == maxData)

  maxims, num = ndimage.label(isMax)
  points = ndimage.find_objects( maxims )
  vals = []

  for dy,dx in points:
    y_cent = (dy.start+dy.stop-1)/2
    x_cent = (dx.start+dx.stop-1)/2

    if( (y_cent < im.shape[0]-boundarySize) and (y_cent > boundarySize)):
      if( (x_cent < im.shape[1]-boundarySize) and (x_cent > boundarySize)):
        vals.append((float(y_cent),float(x_cent)))

  result = map(lambda val : point(val[1],val[0]), vals)
  return result


def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
  '''f_list: a list of feature objects'''
  lum = BW(im)

  blurLum = ndimage.filters.gaussian_filter( lum, sigmaBlurDescriptor )
  descriptors = []

  for corner in cornerL:
    d = descriptor( blurLum, corner, radiusDescriptor)
    descriptors.append( d )

  featureList = zip( cornerL, descriptors )
  result = map(lambda f : feature(f[0],f[1]), featureList)
  return result


def descriptor(blurredIm, P, radiusDescriptor=4):
  '''patch: descriptor around 2-D point P, with size (2*radiusDescriptor+1)^2 in 1-D'''
  r = radiusDescriptor
  patch = blurredIm[ P.y-r : P.y+r+1, P.x-r : P.x+r+1]
  patch = patch - np.mean(patch)
  patch = (1/np.std(patch)) * patch
  return patch

def L2Norm(feature1, feature2):
  eucliDist = feature1.descriptor - feature2.descriptor
  vector = eucliDist.flatten()
  return np.dot(vector, vector)


def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):  
  '''correpondences: a list of correspondences object that associate two feature lists.'''
  correspondences = []

  for feature1 in listFeatures1:
    norms = [ L2Norm(feature1, feature2) for feature2 in listFeatures2 ]

    minArg1 = np.argmin(norms)
    minVal1 = np.min(norms)

    norms.pop(minArg1)

    minArg2 = np.argmin(norms)
    minVal2 = np.min(norms)

    if(minVal2/minVal1 > threshold):
      pt1 = feature1.pt
      pt2 = listFeatures2[minArg1].pt
      correspondences.append( correspondence(pt1,pt2) )

  return correspondences


def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4, acceptableProbFailure=1e-9):
  '''H_best: the best estimation of homorgraphy (3-by-3 matrix)'''
  '''inliers: A list of booleans that describe whether the element in listOfCorrespondences 
  an inlier or not'''
  ''' 6.815 can bypass acceptableProbFailure'''
  cLen = len(listOfCorrespondences)
  maxInliers = 0
  H_best = np.zeros((3,3))

  for i in xrange(Niter):
    samples = rnd.sample( listOfCorrespondences, 4 )
    H = a6.computeHomography( A7PairsToA6Pairs(samples) )

    areInliers = map( mapH, [H]*cLen, listOfCorrespondences, [epsilon]*cLen )
    numInliers = areInliers.count(True)

    if(maxInliers < numInliers):
      H_best = H
      maxInliers = numInliers

    x = float(maxInliers)/cLen
    probFailure = pow( (1-pow(x,4)), i+1 )

    if (probFailure<acceptableProbFailure):
      break

  inliers = map( mapH, [H_best]*cLen, listOfCorrespondences, [epsilon]*cLen )
  return (H_best, inliers)


def mapH( H, corr, epsilon):
  p1 = corr.pt1
  p2 = corr.pt2

  pos = np.dot(H, A7PointToA6Point(p1) )
  if pos[2] == 0:
    pos = np.array( [np.inf, np.inf, 1] )
  else:
    pos /= pos[2]

  posH = pos[ [0,1] ]
  posP = np.array( [p2.y,p2.x] )

  d = posH - posP
  v = d.flatten()
  dist = np.sqrt( np.dot(v,v) )

  if( dist<epsilon ):
    return True
  else:
    return False


def getFeatures(im, blurDescriptor=0.5, radiusDescriptor=4):
  cornerList = HarrisCorners(im)
  featureList = computeFeatures(im, cornerList, blurDescriptor, radiusDescriptor)
  return featureList


def computeNHomographies(L, refIndex, blurDescriptior=0.5, radiusDescriptor=4):
  '''finalH: a list of Homorgraphy relative to L[refIndex]'''
  '''Note: len(finalH) is equal to len(L)'''

  imFeatures = map( getFeatures, L, [blurDescriptior] * len(L), [radiusDescriptor] * len(L) )

  Hlist = []
  for i in range(len(L)-1):
    corrs = findCorrespondences(imFeatures[i], imFeatures[i+1])
    bestH = RANSAC(corrs)[0]
    Hlist.append(bestH)
  identity = np.identity(3)
  Hlist.insert(refIndex, identity)

  finalH = Hlist
  H = Hlist[refIndex]

  for i in range( len(Hlist) ):
      if i < refIndex:
          ind = len(Hlist[:refIndex])-(i+1)
          H = np.dot( np.linalg.inv( Hlist[ind]), H )
      elif i > refIndex:
          ind = i
          H = np.dot( Hlist[ind], H )
      else:
          ind = i
          H = Hlist[ind]
      finalH[ind] = H

  return finalH


def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  '''Use your a6 code to stitch the images. You need to hand in your A6 code'''
  H_list = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
  return a6.compositeNImages(L, H_list)


def separableArray(length):
  fLen = float(length-1)

  interval = 1/fLen
  a = np.arange( -0.5, 0.5 + interval/2, interval)
  b = np.arange( 0.5, -0.5 - interval/2, -interval)
  c = np.ones((length)) - abs(a-b)
  return np.matrix(c)

def weight_map(h,w):
  ''' Given the image dimension h and w, return the hxwx3 weight map for linear blending'''

  widths = separableArray(w)
  heights = separableArray(h)

  w_map = np.dot( heights.T, widths )
  return np.array(w_map)


def linear_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with linear blending'''
  
  H_list = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
  
  w_list = []
  for i in range(len(L)):
    w = weight_map( L[i].shape[0], L[i].shape[1] )
    w_list.append(w)

  return a6.compositeNImages(L, H_list, w_list)


def two_scale_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with two scale blending'''

  H_list = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
  
  w_list = []
  for i in range(len(L)):
    w = weight_map( L[i].shape[0], L[i].shape[1] )
    w_list.append(w)

  return a6.compositeNImages(L, H_list, w_list, True)


# Helpers, you may use the following scripts for convenience.
def A7PointToA6Point(a7_point):
  return np.array([a7_point.y, a7_point.x, 1.0], dtype=np.float64)


def A7PairsToA6Pairs(a7_pairs):
  A7pointList1=map(lambda pair: pair.pt1 ,a7_pairs)
  A6pointList1=map(A7PointToA6Point, A7pointList1)
  A7pointList2=map(lambda pair: pair.pt2 ,a7_pairs)
  A6pointList2=map(A7PointToA6Point, A7pointList2)
  return zip(A6pointList1, A6pointList2)
  


