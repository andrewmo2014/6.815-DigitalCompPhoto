#a2.py
import numpy as np
import math
import imageIO as io


#this file should only contain function definitions.
#It should not call the functions or perform any test.
#Do this in a separate file.

def check_my_module():
   ''' Fill your signature here. When upload your code, check if the signature is correct'''
   my_signature= 'Andrew Moran'
   return my_signature

   
def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)


def pix(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns a pixel.
        If y,x is outside the image and repeatEdge==True , you should return the nearest pixel in the image
        If y,x is outside the image and repeatEdge==False , you should return a black pixel
    '''
    height = im.shape[0]
    width = im.shape[1]
   
    if repeatEdge:
       if (x<0) or (x>=width) or (y<0) or (y>=height):
          return np.array([0,0,0])
       else:
          return im[y,x]
    else:
       clipX = min(width-1, max(x,0))
       clipY = min(height-1, max(y,0))
       return im[clipY, clipX]
   


def scaleNN(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using nearest neighbor interpolation.
    '''
    height = im.shape[0]
    width = im.shape[1]

    out = io.constantIm(height*k, width*k, 0.0)
    for y,x in imIter(out):
       out[y,x] = im[y/k, x/k]
    return out
    

def interpolateLin(im, y, x, repeatEdge=False):
    '''takes an image, y and x coordinates, and a bool
        returns the interpolated pixel value using bilinear interpolation
    '''

    x_int = int(x)
    y_int = int(y)

    dx = x - x_int
    dy = y - y_int
    
    n00 = pix(im, y_int  , x_int  , repeatEdge)
    n01 = pix(im, y_int  , x_int+1, repeatEdge)
    n10 = pix(im, y_int+1, x_int  , repeatEdge)
    n11 = pix(im, y_int+1, x_int+1, repeatEdge)
    
    a = n00*(1-dx) + n01*dx
    b = n10*(1-dx) + n11*dx

    pixel = a*(1-dy) + b*dy
    return pixel

def interpolateBicubic( im, y, x, repeatEdge=False ):

    x_int = int(x)
    y_int = int(y)

    dx = x - x_int
    dy = y - y_int
    normxy = np.array([dy, dx], dtype=np.float64)

    st0 = ((2.0 - normxy)*normxy - 1.0) * normxy
    st1 = (3.0*normxy - 5.0)*normxy*normxy + 2.0
    st2 = ((4.0 -3.0*normxy)*normxy + 1.0)*normxy
    st3 = (normxy-1.0)*normxy*normxy

    p00 = pix(im, y_int-1, x_int-1, repeatEdge)
    p01 = pix(im, y_int-1, x_int  , repeatEdge)
    p02 = pix(im, y_int-1, x_int+1, repeatEdge)
    p03 = pix(im, y_int-1, x_int+2, repeatEdge)
    p10 = pix(im, y_int  , x_int-1, repeatEdge)
    p11 = pix(im, y_int  , x_int  , repeatEdge)
    p12 = pix(im, y_int  , x_int+1, repeatEdge)
    p13 = pix(im, y_int  , x_int+2, repeatEdge)
    p20 = pix(im, y_int+1, x_int-1, repeatEdge)
    p21 = pix(im, y_int+1, x_int  , repeatEdge)
    p22 = pix(im, y_int+1, x_int+1, repeatEdge)
    p23 = pix(im, y_int+1, x_int+2, repeatEdge)
    p30 = pix(im, y_int+2, x_int-1, repeatEdge)
    p31 = pix(im, y_int+2, x_int  , repeatEdge)
    p32 = pix(im, y_int+2, x_int+1, repeatEdge)
    p33 = pix(im, y_int+2, x_int+2, repeatEdge)

    row0 = st0[1]*p00 + st1[1]*p01 + st2[1]*p02 + st3[1]*p03
    row1 = st0[1]*p10 + st1[1]*p11 + st2[1]*p12 + st3[1]*p13
    row2 = st0[1]*p20 + st1[1]*p21 + st2[1]*p22 + st3[1]*p23
    row3 = st0[1]*p30 + st1[1]*p31 + st2[1]*p32 + st3[1]*p33

    pixel = 0.25 * ((st0[0]*row0) + (st1[0]*row1) + (st2[0]*row2) + (st3[0]*row3))
    return pixel


def interpolateBiquad( im, y, x, repeatEdge=False ):

    x_int = int(x)
    y_int = int(y)

    dx = x - x_int
    dy = y - y_int

    qx = 0.5*dx*dx
    qy = 0.5*dy*dy
    dx = 0.5*dx
    dy = 0.5*dy
    
    p00 = pix(im, y_int-1, x_int-1, repeatEdge)
    p01 = pix(im, y_int-1, x_int  , repeatEdge)
    p02 = pix(im, y_int-1, x_int+1, repeatEdge)
    p10 = pix(im, y_int  , x_int-1, repeatEdge)
    p11 = pix(im, y_int  , x_int  , repeatEdge)
    p12 = pix(im, y_int  , x_int+1, repeatEdge)
    p20 = pix(im, y_int+1, x_int-1, repeatEdge)
    p21 = pix(im, y_int+1, x_int  , repeatEdge)
    p22 = pix(im, y_int+1, x_int+1, repeatEdge)

    row0 = p01 + (p02-p00)*dx + (p00 - 2*p01 + p02)*qx
    row1 = p11 + (p12-p10)*dx + (p10 - 2*p11 + p12)*qx
    row2 = p21 + (p22-p20)*dx + (p20 - 2*p21 + p22)*qx
    
    pixel = row1 + (row2-row0)*dy + (row0 - 2*row1 + row2)*qy
    return pixel

   
def scaleLin(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using bilinear interpolation.
    '''
    height = im.shape[0]
    width = im.shape[1]
    newHeight = height*k
    newWidth = width*k

    out = io.constantIm(newHeight, newWidth, 0.0)
    for y,x in imIter(out):
        x_float = x/float(newWidth) * (width)
        y_float = y/float(newHeight) * (height)

        pix = interpolateLin(im, y_float, x_float)
        out[y,x] = pix
    return out
       
def scaleBicubic(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using bilinear interpolation.
    '''
    height = im.shape[0]
    width = im.shape[1]
    newHeight = height*k
    newWidth = width*k

    out = io.constantIm(newHeight, newWidth, 0.0)
    for y,x in imIter(out):
        x_float = x/float(newWidth) * (width)
        y_float = y/float(newHeight) * (height)

        pix = interpolateBicubic(im, y_float, x_float)
        out[y,x] = pix
    return out

def scaleBiquadratic(im, k):
    '''Takes an image and a scale factor. Returns an image scaled using bilinear interpolation.
    '''
    height = im.shape[0]
    width = im.shape[1]
    newHeight = height*k
    newWidth = width*k

    out = io.constantIm(newHeight, newWidth, 0.0)
    for y,x in imIter(out):
        x_float = x/float(newWidth) * (width)
        y_float = y/float(newHeight) * (height)

        pix = interpolateBiquad(im, y_float, x_float)
        out[y,x] = pix
    return out

def rotate(im, theta):
    '''takes an image and an angle in radians as input
        returns an image of the same size and rotated by theta
    '''

    h = im.shape[0]
    w = im.shape[1]
    center = (int(h/2), int(w/2))

    out = io.constantIm(h, w, 0.0)

    for y,x in imIter(out):
       tx = ((x-center[1])*math.cos(theta) - (y-center[0])*math.sin(theta) + center[1])
       ty = ((x-center[1])*math.sin(theta) + (y-center[0])*math.cos(theta) + center[0])
       
       pix = interpolateLin(im, ty, tx, True)
       out[y,x] = pix
    return out


class segment:
    def __init__(self, x1, y1, x2, y2):
        #notice that the ui gives you x,y and we are storing as y,x
        self.P=np.array([y1, x1], dtype=np.float64)
        self.Q=np.array([y2, x2], dtype=np.float64)
        #You can precompute more variables here
        #...
        self.QminusP = np.subtract(self.Q,self.P)

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def perpendicular2D(self, vector):
        (y, x) = vector
        return np.array([x, -y], dtype=np.float64)

    def uv(self, X):
        '''Take the (y,x) coord given by X and return u, v values
        '''
        XminusP = np.subtract(X,self.P) 
        u = np.divide( np.dot(XminusP,self.QminusP), np.square(np.linalg.norm(self.QminusP)))
        v = np.divide( np.dot(XminusP,self.perpendicular2D(self.QminusP)), np.linalg.norm(self.QminusP))
        return u, v

    def dist (self, X):
        '''returns distance from point X to the segment (pill shape dist)
        '''
        (y_o, x_o) = X
        
        px = self.x2 - self.x1
        py = self.y2 - self.x1
        p_dist = px**2 + py**2

        u = ((x_o - self.x1)*px + (y_o - self.y1)*py) / float(p_dist)

        if u>1:
           u=1
        elif u<0:
           u=0

        x = self.x1 + u*px
        y = self.y1 + u*py

        dx = x-x_o
        dy = y-y_o

        distance = math.sqrt(dx**2 + dy**2)
        return distance
        
    
    def uvtox(self,u,v):
        '''take the u,v values and return the corresponding point (that is, the np.array([y, x]))
        '''
        newX = self.P + np.dot(u,self.QminusP) + np.divide( np.dot(v, self.perpendicular2D(self.QminusP)), np.linalg.norm(self.QminusP) ) 
        return newX

def warpBy1(im, segmentBefore, segmentAfter):
    '''Takes an image, one before segment, and one after segment. 
        Returns an image that has been warped according to the two segments.
    '''
    height = im.shape[0]
    width = im.shape[1]

    out = io.constantIm(height, width, 0.0)
    for y,x in imIter(out):
        X = np.array([y,x], dtype=np.float64) 
        (u,v) = segmentAfter.uv(X)
        X_prime = segmentBefore.uvtox(u,v)
        pixel = interpolateLin(im, X_prime[0], X_prime[1])
        out[y,x] = pixel
    return out


def weight(s, X):
    '''Returns the weight of segment s on point X
    '''
    length = np.linalg.norm( s.QminusP )
    dist = s.dist(X)
    a = 10
    b = 1
    p = 1
    weight = ( (length**p) / float(a + dist) )**b
    return weight
    

def warp(im, segmentsBefore, segmentsAfter, a=10, b=1, p=1):
    '''Takes an image, a list of before segments, a list of after segments, and the parameters a,b,p (see Beier)
    '''
    height = im.shape[0]
    width = im.shape[1]

    out = io.constantIm(height, width, 0.0)
    for y,x in imIter(out):
        X = np.array([y,x], dtype=np.float64)

        DSUM = np.array([0,0], dtype=np.float64)
        weightsum = 0

        for i in range(len(segmentsAfter)):
            (u,v) = segmentsAfter[i].uv(X)
            X_prime = segmentsBefore[i].uvtox(u,v)
            D = np.subtract(X_prime, X)
            dist = segmentsAfter[i].dist(X)
            w = weight(segmentsAfter[i], X)
            DSUM = np.add( DSUM, D*w )
            weightsum += w
        X_final = np.add( X, DSUM/float(weightsum) )
        
        pixel = interpolateLin(im, X_final[0], X_final[1])
        out[y,x] = pixel
    return out


def morph(im1, im2, segmentsBefore, segmentsAfter, N=1, a=10, b=1, p=1):
    '''Takes two images, a list of before segments, a list of after segments, the number of morph images to create, and parameters a,b,p.
        Returns a list of images morphing between im1 and im2.
    '''
    sequence=list()
    sequence.append(im1.copy())
    #add the rest of the morph images

    tVals = [float(n)/(N+1) for n in range(N+2)]
    tMids = tVals[1:-1]    #t values for middle images

    for t in tMids:
       segmentsMiddle=list()
       for i in range(len(segmentsAfter)):
          segMidP = np.add( segmentsBefore[i].P*(1.0-t), segmentsAfter[i].P*t )
          segMidQ = np.add( segmentsBefore[i].Q*(1.0-t), segmentsAfter[i].Q*t )
          segmentsMiddle.append(segment( segMidP[1], segMidP[0], segMidQ[1], segMidQ[0] ))

       im1Warp = warp(im1, segmentsBefore, segmentsMiddle, a, b, p)
       im2Warp = warp(im2, segmentsAfter, segmentsMiddle, a, b, p)

       interWarp = (1.0-t)*im1Warp + t*im2Warp
       sequence.append(interWarp)

    sequence.append(im2.copy())
    return sequence
