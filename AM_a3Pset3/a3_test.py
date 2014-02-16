from scipy import misc
import imageIO as io
import numpy as np
import matplotlib.pyplot as plt
import a3
import matplotlib.cm as cm
import time

# No need to touch this class
#class myImageIO:
#  @staticmethod
#  def imread(path='in.png'):
#    from scipy import misc
#    return (misc.imread(path).astype(float)/255)**2.2
#
#  @staticmethod
#  def imreadg(path='in.png'):
#    from scipy import misc
#    return (misc.imread(path).astype(float)/255)
#
#  @staticmethod
#  def imwrite(im_in, path):
#    from scipy import misc
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return misc.imsave(path, im_in**(1/2.2))
#
#  @staticmethod
#  def imwriteg(im_in, path):
#    from scipy import misc
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return misc.imsave(path, im_in)
#  
#  @staticmethod
#  def thresh(im_in):
#    im_in[im_in<0]=0
#    im_in[im_in>1]=1
#    return im_in 

## Test case ## 
## Feel free to change the parameters or use the impulse as input

def test_module():
    print a3.check_module()

def test_box_blur():
    im=io.imread('pru.png')
    out=a3.boxBlur(im, 7)
    io.imwrite(out, 'my_boxblur.png')

def test_convolve_gauss():

    im=io.imread('pru.png')
    gauss3=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel=gauss3.astype(float)
    kernel=kernel/sum(sum(kernel))
    out=a3.convolve(im, kernel)
    io.imwrite(out, 'my_gaussblur.png')

def test_convolve_deriv():
    im=io.imread('pru.png')
    deriv=np.array([[-1, 1]])
    out=a3.convolve(im, deriv)
    io.imwrite(out, 'my_deriv.png')

def test_convolve_Sobel():
    im=io.imread('pru.png')
    Sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    out=a3.convolve(im, Sobel)
    io.imwrite(out, 'my_Sobel.png')

def test_grad():
    im=io.imread('pru.png')
    out=a3.gradientMagnitude(im)
    io.imwrite(out, 'my_gradient.png')

def test_horigauss():
    im=io.imread('pru.png')
    kernel=a3.horiGaussKernel(2,3)
    out=a3.convolve(im, kernel)
    io.imwrite(out, 'my_horigauss.png')

def test_gaussianBlur():
    im=io.imread('pru.png')
    out=a3.gaussianBlur(im, 2,3)
    io.imwrite(out, 'my_gaussBlur2.png')


def test_gauss2D():
    im=io.imread('pru.png')
    out=a3.convolve(im, a3.gauss2D())
    io.imwrite(out, 'my_gauss2DBlur.png')

def test_equal():
    im=io.imread('pru.png')
    out1=a3.convolve(im, a3.gauss2D())
    out2=a3.gaussianBlur(im,2, 3)
    res=abs(out1-out2);
    return (sum(res.flatten())<0.1)
  
def test_unsharpen():
    im=io.imread('zebra.png')
    out=a3.unsharpenMask(im, 1, 3, 1)
    io.imwrite(out, 'my_unsharpen.png')

def test_bilateral():
    im=io.imread('lens-3-med.png')
    out=a3.bilateral(im, 0.3, 1.4)
    io.imwrite(out, 'my_bilateral.png')
    

def test_bilaYUV():
    im=io.imread('lens-3-med.png')
    out=a3.bilaYUV(im, 0.3, 1.4, 6)
    io.imwrite(out, 'my_bilaYUV.png')
  
def impulse(h=100, w=100):
    out=constantIm(h, w, 0.0)
    out[h/2, w/2]=1
    return out


#Uncomment the following function to test your code

test_module()
test_box_blur()
test_convolve_gauss()
test_convolve_deriv()
test_convolve_Sobel()
test_grad()
test_horigauss()

t = time.time()
test_gaussianBlur()
print time.time()-t, 'seconds for gB'
t = time.time()
test_gauss2D()
print time.time()-t, 'seconds for gB2D'
print test_equal()
test_unsharpen()
test_bilateral()
test_bilaYUV()
