import numpy as np
from utils import imageIO as io

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
	im1_dot_im2 = np.sum(im1 * im2)
  	return im1_dot_im2

def applyKernel(im, kernel):
  	''' return Mx, where x is im '''
  	out = convolve3(im, kernel)
  	return out

def applyConjugatedKernel(im, kernel):
  	''' return M^T x, where x is im '''
  	kernelT = np.flipud(np.fliplr(kernel))
  	out = convolve3(im, kernelT)
  	return out

def computeResidual(kernel, x, y):
  	''' return y - Mx '''
  	out = y - applyKernel(x, kernel)
  	return out

def computeStepSize(r, kernel):
	alpha = dotIm(r,r) / dotIm(r, applyConjugatedKernel(applyKernel(r, kernel), kernel))
	return alpha

def deconvGradDescent(im_blur, kernel, niter=10):
  	''' return deblurred image '''

  	x = io.constantIm( im_blur.shape[0], im_blur.shape[1], 0.0)
  	for i in range(niter): 
  		r = applyConjugatedKernel( computeResidual(kernel, x, im_blur), kernel )
  		alpha = computeStepSize(r, kernel)
  		x = x + np.multiply(alpha, r)
  	return x

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
	alpha = dotIm(r,r) / dotIm(d, applyConjugatedKernel(applyKernel(d, kernel), kernel))
	return alpha

def computeConjugateDirectionStepSize(old_r, new_r):
	beta = dotIm(new_r,new_r) / dotIm(old_r, old_r)
	return beta

def deconvCG(im_blur, kernel, niter=10):
  	''' return deblurred image '''

  	x = io.constantIm( im_blur.shape[0], im_blur.shape[1], 0.0)
  	r = applyConjugatedKernel( computeResidual(kernel, x, im_blur), kernel )
  	d = r

	for i in range(niter): 
		
		alpha = computeGradientStepSize(r, d, kernel)
		x = x + np.multiply(alpha, d)
		r1 = r - np.multiply(alpha, applyConjugatedKernel(applyKernel(d, kernel), kernel))
		beta = computeConjugateDirectionStepSize(r, r1)
		d = r1 + np.multiply( beta, d )
		r = r1

	return x

def laplacianKernel():
 	''' a 3-by-3 array '''
 	L = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
 	return L

def applyLaplacian(im):
 	''' return Lx (x is im)'''
 	out = convolve3(im, laplacianKernel())
  	return out

def applyAMatrix(im, kernel):
 	''' return Ax, where A = M^TM'''
 	out = applyConjugatedKernel(applyKernel(im, kernel), kernel)
 	return out

def applyRegularizedOperator(im, kernel, lamb):
  	''' (A + lambda L )x'''
  	out = applyAMatrix(im, kernel) + np.multiply(lamb, applyLaplacian(im))
  	return out


def computeGradientStepSize_reg(grad, p, kernel, lamb):
	alpha = dotIm(grad,grad) / dotIm(p, applyRegularizedOperator(p, kernel, lamb))
	return alpha

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
 	''' return deblurred and regularized im '''
  	
  	x = io.constantIm( im_blur.shape[0], im_blur.shape[1], 0.0)
  	r = applyKernel(im_blur, kernel) - applyRegularizedOperator(x, kernel, lamb)
  	d = r

	for i in range(niter): 
		
		alpha = computeGradientStepSize_reg(r, d, kernel, lamb)
		x = x + np.multiply(alpha, d)
		im = x
		r1 = r - np.multiply(alpha, applyRegularizedOperator(d, kernel, lamb))
		beta = computeConjugateDirectionStepSize(r, r1)
		d = r1 + np.multiply( beta, d )
		r = r1

	return im

    
def naiveComposite(bg, fg, mask, y, x):
 	''' naive composition'''
 	out = bg.copy()

 	mask1 = 1-mask
 	out[ y:y+fg.shape[0], x:x+fg.shape[1] ] *= mask1
 	io.imwrite(out, "naiveDebugBG.png")

 	fg[ mask == 0] = 0
 	bg[ y:y+fg.shape[0], x:x+fg.shape[1] ] = fg
 	io.imwrite(fg, "naiveDebugFG.png")
 	io.imwrite(bg, "naiveDebugBGFG.png")

 	out[y:y+fg.shape[0], x:x+fg.shape[1]] += bg[y:y+fg.shape[0], x:x+fg.shape[1]]

 	return out


def Poisson(bg, fg, mask, niter=200):
	''' Poisson editing using gradient descent'''
	b = applyKernel(fg, laplacianKernel())
	x = (1-mask)*bg

	for i in range(niter):

  		r = b - applyLaplacian(x)
  		r *= mask
  		alpha = dotIm(r,r) / dotIm(r, applyLaplacian(r))
  		x = x + np.multiply(alpha, r)

  	return x


def PoissonCG(bg, fg, mask, niter=200):
	''' Poison editing using conjugate gradient '''
	b = applyKernel(fg, laplacianKernel())
	x = (1-mask)*bg
	r = b - applyLaplacian(x)
	r *= mask
	d = r

	for i in range(niter):

		alpha = dotIm(r,r) / dotIm(d, applyLaplacian(d))
		x = x + np.multiply(alpha, d)
		r1 = r - np.multiply(alpha, applyLaplacian(d))
		r1 *= mask
		beta = computeConjugateDirectionStepSize(r, r1)
		d = r1 + np.multiply( beta, d )
		r = r1

	return x


#==== Helpers. Use them as possible. ==== 

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center) 
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center) 
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center) 
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])



