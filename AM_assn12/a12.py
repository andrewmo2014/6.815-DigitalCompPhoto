import imageIO as io
import numpy as np
from scipy import ndimage
from scipy import signal
import glob

io.baseInputPath = './'

def getPNGsInDir(path):
	'''gets the png's in a folder and puts them in out.'''
	fnames = glob.glob(path+"*.png")
	out=[]
	for f in fnames:
		imi = io.imread(f)
		out.append(imi)
	return out

def convertToNPY(path, pathOut):
    '''converts the png images in a path path to a npy file at pathOut'''
    L=getPNGsInDir(path)
    V=np.array(L)
    np.save(pathOut, V)

def writeFrames(video, path):
    '''writes the frames of video to path.'''
    nFrame=video.shape[0]	
    for i in xrange(nFrame):
        pathi=path+str('%03d'%i)+'.png'
        #if i%10==0: print i 
        io.imwrite(video[i], pathi)
    print 'wrote'+path+'\n'

def RGB2YUV(video):
    '''Convert an RGB video to YUV.'''
    RGB2YUVmatrix=np.transpose([[0.299,  0.587,  0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]])
    return np.dot(video[:, :, :], RGB2YUVmatrix)


def YUV2RGB(video):
    '''Convert an YUV video to RGB.'''
    YUV2RGBmatrix=np.transpose([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    return np.dot(video[:, :, :], YUV2RGBmatrix)

##################### Write the functions below ##############


def lowPass(video, sigma):
    '''This should low pass the spatial frequencies of your video using a gaussian filter with sigma given as the second input parameter.'''
    return ndimage.filters.gaussian_filter( video, [0, sigma, sigma, 0])


def timeBandPass(video, sigmaTlow, sigmaThigh):
    '''Apply a band pass filter to the time dimension of the video.
    Your band passed signal should be the difference between two gaussian
    filtered versions of the signal.
    '''
    return ndimage.filters.gaussian_filter(video, [sigmaThigh, 0, 0, 0]) - ndimage.filters.gaussian_filter(video, [sigmaTlow, 0, 0, 0])


def videoMag(video, sigmaS, sigmaTlow, sigmaThigh, alphaY, alphaUV):
    '''returns the motion magnified video. sigmaS is the sigma for your
    spatial blur. sigmaTlow is the sigma for the larger termporal gaussian,
    sigmaThigh is the sigma for the smaller temporal gaussian. alphaY is
    how much of the bandpassed Y signal you should add back to the video.
    alphaUV is how much of the bandpassed UV signal you should aff back to
    the video.
    You should use lowPass() to apply your spatial filter and timeBandPass()
    to apply your time filter.'''
    yuv = RGB2YUV(video)
    lowFreq = lowPass(yuv, sigmaS)
    out = timeBandPass(lowFreq, sigmaTlow, sigmaThigh)
    out[:,:,:,0]*=alphaY
    out[:,:,:,1:2]*=alphaUV
    final = YUV2RGB(out+yuv)
    return final
    

def timeBandPassButter(video, low, high, order):
    '''    
    B,A = signal.butter(order, [low, high], 'bandpass')
    gives the coefficients used in the butterworth iir filter.
    for a input signal x, the filtered output signal is given
    by the recursion relationship:
    
    A[0]*y[n]= -A[1]*y[n-1]
               -A[2]*y[n-2]
               -A[3]*y[n-3]
                 ...(up to the number of coefficients, which depends on 'order')
               +B[0]*x[n]
               +B[1]*x[n-1]
               +B[2]*x[n-2]
               +B[3]*x[n-2]
               ...(up to the number of coefficients, which depends on 'order')
    '''
    B,A = signal.butter(order, [low, high], 'bandpass')
    out = np.zeros_like(video)
    numberF = video.shape[0]

    for n in xrange(numberF):
    	for i in range(1, A.shape[0]):
    		frame = n-i
	    	if frame < 0:
	    		frame = 0
    		out[n] += (-A[i]*out[frame] + B[i]*video[frame])
    	out[n] += B[0]*video[n]
    	out[n] /= A[0]

    return out


def videoMagButter(video, sigmaS, low, high, order, alphaY, alphaUV):
    '''video magnification using the butterworth iir filter to
    bandpass the signal over time instead of 
    '''
    yuv = RGB2YUV(video)
    lowFreq = lowPass(yuv, sigmaS)
    out = timeBandPassButter(lowFreq, low, high, order)
    out[:,:,:,0]*=alphaY
    out[:,:,:,1:2]*=alphaUV
    final = YUV2RGB(out+yuv)
    return final


def main():
    print 'Yay for computational photography!'
    #convertToNPY('face/face', 'face.npy')
    #return
    v=np.load('Input/face.npy')
    print '    done loading input file, size: ', v.shape
    #out=videoMag(v, 10, 20, 4, 40, 40)
    #writeFrames(out, 'videoOut/frame')
    out=videoMagButter(v, 10, 0.11, 0.134, 2, 2, 50)
    #out=videoMagButter(v, 10, 0.06, 0.067, 2, 50, 50)
    writeFrames(out, 'videoOut/frameButter')


#the usual Python module business
if __name__ == '__main__':
    main()

