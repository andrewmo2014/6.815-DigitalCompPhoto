#===============================================
#Name: Andrew Moran
#MIT Email: andrewmo@mit.edu
#6.815: Assignment 7 - Automatic Stitching and Blending
#10/30/2013
#===============================================


def time_spent():
  '''N: # of hours you spent on this one'''
  return 20

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  ppl = ['None']
  return ppl

def potential_issues():
  a_string = '''
  At first, I thought I had some potential issues with my RANSAC and thought I was missing
  corners and getting the getting the incorrect correspondences.  However after much work, 
  I was able to finally able to get it to work properly, knowing that it was ok if it was not perfect.
  Some issues came for linear and two-scale blending.  At first I had trouble with the weights,
  but then debugged to see that they were fine.  For linear blending, I noticed that there was noticeable
  ghosting and things seemed blurry.  I thought this was ok, however, I also noticed that after doing
  two-scale blending, seems got a bit better but not perfect.  In some cases, they seemed blurry.
  Not sure what the issue was, I feel that I may have calculated the per-pixel
  weight incorrectly somewhere.

  Other concerns were dividng by zero, especially with the wieghts.  When doing tests with my new images,
  my code was able to still run, however, I had RuntimeWarnings.  Also, I changed a6 code and copied it into the a7
  file to get automatic stitching and blending to work properly.

  '''
  return a_string

def extra_credit():
#```` Return the function names you implemended````
#```` Eg. return ['full_sift', 'bundle_adjustment']````
  fun_list = ['6.865_probablistic_terminatiion', '6.865_autoStitchN', 'least_square_adjustment']
  return fun_list


def most_exciting():
  exciting = "Blending the two images in the panorama together after stitching, very satifying"
  return exciting

def most_difficult():
  difficult = "It was hard at first to get the RANSAC to work with the correct correspondences.  After much work, I was able to finally get it to work.  Also, blending required a lot of debugging, especially with the weights."
  return difficult

def my_panorama():
  input_images=['mit0.png', 'mit1.png']
  output_images=['panoramaMine.png', 'linear_blendingMine.png', 'two_scale_blendingMine.png']
  return (input_images, output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.

  Eg
  images=['debug1.jpg', 'debug2jpg']
  my_debug='I used blahblahblah...
  '''

  images=['debugWeightsLow.png', 'debugWeightsHigh.png']
  my_debug = "I had trouble calculating and maintainting the weights properly, epecially when I got into two-scale blending.  I simply debugged by printing out the image of the summedWeight for lowFrequencies and maxedWeight for highFrequencies"

  return (my_debug, images)

#print time_spent()
#print collaborators()
#print potential_issues()
#print extra_credit()
#print most_exciting()
#print most_difficult
#print my_panorama()
#print my_debug()
