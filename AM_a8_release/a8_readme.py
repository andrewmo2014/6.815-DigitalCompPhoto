def time_spent():
  '''N: # of hours you spent on this one'''
  return 6

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  ppl = ['None']
  return ppl

def potential_issues():
  issues = """
  I think everything is working fine, only thing that confused me was covergence of poisson.
  I understand that poisson w/o CG could converge slower.
  Also, more iterations resulted in fg having same tint as bg.
  """
  return issues

def extra_credit():
  #```` Return the function names you implemended````
  #```` Eg. return ['full_sift', 'bundle_adjustment']````
  extras = ['fun_composite']
  return extras

def most_exciting():
  exciting = "Making poisson algorithms to work and having a custom composite"
  return exciting

def most_difficult():
  difficult = "Nothing that much, Pset was well explained.  Had to debug naive compositing though"
  return difficult

def my_composition():
  input_images=['data/boston-skyline.png', 'data/dog_frisbee.png', 'data/dog_frisbee-mask.png']
  output_images='my_poisson_CG.png'
  return (input_images, output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.
  '''
  images=['pru_blur_real25.png', 'psf25.png', 'pru_sharp_CG_real25.png', 'naiveDebugFG.png', 'naiveDebugBG.png', 'naiveDebugBGFG.png', 'my_naive_composite.png', 'my_poisson500.png', 'my_poisson2000.png', 'my_poisson_CG40.png']
  my_debug="""
  Understanding why flipping rather than transposing to get M^T was tricky but got it to work. Retried with 25 iterations.
  I initially had trouble getting the naive composite to work so I broke it up into BG, FG parts.
  When I first did my custom poisson composite of a dog catching frisbee in boston skyline, I noticed a tint.
  Writing images of different interations and with or w/o CG made it the covergence more clear.
  """
  return (my_debug, images)

#print time_spent()
#print collaborators()
#print potential_issues()
#print extra_credit()
#print most_exciting()
#print my_composition()
#print my_debug()
