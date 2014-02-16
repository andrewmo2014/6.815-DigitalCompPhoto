import sys
sys.path.append('utils')
import imageIO as io
import numpy as np
import a5
import multiprocessing as multi



def test_computeWeight():
  im=io.imread('data/design-2.png')
  out=a5.computeWeight(im)
  io.imwrite(out, 'design-2_mask.png')

def test_computeFactor():
  im2=io.imread('data/design-2.png')
  im3=io.imread('data/design-3.png')
  w2=a5.computeWeight(im2)
  w3=a5.computeWeight(im3)
  out=a5.computeFactor(im2, w2, im3, w3)
  if abs(out-50.8426955376)<1 : 
    print 'Correct'
  
def test_makeHDR():
  import glob, time
  inputs=sorted(glob.glob('data/ante0-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_ante0.png')
  

def test_toneMap():
  hdr=np.load('hdr.npy')
  out1=a5.toneMap(hdr, 100, 1, useBila=False)
  io.imwrite(out1, 'tone_map_ante0G.png')
  out2=a5.toneMap(hdr, 100, 3, useBila=True)
  io.imwrite(out2, 'tone_map_ante0BF.png') 


def test_makeHDR_rest():
  import glob, time

  #Image 1
  inputs=sorted(glob.glob('data/ante1-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr1', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_ante1.png')

  #Image 2
  inputs=sorted(glob.glob('data/ante3-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr3', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_ante3.png')

  #Image 3
  inputs=sorted(glob.glob('data/design-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_design', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_design.png')

  #Image 4
  inputs=sorted(glob.glob('data/horse-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_horse', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_horse.png')

  #Image 5
  inputs=sorted(glob.glob('data/sea-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_sea', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_sea.png')

  #Image 6
  inputs=sorted(glob.glob('data/nyc-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_nyc', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_nyc.png')

  #Image 7
  inputs=sorted(glob.glob('data/stairs-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_stairs', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_stairs.png')

  #Image 8
  inputs=sorted(glob.glob('data/vine-*.png'))
  p=multi.Pool(processes=8)
  im_list=p.map(io.imread,inputs)

  hdr=a5.makeHDR(im_list)
  np.save('hdr_vine', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale_vine.png')

def test_toneMap_rest():
  hdr=np.load('hdr1.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_ante1G.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_ante1BF.png')   

  hdr=np.load('hdr3.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_ante3G.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_ante3BF.png') 

  hdr=np.load('hdr_design.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_designG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_designBF.png') 

  hdr=np.load('hdr_horse.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_horseG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_horseBF.png')   

  hdr=np.load('hdr_sea.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_seaG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_seaBF.png') 

  hdr=np.load('hdr_nyc.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_nycG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_nycBF.png') 

  hdr=np.load('hdr_stairs.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_stairsG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_stairsBF.png')   

  hdr=np.load('hdr_vine.npy')
  out1=a5.toneMap(hdr, useBila=False)
  io.imwrite(out1, 'tone_map_vineG.png')
  out2=a5.toneMap(hdr, useBila=True)
  io.imwrite(out2, 'tone_map_vineBF.png') 


# Uncomment the below to test your code

#test_computeWeight()
#test_computeFactor()
#test_makeHDR()
#test_toneMap()
#test_makeHDR_rest()
test_toneMap_rest()

