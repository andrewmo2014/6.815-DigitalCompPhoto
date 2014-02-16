===============================================
Name: Andrew Moran
MIT Email: andrewmo@mit.edu
6.815: Assignment 4 - Denoising and Demosaicking
10/02/2013
===============================================

Q1:How long did the assignment take?:
8 hours


Q2:Potential issues with your solution and explanation of partial completion (for partial credit):
- Offsets For Demosaicking: I noticed there was a discrepancy in the green Offset parameter (whether it was 0 or 1 and if it represented which top-left pixel was green)
I made my code so that it works for the default parameters (green Offset=1, red Offset = 1,1, blue Offset = 0,0)
- MaxOffset in Align: Wasn't sure if properly sliced images so that pixels that are MaxOffset away from boundary
or less are not calculated in norm squared.
- Sergie Alignment & Cropping:  When splitting image into thirds, there may be some pixels not accounted for.  I wasn't sure at first the order from top to bottom of Sergie images (BGR).  
- SNR: there was the possibility of having zero variance, clipped it to .000001.


Q3:Any extra credit you may have implemented:
Just the requirements so far


Q4:Collaboration acknowledgement (but again, you must write your own code):
No help from other students in the class, however, took advantage of piazza and office hours.
Also used this website to help me understand demosaicing: http://www.unc.edu/~rjean/demosaicing/demosaicing.pdf


Q5:What was most unclear/difficult?:
Most was explained in the potential issues.  Mainly, not sure which green Offset to use and how it affected overall demosaicking.  For align, more clarification was needed for using only the inner frame of two images to calculate error and not the outer border containing maxOffset pixels.  (Perhaps more insight on numpy.roll could have been useful).  Also, had trouble at first for SNR with zero variance (fixed with clipping to very small value). 


Q6:What was most exciting?:
Sergie images were cool, using 3 channels to make a colored image.


Q7:What ISO has better SNR?:
ISO 3200 was better, perhaps because there was just more coverage compared to the other ISO


Q8:Which direction you decided to interpolate along for the edgeBasedGreenChannel?:
Whatever direction had the smallest gradient.  The direction with a large gradient determines an edge.  Wanted to interpolate similar pixels along that edge (hence taking the smallest gradient)

