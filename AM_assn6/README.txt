How long did the assignment take?
14 Hours


Potential issues with your solution and explanation of partial completion (for partial credit)
Originally I had trouble computing translations and homographies.  In my stitch function, I noticed that if I took the inverse of the homography, I was able to get the right result.  I wasn't sure why this was the case and looked to see if I accidentally switched source and out image in the parameters.  This may have been related to how I computed the translations and debated whether I should take the negative or positive ty, tx values.  Luckily, I was able to get good results for the undergraduate version of the assignment. 

When I attempted the 6.865 version, I ran into some trouble.  There is currently an issue with my StitchN function, in particular compositeNImages.  After much thought, I understood how to compute the global homographs and chaining them together (take a look at the example 4 image in the comment section of my computeNHomographies function).  What I was confused about was how to actually composite my images now having a homography in global space for each image.  Since our original stitch function accepted only a listOfPairs, I introduced another function in the form stitchH that took in a homography computed from before.  However, similar to stitch I took the inverse.  My thoughts were to chain the stitched images similar to how we chained the homographies.  output = stitchH( image[i], output, H_global[i])  however, this gave me some issues, especially when the first mage wasn't reference.

My attempt to compute stitchN is incomplete, however, feedback and psuedocode of the solution would be very helpful.  I felt like it was very trivial and would like to know what the correct implementation was.   


Any extra credit you may have implemented
- computeHomography accepts >4 multiple pairs of points to construct a homography.
- computeHomography uses SVD to solve system of equations
- Completed parts of 6.865 assignment
	- applyHomographyFast by using bounding box of source image
	- Stitch N Images: Able to properly construct global homographs in computeNHomographies, however, had trouble stitching them together properly in compute N homographies. 


Collaboration acknowledgement (but again, you must write your own code)
None, did all on my own (with help from piazza and 6.815 staff of course)


What was most unclear/difficult?
Stitching N images together was more confusing than I though it should have been.  For example, I overlooked the fact that computeNHomographies was suppose to produce a homography for each image as a "global" reference to the reference image.  I didn't notice the issue until I had to started composite N Images (Mainly, how to properly translate on the global level and when to properly use the inversed homography).  I also had trouble stitching all the images together after computing all the homographs.


What was most exciting?
I loved applying out own homography and panoramas.  I put a sign on the prudential in Fun.png and stitched two images of the MIT Dome in MyPano.png


For 6.865 what was the speed-up from using bounding boxes?
About 3x faster, but did not do extensive testing