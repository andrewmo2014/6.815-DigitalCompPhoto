===============================================
Name: Andrew Moran
MIT Email: andrewmo@mit.edu
6.815: Assignment 3 - Convolution and the Bilateral Filter
9/25/2013
===============================================

1) The assignment took about 10 hours
   (Broken up between 2-3 days) AND lots of debugging


2) Issues with solution
   - I renamed a3_starter.py to a3.py so it can be read in a3_test.py file
   - I imported imageIO.py file in same directory because MyImageIO was not working properly with scipy.misc
   - I followed same format of having Input/Output folders for testing the images
   - bilateral filter may be slow but was close to the results/pictures on pset and submission site.
   - YUV bilateral filter may be much slower than the original bilateral filter because broken up
     into Y, U, V filters individually.  Picture in output file is outdated version and hard to test since took
     long time to compute 


3) Extra Credit & Extra Work
   - Completed YUV Bilateral Filter (part of 6.865)


4) Collaboration acknowledgement
   I worked on this assignment alone, however, there were online resources other than
   those provided in the class that were much help
   - Bilateral Filter: http://cs.brown.edu/courses/cs129/lectures/bf_course_Brown_Oct2012.pdf
   - Gaussian: http://en.wikipedia.org/wiki/Gaussian_filter


5) Bilateral filter gave me the most trouble.  Initially, it was due to the fact that I was working
   on an older version of the pset and the equations given were not fully updated.  I think perhaps
   a better explanation of the bilateral filter would have been helpful in the handout.  I had to go
   to office hours to better understand everything.  There was a lot of info that was not straight-
   forward that had to be accounted for such as the number of "neighborhood" pixels and extracting the 
   weights by multiplying color difference value by gaussian value.  Also, since it takes a while to 
   compute the filter, it was very time-consuming while I was testing it.


6) Seeing the bilateral filter working properly was cool.  I also liked the result of the gradient
   magnitude of an image.


7) Gaussian filters: separable vs. 2D
   {A1: 7.48745012283 seconds } {A2: 41.5839231014 seconds}
