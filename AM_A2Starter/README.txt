===============================================
Name: Andrew Moran
MIT Email: andrewmo@mit.edu
6.815: Assignment 2 - Resampling, Warping and Morphing
9/18/2013
===============================================

1) The assignment took about 10 hours
   (Broken up between 2-3 days)


2) Issues with solution
   - scaleNN(im, k) may be partially correct.  Got nearest neighbor by diving 
     scaled y,x coord by k factor (hoping that dividing as int would round).  
     Seemed to work fine for me
   - dist(self, X) could have been a simpler calculation.  Needed to calculate
     point X to line segment (even when not perpendicular to line).
   - added new function in segment class perpendicular2D to calculate vector
     perpendicular to one given (chose -90 degree angle) 
   - hardcoded a,b,p values in weight(s, X).  Therefore weight will not change if
     change a,b,p in warp or morph functions  


3) Extra Credit
   - Completed rotate (even though not in grad version)
   - Completed Bicubic Interpolation using 16 neighbors
   - Completed Biquadratic Interpolation using 9 neighbors
   - Used ffmpeg to make out.mp4 movie file of png werewolf images where N=3


4) Collaboration acknowledgement
   I worked on this assignment alone, however, there were online resources other than
   those provided in the class that were much help
   - Rotation : http://answers.yahoo.com/question/index?qid=20120611124036AA7rVPI
   - Dist : http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
   - Interpolation: http://www.fho-emden.de/~hoffmann/bicubic03042002.pdf


5) Warping gave me trouble at first.  It would have been easier knowing that a,b,p had
   to be hardcoded in the weight function.  After implementing this, warping was still
   giving incorrect rests and I pinpointed it down to the dist(self, X) function.  I
   wasn't sure how to account for a point whose component did not lie on a line segment.
   More clarification would have been nice.


6) Morphing faces together was very exciting, especially making the movie out of png images.
   Overall, fun assignment
