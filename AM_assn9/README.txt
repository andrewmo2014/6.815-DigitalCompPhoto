Name: Andrew Moran
MIT Email: andrewmo@mit.edu

Q1:How long did the assignment take?:
10 hrs, lots of debugging for computeAngles(im)

Q2:Potential issues with your solution and explanation of partial completion (for partial credit):
Initially had trouble with getting the right thetas for computeAngles(im).  Now it should work.  After getting the larger eigenVector from np.linalg.eigh, I added pi to the computed angle to get the angle of the smaller eigenvector (which is rotated by 90 degrees).  I also had some trouble with rejection sampling.  Using the importance map and accepting probabilities, I kept adding strokes until I reached my desired number of strokes.

Q3:Anything extra you may have implemented:
6.865 oriented paintings

Q4:Collaboration acknowledgement (but again, you must write your own code):
None

Q5:What was most unclear/difficult?:
Having two computeTensor functions was a bit confusing.  Also, it was unclear at first how to get the smaller eigenvector.  I initially tried to extract the eigenvector with the smaller eigenvalue and then use arctan2 to compute the angle.  Not sure why that approach didn't work.  When I extracted the larger eigenvector, computed its angle, but then rotated by 90 degrees to get the angle of the smaller eigenvector, it worked.  Very confusing and hard to debug. 

Q6:What was most exciting?:
Getting the Oriented Painterly to work with correct angles.  Different brush strokes look so cool!
