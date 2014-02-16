===============================================
Name: Andrew Moran
MIT Email: andrewmo@mit.edu
6.815: Assignment 5 - High Dynamic Range Imaging and Tone Mapping
10/09/2013
===============================================

How long did the assignment take?
10 hours (debugging of course)

Potential issues with your solution and explanation of partial completion (for partial credit)
Pset was straight forward, however, there were some small things that I had trouble with that may have had my code produce a not so correct result.
	makeHDR: I had trouble getting the linear scale to work from chaining the darkest(first) to the brightest(last image).  I have been getting these green artifacts in areas that were overexposed.  Suggestions of sorting the input images had me try reversing the passed in imageList, which worked.  I thought this was still ok since chaining (regardless from first or last image) was all relative to the global scale factor.  Keep in mind, I also had to account for clipping of special cases in the darkest and brightest image.
	toneMap: followed the handout and get reasonable results, however, afraid may have minor mistakes in working with base and detail in log domain and then converting it to linear domain.
	Was not sure if 0.3R + 0.6G + 0.1B were the correct weights for luminance.
	Finally, I had trouble clipping images and weights that had zero pixels/values.  For black/white image, weightSum for HDR, and luminance log: I replaced all zeros with the smallest non-zero value (epsilon) so division and log operation could be used.  Not sure how this effects results.  

Any extra credit you may have implemented
None so far

Collaboration acknowledgement (but again, you must write your own code)
I did this all on my own,  I went to office hours to ask conceptual questions (especially about chaining k_i in relation to the globe scale factor for HDR)

What was most unclear/difficult?
When computing the scaling factor for HDR, I had trouble understanding how to chain at first.  I realized that you set the initial image to have a scale factor of 1.0 and you chain the scale factor to get the actual/global scale factor for the next image.
Also, wasn't sure exactly how the detailAmp parameter worked.  At first, I thought my bilateral filter was wrong since it produced very dark,spotty images when detailAmp was high, however, realized that was ok.

What was most exciting?
Getting tone map to work was pretty cool, especially for the cave images.
