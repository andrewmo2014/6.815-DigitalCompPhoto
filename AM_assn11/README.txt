Name: Andrew Moran
MIT Email: andrewmo@mit.edu

Q1:How long did the assignment take?:
Many, many hours (20-30).  Thanks for the extension


Q2:Potential issues with your solution and explanation of partial completion (for partial credit):
	There may be some discrepancy from part I and part II due to the code being updated on stellar.  luckily part I only consisted of tutorials so was able to adjust.  I tried to write the correct python equivalent code for box schedules 5-7 however got confused with compute_at and correct process of producer and consumer (newly posted video helped).  Had trouble figuring out where to put the producer code and how much of the producer was needed so it would be ready for the consumer.
	LocalMax - had trouble clamping and using & rather than 'and', now should work
	GaussianSingleChannel - created horizontalGaussianKernel mathematically, did two blurs along the x, then the y.  Since using sum sugar, divided by kernel_width**2.  Hope this was the correct implementation, got reasonable results.
	Harris - wasn't sure when to clamp, i did it immediately when got image.
had trouble to think of best scheduling for sobel filter and wasn't sure whether to divide by sober.width()**2 since using sum sugar.
Had trouble getting the correct corner output.  Fixed by calculating threshold from response, then localMaximum from response, and finally anding them together.
Scheduling was difficult.  My default scheduling just used compute_root() for all functions (pretty fast).  Made it faster by using tiling and parallelism on the GaussianSingleChannel (tiles of size 256*128 from tutorial6).


Q3:Anything extra you may have implemented:
None added


Q4:Collaboration acknowledgement (but again, you must write your own code):
None, got everything I needed from office hours/piazza


Q5:What was most unclear/difficult?:
Scheduling and compute_at.  Had trouble understanding how it is done in Halide.  However, managed to understand most of it to complete the pset effectively.


Q6:What was most exciting?:
Getting harrisCorners to work properly.


Q7: How long did it take for the 2 schedules for the smooth gradient? 
Tutorial 5 Schedule Smooth Gradient
default time:  		0.0282158851624
gradient fast time:	0.0471119880676
acceleration factor:  	1.66969732817
*Note: gradient fast was actually slower


Q8: Speed in ms per megapixel for the 4 schedules (1 per line)
Tutorial 6 Box Schedule (Summary)
schedule 1, ROOT:		11.28591 ms per megapixel
schedule 2, INLINE:		5.31954 ms per megapixel
schedule 3, TILING:		7.16045 ms per megapixel
schedule 4, TILE & PARALLEL:	2.83702 ms per megapixel

Full printout for reference
"""
 schedule 1, ROOT:
best:  0.276926040649 average:  0.403233194351
11.28591 ms per megapixel (403.2331944 ms for 35 megapixels)

 schedule 2, INLINE:
best:  0.177377939224 average:  0.19006152153
5.31954 ms per megapixel (190.0615215 ms for 35 megapixels)
speedup compared to root: 1.56

 schedule 3: TILING
best:  0.246336936951 average:  0.255835199356
7.16045 ms per megapixel (255.8351994 ms for 35 megapixels)
speedup compared to root: 1.12

 schedule 4: TILE & PARALLEL
best:  0.0957159996033 average:  0.101363801956
2.83702 ms per megapixel (101.3638020 ms for 35 megapixels)
speedup compared to root: 2.89
"""


Q9: What machine did you use (CPU type, speed, number of cores, memory)
MacBook Pro, 15-inch, Mid 2010
Processor: 2.66 GHz Intel Core i7
Memory: 4GB 1067 MHz DDR3


Q10: Speed for the box schedules, and best tile size
Tutorial 10 Schedule Convolution (Summary)
-->Times
default schedule
           took  5.62193160057 seconds

root first stage
           took  4.2982629776 seconds

tile  256 x 256  + interleave
           took  1.49779081345 seconds

tile  256 x 256 + parallel
           took  2.20879440308 seconds

tile  256 x 256  + parallel+vector without interleaving
           took  2.62455501556 seconds

-->Tile Sizes
tile  64 x 128 + parallel
           took  0.859215974808 seconds

tile  128 x 128 + parallel
           took  1.52214121819 seconds

tile  256 x 128 + parallel
           took  0.832024383545 seconds

tile  512 x 128 + parallel
           took  1.50865597725 seconds

tile  1024 x 128 + parallel
           took  0.911333227158 seconds

-->Best tile size: 256 x 128


Q11: How fast did Fredo’s Harris and your two schedules were on your machine?
FredoNumpy harris took 	    61.8687551022  seconds
		       	    1730.59455 ms per megapixel (61868.7551022 ms for 35 megapixels)
scheduleIndex0  harris took 115.734651804  seconds
			    3237.03276 ms per megapixel (115734.6518040 ms for 35 megapixels)
scheduleIndex1  harris took 44.6555728912  seconds
			    1248.99112 ms per megapixel (44655.5728912 ms for 35 megapixels)


Q12: Describe your auto tuner in one paragraph
6.815 - If I were to implement a functional auto tuner, I would follow the same process as tutorial10_convolution scheduling.  I would focus on my gaussian and sobel convolutions because here I can take advantage of locality.  Then I would try different tile sizes other than my best 256*128.

Q13: What is the best schedule you found for Harris. 
6.815 - I just tried a schedule that took advantage of parallelism and locality.  I am sure I can do better.  I thought that focusing on how to schedule GaussianSingleChannel similar to tutorial10_convolution, I could get better results.  By using my best tile_size of (256*128), I was able to tile and take use of parallelism to get faster results.  As explained in my auto tuner, I would try different schedules as well as tile sizes.