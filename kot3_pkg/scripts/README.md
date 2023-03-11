From Chat GPT
If your LiDAR sensor returns an array of 360 values, then it is likely that the sensor has a field of view of 360 degrees, meaning it can capture data from all directions around it. In this case, the mapping between index and angle is likely to be evenly distributed around the full 360 degrees of the sensor's field of view.

Assuming that the LiDAR has a uniform distribution of beams, each beam will cover an angle of 1 degree (i.e., 360 degrees divided by 360 values in the array). Therefore, the angle corresponding to each index in the array can be calculated as follows:

Index 0 corresponds to an angle of 0 degrees (i.e., the center of the field of view)
Index 1 corresponds to an angle of 1 degree clockwise from the center
Index 2 corresponds to an angle of 2 degrees clockwise from the center
Index 3 corresponds to an angle of 3 degrees clockwise from the center
and so on, up to index 359, which corresponds to an angle of 359 degrees clockwise from the center.
It's important to note that this mapping assumes that the LiDAR has a uniform distribution of beams around its full 360-degree field of view. If the LiDAR has a non-uniform distribution of beams or its field of view is restricted in some way, then the mapping between index and angle may be different.