"""
ICL (In-Context Learning) examples for relative depth estimation.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given one image containing multiple labeled points (A, B, C, …) and a depth map visualization showing different regions with color-coded depth information.  
The task: determine which labeled point (A/B/C/…) is closest to the camera.

> [!IMPORTANT]
> WHITE INDICATES CLOSER REGIONS, WHILE BLACK INDICATE FARTHER REGIONS.

One example of how to use the depth map to solve the problem is given below.

Image: (scene with labeled points)
Depth map: Visualization showing depth regions with a black (far) to white (close) colormap
Procedure:
- Identify each labeled point (A, B, C, etc.) in the original image
- For each labeled point, examine the depth map visualization at the corresponding location
- Note the color or depth value at each point's location:
  - White regions = closer to camera
  - Darker regions = farther from camera
- Compare the relative depth values or color intensities at each labeled point
- The point in the more white region is closest to the camera

Example:
- Point A: Located in a light/white region → appears closest to the camera  
- Point B: Located in a dark/black region → appears farthest from the camera  
- Point C: Located in a gray region → appears in between

Comparing visual indicators:  
A (white) > C (gray) > B (black)

Answer: (A)

Now use this and solve the given question.
"""

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given one image containing multiple labeled points (A, B, C, …) and two <perceptionprogram> blocks:  
the first one lists the coordinates of the labeled points, and the second one lists patch centroids along with their depth ranges.  
The task: determine which labeled point (A/B/C/…) is closest to the camera.

> [!IMPORTANT]
> The comparison of closeness is based on **depth values**, not 2D coordinate positions or patch ids.  
> Use the **mean of the depth range** (`r`) from the nearest patch to each labeled point to decide which is closest.

One example of how to use the perception programs to solve the problem is given below.

Image: (scene with labeled points)
PP 1:
    Labeled points: A(620,619) B(674,338) C(574,554);
PP 2: Multiple patches in the scene listed as items. For each item, there are three main fields:  
       the patch id (p); the centroid coordinates of the patch (c); and its depth range (r).  
Procedure:
- Parse PP1: each item represents one labeled point; the `b` field is the label (A/B/C/…) and the `c` field gives its coordinates.
- Parse PP2: each item represents one patch; the `c` field gives the centroid coordinates, and the `r` field gives its depth range `[min, max]`. (Ignore the `p` field. It is just the id)

- For each labeled point in PP1:
  - Iterate through all patch items in PP2.
  - Compute the **Euclidean distance** between the labeled point’s coordinates (`c` in PP1) and each patch’s centroid coordinates (`c` in PP2).
  - Find the **closest patch** (the one with the smallest Euclidean distance).
  - Record the depth range `r` from that patch.

> [!IMPORTANT]
> DO NOT use the patch ID or label to infer proximity — only use centroid coordinates to find the nearest patch.

> [!IMPORTANT]
> The depth range `[r_min, r_max]` represents how far the patch lies in normalized depth space (0 = far, 1 = near).  
> Use the **mean depth** `(r_min + r_max)/2` as the depth estimate for comparison.

- After evaluating all labeled points:
  - Compute the **mean depth** of the nearest patch for each point.
  - The point with the **highest mean depth** value is the one **closest to the camera.**

Example:
- For A(620,619): closest patch = 67 → `r: [0.251, 0.349]` → mean = 0.300  
- For B(674,338): closest patch = 37 → `r: [0.0745, 0.1451]` → mean = 0.110  
- For C(574,554): closest patch = 56 → `r: [0.1529, 0.251]` → mean = 0.202

Comparing mean depths:  
A (0.300) > C (0.202) > B (0.110)

Answer: (A)

Now use this and solve the given question.
"""



__all__ = ['ICL_EXAMPLE_PP', 'ICL_EXAMPLE_TOOL']

