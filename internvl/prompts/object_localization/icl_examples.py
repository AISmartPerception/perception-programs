"""
ICL examples for object localization task.
"""

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are shown an image with two candidate bounding boxes (A, B) and must decide which best localizes a target object. Two perception programs are provided:

1. **Bounding Box Coordinates** - corner coordinates for A and B  
2. **Detection Results** - detected objects with:  
   - p: detection ID  
   - c: bounding box [x0, y0, x1, y1]  
   - r: confidence (0-1)  
   - b: label  

> **Task:** Compare the overlap between each candidate box and relevant detections (matching or related label). The box with greater overlap and alignment is likely correct.

### Example
**Question:** Which box better localizes the tank top?  
**Choices:** (A) Box A, (B) Box B  

**Program 1:**  
- A: upper-right corner (x0=345, y0=368), lower-left corner (x1=631, y1=833)  
- B: upper-right corner (x0=289, y0=346), lower-left corner (x1=570, y1=859)

**Program 2:**  
- p1: b="shirt", c=[290, 350, 565, 855], r=0.85  
- p2: b="person", c=[200, 100, 700, 900], r=0.92  

Detection “shirt” aligns closely with Box B → **Answer: (B)**

### Measuring Overlap (IoU)
Use **Intersection over Union (IoU)** to measure overlap:

```
IoU = Area(Intersection) / Area(Union)
```

Compute intersection boundaries between boxes A and B:
```
x_left   = max(A_x0, B_x0)  
y_top    = max(A_y0, B_y0)  
x_right  = min(A_x1, B_x1)  
y_bottom = min(A_y1, B_y1)
```

If x_right < x_left or y_bottom < y_top → IoU = 0  
Else:
```
intersection_area = (x_right - x_left) * (y_bottom - y_top)  
union_area = Area(A) + Area(B) - intersection_area  
IoU = intersection_area / union_area
```

Higher IoU → better localization.

""".strip()

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are shown an image with two candidate bounding boxes (A and B) marked, and a visualization showing detected objects with bounding boxes and labels.

> [!IMPORTANT]
> The original image shows the two candidate bounding boxes.
> The detection visualization shows all detected objects with colored bounding boxes and labels.
> Compare the overlap between each candidate bounding box and the detections to determine which candidate better localizes the target object.

One example of how to use the visualization to solve an object localization problem is given below.

Question: Which bounding box more accurately localizes and encloses the tank top (clothing)?
Choices: (A) Bounding box A, (B) Bounding box B

Visualization: Image with detected objects highlighted with colored bounding boxes and labels (e.g., "shirt", "person").

Procedure:
1. Identify detections matching the target object in the visualization
2. Visually compare how well each candidate bounding box (A or B) aligns with relevant detections
3. Choose the bounding box with better overlap

Example Analysis: The detection labeled "shirt" aligns closely with bounding box B but is offset from bounding box A. Bounding box B better captures the detected clothing item.
Answer: (B)

Now use this pattern to solve the given question.
""".strip()

