"""
ICL (In-Context Learning) examples for visual correspondence tasks.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two images of the same scene and a visual correspondence map (colored tracks/lines that link matching points between the two frames).  
The task: find which red-circled point in IMAGE 2 (options A/B/C/D) corresponds to the single reference point (REF) in IMAGE 1.

> [!IMPORTANT]
> The correspondence map shows visual flow. Follow the tracks/lines originating close to REF to see where they lead in IMAGE 2.

One example of how to use the visual correspondence map to solve the problem is given below.

Images: (reference frame with REF, target frame with A/B/C/D)
Correspondence map: Visualization with colored tracks connecting matching points across the two images
Procedure:
- Locate the reference point (REF) in IMAGE 1
- Examine the correspondence map and identify the tracks/lines that originate at or near REF
- Follow these tracks to see where they terminate in IMAGE 2
- Note the general direction and endpoint region of the tracks

- Compare each candidate point (A, B, C, D) in IMAGE 2:
  - Check which candidate is closest to where the REF tracks terminate
  - Tracks converging near a candidate indicate correspondence
  - Tracks pointing away from a candidate indicate non-correspondence

Example:
Images: (reference frame with REF, target frame with A/B/C/D)
- REF is located on a construction joint near the bottom-center of IMAGE 1
- Correspondence map shows tracks from REF running horizontally to the right
- These tracks terminate at the same joint region in IMAGE 2
- Candidate A: tracks near A head to the roof area → not aligned with REF region
- Candidate B: tracks near B head to the statue body → not aligned with REF region
- Candidate C: tracks from REF converge near C → aligned with REF region
- Candidate D: tracks near D head into the trees → not aligned with REF region

> [!IMPORTANT]
> THERE MAY NOT BE A TRACK IN THE EXACT POSITION OF THE POINT!!! USE THE CLOSEST ONE YOU CAN!!!

Answer: (C)

Now use this and solve the given question.
""".strip()

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two images of the same scene and two <perceptionprogram> blocks: the first one lists the coordinates of some markings in the two images; the second one lists correspondences between points in the two images.
The task: find which red-circled point in IMAGE 2 (options A/B/C/D) corresponds to the single
reference point (REF) in IMAGE 1.

> [!IMPORTANT]
> DO NOT USE EUCLIDIAN DISTANCE BETWEEN THE COORDINATES OF DIFFERENT IMAGES TO CONCLUDE INFORMATION ABOUT WHICH IS CLOSER TO WHICH!!! THIS WILL ONLY WORK IF THERE IS LITTLE MOVEMENT BETWEEN THE IMAGES, WHICH WILL NOT ALWAYS BE THE CASE

One example of how to use perception program to solve the problem is given below.

Images: (reference, target)
PP 1:
    Image 1: REF at (546,795)
    Image 2: candidates at A(46,68) B(142,849) C(547,797) D(765,186);
PP 2: Multiple correspondences between points in the two images listed as items. For each item, there are two pairs of coordinates: the coordinates in image 1 (c) and their equivalent in image 2 (r).
Procedure:
- Find the reference point (REF) in PP1
- Iterate through the items of PP2. Look for the c fields, they have positions in the reference image
- For each of them, calculate the Euclidian distance to REF using the coordinates from before: PP1 REF and PP2 c
- Find out one point that is close to REF. This is the neighbour of REF.

> [!IMPORTANT]
> DO NOT LOOK FOR THE EXACT POINTS IN THE CORRESPONDENCE PERCEPTION PROGRAM 2!!! THEY WILL NOT ALWAYS BE THERE!!! LOOK FOR A CLOSE ONE!!!

> [!IMPORTANT]
> DO NOT EXHAUSTIVELY CHECK THE POINTS. THIS CAN BLOW UP COMPUTATION. FIND ONE NEIGHBOUR THAT SEEMS CLOSE ENOUGH.
> THERE MAY NOT BE ANY POINT VERY CLOSE TO THE REF. IN THAT CASE, USE THE CLOSEST YOU CAN.

- Considering the neighbour, look at the r field to see where it ended up in the second image
- Go back to PP 1. Now iterate the points from this PP which live in the second image: A, B, C, D
- For each of them, calculate the Euclidian distance to the neighbour you found before 
- The candidate with the smallest distance is the point in image 2 that corresponds to REF. In this example, it is C.
Answer: (C)

Now use this and solve the given question.
""".strip()

