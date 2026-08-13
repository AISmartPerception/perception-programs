"""
ICL (In-Context Learning) examples for semantic correspondence tasks.
These are adapted to InternVL3 prompt style (concise instructions + one worked example),
not Qwen-style meta-instructions.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two images of the same scene and a visual similarity map over IMAGE 2.
The task: pick which red-circled point in IMAGE 2 (A/B/C/D) corresponds to the source point (REF) in IMAGE 1.

> [!IMPORTANT]
> Brighter/hotter areas in the similarity map indicate stronger match to REF. Use it as your primary signal.

One example of how to use the similarity map to solve the problem is given below.

Images: (IMAGE 1 with REF, IMAGE 2 with A/B/C/D)
Similarity map: heat overlay on IMAGE 2
Procedure:
- In IMAGE 2, find the hottest/highest-intensity area on the similarity map
- See what candidate point among A/B/C/D is either in or close to that area
- If the hotspot falls closest to a candidate, that candidate is the answer

Example:
Images: (REF on a lamp edge in IMAGE 1; candidates over building façade in IMAGE 2)
- Similarity map shows a clear hotspot near the mid-right façade
- Candidate C lies at the hotspot center; A/B/D are far from high-intensity region
Answer: (C)

Now use this and solve the given question.
""".strip()

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two images and two <perceptionprogram> blocks: first has a source point in IMAGE 1 and second has a semantic similarity scores for points in IMAGE 2.
The task: pick which red-circled point in IMAGE 2 (A/B/C/D) corresponds to the source point (REF) in IMAGE 1.

One example of how to use perception program to solve the problem is given below.

> [!IMPORTANT]
> DO NOT USE THE COORDINATE POSITIONS (key c in perception program) TO DECIDE THE ANSWER!!! USE THE SIMILARITY SCORE (key r in perception program)!!! 

Images: (IMAGE 1 with REF, IMAGE 2 with A/B/C/D)
PP:
  - PP 1 (point-detection): REF coordinates in IMAGE 1
  - PP 2 (semantic): per-region similarity scores for IMAGE 2. Each item has label p, coordinates c and similarity r
Procedure:
- From PP 2, identify the point with highest similarity r
- Choose the candidate with the highest score as the answer

Example:
PP 2 indicates the highest semantic score is in the item listed as B.
Answer: (B)

Now use this and solve the given question.
""".strip()


