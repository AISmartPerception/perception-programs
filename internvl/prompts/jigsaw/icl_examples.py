"""
ICL (In-Context Learning) examples for jigsaw tasks.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are given an image with a missing corner, two candidate pieces, and two stitched previews showing how each candidate would look if placed in the missing spot. Decide which candidate fits best with the original image.
(A) the second image  (B) the third image

> [!IMPORTANT]
> The stitched previews show complete images where each candidate piece has been placed into the missing region. Examine these previews carefully to see which stitched result appears most natural and consistent with the original image's patterns, edges, and colors.

One example of how to use the stitched previews to solve the problem is given below.

Images: (main image with missing corner, candidates A & B, and two stitched previews)
Stitched Preview Analysis: The stitched image with candidate B shows better alignment along the borders - the skyline, textures, and color transitions match seamlessly with the surrounding region.
Answer: (B)

Now use this and solve the given question.
""".strip()

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are given an image with a missing corner, two candidate pieces, and a <perceptionprogram> block listing compatibility scores on the borders (left and top) of the hole. Decide which candidate fits best:
(A) the second image  (B) the third image

> [!IMPORTANT]
> The perception program provides quantitative similarity metrics for each border of the hole. Each item shows:
> - p: which border ('left' or 'top')
> - i: candidate ID ("A" or "B")
> - r: similarity scores (structural, color, and edge similarity in [0,1])
> Higher scores indicate better fit. Compare the average scores across borders for each candidate.

One demonstrative example of how to use perception program to solve a certain problem is given below.

Images: (main image with missing corner, candidates A & B)
Perception: For this particular example, candidate B has higher average similarity scores (averaging across left and top borders), indicating that (B) is the correct answer.
Answer: (B)

Important Note: The answer above belongs to the example and must not be copied.
You will now receive a new question. For it compute your own choice and output the answer.
""".strip()

