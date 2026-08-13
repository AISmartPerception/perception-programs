"""
ICL (In-Context Learning) examples for visual correspondence / multi-view reasoning tasks.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two frames from a static scene and an optical flow map between them.
Decide the global camera motion:
(A) left    (B) right

> [!IMPORTANT]
> The optical flow map visualizes motion between frames using color coding. Warm hues (red/orange/yellow) typically indicate rightward motion (positive horizontal flow), while cool hues (blue/cyan) indicate leftward motion (negative horizontal flow). The dominant color pattern across the scene reveals the overall camera motion. Additionally, notice that the question may sometimes say clockwise or counter-clockwise. Clockwise here is used interchangeably with `left`, while counter-clockwise with `right`.

DO NOT USE THE COLOR AS IF IT WAS OBJECT MOVEMENT!!! IT REPRESENTS CAMERA MOVEMENT!!!
DO NOT REVERSE THE LOGIC!!! IF THE COLORS INDICATE LEFT, THIS MEANS THAT THE **CAMERA** MOVED LEFT, NOT THE POINTS. DO NOT CONCLUDE THAT THE CAMERA MOVED IN THE OPPOSITE DIRECTION AS THE COLORS!!!

One example of how to use the optical flow to solve the problem is given below.

Images: (beginning frame, end frame)
Flow: dominated by warm hues (red/orange/yellow) across the scene, indicating positive horizontal flow (rightward).
Justification: The optical-flow map shows global rightward motion (positive u), so the camera moved to the right.
Answer: (B)

Now use this and solve the given question.
""".strip()

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You're given two frames from a static scene and a <perceptionprogram> block that lists per-patch horizontal motion
("left" or "right") on a 10x10 grid. Decide the global camera motion:
(A) left    (B) right

> [!IMPORTANT]
> All items in the perception program are relative to patches in the first image (i.e. the one taken before). Each patch has either left or right written in its properties, indicating what will happen to the camera. This value is already manipulated such that the information there is about camera motion, not patch motion or motion from objects in the scene. So, if many patches have "right", this means that the camera is likely moving right. The same logic can be applied to left movement. Additionally, notice that the question may sometimes say clockwise or counter-clockwise. Clockwise here is used interchangeably with `left`, while counter-clockwise with `right`.

DO NOT USE THE MOTION UNDER PATCH AS IF IT WAS PATCH MOVEMENT!!! IT REPRESENTS CAMERA MOVEMENT!!!
DO NOT REVERSE THE LOGIC!!! IF THE PATCHES SAY LEFT IN THE PP, THIS MEANS THAT THE **CAMERA** MOVED LEFT, NOT THE PATCHES. DO NOT CONCLUDE THAT THE CAMERA MOVED IN THE OPPOSITE DIRECTION AS THE MAJORITY!!!

One example of how to use perception program to solve the problem is given below.

Images: (beginning frame, end frame)
Perception: more patches are pointing towards "right" than towards "left"
Justification: The rows overwhelmingly show 'right' motion and overall the grid has a right majority (roughly 60+ vs 40), indicating global camera motion to the right.
Answer: (B)

Now use this and solve the given question.
        """.strip()

