## Solution Requirements
- PyGLM (pip install pyglm)
- NumPy (pip install numpy)
- OpenMesh (pip install openmesh)
- Numpy-STL (pip install numpy-stl)

Run the script 'skirting.py' in directory with 'Part.stl'; the output, 'input_skirted.stl', will be written to the same dir.

<p align="center">-----</p>

# CompGeo_Add_Skirt
## Objective
Modify a shape, a 3D geometry, so that it can be "formed" using our Dual Sided Incremental Forming (DSIF) process. 
<p align="center"><img src=https://user-images.githubusercontent.com/91622575/174198770-e7d3b0c8-156c-4757-8b05-1a29f07e73e5.png></p>

## Context
At Machina Labs, we use our robots to push, stretch, bend and fold flat sheets of metal into various shapes. This "forming" process results in the originally flat sheet of metal, now having a 3D shape protruding out of it. Typically, the desired "part" is only a portion of the protruding shape, and which eventually gets cut out of it. 
<p align="center"><img src=https://user-images.githubusercontent.com/91622575/174198462-d68516ca-4f35-4b5a-b535-1a319c6571ba.png></p>


To be "formable" or manufacturable using our DSIF process, the geometry must resemble a protrusion off a flat surface. In other words, all its edges or the entire perimeter of the geometry must lie in a single plane. Typically however, the parts required by our clients do not satisfy this requirement! So, in order to make it manufacturable, the "part" geometry provided to us by clients must be adapted. The process of adapting the given geometry is to first, reorient the geometry to be optimal for the following steps. Next, project edges of the geometry onto a flat surface. This creates extension surfaces with perimeter ending in a single plane. The edges thus extended with the new surfaces, fulfil the criteria of being a protrusion. However, this surface will need additional modifications as explained below. Such an extension surface is what we call a "skirt" - for obvious reasons.
<p align="center"><img src=https://user-images.githubusercontent.com/91622575/174199005-22c1a9ef-db98-4096-b50f-51f5c8893031.png></p>
<p align="center"><img src=https://user-images.githubusercontent.com/91622575/174199072-42ceecb8-fc4c-4504-9f33-cdafa2a7327d.png></p>


In addition to its perimeter lying in a single plane, an ideal manufacturable surface typically has a draft, or is slanted less than 90 degrees towards the plane. Typically the transition from part surface to extended surface, over the original edge, causes a change in direction. For manufacturability, it is best for the newly added skirt surface to extend smoothly beyond the edge of the orignal part. Often this can be achieved by orienting the part optimally before extending its edges. As such, to be manufacturable, there are several other requirements and constraints - but those are beyond the scope of this assignment.
## Task
Create a program that...
- Takes in a 3D geometry in form of a .STL file. Use provided file - part.STL.
- Programmatically adds a "skirt" to the provided 3D geometry.
- Outputs a new .STL file - part_skirted.STL.
## Constraints, Assumptions and Terminology
- There are several shapes which are impossible to form, and are assumed not to be used as input. Some examples are...
  * Shapes that form a closed volume - a sphere,
  * Shapes with intersecting planes,
  * Mobius strip, etc.
- To understand the requirements below, assume the part to be in a 3D cartesian co-ordinate space, with the part entirely in negative z space (the convention we use).
- Once the part is skirted, its new perimeter, formed by extending the edges is located in the z=0 plane.
- Imagine the new skirt to be a slanting wall falling from the z=0 plane. The acute angle this wall makes with the z=0 plane will be referred to as the "wall-angle" and will be a constrained value. 
- The edges shown in images here are filleted - another requirement of manufacturable geometries. But you can ignore this requirement and create edges without fillets.
## Requirements
- The program can be a simple command line script, or optionally include a user-friendly GUI.
- Develop using your language of choice (we prefer Python).
- Constraint: The newly added surfaces will have a wall-angle no greater than 70 degrees with respect to the z=0 plane.
- The provided geometry (part.STL) may be re-oriented using your choice of CAD tools, before being used as input to your program.
- The deliverables must include documentation and all artifacts necessary to setup and run the program.
- To enable review, deliverables must include code - not just executables. And per good programming practices, must be sufficiently documented for ease of understanding.
## Optional Challenge (any or all of the features for bonus points)
- Update your program to accept any .STL file (limited by the constraints described above) and output a skirted geometry.
- Create a GUI that includes display windows to visualize the input and output 3D geometries.
- Make the visualization window interactive, allowing user to manipulate the geometry displayed in the window.
- Allow user to re-orient and update the input geometry.
- Add user controls so the constraints can be adjusted and applied to update output geometry.
## Submission
In order to submit the assignment, do the following:

1. Navigate to GitHub's project import page: [https://github.com/new/import](https://github.com/new/import)

2. In the box titled "Your old repository's clone URL", paste the homework repository's link: [https://github.com/Machina-Labs/CompGeo_Add_Skirt](https://github.com/Machina-Labs/CompGeo_Add_Skirt)

3. In the box titled "Repository Name", add a name for your local homework (ex. `Add_skirt_soln`)

4. Set privacy level to "Public", then click "Begin Import" button at bottom of the page.

5. Develop your homework solution in the cloned repository and push it to Github when you're done. Extra points for good Git hygiene.

6. Send us the link to your repository.
