CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Joseph Klinger
* Tested on: Windows 10, i5-7300HQ (4 CPUs) @ ~2.50GHz, GTX 1050 6030MB (Personal Machine)

### README

This week, I took on the task of implementing a rasterizer in CUDA. I have already written a CPU rasterizer (almost 2 years ago, in the introductory
graphics course CIS 460), but implementing a basic graphics pipeline on the GPU was a different beast.

The features included in this rasterizer are:
- Texture mapping
- Supersampling Antialiasing
- Color interpolation across triangles

[Demo video here.](https://vimeo.com/238849683)

Rasterization, in very brief summary, is taking a 3d shape and deciding how to color the pixels that the object overlaps. In this project, that involves transforming
the input GLTF models' vertex data, creating triangles from that data, projecting the triangles into view->clip->NDC/screen->viewport space, computing line intersection
with the edges of the triangle, and shading the overlapping fragments.

Here is an image of the given Duck GLTF model rasterized with texture mapping:

![](/renders/duck_noaa.PNG)

For comparison, here is the same Duck but rendered with SSAA (supersampling antialiasing). This process involves simply rendering to an image of higher resolution than the 
screen, then downsampling that information into the final image:

![](/renders/duck_ssaa.PNG)

### Performance Analysis











### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
