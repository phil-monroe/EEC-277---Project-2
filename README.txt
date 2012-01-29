
1/4/2010
E. Wes Bethel, Lawrence Berkeley National Laboratory

This distribution contains two codes, wesBench and wesBench-vbo, that
are useful for measuring vertex and fill rate performance of GPUs. The
two codes have identical command-line arguments and measure/report the
same information. There are two applications, rather than one, since
the use of OpenGL vertex buffer objects requires some additional
logic. Long term, it should be straightforward to combine these two
codes into one application.


wesBench and wesBench-vbo command line arguments:


[-a AAAA] - sets triangle area in pixels (double precision
value). Default triangle area is 128.0 pixels (computed as w*h/2).

[-tl LLLL] - sets maximum number of triangles (long int
value). Default limit is 1G triangles.

[-ms NNNN] - sets the duration of the test in milliseconds. Default
test duration is 5000 ms (five seconds).

[-w WWW -h HHH] -  sets the display window size. Default window size
is 1K by 1K pixels.

[-tx WWW] -  sets the texel resolution of the texture, the texture
will be WWWxWWW texels. WWW must be an even power of 2.  Default is
zero (no texturing).

[-light] - turns on lighting (no args). Default is no lighting.

[-cm] - turns on Color Material/per-vertex colors will be used in
diffuse term of lighting (no args). Default is to not use color
material/per-vertex colors when lighting is enabled.

[-df fname] - sets the name of the dumpfile for performance
statistics. Default value is "wesBench-dump.txt". The app will write
to this dump file if doing any of the following test batteries: area
test, vertex buffer size test, texture size test (see below for
details). 

[-tt (0, 1, 2, 3)] - set triangle type: 0=disjoint, 1=tstrip,
2=indexed disjoint, 3=indexed tstrip. Default is 0=disjoint.

[-retained] - Use OpenGL display lists. Default is to do immediate
mode rendering.

[-clear] - Clear the frame after each drawing. Since we are using a
single-buffered FB, you will likely see "tearing" as multiple
clear/draw cycles happen within one vertical retrace interval. Default
is no framebuffer clearing between successive frame draws.

[-line] - Set polygon mode to GL_LINE to draw triangle outlines, no
fill. Default is to draw filled triangles.


Automated tests (choose one or none):
[-doareatest] - Will iterate over a number of triangle area sizes:
1..128K in even power-of-two steps. Output is written to the dump file
specified by [-df fname]. 

[-dovbtest] - Iterate over different buffer sizes/number of verts sent
with each glDrawArraysCall. Will result in 'partial rendering' of base
mesh. Output is written to the dump file specified by [-df
fname]. Test iterates over a number of vertex buffer sizes ranging
from 64, 128, ... 8M in powers of two. Can be changed by modifying the code.

[-dotxtest] - Iterate over a set of texture image sizes. Output is
written to the dump file specified by [-df fname]. Iterates over
texture sizes 8x8, 16x16, ... , 2Kx2K, 4Kx4K. Set of tests can be
modified by changing the code.



3rd party software requirements:
- An OpenGL implementation (libraries, headers). OpenGL 1.5 or better
is required to support vertex buffer objects.
- GLUT (libraries, headers)

System requirements:
- Tested on several different Linux platforms. Could be ported to
Windows though the timer code will need to be updated.
- POSIX-compliant C compiler (e.g., gcc)