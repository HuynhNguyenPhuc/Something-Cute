# 3D Rendering

## How to represent an 3D object
We have to follow three steps
### Modeling
Convert the 3D object into a set of multiple polygons (usually triangles), a polygon can be called as a face.
### Texturing
Fill the face using the color information and textures.
### Rig or Animation
Build a backbone structure for the 3D object for the ability to represent a motion.

## Polygon mesh
### Main components
* **Vertex**: A position (usually in 3D space) along with other information such as color, normal vector and texture coordinates.
* **Edge**: A connection between two vertices.
* **Face**: A closed set of edges, in which a triagle face has three edges, a quad face has four edges, etc.

### Rendering
Render the polygon mesh into the 2D environment such as computer screen, mobile screen, etc.

### Primitives
A primitive is formed by one or more vertices.
* Vertex: 3D point v(x, y, z)
* Triangle: A set of three points v<sub>1</sub>, v<sub>2</sub>, v<sub>3</sub>
* Normal vector: A set of three vectors n<sub>1</sub>, n<sub>2</sub>, n<sub>3</sub> - describing the surface orientation n = (n<sub>x</sub>, n<sub>y</sub>, n<sub>z</sub>)

### Fragments
Grid-aligned fragments are interpolated from primitives after rasterizing.


## The graphics pipeline
### Vertex Processing or Vertex Shader (Programmable)
![Vertex transformation pipeline](Images/vertex-transformation-pipeline.png)
> In essence, in this step, we have to perform some linear transformation operations.
#### Model Transform
Transform a point from the model space (local coordinate system of an object) to the world space (global coordinate system).
> Homogeneous coordinates: \
> $`v=\begin{pmatrix}x\\y\\z\\1\end{pmatrix}`$ \
> Ensure homogeneous when all operations are matrix multiplication (or linear transformation). Therefore, it makes the calculations faster and can inverse. \
> $$f_1 \circ f_2 \circ \cdots \circ f_n(x) = M_1 \times M_2 \times \cdots \times M_N \times x = M \times x$$

Three operations (applied for homogeneous coordinates):
* Scaling: \
$`S(s).v = \begin{pmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix} = \begin{pmatrix} s_x x \\ s_y y \\ s_z z \\ 1 \end{pmatrix}`$
* Rotation: \
$`R_z(\theta) = \begin{pmatrix} \cos(\theta) & -\sin(\theta) & 0 & 0 \\ \sin(\theta) & \cos(\theta) & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}`$ \
$`R_x(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\theta) & -\sin(\theta) & 0 \\ 0 & \sin(\theta) & \cos(\theta) & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}`$ \
$`R_y(\theta) = \begin{pmatrix} \cos(\theta) & 0 & \sin(\theta) & 0 \\ 0 & 1 & 0 & 0 \\ -\sin(\theta) & 0 & \cos(\theta) & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}`$
* Translation \
$`T(d) = \begin{pmatrix} 1 & 0 & 0 & d_x \\ 0 & 1 & 0 & d_y \\ 0 & 0 & 1 & d_z \\ 0 & 0 & 0 & 1 \end{pmatrix}`$

The sequence of operations or **Model Matrix**: \
$`A_1 = M \times A = T \times R \times S \times A`$ \
With:
* A is a set of vectors in **model space** (**model coordinates**)
* A<sub>1</sub> is a set of vectors in **world space** (**world coordinates**)
* M is the **model transformation matrix**

#### View Transform
Convert from **world space** (**world coordinates**) into **eye space** (**eye coordinates**)
> **Camera information**
> * Camera position: \
>   $`P = \begin{pmatrix} x \\ y \\ z \end{pmatrix}`$
> * Forward vector: \
>   $`f = \begin{pmatrix} x_f \\ y_f \\ z_f \end{pmatrix}`$
> * Up vector: \
>   $`u = \begin{pmatrix} x_u \\ y_u \\ z_u \end{pmatrix}`$

We have to follow some steps:
* Calculate the forward vector (must normalize it).
* Normalize the up vector u.
* Perform *translation* operation to convert P(x, y, z, 1) (in homogeneous coordinate form) into (0, 0, 0, 1).
* Calculate vector l using the cross product of f and u.
* Calculate vector u' using the cross product of l and f.
* l, u', f formed into a basis, next step we convert into <bold>R</bold><sup>3</sup> using M<sub>R</sub> (Multi-rotation operation).

> **View Matrix** \
> $`V = M_R \times T(-P) = \begin{pmatrix} f_x & u_x & l_x & 0 \\ f_y & u_y & l_y & 0 \\ f_z & u_z & l_y & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}^{-1} \times \begin{pmatrix} 1 & 0 & 0 & -x \\ 0 & 1 & 0 & -y \\ 0 & 0 & 1 & -z \\ 0 & 0 & 0 & 1 \end{pmatrix}`$

#### Normal Vector Transform


#### Projection Transform
Convert from **eye space** (**eye coordinates**) into **clip space** (**clip coordinates**)

![Perspective frustum](Images/perspective-frustum.png)
In perspective projection, a 3D point in a truncated pyramid frustum (eye coordinates) is mapped to a cube (NDC); the range of x-coordinate from [l, r] to [-1, 1], the y-coordinate from [b, t] to [-1, 1] and the z-coordinate from [-n, -f] to [-1, 1].

### Rasterizer

### Fragment Processor or Fragment Shader (Programmable)
#### Phong Lighting Model or Phong Shading
The major building blocks of the Phong lighting model consist of 3 components: **ambient**, **diffuse** and **specular** lighting. Below you can see what these lighting components look like on their own and combined:
* **Ambient lighting**: even when it is dark there is usually still some light somewhere in the world (the moon, a distant light) so objects are almost never completely dark. To simulate this we use an ambient lighting constant that always gives the object some color.
* **Diffuse lighting**: simulates the directional impact a light object has on an object. This is the most visually significant component of the lighting model. The more a part of an object faces the light source, the brighter it becomes.
* **Specular lighting**: simulates the bright spot of a light that appears on shiny objects. Specular highlights are more inclined to the color of the light than the color of the object.
In Phong shading, the light we saw is the combination of three above types. 

### Output merging

## Reference
* OpenGL pipeline - Lecture 2, EE267: Virtual Reality course, Stanford University. Get the detail [here](https://stanford.edu/class/ee267/lectures/lecture2.pdf)
* Songho.ca Blog. Get the detail [here](http://www.songho.ca/index.html)
