# Euler Angles and Axis Angle

## Euler Angles
Using three different angles (independent variables) for rotation

$$R = R_z \cdot R_y \cdot R_x$$

$$= \begin{pmatrix} \cos(\gamma) & -\sin(\gamma) & 0 & 0 \\ 
\sin(\gamma) & \cos(\gamma) & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 \end{pmatrix} 
\cdot \begin{pmatrix} \cos(\beta) & 0 & \sin(\beta) & 0 \\ 
0 & 1 & 0 & 0 \\ 
-\sin(\beta) & 0 & \cos(\beta) & 0 \\ 
0 & 0 & 0 & 1 \end{pmatrix} 
\cdot \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) & 0 \\ 
0 & \sin(\alpha) & \cos(\alpha) & 0 \\ 
0 & 0 & 0 & 1 \end{pmatrix}$$

May face the Gimbal lock problem, if the value of $`\beta`$ is $`\frac{\pi}{2}`$.
> **Gimbal lock**
> 
> $$R = R_z \cdot R_y \cdot R_x$$
> 
> $$= \begin{pmatrix} \cos(\gamma) & -\sin(\gamma) & 0 & 0 \\
> \sin(\gamma) & \cos(\gamma) & 0 & 0 \\
> 0 & 0 & 1 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}
> \begin{pmatrix} \cos(\frac{\pi}{2}) & 0 & \sin(\frac{\pi}{2}) & 0 \\
> 0 & 1 & 0 & 0 \\
> -\sin(\frac{\pi}{2}) & 0 & \cos(\frac{\pi}{2}) & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}
> \begin{pmatrix} 1 & 0 & 0 & 0 \\
> 0 & \cos(\alpha) & -\sin(\alpha) & 0 \\
> 0 & \sin(\alpha) & \cos(\alpha) & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}$$
> 
> $$= \begin{pmatrix} \cos(\gamma) & -\sin(\gamma) & 0 & 0 \\
> \sin(\gamma) & \cos(\gamma) & 0 & 0 \\
> 0 & 0 & 1 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}
> \begin{pmatrix} 0 & 0 & 1 & 0 \\
> 0 & 1 & 0 & 0 \\
> -1 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}
> \begin{pmatrix} 1 & 0 & 0 & 0 \\
> 0 & \cos(\alpha) & -\sin(\alpha) & 0 \\
> 0 & \sin(\alpha) & \cos(\alpha) & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}$$
> 
> $$= \begin{pmatrix} 0 & -\sin(\gamma) & \cos(\gamma) & 0 \\
> 0 & \cos(\gamma) & \sin(\gamma) & 0 \\
> -1 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}
> \begin{pmatrix} 1 & 0 & 0 & 0 \\
> 0 & \cos(\alpha) & -\sin(\alpha) & 0 \\
> 0 & \sin(\alpha) & \cos(\alpha) & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}$$
> 
> $$= \begin{pmatrix} 0 & -\sin(\gamma)\cos(\alpha) + \cos(\gamma)\sin(\alpha) & \sin(\gamma)\sin(\alpha) + \cos(\gamma)\cos(\alpha) & 0 \\
> 0 & \cos(\gamma)\cos(\alpha) + \sin(\gamma)\sin(\alpha) & -\cos(\gamma)\sin(\alpha) + \sin(\gamma)\cos(\alpha) & 0 \\
> -1 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}$$
> 
> $$= \begin{pmatrix} 0 & \sin(\gamma - \alpha) & \cos(\gamma - \alpha) & 0 \\
> 0 & \cos(\gamma - \alpha) & -\sin(\gamma - \alpha) & 0 \\
> -1 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 1 \end{pmatrix}$$
> 
We can see here also one independent variable was left, althought we want to have two independent variables.

## Axis Angle
To perform smoother rotations, we need to use another method called axis-angle rotation. This method uses one axis as the axis of the rotation operation, and peforms an anti-clockwise rotation with angle $`\theta`$. We can represent the rotation as a matrix transformation like below.

**Axis-angle represents**
* The axis is given by a unit vector $`\mathbf{u} = (u_x, u_y, u_z)`$
* The angle of rotation is given by the value $`\theta`$

**Rotation matrix**

The rotation matrix R corresponding to a rotation of $`\theta`$ about the axis u is given by:

$$ R = I + \sin(\theta) \mathbf{U} + (1 - \cos(\theta)) \mathbf{U}^2$$

where I is the identity matrix and U is the skew-symmetric matrix of u, defined as:

$$
\mathbf{U} = \begin{pmatrix}
0 & -u_z & u_y \\
u_z & 0 & -u_x \\
-u_y & u_x & 0
\end{pmatrix} 
$$

**Expanded Rotation Matrix**:

Substituting U into the rotation matrix formula, we get:

$$
R = \begin{pmatrix}
1 + (1 - \cos \theta) (u_x^2 - 1) & -u_z \sin \theta + (1 - \cos \theta) u_x u_y & u_y \sin \theta + (1 - \cos \theta) u_x u_z \\
u_z \sin \theta + (1 - \cos \theta) u_x u_y & 1 + (1 - \cos \theta) (u_y^2 - 1) & -u_x \sin \theta + (1 - \cos \theta) u_y u_z \\
-u_y \sin \theta + (1 - \cos \theta) u_x u_z & u_x \sin \theta + (1 - \cos \theta) u_y u_z & 1 + (1 - \cos \theta) (u_z^2 - 1)
\end{pmatrix} 
$$

In a more concise form, this can be written as:

$$
R = \begin{pmatrix}
\cos \theta + u_x^2 (1 - \cos \theta) & u_x u_y (1 - \cos \theta) - u_z \sin \theta & u_x u_z (1 - \cos \theta) + u_y \sin \theta \\
u_y u_x (1 - \cos \theta) + u_z \sin \theta & \cos \theta + u_y^2 (1 - \cos \theta) & u_y u_z (1 - \cos \theta) - u_x \sin \theta \\
u_z u_x (1 - \cos \theta) - u_y \sin \theta & u_z u_y (1 - \cos \theta) + u_x \sin \theta & \cos \theta + u_z^2 (1 - \cos \theta)
\end{pmatrix} 
$$

This matrix R can then be used to rotate a vector v in three-dimensional space.

For homogeneous coordinates, we can write the rotation matrix in the form:

$$
R = \begin{pmatrix}
\cos \theta + u_x^2 (1 - \cos \theta) & u_x u_y (1 - \cos \theta) - u_z \sin \theta & u_x u_z (1 - \cos \theta) + u_y \sin \theta & 0 \\
u_y u_x (1 - \cos \theta) + u_z \sin \theta & \cos \theta + u_y^2 (1 - \cos \theta) & u_y u_z (1 - \cos \theta) - u_x \sin \theta & 0 \\
u_z u_x (1 - \cos \theta) - u_y \sin \theta & u_z u_y (1 - \cos \theta) + u_x \sin \theta & \cos \theta + u_z^2 (1 - \cos \theta) & 0 \\
0 & 0 & 0 & 1
\end{pmatrix} 
$$

### Proof
First, we need to perform a matrix transformation to convert the given axis into the z-axis $`(0,0,1)`$. We denote this transformation as T and it will convert the rotation about the given axis into a rotation about the z-axis (still the same angle $`\theta`$)

The rotation transformation about the z-axis will be:

$$R_z(\theta) = \begin{pmatrix}
\cos(\theta) & -\sin(\theta) & 0 \\ 
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1
\end{pmatrix}$$

So, the final matrix transformation that rotates a vector around the given axis with angle $`\theta`$ is:

$$M = T^{-1}R_z(\theta)T$$

We will compute the value of T. This transformation is a combination of two rotations: one around the x-axis and one around the y-axis. We can represent it as follows:

1. **Rotate v into the zx-plane**: 
   The first rotation, $`R_x(\alpha)`$, rotates the vector u into the zx-plane. The angle $`\alpha`$ is the angle between v and the zx-plane.
   
$$
R_x(\theta) = \begin{pmatrix}
1 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) \\
0 & \sin(\alpha) & \cos(\alpha)
\end{pmatrix}
$$

2. **Rotate v to align with the z-axis**: 
   The second rotation, $`R_y(\beta)`$, rotates the vector u in the zx-plane to align with the z-axis. The angle $`\beta`$ is the angle between u in the zx-plane and the z-axis.
   
$$
R_y(\beta) = \begin{pmatrix}
\cos(\beta) & 0 & \sin(\beta) \\
0 & 1 & 0 \\
-\sin(\beta) & 0 & \cos(\beta)
\end{pmatrix}
$$

3. **Combine the rotations**: 
   The transformation matrix $`T`$ is the product of these two rotations:
     
$$
T = R_y(\beta) R_x(\alpha)
$$

Thus, the transformation matrix $`T`$ is:

$$
T = \begin{pmatrix}
\cos \beta & 0 & \sin \beta \\
0 & 1 & 0 \\
-\sin \beta & 0 & \cos \beta
\end{pmatrix}
\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos \alpha & -\sin \alpha \\
0 & \sin \alpha & \cos \alpha
\end{pmatrix}
$$

To find the exact angles $`\alpha`$ and $`\beta`$, we use the components of the unit vector $`\mathbf{u} = (u_x, u_y, u_z)`$:
- $`\cos(\alpha) = \frac{u_z}{\sqrt{u_z^2 + u_y^2}}`$, $`\sin(\alpha) = \frac{u_y}{\sqrt{u_z^2 + u_y^2}}`$
- $`\cos(\beta) = \frac{u_z}{\sqrt{u_z^2 + u_x^2}}`$, $`\sin(\beta) = \frac{u_x}{\sqrt{u_z^2 + u_y^2}}`$

Finally,

$$M = T^{-1}R_z(\theta)T = R_x(-\alpha)R_y(-\beta)R_z(\theta)R_y(\beta)R_x(\alpha)$$

$$ = \begin{pmatrix} 
u_x^2 + \cos(\theta)(u_y^2 + u_z^2) & u_xu_y(1-\cos(\theta)) - u_z\sin(\theta) & u_xu_z(1-\cos(\theta)) + u_y\sin(\theta) \\
u_xu_y(1-\cos(\theta)) + u_z\sin(\theta) & u_y^2 + \cos(\theta)(u_x^2 + u_z^2) & u_yu_z(1-\cos(\theta)) - u_x\sin(\theta) \\
u_xu_z(1-\cos(\theta)) - u_y\sin(\theta) & u_yu_z(1-\cos(\theta)) + u_x\sin(\theta) & u_z^2 + \cos(\theta)(u_x^2 + u_y^2)
\end{pmatrix} $$

$$ = \begin{pmatrix}
\cos \theta + u_x^2 (1 - \cos \theta) & u_x u_y (1 - \cos \theta) - u_z \sin \theta & u_x u_z (1 - \cos \theta) + u_y \sin \theta \\
u_y u_x (1 - \cos \theta) + u_z \sin \theta & \cos \theta + u_y^2 (1 - \cos \theta) & u_y u_z (1 - \cos \theta) - u_x \sin \theta \\
u_z u_x (1 - \cos \theta) - u_y \sin \theta & u_z u_y (1 - \cos \theta) + u_x \sin \theta & \cos \theta + u_z^2 (1 - \cos \theta)
\end{pmatrix} 
$$

## Quaternion
We have learned that a rotation in $`\mathbb{R}^3`$ about an axis can be represented as a 3x3 matrix (or 4x4 if we use homogeneous coordinates). However, the matrix representation seems redundant because only four of its nine (or sixteen) are independent and the geometric interpretation of such a matrix is not clear until we carry out several steps of calculation to extract the rotation axis and angle. Furthermore, to compose two or more rotations, we need to compute the product of all corresponding matrices, which requires twenty-seven multiplications and eighteen
additions.

**Quaternions** are very efficient for analyzing situations where rotations in $`\mathbb{R}^3`$ are involved. A
quaternion is a 4-tuple, which is a more concise representation than a rotation matrix. Its geometric meaning is also more obvious as the rotation axis and angle can be trivially recovered. The quaternion algebra to be introduced will also allow us to easily compose rotations. This is because
quaternion composition takes merely sixteen multiplications and twelve additions.

### Quaternion Algebra





