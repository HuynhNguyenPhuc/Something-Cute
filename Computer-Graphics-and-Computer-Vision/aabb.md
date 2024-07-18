# Axis-Aligned Bounding Box (AABB)

Axis-Aligned Bounding Box (AABB) is a rectangular box that is aligned with the coordinate axes and used to enclose a set of points (an example is convex hull) or geometric objects. It is commonly used in computer graphics, collision detection, and spatial indexing due to its simplicity and computational efficiency.

## Properties

- **Axis-Aligned**: The edges of the bounding box are parallel to the coordinate axes.
- **Bounding**: It completely encloses the given set of points or objects.
- **Box**: It is defined as a rectangular shape in 2D or a rectangular prism in 3D.

## Representation

In 2D, an AABB is represented by:
- **Minimum coordinates**: `(min_x, min_y)`
- **Maximum coordinates**: `(max_x, max_y)`

In 3D, an AABB is represented by:
- **Minimum coordinates**: `(min_x, min_y, min_z)`
- **Maximum coordinates**: `(max_x, max_y, max_z)`

## Construction

Given a set of points or geometric objects, the AABB can be constructed by determining the minimum and maximum coordinates along each axis. For a set of points $`(x_i, y_i, z_i)`$, the AABB can be constructed as:

- $`\text{min}_x = \min(x_i)`$
- $`\text{min}_y = \min(y_i)`$
- $`\text{min}_z = \min(z_i)`$ (for 3D)

- $`\text{max}_x = \max(x_i)`$
- $`\text{max}_y = \max(y_i)`$
- $`\text{max}_z = \max(z_i)`$ (for 3D)

## Applications

- **Collision Detection**: AABBs are used to quickly determine if two objects might intersect. If their bounding boxes do not overlap, the objects cannot collide.
- **Ray Tracing**: AABBs can be used to accelerate ray-object intersection tests by quickly eliminating objects that are not intersected by the ray. We will dicuss it in detail in another topic.
- **Spatial Indexing**: AABBs are used in spatial data structures like Quadtrees and Octrees to organize and query spatial information efficiently.

## Advantages

- **Simplicity**: AABBs are easy to compute and represent.
- **Efficiency**: Intersection tests between AABBs are computationally inexpensive, making them ideal for real-time applications.

## Limitations

- **Coarse Approximation**: Since AABBs are axis-aligned, they may not tightly fit the enclosed objects, especially if the objects are rotated or have irregular shapes.
- **Static Nature**: AABBs need to be recomputed if the enclosed objects move or change shape, which can be inefficient for dynamic scenes.
