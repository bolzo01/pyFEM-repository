// Porous lattice structure: discretize domain from STL triangulated surfaces
//
// pipe_mesh.stl: https://gitlab.onelab.info/gmsh/gmsh/-/issues/1598

// Generate mesh with
// gmsh pipe.geo -3 -format msh2 -o pipe.msh

// Geometrical details for pipe_mesh.stl:

// Bounding box dimensions

// Axis Minimum     Maximum     Length
// X	−0.54028	28.54623	29.08651
// Y	−0.54546	28.53871	29.08417
// Z	−0.54636	28.55665	29.10301

// Overall dimensions
// The object occupies a cube-like region of approximately
// 29.09 × 29.08 × 29.10 units^3

// Coordinate bounds
// X: -0.54028  ->  28.54623
// Y: -0.54546  ->  28.53871
// Z: -0.54636  ->  28.55665

// Meshing works if your STL is closed/watertight (no holes).
// If you get "Non-manifold surface" or "Surface loop wrong", the STL is not watertight.
// Solution: use Geometry.AutoCoherence = 1; (already added), or fix STL externally (MeshLab, Blender, FreeCAD).


Merge "pipe_mesh.stl";

// Ensure STL surfaces are merged/coherent if there are gaps
Geometry.AutoCoherence = 1;

// Build model from STL triangulated surfaces
Surface Loop(1) = { Surface{:} };
Volume(1) = {1};

// Mesh parameters
Mesh.ElementOrder = 1; // Linear tetrahedral elements
Mesh.CharacteristicLengthMin = 5;
Mesh.CharacteristicLengthMax = 10;

// Optional: Improve Mesh Quality
// Mesh.Algorithm3D = 4; // Delaunay
// Mesh.Optimize = 1;
// Mesh.OptimizeNetgen = 1;
// Mesh.Smoothing = 10;
// Mesh.MinimumQuality = 0.2;

// Physical groups for FEM
Physical Surface("walls") = Surface{:};
Physical Volume("solid") = {1};

// Export mesh as Gmsh v2
Mesh.MshFileVersion = 2.2;

// Generate 3D tetrahedral mesh
Mesh 3;
