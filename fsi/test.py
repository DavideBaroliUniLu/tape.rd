from dolfin import *
from boundary_mesh import py_BoundaryMesh
import os

taperd_dir = '..'
# Fluid domain setup
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), os.path.join(taperd_dir, 'mesh/CYLINDER/cylinder_2.h5'), 'r')
hdf.read(mesh, '/mesh', False)
boundaries = FacetFunction('size_t', mesh)
hdf.read(boundaries, '/boundaries')

# Solid domain setup
bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, [1], True)

plot(bmesh, interactive=True)
