from dolfin import *
from utils import FSIProblem
from utils import pressure_transition
from boundary_mesh import py_BoundaryMesh
from solid_model import SolidMembrane
from collections import namedtuple


class CylinderShell(FSIProblem):
    def __init__(self, params=None):
        FSIProblem.__init__(self, params)
        N = self.params.N
        self.Epsilon = 1   # FSI
        self.left = 2
        self.right = 3

        # Fluid domain setup
        mesh = Mesh()
        hdf = HDF5File(mesh.mpi_comm(), './meshes/CYLINDER/cylinder_%d.h5' % N, 'r')
        hdf.read(mesh, '/mesh', False)
        boundaries = FacetFunction('size_t', mesh)
        hdf.read(boundaries, '/boundaries')
        self.initialize_geometry(mesh, facet_domains=boundaries)

        # Solid domain setup
        bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, 1, True)
        self.bmesh = bmesh
        self.bmesh_boundaries = bmesh_boundaries
        self.emap = emap
        # Which solid model to use
        self.solid_model = SolidMembrane

        # Forcing
        self.ExternalPressure = pressure_transition(pLeft=0, pRight=0, w=1)
        # Outflow part of the boundary might require special trearment
        self.outflow_domains = (self.right, )

    @classmethod
    def default_params(cls):
        params = FSIProblem.default_params()
        params.replace(
            E=4.07E6,        # Young
            nu=0.5,          # Poisson
            rho_s=1.1,       # Solid density
            mu=0.035,       
            rho=1.0,         
            T=1.0,
            dt=1e-4)
 
        params.update(
            stress_amplitude=2e2,
            stress_time=5e-3,
            N=2,
            Q=1,
            eps=1E-13,       # Thickness
            k=5./6)         # Traverse shear parameter

        return params

    def initial_conditions(self, spaces, controls):
        u = (Constant(0), Constant(0), Constant(0))
        p = Constant(0)
        eta = Constant((0, 0, 0))    # Init for solid equation
        return (u, p, eta)

    def boundary_conditions(self, spaces, u, p, t, controls):
        '''Bcs borrowing outflow on pressure.'''
        # No equation for velocity in N-S
        bcu = []    
        # The flow is pressure driven
        bcp = [(Expression("time>stress_time ? 0.0 : A*sin(pi*time/stress_time)",
                           A=self.params.stress_amplitude, time=float(t),
                           stress_time=self.params.stress_time),
                self.left)]
        # Here we only specify DirichltBCs for (normal component) of solid
        # displacement, i.e. scalar functions. If not specified the boundary
        # is Neumann
        bcsolid = [(Constant((0, 0, 0)), self.left),
                   (Constant((0, 0, 0)), self.right)]

        BcsTuple = namedtuple('BcsTuple', ['u', 'p', 'solid'])
        return BcsTuple(bcu, bcp, bcsolid)

    def update(self, spaces, u, p, t, timestep, bcs, *args, **kwargs):
        # The only time-dependent bc in this case is pressure
        for value, tag in bcs.p:
            if hasattr(value, 'time'): value.time = float(t)
