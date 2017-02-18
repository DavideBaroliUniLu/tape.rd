from dolfin import *
from utils import FSIProblem
from utils import pressure_transition
from boundary_mesh import py_BoundaryMesh
from models.scalar_visco_elasticity import ScalarViscoElastic
# from models.vector_visco_elasticity import VectorViscoElastic
# from models.membrane import SolidMembrane
from inflow_bcs_flux import InflowFromFlux
from collections import namedtuple
import numpy as np


class CSF(FSIProblem):
    def __init__(self, params=None):
        FSIProblem.__init__(self, params)
        self.Epsilon = [2, 3]   # FSI
        self.inflow = 1
        self.outflow = 4

        # Fluid domain setup
        mesh = Mesh()
        hdf = HDF5File(mesh.mpi_comm(), params['meshFile'], 'r')
        hdf.read(mesh, '/mesh', False)
        boundaries = FacetFunction('size_t', mesh)
        hdf.read(boundaries, '/boundaries')
        self.initialize_geometry(mesh, facet_domains=boundaries)

        # Setup up inflow bcs
        fluxes = np.loadtxt(params['fluxFile'])
        fluxes = fluxes[:, :2]
        fluxes[:, -1] *= -1.

        inflow_foo = InflowFromFlux(mesh,
                                    boundaries, marker=self.inflow, n=Constant((-1, 0, 0)),
                                    fluxes=fluxes,
                                    source=params['inflowFile'])
        self.inflow_foo = inflow_foo

        # Solid domain setup
        bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, self.Epsilon, True)
        self.bmesh = bmesh
        self.bmesh_boundaries = bmesh_boundaries
        self.emap = emap
        # Which solid model to use
        self.solid_model = ScalarViscoElastic

        # Forcing
        self.ExternalPressure = Constant(0)
        # Outflow part of the boundary might require special trearment
        self.outflow_domains = (self.outflow, )

    @classmethod
    def default_params(cls):
        params = FSIProblem.default_params()
        params.replace(
            E=0.75*1e6,
            h=0.1,
            nu=0.5,
            rho_s=1.1,
            mu=0.035,
            rho=1.0,
            T=1.0,
            dt=1e-4,
            R=1.45)

        params.update(
            meshFile='',
            inflowFile='',
            fluxFile='',
            kk=5./6.,    # Specific to membrane
            Q=1)

        return params

    def initial_conditions(self, spaces, controls):
        u = (Constant(0), Constant(0), Constant(0))
        p = Constant(0)
        eta = Constant((0, 0, 0))    # Init for solid equation
        return (u, p, eta)

    def boundary_conditions(self, spaces, u, p, t, controls):
        '''Bcs borrowing outflow on pressure.'''
        bcu = [(self.inflow_foo, self.inflow)]    

        bcp = []
        # Here we only specify DirichltBCs for (normal component) of solid
        # displacement, i.e. scalar functions. If not specified the boundary
        # is Neumann
        bcsolid = [(Constant((0, 0, 0)), self.inflow),
                   (Constant((0, 0, 0)), self.outflow)]

        BcsTuple = namedtuple('BcsTuple', ['u', 'p', 'solid'])
        return BcsTuple(bcu, bcp, bcsolid)

    def update(self, spaces, u, p, t, timestep, bcs, *args, **kwargs):
        # The only time-dependent bc in this case is pressure
        for value, tag in bcs.u:
            value.time = float(t)
