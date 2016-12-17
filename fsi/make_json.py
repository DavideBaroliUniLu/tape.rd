from utils import ExternalForcing, StepFoo, FSIProblem, AbsorbingStress, PressureInterpolator
from models.scalar_visco_elasticity import ScalarViscoElastic
from cbcpost import PostProcessor, SolutionField, PointEval
from cbcflow import make_womersley_bcs
from boundary_mesh import py_BoundaryMesh
from collections import namedtuple
from fsi_decoupled import FSI_Decoupled
from dolfin import Constant, Expression, Mesh, FacetFunction, HDF5File
import numpy as np
import json
from fields import *


def make_json_scheme(model):
    '''Just decide if direct or iterative solver should be used.'''
    scheme = FSI_Decoupled(dict(r=1, s=0, u_degree=1, tol=model['solver']['tolerance']))
    return scheme


def make_json_postprocessor(model, problem):
    '''Build a cbcpost postprocessor based on model specs.'''
    output = model['output']
    name = output['sim_name'].encode('ascii', 'ignore')
    pp = PostProcessor(dict(casedir="./Results/%s" % name, clean_casedir=True))

    # Handle fields
    # NOTE: Don't really save here - just have a (lambda u: u) available. 
    pp.add_field(SolutionField("Displacement", dict(save=False)))
    for f in output['fields']:
        name, dt = f['name'], f['dt']

        if dt == 0:
            params = dict(save=False)
        else:
            params = dict(save=True, save_as="xdmf", stride_timestep=dt)
        # Dispatch
        if name in ('Velocity', 'Pressure'):
            field = SolutionField(name.encode('ascii', 'ignore'), params)
        else:
            field = SolidDisplacement(params)

        pp.add_field(field)

    # Get stuff for validating the probes
    mesh = problem.mesh
    tree = mesh.bounding_box_tree()
    num_cells = mesh.topology().size_global(mesh.topology().dim())

    # Point probes
    point_probes = (('pressure_probes', 'Pressure'), ('velocity_probes', 'Velocity'))
    comm_py = mesh.mpi_comm().tompi4py()
    for key, field in point_probes:
        probes = output.get(key, [])
        for probe in probes:
            x, name, dt = np.array(probe['x'], dtype=float), probe['name'], probe['dt']

            idd = tree.compute_first_entity_collision(Point(x)) < num_cells
            found_local = int(tree.compute_first_entity_collision(Point(x)) < num_cells)
            found_global = comm_py.allreduce(found_local)
            if found_global:
                params = dict(save=True, stride_timestep=dt)
                pp.add_field(PointEval(field, [x], label=name, params=params))
            else:
                info('Skipping probe at %s' % x)
    
    # Integral probes inside
    dispatch_table = {'velocity_profiles': {True: VelocityProfile,
                                            False: VelocityProfileExternal},
                      'flow_rates': {True: FlowRate,
                                     False: FlowRateExternal},
                      'areas': {True: AreaRate,
                                False: AreaRateExternal}}
    
    boundaries = problem.boundaries
    for key in dispatch_table:
        probes = output.get(key, {})
        # Overkill way to get the probe directory name
        prefix = ''.join(map(lambda s: s[0].upper()+s[1:], key.split('_')))

        for probe in probes:
            name, dt = probe['name'].encode('ascii', 'ignore'), probe['dt']
            name = '_'.join([prefix, name])
            
            params = dict(save=True, stride_timestep=dt)
            # External
            if 'tag' in probe:
                field = dispatch_table[key][False](facet_f=boundaries,
                                                   tag=probe['tag'],
                                                   name=name,
                                                   params=params)
            # Internal
            else:
                x = np.array(probe['x'], dtype=float)
                n = np.array(probe['normal'], dtype=float)
                field = dispatch_table[key][True](mesh=mesh,
                                                  point=x, 
                                                  n=n, 
                                                  name=name, 
                                                  params=params)
            # Check empty mesh
            if field.mesh.topology().size_global(0) > 0: pp.add_field(field)
    return pp


class JSONProblem(FSIProblem):
    '''FSI problem specified by JSON'''
    def __init__(self, model):
        json_dict = model.copy()
        # Material and properties
        params = {'rho': json_dict['fluid_properties']['density'],
                  'mu': json_dict['fluid_properties']['viscosity'],
                  'rho_s': json_dict['solid_properties'].pop('density')}
        # Transfer the solid (no density)
        params.update(json_dict["solid_properties"])
        # Some parameters for solver
        params.update({new_k : json_dict["solver"][k] for new_k, k in zip(("dt", "T"), ("time_step", "Tfinal"))})

        # Geometry
        # Tags
        self.Epsilon = json_dict["geometry"]["tags"]["walls"]
        self.inlet_domains = json_dict["geometry"]["tags"]["inlets"]
        self.outlet_domains = json_dict["geometry"]["tags"]["outlets"]
        # Fluid domain setup. # FIXME: that the path is valid w.r.t to exec directory
        mesh = Mesh()
        hdf = HDF5File(mesh.mpi_comm(), json_dict["geometry"]["path"].encode("ascii"), 'r')
        hdf.read(mesh, '/mesh', False)
        boundaries = FacetFunction('size_t', mesh)
        hdf.read(boundaries, '/boundaries')
        self.initialize_geometry(mesh, facet_domains=boundaries)
        self.boundaries=boundaries
        # Solid domain setup
        bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, self.Epsilon, True)
        self.bmesh = bmesh
        self.bmesh_boundaries = bmesh_boundaries
        self.emap = emap

        # Which solid model to use
        self.solid_model = ScalarViscoElastic

        # Forcing: setup force for each marked domain
        marker_values = {}
        for force in json_dict["forcing"]:
            # Constant set up as StepFoo
            value = force["value"]
            if isinstance(value, (int, float)):
                marker_values.update({force["tag"]: StepFoo(v0=value)})
            else:
                marker_values.update({force["tag"]: StepFoo(**value)})

        self.ExternalForcing = ExternalForcing(mesh, family='DG', degree=0,
                                               boundaries=boundaries,
                                               marker_values=marker_values)

        # Done    
        self.json_dict = json_dict
        FSIProblem.__init__(self, params)

    @classmethod
    def default_params(cls):
        params = FSIProblem.default_params()
        # Make sure all necessary items are present in params before replace 
        # Setup to something which will make the program crash if not overriden
        crash = np.nan
        params.update(epsilon=crash, rho_f=crash, mu=crash, alpha0=crash,
                      alpha1=crash, E=crash, rho_s=crash, R=crash, nu=crash)
        return params

    def initial_conditions(self, spaces, controls):
        u = (Constant(0), Constant(0), Constant(0))
        p = Constant(0)
        eta = Constant((0, 0, 0))    # Init for solid equation
        return (u, p, eta)

    def boundary_conditions(self, spaces, u, p, t, controls):
        '''All bcs including outflow/absorbing'''
        bcs = self.json_dict["boundary_conditions"]

        # Solid - Derichlet guys
        bcsolid = [(Constant((0, 0, 0)), bc["tag"]) for bc in bcs["solid"]]
        # TODO: Check that we do not assume more than is tested 
        # Fluid
        fluidBcs = bcs["fluid"]
        # Pressure
        pressure_bcs = fluidBcs.get('pressure', [])
        bcp = []
        for bc in pressure_bcs:
            value = bc["value"]
            # A constant, use StepFoo for consistency
            if isinstance(value, (int, float)):
                foo = StepFoo(v0=value)
            # Exresson
            elif value.has_key("body"):
                body = value['body'].encode('ascii', 'ignore')
                foo = Expression(body, degree=value["degree"] , t=0.0)
            # Tabulated
            else:
                # NOTE value.get("period") returns None if not value.has_key("period")
                foo = PressureInterpolator(value["path"], value.get("period"))
            bcp.append((foo, bc["tag"]))
        # Pressure
        velocity_bcs = fluidBcs.get('velocity', [])
        bcu = []
        for bc in velocity_bcs:
            data = np.loadtxt(bc["path"])  # Parser should point to single col. data

            tvalues = np.linspace(0.0, bc["period"], data.size)
            Q_coeffs = zip(tvalues, data)

            nu = self.params.mu/self.params.rho
            wom = make_womersley_bcs(Q_coeffs, self.mesh, bc["tag"], \
                    nu, None, self.facet_domains)
            for comp in wom:
                comp.set_t(0.0)
            bcu.append((wom, bc["tag"]))
        # Finally setup of Absorbing bcs: map outlet tag to value which is some
        # pressure like function
        absorbing_bcs = bcs.get('outlet', {})
        bcabsorbing = {}
        for bc in absorbing_bcs:
            value = bc["value"]
            # A constant, use StepFoo for consistency
            if isinstance(value, (int, float)):
                p_ext = StepFoo(v0=value)
            # Exresson
            elif value.has_key("body"):
                body = value['body'].encode('ascii', 'ignore')
                p_ext = Expression(body, degree=value["degree"] , t=0.0)
            # Tabulated
            else:
                # NOTE value.get("period") returns None if not value.has_key("period")
                p_ext = PressureInterpolator(value["path"], value.get("period"))
            bcabsorbing[bc['tag']] = AbsorbingStress(self, self.facet_domains, bc['tag'], p_ext) 

        BcsTuple = namedtuple('BcsTuple', ['u', 'p', 'solid', 'absorbing'])
        return BcsTuple(bcu, bcp, bcsolid, bcabsorbing)

    def update(self, spaces, u, p, DF, t, timestep, bcs, *args, **kwargs):
        '''Perform update of everything that is time dependent'''
        time = float(t)
        # That is forcing
        self.ExternalForcing.t = time

        # Boundary conditions
        # for pressure
        for value, _ in bcs.p: value.t = time
        # for velocity 
        for value, _ in bcs.u:
            warning("womersley update not verified")
            for comp in value: comp.set_t(float(t))

        # Absorbing boundary condtions, dict
        for bc in bcs.absorbing.values(): bc.update(u, DF, t)


def make(path):
    '''
    We take path to validated json and create instances of solver, problem
    and postprocessor.
    '''
    try:
        with open(path) as f: model = json.load(f)
    except Exception as e:
        assert False, 'Error parsing %s:' % 'our.json' + str(e)

    scheme = make_json_scheme(model)
    problem = JSONProblem(model)
    postprocessor = make_json_postprocessor(model, problem)

    return scheme, problem, postprocessor
