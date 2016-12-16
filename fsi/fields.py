from dolfin import *
from cbcpost import Field
from cbcpost.utils import create_slice, create_submesh
import numpy as np
try:
    from fenicstools import interpolate_nonmatching_mesh
except ImportError as e:
    print "cannot compute velocity profile", e

# NOTE:
# The following classes work with the assumption that FSI_Decoupled returns a
# ParamDict with u=U, p=P, d=(Usolid, DF), i.e. d holds a tuple of elements in
# particular order
class SolidDisplacement(Field):
    '''Donothing for storing displacement of the shell.'''
    def compute(self, get): return get("Displacement")[0]


class AleDisplacement(Field):
    '''Donothing for storing of mesh displacement.'''
    def compute(self, get): return get("Displacement")[1]


# Internal surface probes
class VelocityProfile(Field):
    '''
    Velocity profile on a plane crossection of the mesh. Plane is specified by (x, n)
    '''
    def __init__(self, mesh, point, n, name, params):
        # Create the mesh as a slice of mesh by plane (x, n)
        slice_mesh = create_slice(mesh, point, n)
        assert slice_mesh.topology().size_global(0) > 0, 'Empty slice mesh'
        self.mesh = slice_mesh
        # Auxliary space
        self.V = None

        Field.__init__(self, params, name, label=None)

    def compute(self, get):
        '''Just restrict'''
        u = get('Velocity')
        # See about auxliary space
        if self.V is None:
            elV = u.function_space().ufl_element()
            assert len(elV.value_shape()) == 1
            self.V = VectorFunctionSpace(self.mesh, elV.family(), elV.degree(), elV.value_size())
            self.u0 = Function(self.V)
        u.set_allow_extrapolation(True)
        self.u0.assign(interpolate(u, self.V))
        u.set_allow_extrapolation(False)
        return self.u0


class FlowRate(Field):
    '''Flux through planar crossection of a mesh. Plane is (point, n)'''
    def __init__(self, mesh, point, n, name, params):
        # Create the mesh as a slice of mesh by plane (x, n)
        slice_mesh = create_slice(mesh, point, n)
        assert slice_mesh.topology().size_global(0) > 0, 'Empty slice mesh'
        # Avoid too tiny cells
        cell_f = CellFunction('size_t', slice_mesh, 0)
        cell_f = mark_measure0(cell_f, marker=1, tol=0.999)

        # The normal better be unit
        n = n/np.linalg.norm(n)
        self.n = Constant(n)

        self.mesh = slice_mesh
        # Measure that avoids wrong cells
        self.dx = Measure('dx', domain=slice_mesh, subdomain_data=cell_f, subdomain_id=1)
        # Auxliary space
        self.V, self.D = None, None

        Field.__init__(self, params, name, label=None)

    def compute(self, get):
        '''
        int_{interior_surface} v.n dS in the deformed domain is mapped to
        reference domain.
        '''

        u = get('Velocity')
        DF = get('Displacement')[1]  # NOTE: ALE is passed as second argument


        if self.V is None and self.D is None:
            self.V = VectorFunctionSpace(self.mesh,
                                         u.function_space().ufl_element().family(),
                                         u.function_space().ufl_element().degree(),
                                         u.value_size())

            self.D = VectorFunctionSpace(self.mesh,
                                         DF.function_space().ufl_element().family(),
                                         DF.function_space().ufl_element().degree(),
                                         DF.value_size())
            
            self.u0 = Function(self.V)
            self.DF0 = Function(self.D)

        # Restrict
        u.set_allow_extrapolation(True)
        self.u0.assign(interpolate(u, self.V))
        u.set_allow_extrapolation(False)

        DF.set_allow_extrapolation(True)
        self.DF0.assign(interpolate(DF, self.D))
        DF.set_allow_extrapolation(False)

        DF0 = self.DF0
        u0 = self.u0

        F = grad(DF0)
        shape = set(F.ufl_shape)
        assert len(shape) == 1
        gdim = shape.pop()
        F = Identity(gdim) + F

        form = inner(u0, dot(cofac(F), self.n))*self.dx
        value = assemble(form)

        return value


class AreaRate(FlowRate):
    '''Area of a planar crossection of the mesh Plane is (point, n)'''
    def compute(self, get):
        '''Compute on reference.'''
        DF = get('Displacement')[1]  # NOTE: ALE is passed as second argument

        if self.D is None:
            self.D = VectorFunctionSpace(self.mesh,
                                         DF.function_space().ufl_element().family(),
                                         DF.function_space().ufl_element().degree(),
                                         DF.value_size())
            self.DF0 = Function(self.D)

        # Restrict
        DF.set_allow_extrapolation(True)
        self.DF0.assign(interpolate(DF, self.D))
        DF.set_allow_extrapolation(False)

        DF0 = self.DF0

        F = grad(DF0)
        shape = set(F.ufl_shape)
        assert len(shape) == 1
        gdim = shape.pop()
        F = Identity(gdim) + F

        form = sqrt(inner(dot(self.n, cofac(F)), dot(self.n, cofac(F))))*self.dx
        value = assemble(form)

        return value


# Extrnal surface probes
class VelocityProfileExternal(Field):
    '''Velocity profile on some external surface of mesh.'''
    def __init__(self, facet_f, tag, name, params):
        # Need to create mesh for that tag. 
        local_any = int(any(1 for f in SubsetIterator(facet_f, tag)))
        global_any = facet_f.mesh().mpi_comm().tompi4py().allreduce(local_any)
        assert global_any, 'No facets marked as %d' % tag
        # We go a long way mesh -> boundarymesh -> submesh
        mesh = facet_f.mesh()
        bmesh = BoundaryMesh(mesh, 'exterior')
        bmesh_cell_f = CellFunction('size_t', bmesh, 0)
        for bmesh_cell_index, mesh_facet_index in enumerate(bmesh.entity_map(2)):
            bmesh_cell_f[bmesh_cell_index] = facet_f[int(mesh_facet_index)]

        slice_mesh = create_submesh(bmesh, bmesh_cell_f, tag)
        self.mesh = slice_mesh

        # Auxliary space
        self.V = None
        Field.__init__(self, params, name, label=None)

    def compute(self, get):
        '''Just restrict'''
        u = get('Velocity')
        # See about auxliary space
        if self.V is None:
            elV = u.function_space().ufl_element()
            assert len(elV.value_shape()) == 1
            self.V = VectorFunctionSpace(self.mesh, elV.family(), elV.degree(), elV.value_size())
        # Restrict to mesh
        u0 = interpolate_nonmatching_mesh(u, self.V)
        return u0


class FlowRateExternal(Field):
    '''Flux external surface of mesh specified by some facet_f, tags'''
    def __init__(self, facet_f, tag, name, params):
        mesh = facet_f.mesh()
        local_any = int(any(1 for f in SubsetIterator(facet_f, tag)))
        global_any = mesh.mpi_comm().tompi4py().allreduce(local_any)
        assert global_any, 'No facets marked as %d' % tag
        self.n = FacetNormal(mesh)
        # API consistency - all these field have mesh attribute
        self.mesh = mesh

        # Measure that avoids wrong cells
        self.ds = Measure('ds', domain=mesh, subdomain_data=facet_f, subdomain_id=tag)

        Field.__init__(self, params, name, label=None)


    def compute(self, get):
        '''
        int_{interior_surface} v.n dS in the deformed domain is mapped to
        reference domain.
        '''
        u = get('Velocity')
        DF = get('Displacement')[1]  # NOTE: ALE is passed as second argument

        F = grad(DF)
        shape = set(F.ufl_shape)
        assert len(shape) == 1
        gdim = shape.pop()
        F = Identity(gdim) + F

        form = inner(u, dot(cofac(F), self.n))*self.ds
        value = assemble(form)

        return value


class AreaRateExternal(FlowRateExternal):
    '''Area of a planar crossection of the mesh Plane is (point, n)'''
    def compute(self, get):
        '''Compute on reference.'''
        DF = get('Displacement')[1]  # NOTE: ALE is passed as second argument

        F = grad(DF)
        shape = set(F.ufl_shape)
        assert len(shape) == 1
        gdim = shape.pop()
        F = Identity(gdim) + F

        form = sqrt(inner(dot(self.n, cofac(F)), dot(self.n, cofac(F))))*self.ds
        value = assemble(form)

        return value

# ----------------------------------------------------------------------------

def mark_measure0(cell_f, marker=1, tol=0.99):
    '''
    Mark as marker those cells who when sorted by volume contribute to the mesh
    which has tol*volume of the total volume of the mesh.
    '''
    mesh = cell_f.mesh()
    comm = mesh.mpi_comm().tompi4py()

    local_volumes = [cell.volume() for cell in cells(mesh)]
    # Collect cell volumes on root
    all_volumes = comm.gather(local_volumes)

    # On master we decide on which cells should be rejected. We do this by the
    # following reasoning. Let V be the total volume of the slice. If I sort out
    # cells by their volume, then start adding the volumes of n largest cells the
    # for some n I have a domain whose volume is within TOL close to V. The
    # remaining cells are rejected.
    rejected = []
    if comm.rank == 0:
        chunks = [0] + map(len, all_volumes)
        pieces = np.cumsum(chunks)
       
        all_volumes = np.hstack(all_volumes)
        sorted_indices = np.argsort(all_volumes)[::-1]

        all_volumes = all_volumes[sorted_indices]
        total_volume = sum(all_volumes)
        cum_volumes = np.cumsum(all_volumes)
        approximation = cum_volumes/total_volume

        # It can happen that each element contributes significantly (e.g. equal size)
        cut_off = np.where(approximation > tol)[0]
        if len(cut_off):
            cut_off = cut_off[0]
        else:
            cut_off = 0
        rejected = sorted_indices[cut_off:]

        # It remains to find out where the rejected elements lived
        def find_proc(index, pieces=pieces):
            '''Find process and local index from global one.'''
            for proc in range(comm.size):
                first, last = pieces[proc], pieces[proc+1]
                if first <= index < last:
                    return proc, index-first
            assert False

        proc_map = {proc: [] for proc in range(comm.size)}
        for index in rejected:
            proc, local = find_proc(index)
            proc_map[proc].append(local)
        # Collapse for scattering    
        rejected = [proc_map[proc] for proc in range(comm.size)]
    rejected = comm.scatter(rejected)

    # Now we can avoid cells
    for cell in cells(mesh): cell_f[cell] = marker*int(cell.index() not in rejected)

    return cell_f

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from cbcpost.utils import create_slice
    import numpy as np

    mesh = UnitCubeMesh(10, 10, 10)

    x = np.array([0.5, 0.5, 0.5])
    n = np.array([1., 1., 1.])

    V = VectorFunctionSpace(mesh, 'CG', 1)

    ######################
    # Internal
    ######################
    # AREA
    # Without motion we should be close to hexagon area
    get = lambda key: {'Displacement': (None, 
                                        interpolate(Constant((0, 0, 0)), V))}[key]
    a = AreaRate(mesh, point=x, n=n, name='test', params={})
    assert abs(a.compute(get) - 3*sqrt(3)/4) < 1E-13

    # Simple moving
    get = lambda key: {'Displacement': (None, 
                                        interpolate(Expression(('x[0]',
                                                                'x[1]',
                                                                'x[2]'),
                        degree=1), V))}[key]
    a = AreaRate(mesh, point=x, n=n, name='test', params={})
    assert abs(a.compute(get) - 3*sqrt(3)) < 1E-13

    # FLOW RATE
    # Flow through constant and parallel flow.
    A = sqrt(3)/3
    get = lambda key: {'Velocity': interpolate(Constant((A, A, A)), V),
                       'Displacement': (None, 
                                        interpolate(Constant((0, 0, 0)), V))}[key]
    a = FlowRate(mesh, point=x, n=n, name='test', params={})
    assert abs(a.compute(get) - 3*sqrt(3)/4) < 1E-13

    # Orthogonal flow
    get = lambda key: {'Velocity': interpolate(Expression(('x[0]-x[2]',
                                                           'x[1]+x[2]',
                                                           '-x[0]-x[1]'), degree=1), V),
                       'Displacement': (None, 
                                        interpolate(Constant((0, 0, 0)), V))}[key]
    a = FlowRate(mesh, point=x, n=n, name='test', params={})
    assert abs(a.compute(get)) < 1E-13

    ######################
    # External
    ######################
    mesh = UnitCubeMesh(10, 10, 10)
    facet_f = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[2], 0)').mark(facet_f, 1)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    # AREA
    # Without motion we should be close to hexagon area
    get = lambda key: {'Displacement': (None, 
                                        interpolate(Constant((0, 0, 0)), V))}[key]
    a = AreaRateExternal(facet_f, 1, name='test', params={})
    assert abs(a.compute(get) - 1) < 1E-13

    # Simple moving
    get = lambda key: {'Displacement': (None, 
                                        interpolate(Expression(('x[0]',
                                                                'x[1]',
                                                                'x[2]'),
                        degree=1), V))}[key]
    a = AreaRateExternal(facet_f, 1, name='test', params={})
    assert abs(a.compute(get) - 4) < 1E-13

    # FLUX
    get = lambda key: {'Velocity': interpolate(Expression(('0', '0', 'x[0]+x[1]'), degree=1), V),
                       'Displacement': (None, 
                                        interpolate(Constant((0, 0, 0)), V))}[key]
    a = FlowRateExternal(facet_f, 1, name='test', params={})
    assert abs(a.compute(get) + 1) < 1E-13

    a = VelocityProfileExternal(facet_f, 1, name='test',
            params=dict(save=True, save_as='xdmf'))
    u = a.compute(get)
    assert errornorm(Expression(('0', '0', 'x[0]+x[1]'), degree=1), u, 'L2') < 1E-13
