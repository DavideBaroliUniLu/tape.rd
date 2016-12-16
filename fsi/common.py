from cbcpost.utils import boundarymesh_to_mesh_dofmap, mesh_to_boundarymesh_dofmap
from distutils.version import LooseVersion
from cbcpost.utils import get_set_vector
from dolfin import *
import numpy as np
import ufl

try:
    traverse = ufl.algorithms.traverse_unique_terminals
except AttributeError:
    traverse = ufl.algorithms.traversal.iter_expressions


__TOL__ = 1E-12


class BoundaryProlongator(object):
    '''
    Set function f in V(domain) such that it matches fb in Vb(boundary) on
    boundary.
    '''
    def __init__(self, Vb, bmesh, V, tol=__TOL__):
        '''Precompute the maps. Compatibility of Vb, V in Oywind's code.'''
        local_dofmapping = d_boundarymesh_to_mesh_dofmap(bmesh, Vb, V)
        _keys, _values = zip(*local_dofmapping.iteritems())
        self._keys = np.array(_keys, dtype=np.intc)
        self._values = np.array(_values, dtype=np.intc)

        self.Vb = Vb
        self.V = V
        if tol > __TOL__:
            print 'Boundary->Domain should be more accurate using %g' % __TOL__
            tol = __TOL__
        self.tol = tol

    def map(self, fb, f):
        '''Map fb to f'''
        V = f.function_space()
        assert V.ufl_element() == self.V.ufl_element()
        assert V.dim() == self.V.dim()
        
        # Function to Function
        if isinstance(fb, Function):
            Vb = fb.function_space()
            assert Vb.ufl_element() == self.Vb.ufl_element()
            assert Vb.dim() == self.Vb.dim()

            get_set_vector(f.vector(), self._keys, fb.vector(), self._values)
            return -1

        # This might be UFL-expression to function
        else:
            # Make sure that no Trial/Test Functions are present
            assert not any(isinstance(arg, Argument) for arg in traverse(fb))

            # Assemble the form on Vb (f, q)
            q = TestFunction(self.Vb)
            if not hasattr(self, 'b'):
                self.b = assemble(inner(fb, q)*dx)
                self.x = Function(self.V).vector()
            else:
                assemble(inner(fb, q)*dx, self.b)

            # Prolongate
            get_set_vector(self.x, self._keys, self.b, self._values)

            # Scale back
            if not hasattr(self, 'solver'):
                u = TrialFunction(self.V)
                v = TestFunction(self.V)
                M = assemble(inner(u, v)*ds+Constant(0)*inner(u, v)*dx)
                M.ident_zeros()  
                # Otherwise singular mat. NOTE: breaks symmetry - so bicgstab
                # FIXME: petsc has zeroRowsColums that could help here
                if self.tol < 0:
                    solver = PETScLUSolver('superlu_dist')
                    solver.set_operator(M)
                    solver.parameters['reuse_factorization'] = True
                else:
                    solver = KrylovSolver('bicgstab', 'hypre_amg')
                    solver.set_operators(M, M)
                    solver.parameters['relative_tolerance'] = self.tol
                    solver.parameters['absolute_tolerance'] = self.tol
                self.solver = solver

            niters = self.solver.solve(f.vector(), self.x) 

            return niters


class BoundaryRestrictor(object):
    '''
    Set function fb in Vb(boundary) to be a boundary restriction of f in
    V(domain)
    '''
    def __init__(self, V, bmesh, Vb, tol=__TOL__):
        '''Precompute the maps. Compatibility of Vb, V in Oywind's code.'''
        local_dofmapping = d_mesh_to_boundarymesh_dofmap(bmesh, V, Vb)
        self._keys = np.array(local_dofmapping.keys(), dtype=np.intc)
        self._values = np.array(local_dofmapping.values(), dtype=np.intc)
        self._temp_array = np.zeros(len(self._keys), dtype=np.float_)

        self.Vb = Vb
        self.V = V
        if tol > __TOL__:
            print 'Domain->Boundary should be more accurate using %g' % __TOL__
            tol = __TOL__
        self.tol = tol

    def map(self, f, fb):
        '''Map f to fb'''
        Vb = fb.function_space()
        assert Vb.ufl_element() == self.Vb.ufl_element()
        assert Vb.dim() == self.Vb.dim()
        
        # Function to Function
        if isinstance(f, Function):
            V = f.function_space()
            assert V.ufl_element() == self.V.ufl_element()
            assert V.dim() == self.V.dim()

            get_set_vector(fb.vector(), self._keys, f.vector(), self._values, self._temp_array)
            return 0

        # This might be UFL-expression to function
        else:
            # Make sure that no Trial/Test Functions are present
            assert not any(isinstance(arg, Argument) for arg in traverse(f))

            # Assemble the form on V (f, q)
            q = TestFunction(self.V)
            if not hasattr(self, 'b'):
                self.b = assemble(inner(f, q)*ds)
                self.x = Function(self.Vb).vector()
            else:
                assemble(inner(f, q)*ds, self.b)

            # Prolongate
            get_set_vector(self.x, self._keys, self.b, self._values)

            # Scale back
            if not hasattr(self, 'solver'):
                u = TrialFunction(self.Vb)
                v = TestFunction(self.Vb)
                M = assemble(inner(u, v)*dx)

                if self.tol < 0:
                    solver = PETScLUSolver('superlu_dist')
                    solver.set_operator(M)
                    solver.parameters['reuse_factorization'] = True
                else:
                    solver = KrylovSolver('cg', 'hypre_amg')
                    solver.set_operators(M, M)
                    solver.parameters['relative_tolerance'] = self.tol
                    solver.parameters['absolute_tolerance'] = self.tol
                self.solver = solver

            niters = self.solver.solve(fb.vector(), self.x) 

            return niters

# The following code is taken from cbcpost but there is slight modification
# because we need a different API. Specifically, we shall pass as boundary the
# dictionary of entitity maps. This allows for usage not just with BoundaryMesh.
def d_boundarymesh_to_mesh_dofmap(boundary, Vb, V):
    "Find the mapping from dofs on boundary FS to dofs on full mesh FS"
    # Call cbc
    if isinstance(boundary, BoundaryMesh):
        mapping = mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own="bdof")
    else:
        assert isinstance(boundary, dict)
        mapping = d_mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own="bdof")

    mapping = dict((v,k) for k,v in mapping.iteritems())
    return mapping


def d_mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own="cdof"):
    "Find the mapping from dofs on full mesh FS to dofs on boundarymesh FS"
    if isinstance(boundary, BoundaryMesh):
        return mesh_to_boundarymesh_dofmap(boundary, V, Vb, _should_own)

    assert isinstance(boundary, dict)
    assert V.ufl_element().family() == Vb.ufl_element().family()
    assert V.ufl_element().degree() == Vb.ufl_element().degree()
    assert _should_own in ["cdof", "bdof"]

    # Currently only CG1 and DG0 spaces are supported
    assert V.ufl_element().family() in ["Lagrange", "Discontinuous Lagrange"]
    if V.ufl_element().family() == "Discontinuous Lagrange":
        assert V.ufl_element().degree() == 0
    else:
        assert V.ufl_element().degree() == 1

    mesh = V.mesh()
    bmesh = Vb.mesh()
    D = bmesh.topology().dim()

    V_dm = V.dofmap()
    Vb_dm = Vb.dofmap()

    dofmap_to_boundary = {}

    # Extract maps from boundary to mesh
    vertex_map = boundary[0]
    cell_map = boundary[D]

    for i in xrange(len(cell_map)):
        boundary_cell = Cell(bmesh, i)
        mesh_facet = Facet(mesh, cell_map[i])
        mesh_cell_index = mesh_facet.entities(D+1)[0]
        mesh_cell = Cell(mesh, mesh_cell_index)

        cell_dofs = V_dm.cell_dofs(mesh_cell_index)
        boundary_dofs = Vb_dm.cell_dofs(i)

        if V_dm.num_entity_dofs(0) > 0:
            for v_idx in boundary_cell.entities(0):

                mesh_v_idx = vertex_map[int(v_idx)]
                mesh_list_idx = np.where(mesh_cell.entities(0) == mesh_v_idx)[0][0]
                boundary_list_idx = np.where(boundary_cell.entities(0) == v_idx)[0][0]

                bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(0, boundary_list_idx)]
                cdofs = cell_dofs[V_dm.tabulate_entity_dofs(0, mesh_list_idx)]

                for bdof, cdof in zip(bdofs, cdofs):
                    if LooseVersion(dolfin_version()) > LooseVersion("1.4.0"):
                        bdof = Vb_dm.local_to_global_index(bdof)
                        cdof = V_dm.local_to_global_index(cdof)

                    if _should_own == "cdof" and not (V_dm.ownership_range()[0] <= cdof < V_dm.ownership_range()[1]):
                        continue
                    elif _should_own == "bdof" and not (Vb_dm.ownership_range()[0] <= bdof < Vb_dm.ownership_range()[1]):
                        continue
                    else:
                        dofmap_to_boundary[bdof] = cdof

        if V_dm.num_entity_dofs(D+1) > 0 and V_dm.num_entity_dofs(0) == 0:
            bdofs = boundary_dofs[Vb_dm.tabulate_entity_dofs(D,0)]
            cdofs = cell_dofs[V_dm.tabulate_entity_dofs(D+1,0)]
            for bdof, cdof in zip(bdofs, cdofs):
                if LooseVersion(dolfin_version()) > LooseVersion("1.4.0"):
                    bdof = Vb_dm.local_to_global_index(bdof)
                    cdof = V_dm.local_to_global_index(cdof)

                dofmap_to_boundary[bdof] = cdof

    return dofmap_to_boundary

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from boundary_mesh import py_BoundaryMesh
    
    # NOTE: the tests below check that mapping functions by prolongations and
    # restructions is exact. For forms the error depends on the tolerance of the
    # linear iterative solver. Still what is observed is well below 1E-10. For
    # mapping of the traction force there is order 1 convergence in the average
    # error of the traction vectors in points of Vb. The error is mosly located
    # at corners
    all_errors = []
    for n_cells in (64, 128, 256, 512):
        errors = []
        mesh = UnitSquareMesh(n_cells, n_cells)

        if False:
            bmesh = BoundaryMesh(mesh, 'exterior')
            emap = bmesh
        else:
            markers = FacetFunction('size_t', mesh, 1)
            bmesh, emap = py_BoundaryMesh(mesh, markers, 1)

        V = FunctionSpace(mesh, 'CG', 1)
        Vb = FunctionSpace(bmesh, 'CG', 1)
        comm = mesh.mpi_comm().tompi4py()

        # Check prolongation
        P = BoundaryProlongator(Vb, emap, V, -1)
        # Foo to foo
        fb = Function(Vb)
        fb.vector().set_local(np.random.rand(fb.vector().local_size()))
        fb.vector().apply('insert')

        f = Function(V)

        P.map(fb, f)

        e = max(abs(f(v.x(0), v.x(1)) - fb(v.x(0), v.x(1))) for v in vertices(bmesh))
        e = comm.reduce(e)
        if comm.rank == 0: errors.append(e)
        
        # UFL to foo
        fb = Expression('x[0]+x[1]', degree=1)
        f.vector().zero()
        P.map(fb, f)

        e = max(abs(f(v.x(0), v.x(1)) - fb(v.x(0), v.x(1))) for v in vertices(bmesh))
        e = comm.reduce(e)
        if comm.rank == 0: errors.append(e)

        # Check restriction
        R = BoundaryRestrictor(V, emap, Vb)
        # Foo to foo
        f = Function(V)
        f.vector().set_local(np.random.rand(f.vector().local_size()))
        f.vector().apply('insert')

        fb = Function(Vb)

        R.map(f, fb)

        e = max(abs(f(v.x(0), v.x(1)) - fb(v.x(0), v.x(1))) for v in vertices(bmesh))
        e = comm.reduce(e)
        if comm.rank == 0: errors.append(e)

        # UFL to foo
        f = interpolate(Expression('x[0]+x[1]', degree=1), V)
        fb.vector().zero()
        R.map(f, fb)

        e = max(abs(f(v.x(0), v.x(1)) - fb(v.x(0), v.x(1))) for v in vertices(bmesh))
        e = comm.reduce(e)
        if comm.rank == 0: errors.append(e)

        # Something what we have in mind
        x, y = SpatialCoordinate(mesh)

        u = as_vector((x**2, x+y))
        p = x + y
        F = as_matrix(((1, 2), (3, 4)))
        viscosity = Constant(2)
        n = FacetNormal(mesh)
       
        sigma_f = lambda u, p, F, viscosity: -p*Identity(F.ufl_shape[0])+\
                                             2*viscosity*sym(dot(grad(u), inv(F)))
        stress = sigma_f(u, p, F, viscosity)
        form = dot(dot(stress, cofac(F)), n)

        # Compute stress on the boundary
        V = VectorFunctionSpace(mesh, 'CG', 1)
        Vb = VectorFunctionSpace(bmesh, 'CG', 1)
        R = BoundaryRestrictor(V, emap, Vb)
        fb = Function(Vb)
        R.map(form, fb)

        # Let's see about the error. Look at right edge where normal is (1, 0)
        right = CompiledSubDomain('near(x[0], 1.)')
        n = Constant((1, 0))
        # The force in n direction 
        form = dot(dot(stress, cofac(F)), n)
        f = project(form, V)

        # We look at the average error. Converges.
        Errors = []
        for vertex in vertices(bmesh):
            x = np.array([vertex.x(0), vertex.x(1)])
            if right.inside(x, False):
                error = np.linalg.norm(fb(x)-f(x))
                # print f(x), fb(x), 'error', error, '@', x
                Errors.append(error)
        e = 0.
        e, n = float(sum(Errors)), len(Errors)
        e, n = comm.reduce(e), comm.reduce(n)
        if comm.rank == 0: errors.append(e/n)

        ##########################
        # Getting normal component
        ##########################
        # Fixme error here
        V = VectorFunctionSpace(mesh, 'CG', 1)
        Vb = VectorFunctionSpace(bmesh, 'CG', 1)
        Q = FunctionSpace(mesh, 'CG', 1)
        Qb = FunctionSpace(bmesh, 'CG', 1)
      
        n = FacetNormal(mesh)
        Q_to_Qb = BoundaryRestrictor(Q, emap, Qb)
        # The force in n direction 
        form = dot(dot(dot(stress, cofac(F)), n), n)
        normal_t = Function(Qb)
        Q_to_Qb.map(form, normal_t)
        # Let's see about the error
        n = Constant((1, 0))
        # The force in n direction 
        form = dot(dot(dot(stress, cofac(F)), n), n)
        normal_t0 = project(form, Q)
        # We look at the average error. Converges.
        Errors = []
        for vertex in vertices(bmesh):
            x = np.array([vertex.x(0), vertex.x(1)])
            if right.inside(x, False):
                error = np.linalg.norm(normal_t0(x)-normal_t(x))
                Errors.append(error)
        e = 0.
        e, n = float(sum(Errors)), len(Errors)
        e, n = comm.reduce(e), comm.reduce(n)
        if comm.rank == 0: errors.append(e/n)

        # Resctrict normal vector to Vb
        n = FacetNormal(mesh)
        V = VectorFunctionSpace(mesh, 'DG', 0)
        Vb = VectorFunctionSpace(bmesh, 'DG', 0)
        V_to_Vb = BoundaryRestrictor(V, emap, Vb)
        normal_Vb = Function(Vb)
        V_to_Vb.map(Constant(1)*n, normal_Vb)

        Errors = []
        for cell in cells(bmesh):
            x, y = cell.midpoint().x(), cell.midpoint().y()
            if near(x, 0.0):
                n = [-1, 0]
            elif near(x, 1.0):
                n = [1, 0]
            elif near(y, 0):
                n = [0, -1]
            else:
                n = [0, 1]
            n = np.array(n)
            x = [x, y]
            e = np.linalg.norm(normal_Vb(x)-n)
            Errors.append(e)
        e = 0.
        e, n = float(sum(Errors)), len(Errors)
        e, n = comm.reduce(e), comm.reduce(n)
        if comm.rank == 0: errors.append(e/n)

        # Prolongate it back
        mag = interpolate(Constant(1), Qb)
        foo = Function(V) 
        Vb_to_V = BoundaryProlongator(Vb, emap, V)
        bar = normal_Vb*mag
        Vb_to_V.map(bar, foo)

        Errors = []
        for cell in cells(bmesh):
            x, y = cell.midpoint().x(), cell.midpoint().y()
            if near(x, 0.0):
                n = [-1, 0]
            elif near(x, 1.0):
                n = [1, 0]
            elif near(y, 0):
                n = [0, -1]
            else:
                n = [0, 1]
            n = np.array(n)
            x = [x, y]
            e = np.linalg.norm(foo(x)-n)
            Errors.append(e)
        e = 0.
        e, n = float(sum(Errors)), len(Errors)
        e, n = comm.reduce(e), comm.reduce(n)
        if comm.rank == 0: errors.append(e/n)

        all_errors.append(errors)
    # Let root see about convergence
    if comm.rank == 0:
        all_errors = np.array(all_errors)
        nrows, ncols = all_errors.shape
        # 'Exact'
        for row in range(nrows): assert all(all_errors[row, col] < 1E-10 for col in range(4))
        # 'Convergece'
        for row in range(1, nrows):
            assert (all_errors[row, col] < all_errors[row-1, col] for col in range(4, ncols))
