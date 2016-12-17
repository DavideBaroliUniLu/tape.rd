from boundary_mesh import py_BoundaryMesh
from itertools import combinations, groupby
from collections import defaultdict
from sys import maxsize
import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from cbcpost import *
from cbcpost.utils import *
from cbcflow import *
from cbcflow.schemes.utils import *
from dolfin import *
from sympy.printing import ccode


class FSIProblem(NSProblem):
    @classmethod
    def default_params(cls):
        """Returns the default parameters for an FSI problem.
        (Extends NSProblem.default_params())

        Explanation of parameters:

        Physical parameters:

          - E: float, kinematic viscosity
          - rho_s: float, mass density
          - R: float, Radius to use
          - h: float, Wall thickness
          - nu: float, Poisson ratio

        """
        params = NSProblem.default_params()
        params.update(
            E=1.0,
            rho_s=1.0,
            R = 1.0,
            h = 1.0,
            nu = 0.3,
            alpha0 = 1,
            alpha1 = 1e-3)

        return params


##############################################################################
# UFL short-cuts
##############################################################################
def par(f, n):
    "Return parallel/tangential component of f"
    return f - dot(f, n)*n


def Epsilon(u, F):
    'Dynamic fluid stress with grad expressed in ALE domain'
    return 0.5*(grad(u)*inv(F) + inv(F).T*grad(u).T)


def Sigma(mu, u, p, F):
    'Fluid stress with grad expressed in ALE domain'
    return -p*Identity(F.ufl_shape[0]) + 2*mu*Epsilon(u, F)

##############################################################################
# Utils of scheme and bcs
##############################################################################
class Extrapolation(Function):
    "Helper class for creating an extrapolated function"
    def __init__(self, V, k):
        Function.__init__(self, V)
        self.k = k
        self.funcs = [Function(V) for i in range(k)]

    def update(self, f):
        if self.k == 0:
            return
        for i in range(self.k-1):
            self.funcs[i] = self.funcs[i+1]

        self.funcs[-1].assign(f)
        self.vector().zero()

        if self.k == 1:
            self.vector().axpy(1.0, self.funcs[0].vector())
        elif self.k == 2:
            self.vector().axpy(2.0, self.funcs[0].vector())
            self.vector().axpy(-1.0, self.funcs[1].vector())


class AbsorbingStress(Constant):
    "Implemented from Nobile and Vergara paper"
    def __init__(self, problem, facet_domains, indicator, p_ext):
        Constant.__init__(self, 0)
        self.ds = Measure('ds',
                          domain=problem.mesh,
                          subdomain_data=problem.facet_domains,
                          subdomain_id=indicator)

        self.problem = problem
        self.A0 = assemble(Constant(1)*self.ds)
        self.rho_f = problem.params.rho
        self.n = FacetNormal(problem.mesh)
        self.p_ext = p_ext

    def update(self, u, DF, t):
        # Update time for the external pressure
        time = float(t)
        self.p_ext.t = time

        problem = self.problem
        dim = problem.mesh.geometry().dim()
        F = Identity(dim) + grad(DF)
        n = self.n

        form = inner(dot(n, cofac(F)), dot(n, cofac(F)))*self.ds
        An = assemble(form)
        Fn = assemble(dot(u, dot(cofac(F), self.n))*self.ds)

        Rn = sqrt(An/np.pi)  # Assuming the circle is more-or-less preserved
        beta = problem.params.E*problem.params.h/(1-problem.params.nu**2)*1.0/Rn**2
        # Computed from NS.
        val = ((sqrt(self.rho_f)/(2*sqrt(2))*Fn/An + sqrt(beta*sqrt(self.A0)))**2 - beta*sqrt(self.A0))
        # Add the current external stuff
        val += self.p_ext(time)         # or 0, p_ext is supposed to be constant in space
        self.assign(val)


##############################################################################
# External forcing
##############################################################################
def pressure_transition(pLeft, pRight, w, nderivs=2, degree=None):
    '''
    Smooth transition from constant pLeft value at -w to pRight value at w.
    The transition is done by a polynomial with support [-w, w] which mathces
    p* values and their (constant) derivatives of order 1, ... nderics.

    The output is a an Expression which is
            / pLeft x < w
    f(x) =  | polynomial x in (-w, w)
            \ pRight x > w

    The Expression is such that for given nderivs/polynomial degree the JIT is
    triggered only once and different pLeft, pRight, w values and the polynomial
    coeffs are substituted as arguemnts.
    '''
    n = nderivs + 1
    x = sp.symbols('x[0]')
    ais = sp.symbols(','.join(['a%d' % i for i in range(2*n)]))
    p = sum(ai*x**i for i, ai in enumerate(ais))

    # Build the equations: match values at ends, 0 derivs ...
    b = sp.Matrix([pLeft, pRight] + [0]*(2*nderivs))

    rows = []
    for i in range(n):
        row = [p.diff(x, i).subs(x, -w).coeff(ai) for ai in ais]
        rows.append(row)

        row = [p.diff(x, i).subs(x, w).coeff(ai) for ai in ais]
        rows.append(row)

    A = sp.Matrix(rows)
    coefs = A.solve(b)

    # If there is no degree, we are exact
    degree = len(coefs) if degree is None else degree

    args = ', '.join(['%s=%g' % kv for kv in zip(ais, coefs)])
    cases = "Expression('(x[0] < -w) ? pL : ((x[0] > w) ? pR : %s)'"
    body = cases % ccode(p)
    f = eval("%s, %s, pL=%g, pR=%g, w=%g, degree=%d)" % (body, args, pLeft, pRight, w, degree))

    return f


def external_pressure(yLeft, xLeft, yRight, xRight, w, yMiddle=0, nderivs=2, degree=None):
    '''
                / yLeft       x < xLeft -w
    p_ext(x) =  | yMiddle     xLeft + w < x < xRight - w 
                \ yRight      x > xRight + w

    There is a smooth transition between.
    The difference between here and pressure transition is that here there are
    two transition regions.
    '''   
    n = nderivs + 1
    x = sp.symbols('x[0]')

    # Handle left
    ais = sp.symbols(','.join(['a%d' % i for i in range(2*n)]))
    p = sum(ai*x**i for i, ai in enumerate(ais))
    # Build the equations: value, 0
    b = sp.Matrix([yLeft, 0] + [0]*(2*nderivs))

    rows = []
    for i in range(n):
        row = [p.diff(x, i).subs(x, xLeft-w).coeff(ai) for ai in ais]
        rows.append(row)

        row = [p.diff(x, i).subs(x, xLeft+w).coeff(ai) for ai in ais]
        rows.append(row)
     
    A = sp.Matrix(rows)
    coefs = A.solve(b)
    # If there is no degree, we are exact
    degree = len(coefs) if degree is None else degree
    args = ', '.join(['%s=%g' % kv for kv in zip(ais, coefs)])
    cases = "Expression('(x[0] < xL - w) ? yL : ((x[0] < xL + w) ? %s : yM)'"
    body = cases % ccode(p)
    fL = eval("%s, %s, xL=%g, w=%g, yL=%g, yM=%g, degree=%d)" % (body, args, xLeft, w, yLeft, yMiddle, degree))

    # Handle right
    bis = sp.symbols(','.join(['b%d' % i for i in range(2*n)]))
    p = sum(bi*x**i for i, bi in enumerate(bis))
    # Build the equations: value, 0
    b = sp.Matrix([0, yRight] + [0]*(2*nderivs))

    rows = []
    for i in range(n):
        row = [p.diff(x, i).subs(x, xRight-w).coeff(bi) for bi in bis]
        rows.append(row)

        row = [p.diff(x, i).subs(x, xRight+w).coeff(bi) for bi in bis]
        rows.append(row)
    
    A = sp.Matrix(rows)
    coefs = A.solve(b)
    # If there is no degree, we are exact
    args = ', '.join(['%s=%g' % kv for kv in zip(bis, coefs)])
    cases = "Expression('(x[0] > xR + w) ? yR : ((x[0] > xR - w) ? %s : yM)'"
    body = cases % ccode(p)
    fR = eval("%s, %s, xR=%g, w=%g, yR=%g, yM=%g, degree=%d)" % (body, args, xRight, w, yRight, yMiddle, degree))

    # Combine
    f = Expression('(x[0] < xM) ? fL : fR', xM=0.5*(xLeft+xRight), fL=fL, fR=fR, degree=degree)

    return f, fL, fR


def characteristic_function(domain, subdomain, subdomain_id):
    '''
    Scalar DG0 function over domain which takes value 1 where 
    subdomain[cell] == subdomain_id.
    '''
    assert isinstance(subdomain, MeshFunctionSizet), str(type(subdomain))
    gdim, tdim = domain.geometry().dim(), domain.topology().dim()
    assert subdomain.dim() == tdim

    V = FunctionSpace(domain, 'DG', 0)
    v = TestFunction(V)
    dm = dx(domain=domain, subdomain_data=subdomain, subdomain_id=subdomain_id)
    form = inner(Constant(1)/CellVolume(domain), v)*dm
    f = Function(V)
    assemble(form, tensor=f.vector())

    return f

##############################################################################
# Vittamed forcing stuff
##############################################################################
class StepFoo(Constant):
    '''
    f(t) which starting from v0 increments every dt its value by dv.
    '''
    def __init__(self, v0=0, delta=0, dt=1e9):
        """
        v0    : initial value
        dt    : How often to update value
        delta : How much to increae value each time step
        NB! changed names to correspond to json
        """
        self.v0 = v0
        # Catch when StepFoo is intended to be used as a constant
        self.dv = None if delta == 0 else delta
        self.dt = dt
        self._t = 0.
        Constant.__init__(self, v0)

    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        '''Set time to t thereby increasing the value.'''
        self._t = t
        # dv was 0
        if self.dv is None:
            self.assign(self.v0)
        # Something else
        else:
            self.assign(self.v0 + self.dv*int((t/self.dt)))

# -----------

class ExternalForcing(Function):
    '''
    Function where the spatial dependence is based on values of boundaries;
    for each marker there is a different constant value.
    '''
    def __init__(self, mesh, family, degree, boundaries, marker_values, debug=False):
        '''
        V is in FunctionSpace(mesh, family, degree) taking value
        marker_values[marker] for x in that is in enitity(facet)
        boundaries[facet] == marker.
        '''
        assert mesh.geometry().dim() == 3 and mesh.topology().dim() == 3
        assert all(isinstance(v, Constant) for v in marker_values.values())
        if family == 'DG': assert degree == 0
        # There's more ways to do it. Here bdry_markers which is facet_f of mesh
        # is translated to cell_f of bmesh. Interfaces are found and introduced
        # into cell_f. Consequently dofs on the V facet_s are collected for each
        # marker.
        markers = marker_values.keys()
        bmesh, emap  = py_BoundaryMesh(mesh, boundaries, markers)
        # IDEA: Interface between domains with different markers can be more 
        # cheaply computed on the boundary mesh. Transfer the markers from mesh 
        # facet to bmesh cells.
        cell_2_facet = emap[2]
        # Transfer
        cell_f = CellFunction('size_t', bmesh, maxsize)
        for cell in cells(bmesh): cell_f[cell] = boundaries[int(cell_2_facet[cell.index()])]
        # Next we want to find interfaces: they are given as new markers in
        # cell_f, also need to know which values (i.e. marker_values[marker]) to
        # used
        cell_f, iface_map = find_interfaces(bmesh, cell_f, markers)
        # Now we can collect stuff for populating the function
        V = FunctionSpace(mesh, family, degree)
        Function.__init__(self, V)
        # New markers
        markers.extend(iface_map.keys())

        # Find dofs of the marked facets
        dofmap = V.dofmap()
        first, last = dofmap.ownership_range()
        mesh.init(2, 3)
        mfacet_2_mcell = mesh.topology()(2, 3)

        marker_dofs = {}
        for m in markers:
            m_dofs = []
            for bmesh_cell in SubsetIterator(cell_f, m):
                mesh_facet = cell_2_facet[bmesh_cell.index()]
                mesh_cell = mfacet_2_mcell(mesh_facet)
                assert len(mesh_cell) == 1
                mesh_cell = int(mesh_cell)

                cell_dofs = dofmap.cell_dofs(mesh_cell)
                if not family == 'DG':
                    for index, facet in enumerate(facets(Cell(mesh, mesh_cell))):
                        if facet.index() == mesh_facet:
                            cell_dofs = cell_dofs[dofmap.tabulate_facet_dofs(index)]
                            break
                m_dofs.extend(cell_dofs.tolist())
            # Only local unique
            m_dofs = set(m_dofs)
            m_dofs = filter(lambda dof: first <= dofmap.local_to_global_index(dof) < last, m_dofs)
            marker_dofs[m] = m_dofs
        # Remember
        self.marker_values = marker_values
        self.iface_values = iface_map
        self.marker_dofs = marker_dofs
        self.array = self.vector().get_local()
        self._t = 0
        self.t = self._t  # Sets the values of the function in marked domains
           
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        self._t = t
        array = self.array
        # Those who update
        for m, value in self.marker_values.items():
            if isinstance(value, StepFoo): 
                value.t = self._t
            array[self.marker_dofs[m]] = value(0)
        # Iface
        for m, values in self.iface_values.items():
            v0, v1 = values
            value = 0.5*(self.marker_values[v0](0) + self.marker_values[v1](0))
            array[self.marker_dofs[m]] = value

        self.vector().zero()
        self.vector().set_local(array)
        self.vector().apply('insert')

def find_interfaces(bmesh, cell_f, markers):
    # We will only look at interfaces of 2 domain!
    # This will be a unique marker within bmesh cell markers
    root = max(markers) + 1
    interfaces = {i: m for m, i in enumerate(combinations(markers, 2), root)}

    # We want to find interfaces. A vertex is on an interface if the cells that
    # share it have different markers
    bmesh.init(0, 2)
    bmesh.init(2, 0)
    topology = bmesh.topology()

    c2v = topology(2, 0)
    v2c = topology(0, 2)
    # Cell-cell connectivity is defined via vertices
    c2c = {cell.index(): map(int, set(sum((v2c(v).tolist() for v in c2v(cell.index())), [])))
           for cell in cells(bmesh)}
    # Find the interface and assign marker to it
    interface_cells = defaultdict(list)
    used = set([])
    for cell in cells(bmesh):
        values = set(cell_f[c] for c in c2c[cell.index()])
        count = len(values)
        # Definitely an interface
        if count > 1:
            values = tuple(values)
            assert values in interfaces, '%s has some unknown marker %s' % (values, markers)
            interface_cells[interfaces[values]].append(int(cell.index()))
            used.add(values)

    # Now handle the case of interfaces which are on CPU boundaries.
    # Figure out what markers do the cells on different CPUs have that share the
    # vertex. Then classify (global) uniquely the marker. The value is transfered 
    # to local cells
    local_shared_vertices = topology.shared_entities(0).keys()
    local_2_global_vertex = topology.global_indices(0)
    # Look up markers of locally connected cells
    gv_markers, shared_local_2_global = [], []
    for lv in local_shared_vertices:
        lv_markers = list(set(cell_f[c] for c in map(int, v2c(lv))))
        gv = local_2_global_vertex[lv]
        gv_markers.append((gv, lv_markers))
        shared_local_2_global.append(gv)
    # Communicate these findings and group by global vertex 
    comm = bmesh.mpi_comm().tompi4py()
    gv_markers = comm.allreduce(gv_markers)

    first = lambda iterable: next(iter(iterable))
    gv_tag = {}
    for gv, grouped in groupby(gv_markers, first):
        values = set(sum((m for _, m in grouped), []))
        count = len(values)
        if count > 1:
            values = tuple(values)
            assert values in interfaces, values
            gv_tag[gv] = interfaces[values]

    # Introduce interface markers to bmesh markers
    for marker, icells in interface_cells.items():
        for c in icells: cell_f[c] = marker
    # Now the local cells connected to labelled vertex can be labelled too
    for lv, gv in zip(local_shared_vertices, shared_local_2_global):
        connected_cells = map(int, v2c(lv))
        if gv in gv_tag:
            tag = gv_tag[gv]
            for cell in connected_cells: cell_f[cell] = tag
    
    return cell_f, {v: k for k, v in interfaces.items() if k in used}


class PressureInterpolator(Constant):   
    '''Obtain the value at time t from interpolated data.'''
    def __init__(self, filepath, period=None):
        self._time = 0.
        self.period = period
        self.f = self.interpolate(filepath)
        # Dummy value
        Constant.__init__(self, 0.)
        # Set the value by time
        self.t = 0.

    def interpolate(self, filepath):     
        '''Interpolant from loaded data.'''
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            assert self.period is not None
            x = np.linspace(0, self.period, len(data))
            y = data
        else:
            assert data.ndim == 2
            x, y = data[:, 0], data[:, 1]
            self.period = max(x) - min(x)
        return interp1d(x, y)

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, t):
        self._time = t
        # Periodicity
        while self._time > self.period: self._time -= self.period
        self.assign(float(self.f(self._time)))

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from common import BoundaryRestrictor
    # TESTS
    # Characteristic function should integrate to marked area
    mesh = UnitCubeMesh(10, 10, 10)
    bmesh = BoundaryMesh(mesh, 'exterior')
    cell_f = CellFunction('size_t', bmesh, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0.)').mark(cell_f, 1)

    f = characteristic_function(bmesh, cell_f, 1)
    value = assemble(f*dx)
    assert near(value, 2., 1E-10)

    # Transition integration
    # ----      ----
    #     \____/ 
    f, _, _ = external_pressure(yLeft=1, xLeft=1.5, 
                                yRight=1, xRight=3.5, w=0.5, yMiddle=0, nderivs=0, 
                                degree=1)
    mesh = IntervalMesh(1000, 0, 5)
    value = assemble(f*dx(domain=mesh))
    assert near(value, 3., 1E-10)

    # Forcing
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), './meshes/336add920d9d3466f3498a2ecbab3e2eec43dd734749fca2fbadf8c7.h5', 'r')
    hdf.read(mesh, '/mesh', False)
    boundaries = FacetFunction('size_t', mesh)
    hdf.read(boundaries, '/boundaries')
    
    # Sanity
    f = ExternalForcing(mesh, family='DG', degree=0, boundaries=boundaries,
                        marker_values={11: StepFoo(1, 0.2, 0.5), 
                                       12: StepFoo(1, 0.5, 0.2),
                                       13: StepFoo(1, 0.1, 0.1)})

    ds = ds(domain=mesh, subdomain_data=boundaries)
    areas = [assemble(f*ds(i)) for i in (11, 12, 13)]

    # How we intend to use it
    bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, [11, 12, 13], True)
    D = VectorFunctionSpace(mesh, 'CG', 1)
    Dgb = VectorFunctionSpace(bmesh, 'CG', 1)

    D_to_Dgb = BoundaryRestrictor(D, emap, Dgb)
    g = Function(Dgb)
    
    n = FacetNormal(mesh)
    D_to_Dgb.map(f*n, g)

    assert g.vector().norm('l2') > 0

    # Check pressure interpolation
    x = np.linspace(0, 1, 1000)
    y = np.sin(2*np.pi*x)
    data = np.savetxt('foo.dat', np.c_[x, y])
    f = PressureInterpolator('foo.dat')

    t = np.linspace(0, 2, 40)
    y = []
    for ti in t:
        f.t = ti
        y.append(f(0))
    y = np.array(y)
    y0 = np.sin(2*np.pi*t)

    assert np.linalg.norm(y-y0, np.inf) < 1E-4
