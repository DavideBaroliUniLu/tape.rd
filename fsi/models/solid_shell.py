from boundary_mesh import py_BoundaryMesh
from common import BoundaryRestrictor
from dolfin import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True

_STATIC = False

# UFL shortcuts needed to express stress and strain of the shell membrane
def euclid_norm(v):
    '''Euclidean norm of the vector.'''
    return sqrt(sum(vi**2 for vi in v))


def surface_normal_tangents(n):
    '''
    Given facet normal of a surface we compute here a normal and two tangents
    (all normalized to 1) of the surface.
    '''
    assert n.ufl_shape[0] == 3
    nn = n/euclid_norm(n)            # Normalized normal
    t1 = as_vector((n[1]-n[2], n[2]-n[0], n[0]-n[1])) 
    t1n = t1/euclid_norm(t1)         # First tangent normalized
    # Final tangent is orthogonal to the last two guys
    t2 = cross(n, t1)
    t2n = t2/euclid_norm(t2)

    return nn, t1n, t2n


def strain(u, n):
    '''
    Strain in membrane has some components zero and some are assumed to be
    related. Local strain on surface with normal n is mapped to global frame of
    reference as follows
    '''
    n, t1, t2 = surface_normal_tangents(n)
    e_local = grad(u)

    return as_vector((inner(t1, dot(e_local, t1)),
                      inner(t2, dot(e_local, t2)),
                      inner(t1, dot(e_local, t2)) + inner(t2, dot(e_local, t1)),
                      inner(n, dot(e_local, t1)),
                      inner(n, dot(e_local, t2))))

    
def stress(u, n, E, nu, k):
    '''
    Shell membrane stress: E[Young's modulus], nu[Poisson ratio], k[parameter
    for traverse shear].
    '''
    E, nu, k = map(Constant, (E, nu, k))
    scale = Constant((1-nu)/2.)

    D = ((Constant(1), Constant(nu), Constant(0), Constant(0), Constant(0)),
         (Constant(nu), Constant(1), Constant(0), Constant(0), Constant(0)),
         (Constant(0), Constant(0), scale, Constant(0), Constant(0)),
         (Constant(0), Constant(0), Constant(0), scale*k, Constant(0)),
         (Constant(0), Constant(0), Constant(0), Constant(0), scale*k))
    D = as_matrix(D)
    D = Constant(E/(1.-nu**2))*D

    e_u = strain(u, n)

    return D*e_u

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), './meshes/CYLINDER/cylinder_3.h5', 'r')
hdf.read(mesh, '/mesh', False)
boundaries = FacetFunction('size_t', mesh)
hdf.read(boundaries, '/boundaries')

bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, 1, True)

W = VectorFunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(bmesh, 'CG', 1)
u = TrialFunction(V)
w = TestFunction(V)
bcs = [DirichletBC(V, Constant((0, 0, 0)), bmesh_boundaries, 2)]

E = 4.07E6
k = 5./6
nu = 0.5
eps = Constant(0.03)

n = FacetNormal(mesh)
nb = Function(V)
W_to_V = BoundaryRestrictor(W, emap, V)
W_to_V.map(n, nb)

f = Expression(('0', 'A*sin(pi*x[1])*pow(x[2], 2)*(x[0]-6)', '0'), degree=4, A=1E4)

uh = Function(V)
if _STATIC:
    a = eps*inner(stress(u, nb, E, nu, k), strain(w, nb))*dx(domain=bmesh)
    L = eps*inner(f, w)*dx(domain=bmesh)

    solve(a == L, uh, bcs, form_compiler_parameters={'quadrature_degree': 6})
    # FIXME: Dynamic displacement

    plot(uh, mode='displacement')
    interactive()
else:
    dt = Constant(1E-4)
    rho_s = Constant(1.)
    # Newmark params
    beta = 0.25
    gamma = 0.5

    u0, v0, a0 = Function(V), Function(V), Function(V)
    # Newmark relations
    acc = Constant(1./beta/dt/dt)*(u-u0-dt*v0) - Constant(1./2/beta - 1)*a0
    vel = Constant(gamma/beta/dt)*(u-u0) - Constant(gamma/beta-1)*v0 - Constant(dt*(gamma/2/beta-1))*a0

    form = rho_s*eps*inner(acc, w)*dx(domain=bmesh)+\
           eps*inner(stress(u, nb, E, nu, k), strain(w, nb))*dx(domain=bmesh)-\
           eps*inner(f, w)*dx(domain=bmesh)

    a, L = lhs(form), rhs(form)
    time = 0.

    A, b = PETScMatrix(), PETScVector()
    assembler = SystemAssembler(a, L, bcs,
                                form_compiler_parameters={'quadrature_degree': 8})
    assembler.assemble(A)

    solver = PETScLUSolver(A, 'mumps')
    solver.parameters['reuse_factorization'] = True

    uh_vec = uh.vector()
    u0_vec, v0_vec, a0_vec = u0.vector(), v0.vector(), a0.vector()
    # pp_plot = plot(uh, mode='displacement', range_min=0., range_max=1E-3)

    out = XDMFFile('foo.xdmf')
    while time < 0.2:
        time += dt(0)
        print time

        # Get the new displacement
        assembler.assemble(b)
        solve(A, uh.vector(), b)
    
        # Update velocity and acceleration
        acc_vec = (1./beta/dt(0)/dt(0))*(uh_vec-u0_vec-dt(0)*v0_vec) - (1./2/beta-1)*a0_vec
        vel_vec = (gamma/beta/dt(0))*(uh_vec-u0_vec) - (gamma/beta-1)*v0_vec - (dt(0)*(gamma/2/beta-1))*a0_vec
        
        a0_vec[:] = acc_vec
        v0_vec[:] = vel_vec
        u0_vec[:] = uh_vec
        # pp_plot.plot(uh)
        out.write(uh, dt(0))
    # interactive()

