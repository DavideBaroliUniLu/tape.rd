from model_base import SolidModelBase 
from dolfin import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True


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


class SolidMembrane(SolidModelBase):
    '''
    Elastic membrane from 
        A coupled momentum method for modelling blood flow in 3d deformable
        arteries.
    '''
    def __init__(self, solution, traction, n, dt, bcs, params, tol=1E-8):
        V = solution.function_space()
        u = TrialFunction(V)
        w = TestFunction(V)
        # Solid parameters
        rho_s = Constant(params.rho_s)  # Dendity
        E = Constant(params.E)          # Young
        nu = Constant(params.nu)        # Poisson
        k = Constant(params.kk)         # Constant of traverse shear
        h_s = Constant(params.h)      # Thickness

        u0 = Function(V)
        u0.assign(solution)
        # Previous, previous is zero
        u1 = Function(V)
        # Transient linear-elasticity
        form = rho_s*h_s*inner((u-2*u0+u1)/dt**2, w)*dx+\
               h_s*inner(stress(u, n, E, nu, k), strain(w, n))*dx+\
               inner(traction, w)*dx

        a, L = lhs(form), rhs(form)

        # Create boundary conditions
        bcs = [DirichletBC(V, value, bdry, tag) for (value, bdry, tag) in bcs]

        A, b = PETScMatrix(), PETScVector()
        solver = PETScKrylovSolver('gmres', 'hypre_euclid')

        # What we need to remember
        # Scalar solutions
        self.uh, self.u0, self.u1 = solution, u0, u1
        self.a, self.L = a, L
        self.A, self.b = A, b
        self.solver = solver
        self.bcs = bcs

    def solve(self):
        '''Traction is updated outside'''
        # Step for the normal component
        assemble(self.a, tensor=self.A)
        assemble(self.L, tensor=self.b)

        for bc in self.bcs: bc.apply(self.A, self.b)

        niters = self.solver.solve(self.A, self.uh.vector(), self.b)

        # Next round
        self.u1.assign(self.u0)
        self.u0.assign(self.uh)

        return niters
