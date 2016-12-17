from dolfin import *
from model_base import *


class VectorViscoElastic(SolidModelBase):
    '''
    Pretend that all the components behave like visco-elastic spring-dashpot
    system. Same params for each component.
    '''
    def __init__(self, solution, traction, n, dt, bcs, params, tol=1E-8):
        V = solution.function_space()
        # Solid parameters
        R = Constant(params.R)
        rho_s = Constant(params.rho_s) 
        h_s = Constant(params.h)        
        # Damping etc
        alpha0 = Constant(params.alpha0)
        alpha1 = Constant(params.alpha1)
        # Derived solid parameters
        lambda0 = Constant(params.E*h_s/(1 - params.nu**2)/R**2)
        lambda1 = Constant(0.5*params.E*h_s/(1 + params.nu))

        # The solid model
        eta = TrialFunction(V)
        x = TestFunction(V)
        # Previous
        dgb1 = Function(V)
        dgb1.vector()[:] = solution.vector()
        # Previous, previous is zero
        dgb2 = Function(V)

        # Weak form for normal component
        # elastic operator
        form = rho_s*h_s/dt**2*inner(eta - 2*dgb1 + dgb2, x)*dx
        form += inner(lambda0*eta, x)*dx
        form += inner(lambda1*grad(eta), grad(x))*dx
        # viscous operator
        form += inner(alpha0*rho_s*h_s*(eta - dgb1)/dt, x)*dx
        form += inner(alpha1*lambda1*grad((eta - dgb1)/dt), grad(x))*dx
        # force term
        form += inner(traction, x)*dx  # Note full traction

        a_s, L_s = lhs(form), rhs(form)
        A_s, b_s = PETScMatrix(), PETScVector()
        solver_s = PETScKrylovSolver('gmres', 'hypre_amg')

        # What we need to remember
        # Scalar solutions
        self.dgb, self.dgb1, self.dgb2 = solution, dgb1, dgb2
        self.a_s, self.L_s = a_s, L_s
        self.A_s, self.b_s = A_s, b_s
        self.solver_s = solver_s
        self.solution = solution
        self.bcs = [DirichletBC(V, value, boundary, tag) for value, boundary, tag in bcs]

    def solve(self):
        '''Traction is updated outside'''
        # Step for the normal component
        assemble(self.a_s, tensor=self.A_s)
        assemble(self.L_s, tensor=self.b_s)

        for bc in self.bcs: bc.apply(self.A_s, self.b_s)

        niters = self.solver_s.solve(self.A_s, self.dgb.vector(), self.b_s)

        # Next round
        self.dgb2.assign(self.dgb1)
        self.dgb1.assign(self.dgb)

        return niters
