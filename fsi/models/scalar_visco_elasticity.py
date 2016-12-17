from dolfin import *
from model_base import *


class ScalarViscoElastic(SolidModelBase):
    '''
    Normal component of displacement is governed by 1d spring-dashopot model. So
    we solve smaller (compared to vector) problem but there is a price to pay in
    the form of mapping ics and bcs.
    '''
    def __init__(self, solution, traction, n, dt, bcs, params, tol=1E-8):
        W = solution.function_space()
        # We want to make a scalar model for the normal component
        family = W.ufl_element().family()
        degree = W.ufl_element().degree()
        V = FunctionSpace(W.mesh(), family, degree)

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

        # Let's solve auxliary problems to get ics from vector and bc values.
        # NOTE: for now bc values are not allowed to depend on time - with time
        # dependence the bcs would have to be projected at each time step
        p, q = TrialFunction(V), TestFunction(V)
        a = inner(p, q)*dx
        A = assemble(a)
        # ic
        L = inner(dot(solution, n), q)*dx
        b = assemble(L)
        dgb1 = Function(V)
        solve(A, dgb1.vector(), b, 'cg', 'amg')
        # bcs
        bc_values = []
        for (value, bdry, tag) in bcs:
            assert isinstance(value, Constant) or not is_tdepExpr(value)
                   
            L = inner(dot(value, n), q)*dx
            b = assemble(L)
            foo = Function(V)
            solve(A, foo.vector(), b, 'cg', 'amg')
            bc_values.append(foo)

        # The solid model
        eta = TrialFunction(V)
        x = TestFunction(V)
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
        form += inner(dot(traction, n), x)*dx  # Note only normal part

        a_s, L_s = lhs(form), rhs(form)
        A_s, b_s = PETScMatrix(), PETScVector()
        solver_s = PETScKrylovSolver('gmres', 'hypre_amg')
        # This will be solved for
        dgb = Function(V)

        # To get to the vector there needs to projection step
        u, v = TrialFunction(W), TestFunction(W)
        a_v = inner(u, v)*dx
        L_v = inner(dgb*n, v)*dx
        assembler_v = SystemAssembler(a_v, L_v)
        A_v, b_v = PETScMatrix(), PETScVector()
        assembler_v.assemble(A_v)
        # Since A_v is constant in simulation we can setup preconditioner now
        solver_v = PETScKrylovSolver('cg', 'hypre_amg')
        solver_v.set_operators(A_v, A_v)
        solver_v.parameters['relative_tolerance'] = tol
        solver_v.parameters['absolute_tolerance'] = tol

        # What we need to remember
        # Scalar solutions
        self.dgb, self.dgb1, self.dgb2 = dgb, dgb1, dgb2
        # Scalar form and matrices, solver
        self.a_s, self.L_s = a_s, L_s
        self.A_s, self.b_s = A_s, b_s
        self.solver_s = solver_s
        self.solution = solution
        # For vector projection: b, assembler and the solver
        self.b_v = b_v
        self.solver_v = solver_v
        self.assembler_v = assembler_v
        # Finally it remains to make the boundary conditions and time step
        self.bcs = [DirichletBC(V, value, boundary, tag)
                    for value, (_, boundary, tag) in zip(bc_values, bcs)]

    def solve(self):
        '''Traction is updated outside'''
        # Step for the normal component
        assemble(self.a_s, tensor=self.A_s)
        assemble(self.L_s, tensor=self.b_s)

        for bc in self.bcs: bc.apply(self.A_s, self.b_s)

        niters = self.solver_s.solve(self.A_s, self.dgb.vector(), self.b_s)

        # Displacement as vector
        self.assembler_v.assemble(self.b_v)
        self.solver_v.solve(self.solution.vector(), self.b_v)
        
        # Next round
        self.dgb2.assign(self.dgb1)
        self.dgb1.assign(self.dgb)

        return niters
