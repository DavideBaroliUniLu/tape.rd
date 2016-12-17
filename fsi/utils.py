from cbcpost import *
from cbcpost.utils import *
from cbcflow import *
from cbcflow.schemes.utils import *
from dolfin import *
import numpy as np
import sympy as sp
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
    def __init__(self, problem, facet_domains, indicator):
        Constant.__init__(self, 0)
        self.ds = Measure('ds',
                          domain=problem.mesh,
                          subdomain_data=problem.facet_domains,
                          subdomain_id=indicator)

        self.problem = problem
        self.A0 = assemble(Constant(1)*self.ds)
        self.rho_f = problem.params.rho
        self.n = FacetNormal(problem.mesh)

    def update(self, u, DF):
        problem = self.problem
        dim = problem.mesh.geometry().dim()
        F = Identity(dim) + grad(DF)
        n = self.n

        form = inner(dot(n, cofac(F)), dot(n, cofac(F)))*self.ds
        An = assemble(form)
        Fn = assemble(dot(u, dot(cofac(F), self.n))*self.ds)

        Rn = sqrt(An/np.pi)  # Assuming the circle is more-or-less preserved
        beta = problem.params.E*problem.params.h/(1-problem.params.nu**2)*1.0/Rn**2
        val = ((sqrt(self.rho_f)/(2*sqrt(2))*Fn/An + sqrt(beta*sqrt(self.A0)))**2 - beta*sqrt(self.A0))
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


# ----------------------------------------------------------------------------


if __name__ == '__main__':
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

