from dolfin import *
from cbcpost import Field

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


class FlowRate(Field):
    '''
    Flux through oriented surface of a problem geometry futher specified by 
    normal and tag. It is also necesarry to specify whether the surface is 
    intenal or external.
    '''
    def __init__(self, problem, n, tag, name, params, exterior=True):
        self.n = n

        domain = problem.mesh
        subdomain_data = problem.facet_domains
        self.dm = Measure('ds' if exterior else 'dS', 
                          domain=domain,
                          subdomain_data=subdomain_data,
                          subdomain_id=tag)

        self.gdim = domain.geometry().dim()
        Field.__init__(self, params, name, label=None)

    def compute(self, get):
        '''
        int_{interior_surface} v.n dS in the deformed domain is mapped to
        reference domain.
        '''
        n, gdim, dm = self.n, self.gdim, self.dm

        u = get('Velocity')
        DF = get('Displacement')[1]  # NOTE: ALE is passed as second argument
        # Deformation gradient
        F = Identity(gdim) + grad(DF)

        # Switch the form according to measure
        if dm.integral_type() == 'exterior_facet':
            form = inner(u, dot(n, cofac(F)))*dm
        else:
            form = inner(avg(u), dot(avg(n), cofac(avg(F))))*dm
        value = assemble(form)

        return value


class SurfaceArea(FlowRate):
    '''
    Surface area of an oriented surface of a problem geometry futher specified by 
    normal and tag. It is also necesarry to specify whether the surface is 
    intenal or external.
    '''
    def compute(self, get):
        '''
        int_{interior_surface} 1 dS in the deformed domain is mapped to
        reference domain.
        '''
        n, gdim, dm = self.n, self.gdim, self.dm

        DF = get('Displacement')[1]
        F = Identity(gdim) + grad(DF)
        
        # Switch the form according to measure
        if dm.integral_type() == 'exterior_facet':
            form = sqrt(inner(dot(n, cofac(F)), dot(n, cofac(F))))*dm
        else:
            form = sqrt(inner(dot(avg(n), cofac(avg(F))), dot(avg(n), cofac(avg(F)))))*dm
        value = assemble(form)

        return value

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # TESTS, with fake problem
    ncells = 100
    mesh = UnitSquareMesh(ncells, ncells)
    boundaries = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], 1) && on_boundary').mark(boundaries, 1)
    CompiledSubDomain('near(x[1], 0.5, 1E-8)').mark(boundaries, 2)

    class FakeProblem(object):
        def __init__(self, mesh, facet_domains):
            assert set(map(int, set(facet_domains.array()))) <= set([0, 1, 2])
            self.mesh = mesh
            self.facet_domains = facet_domains


    problem = FakeProblem(mesh, boundaries)
    cell = problem.mesh.ufl_cell()
    get = lambda key: {'Velocity': Constant((2, 0), cell=cell),
                       'Displacement': (0, Constant((0, 0), cell=cell))}[key]
    
    n = Constant((1, 0))
    foo = FlowRate(problem=problem, n=n, tag=1, name='x', params={})
    value = foo.compute(get)
    assert near(value, 2.0, 1E-10), str(value)

    n = Constant((0, 1))
    foo= FlowRate(problem=problem, n=n, tag=2, name='y', params={}, exterior=False)
    value = foo.compute(get)
    assert near(value, 0.0, 1E-10), str(value)

    n = Constant((1, 0))
    bar = SurfaceArea(problem=problem, n=n, tag=1, name='x', params={})
    value = bar.compute(get)
    assert near(value, 1.0, 1E-10), str(value)

    n = Constant((0, 1))
    bar = SurfaceArea(problem=problem, n=n, tag=2, name='x', params={}, exterior=False)
    value = bar.compute(get)
    assert near(value, 1.0, 1E-10), str(value)
