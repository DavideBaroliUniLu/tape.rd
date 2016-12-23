from dolfin import *
from boundary_mesh import py_SubMesh


class InflowFromFlux(Expression):
    def __init__(self, mesh, boundaries, marker, n, fluxes, degree):
        bmesh = BoundaryMesh(mesh, 'exterior')
        c2f = bmesh.entity_map(2)

        bmesh_subdomains = CellFunction('size_t', bmesh, 0)
        for cell in cells(bmesh):
            bmesh_subdomains[cell] = boundaries[c2f[int(cell.index())]]

        inlet_mesh = py_SubMesh(bmesh, bmesh_subdomains, marker)

        if hasattr(inlet_mesh, '__iter__'): inlet_mesh = inlet_mesh[0]
        
        # plot(inlet_mesh)
        # interactive()

        # foo = FacetFunction('size_t', inlet_mesh, 0)
        # DomainBoundary().mark(foo, 1)
        # xx = inlet_mesh.coordinates().reshape((-1, 3))
        # for facet in SubsetIterator(foo, 0): print xx[facet.entities(0)]

        V = VectorElement('Lagrange', inlet_mesh.ufl_cell(), 1)
        Q = FiniteElement('Real', inlet_mesh.ufl_cell(), 0)
        W = MixedElement([V, Q])
        W = FunctionSpace(inlet_mesh, W)

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        f = Constant((0, 0, 0))
        flux = Constant(0)

        # Normal of the surface - FIXME: CellNormal?
        a = inner(grad(u), grad(v))*dx + p*inner(v, n)*dx + q*inner(u, n)*dx
        L = inner(f, v)*dx + inner(-flux, q)*dx
        bc = DirichletBC(W.sub(0), Constant((0, 0, 0)), 'on_boundary')

        assembler = SystemAssembler(a, L, bc)
        A = Matrix()
        assembler.assemble(A)

        # Different flux - only rhs is modified so reuse here
        solver = LUSolver(A, 'mumps')
        solver.parameters['reuse_factorization'] = True
        
        times, snapshots = [], []
        b = Vector()
        for time, flux_value in fluxes:
            wh = Function(W)
            flux.assign(flux_value)

            assembler.assemble(b)
            solver.solve(wh.vector(), b)

            uh, ph = wh.split(deepcopy=True)
            uh.set_allow_extrapolation(True)
            
            times.append(time)
            snapshots.append(uh)

        self._time = 0.
        self.period = times[-1]
        self.times = times
        self.snapshots = snapshots
        self.t = 0.

        inlet_mesh.bounding_box_tree()
        self.comm = inlet_mesh.mpi_comm().tompi4py()

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, t):
        self._time = t
        # Periodicity
        while self._time > self.period: self._time -= self.period

        times = self.times
        # Now find the snapshot indices
        index = [i 
                 for i in range(len(times)-1)
                 if between(self._time, (times[i], times[i+1]))].pop()

        self.index = index

    def eval(self, value, x):
        t = self.t

        index = self.index
        uh, Uh = self.snapshots[index], self.snapshots[index+1]
        uh, Uh = uh(x), Uh(x)
        ts, Ts = self.times[index], self.times[index+1]   # Snaphost times
        # Now perform linear interpolation
        ans = uh*(Ts-t)/(Ts-ts) + Uh*(t-ts)/(Ts-ts)
        value[:] = ans

    def value_shape(self):
        return (3, )

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # Note that Erika has axis oriented towards brain, here it is opposite
    import numpy as np

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(),
                   '../mesh/HOLLOW-ELLIPSOID-HEALTY/hollow-ellipsoid-healty_2.h5',
                   'r')
    hdf.read(mesh, '/mesh', False)
    boundaries = FacetFunction('size_t', mesh)
    hdf.read(boundaries, '/boundaries')

    plot(boundaries, interactive=True)

    fluxes = np.loadtxt('input_data/Vegards01_VolumetricCSFFlow.txt')
    fluxes = fluxes[:, :2]
    fluxes[:, -1] *= -1.

    foo = InflowFromFlux(mesh,
                         boundaries, marker=1, n=Constant((-1, 0, 0)),
                         fluxes=fluxes, degree=1)

    if False:
        import matplotlib.pyplot as plt

        times = fluxes[:, 0]

        plt.figure()
        plt.plot(times, fluxes[:, 1])

        times = np.linspace(times[0], times[-1], 30)
        y = []
        x =  np.array([0.00000000e+00,  0.786588, -0.14115114])
        for t in times:
            foo.t = t
            y.append(foo(x))

        plt.plot(times, y)
        plt.show()

    # Use in boundary value problem
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        zero = Constant((0, 0, 0))

        a = inner(grad(u), grad(v))*dx
        L = inner(zero, v)*dx
        bc0 = [DirichletBC(V, zero, boundaries, i) for i in (1, 2, 3)]
        bci = [DirichletBC(V, foo, boundaries, 4)]
        bcs = bc0 + bci

        assembler = SystemAssembler(a, L, bcs)
        A = PETScMatrix();
        assembler.assemble(A)

        solver = PETScKrylovSolver('cg', 'amg')
        solver.set_operators(A, A)
        
        b = PETScVector()
        uh = Function(V)
        x = uh.vector()
        for t in np.linspace(0, 2, 41):
            foo.t = t
            assembler.assemble(b)

            solver.solve(x, b)
            
            plot(uh, title='t = %gs' % t)
        interactive()

    # FIXME: plot @ point vs. Erika to make sure this is okay
    # FIXME: the mesh is okay if master == (rank zero process). Must make it
    # such that master can be anybody
    # TODO: use with simulations
