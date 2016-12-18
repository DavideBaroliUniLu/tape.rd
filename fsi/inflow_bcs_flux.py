from dolfin import *
from cbcpost.utils import create_submesh


class InflowFromFlux(Expression):
    def __init__(self, mesh, boundaries, marker, n, fluxes, degree):
        bmesh = BoundaryMesh(mesh, 'exterior')
        c2f = bmesh.entity_map(2)

        bmesh_subdomains = CellFunction('size_t', bmesh, 0)
        for cell in cells(bmesh):
            bmesh_subdomains[cell] = boundaries[c2f[int(cell.index())]]

        inlet_mesh = create_submesh(bmesh, bmesh_subdomains, marker)

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
        L = inner(f, v)*dx + inner(flux, q)*dx
        bc = DirichletBC(W.sub(0), Constant((0, 0, 0)), 'on_boundary')

        assembler = SystemAssembler(a, L, bc)
        A = Matrix()
        assembler.assemble(A)

        # Different flux - only rhs is modified so reuse here
        solver = LUSolver(A, 'mumps')
        solver.parameters['reuse_factorization'] = True
        
        snapshots = []
        b = Vector()
        for time, flux_value in fluxes:
            wh = Function(W)
            flux.assign(flux_value)

            assembler.assemble(b)
            solver.solve(wh.vector(), b)

            uh, ph = wh.split(deepcopy=True)
            uh.set_allow_extrapolation(True)

            snapshots.append((time, uh))

        self._time = 0.
        # What we need to rememeber are snapshots, period, out time
        self.snapshots = snapshots
        self.period = time  # Because that is the last one in the for loop
        self.t = 0.
        pass

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, t):
        self._time = t
        # Periodicity
        while self._time > self.period: self._time -= self.period
        # Now find the snapshot indices
        times = [p[0] for p in self.snapshots]
        
        index = [i 
                 for i in range(len(times)-1)
                 if between(self._time, (times[i], times[i+1]))].pop()

        self.index = index

    def eval(self, value, x):
        _, uh = self.snapshots[self.index]
        value[:] = uh(x)
        pass

    def value_shape(self):
        return (3, )

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(),
                   '../mesh/HOLLOW-ELLIPSOID-HEALTY/hollow-ellipsoid-healty_1.h5',
                   'r')
    hdf.read(mesh, '/mesh', False)
    boundaries = FacetFunction('size_t', mesh)
    hdf.read(boundaries, '/boundaries')

    fluxes = np.loadtxt('input_data/Vegards01_VolumetricCSFFlow.txt')
    fluxes = fluxes[:, :2]

    foo = InflowFromFlux(mesh,
                         boundaries, marker=1, n=Constant((-1, 0, 0)),
                         fluxes=fluxes, degree=1)


    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    zero = Constant((0, 0, 0))

    a = inner(grad(u), grad(v))*dx
    L = inner(zero, v)*dx
    bc0 = [DirichletBC(V, zero, boundaries, i) for i in (2, 3, 4)]
    bci = [DirichletBC(V, foo, boundaries, 1)]
    bcs = bc0 + bci

    uh = Function(V)
    for t in np.linspace(0, 4, 30):
        foo.t = t
        solve(a == L, uh, bcs)
        plot(uh, title='t = %gs' % t)
    interactive()

    # FIXME: plot @ point vs. Erika to make sure this is okay
    # TODO: use with simulations


