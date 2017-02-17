from dolfin import *
from boundary_mesh import py_SubMesh
from itertools import chain


class InflowFromFlux(object):
    '''
    Given a time series of [fluxes] over a surface (a collection of
    facets[boundaries] with a common marker) with normal [n] we compute
    a function vector valued function on mesh which has the correct values
    of the 'velocity'.
    '''
    def __init__(self, mesh, boundaries, marker, n, fluxes):
        # How we talk to the 3d problem
        # Let us first compute the 'velocity' as foo on the boundary 
        bmesh = BoundaryMesh(mesh, 'exterior')
        c2f = bmesh.entity_map(2)
        # Marker is where we want to integrate, the rest will be killed of
        # by bcs
        bmesh_subdomains = CellFunction('size_t', bmesh, 0)
        found_markers = set()
        for cell in cells(bmesh):
            value = boundaries[c2f[int(cell.index())]]
            found_markers.add(value)
            bmesh_subdomains[cell] = value 
        found_markers = list(found_markers)
        # In parallel different CPUs have different boundaries
        comm = bmesh.mpi_comm().tompi4py()
        found_markers = comm.allreduce(found_markers)
        found_markers = list(set(found_markers))
        found_markers.remove(marker)

        V = VectorElement('Lagrange', bmesh.ufl_cell(), 1)
        Q = FiniteElement('Real', bmesh.ufl_cell(), 0)
        W = MixedElement([V, Q])
        W = FunctionSpace(bmesh, W)

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        f = Constant((0, 0, 0))
        flux = Constant(1)

        dx = Measure('dx', domain=bmesh, subdomain_data=bmesh_subdomains)

        a = inner(grad(u), grad(v))*dx(marker)\
            + p*inner(v, n)*dx(marker)\
            + q*inner(u, n)*dx(marker)
        L = inner(f, v)*dx(marker) + inner(-flux, q)*dx(marker)

        # For bc the cell function must be translated into a facet function.
        # Bc constrained regions get tag 1
        bmesh.init(2, 1)
        facet_f = FacetFunction('size_t', bmesh, 0)
        for c in chain(*[SubsetIterator(bmesh_subdomains, m) for m in found_markers]):
            for facet in facets(c): facet_f[facet] = 1

        bc = DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_f, 1)
        # Obtain 'velocity' once than scale it
        A, b = assemble_system(a, L, bc)

        wh = Function(W)
        solver = LUSolver(A, 'mumps')
        solver.solve(wh.vector(), b)

        uh_b, _ = wh.split(deepcopy=True)
        plot(uh_b, interactive=True)
        Vb = uh_b.function_space()
        # We only need the values which are CONSTANT in time
        array_b = uh_b.vector().get_local()
        print [v for v in array_b if abs(v) > 1E-10]

        # Now we a way to talk to the world
        V = VectorFunctionSpace(mesh, 'CG', 1)
        uh = Function(V)
        array = uh.vector().get_local()  # Alloc

        # Next we would like to have a mapping for how to update values of uh
        # based on uh_b based on time dependent flux values
        self._time = 0.
        self.times = fluxes[:, 0]
        self.fluxes = fluxes[:, 1]
        self.period = self.times[-1]
        self.uh = uh

        # THIS should work in parallel. For now it does not even work in serial
        # FIXME
        dofb_vb = np.array(dof_to_vertex_map(Vb), dtype=int)
        vb_v = np.repeat(np.array(bmesh.entity_map(0), dtype=int), 3)
        v_dof = np.array(vertex_to_dof_map(V), dtype=int)

        def _update_impl(t, (t0, t1), (q0, q1)):
            qt = q1*(t-t0)/(t1-t0) + q0*(t-t1)/(t1-t0)
            print t, qt,
            array[v_dof[vb_v[dofb_vb]]] = array_b*qt
            print max(np.abs(array_b*qt)), max(np.abs(array)), '><', id(array), v_dof[vb_v[dofb_vb]],
            uh.vector().set_local(array)
            uh.vector().apply('insert')
            print uh.vector().norm('l2')
        self._update = _update_impl

        self.t = 0.

    @property
    def t(self):
        return self._time

    @t.setter
    def t(self, t):
        self._time = t
        # Periodicity
        while self._time > self.period: self._time -= self.period

        times = self.times
        fluxes = self.fluxes
        # Now find the snapshot indices
        index = [i 
                 for i in range(len(times)-1)
                 if between(self._time, (times[i], times[i+1]))].pop()
        (t0, t1) = times[index], times[index+1]
        (q0, q1) = fluxes[index], fluxes[index+1]
        # Now update the values of f
        self._update(t, (t0, t1), (q0, q1))

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

    # plot(boundaries, interactive=True)

    fluxes = np.loadtxt('input_data/Vegards01_VolumetricCSFFlow.txt')
    fluxes = fluxes[:, :2]
    fluxes[:, -1] *= -1.

    inflow = InflowFromFlux(mesh,
                            boundaries, marker=1, n=Constant((-1, 0, 0)),
                            fluxes=fluxes)
    inflow_f = inflow.uh

    bmesh = BoundaryMesh(mesh, 'exterior')
    Vb = VectorFunctionSpace(bmesh, 'CG', 1)
    if True:
        V = VectorFunctionSpace(mesh, 'CG', 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        zero = Constant((0, 0, 0))

        a = inner(grad(u), grad(v))*dx
        L = inner(zero, v)*dx

        bci = [DirichletBC(V, inflow_f, boundaries, 1)]
        bc0 = [DirichletBC(V, zero, boundaries, i) for i in (4, 2, 3)]
        bcs = bc0 + bci

        assembler = SystemAssembler(a, L, bcs)
        A = PETScMatrix();
        assembler.assemble(A)

        solver = PETScKrylovSolver('cg', 'amg')
        solver.set_operators(A, A)
        
        b = PETScVector()
        uh = Function(V)
        x = uh.vector()
        p = plot(uh)
        for t in np.linspace(0, 1, 20):
            inflow.t = t
            assembler.assemble(b)

            solver.solve(x, b)
            p.plot(uh)
        interactive()

    # FIXME: plot @ point vs. Erika to make sure this is okay
    # FIXME: the mesh is okay if master == (rank zero process). Must make it
    # such that master can be anybody
    # TODO: use with simulations
