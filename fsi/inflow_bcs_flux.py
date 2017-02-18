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
    def __init__(self, mesh, boundaries, marker, n, fluxes, source=''):
        # Load from file
        if source: 
            V = VectorFunctionSpace(mesh, 'CG', 1)
            uh = Function(V)

            infile = HDF5File(mesh.mpi_comm(), source, 'r')
            infile.read(uh, 'uh')
            infile.close()

            self.uh = uh
        # Actual computations can for now be done only in serial
        else:
            assert mesh.mpi_comm().size == 1

            bmesh = BoundaryMesh(mesh, 'exterior')
            c2f = bmesh.entity_map(2)
            # Marker is where we want to integrate, the rest will be killed of
            bmesh_subdomains = CellFunction('size_t', bmesh, 0)
            found_markers = set()
            for cell in cells(bmesh):
                value = boundaries[c2f[int(cell.index())]]
                found_markers.add(value)
                bmesh_subdomains[cell] = value 
            found_markers = list(found_markers)
            found_markers.remove(marker)    # Now we can kill off

            # We solve on a full skeleton
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
           
            A, b = assemble_system(a, L, bc)
            wh = Function(W)
            
            solver = LUSolver(A, 'mumps')
            print 'System size is', A.size(0)
            solver.solve(wh.vector(), b)

            # plot(wh, interactive=True)

            uhb, _ = wh.split(deepcopy=True)
            Vb = uhb.function_space()

            # Okay now we need to extend to full mesh
            V = VectorFunctionSpace(mesh, 'CG', 1)
            uh = Function(V)
            
            # We want to transer values from boundary space to space
            dofb_vb = np.array(dof_to_vertex_map(Vb), dtype=int)
            vb_v = np.array(bmesh.entity_map(0), dtype=int)
            v_dof = np.array(vertex_to_dof_map(V), dtype=int)

            array = uh.vector().array()
            in_array = uhb.vector().get_local()

            dim = nsubs = 3
            vb_v = np.repeat(vb_v, dim)
            maps = []
            for i in range(nsubs):
                maps.append(v_dof[i::dim][vb_v[dofb_vb[i::dim]]])
            mapping = np.array(sum(map(list, zip(*maps)), []))

            array[mapping] = in_array
            uh.vector().set_local(array)
            uh.vector().apply('insert')

            ofile = HDF5File(mesh.mpi_comm(), 'inflow_%d.h5' % mesh.num_cells(), 'w')
            ofile.write(uh, 'uh')
            ofile.close()

        # Next we would like to have a mapping for how to update values of uh
        # based on uh_b based on time dependent flux values
        self._time = 0.
        self.times = fluxes[:, 0]
        self.fluxes = fluxes[:, 1]
        self.period = self.times[-1]
        self.uh = uh  # The interface, values changed in time
        self.U = uh.vector().get_local()  # Values constant in time

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

        # Now update the values of uh
        qt = (q1*(t-t0)/(t1-t0) + q0*(t1-t)/(t1-t0))

        Uqt = self.U*qt
        self.uh.vector().set_local(Uqt)
        self.uh.vector().apply('insert')

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
                            fluxes=fluxes,
                            source='inflow_72415.h5')
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
        bc0 = [DirichletBC(V, zero, boundaries, i) for i in (0, 4, 2, 3)]
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
