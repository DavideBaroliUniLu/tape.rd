from mpi4py import MPI as pyMPI
from functools import partial
from itertools import chain
from dolfin import *
import numpy as np
import os


RED = '\033[1;37;31m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'


def py_BoundaryMesh(mesh, mesh_facet_f=None, marker=None, boundaries=False):
    '''
    py_BoundaryMesh(mesh) returns BoundaryMesh(mesh, 'exterior') while if
    mesh_facet_f and marker are present only the `marker` facets are used to
    create the mesh of the bounding surface of mesh. Moreover structure that
    mirrors BoundaryMesh.entity_map is returned. Finaly boundaries facets of the
    boundary mesh are tagged following mesh_facet_f where the entity value is
    not marker.

    In serial and with single marker we rely on dolfin.SubMesh.
    '''
    comm = mesh.mpi_comm().tompi4py()
    # We start from BoundaryMesh - root.
    tic = Timer('BoundaryMesh')
    root = BoundaryMesh(mesh, 'exterior')
    dt = tic.stop()
    dt = max(comm.allgather(dt))
    if comm.rank == 0: info(GREEN % ('BoundaryMesh took %g s' % dt))

    tdim = mesh.topology().dim()
    c2f = root.entity_map(tdim-1)
    v2v = root.entity_map(0)

    # Pay up for API generality
    if marker is None: 
        marker = []
    elif isinstance(marker, int):
        marker = [marker]

    # Mesh for the whole bounding surface was requested so we are done
    if mesh_facet_f is None or len(marker) == 0: 
        # Extract the maps into dictionary for consistency of return value
        entity_map = {0: v2v, tdim-1: c2f}
        return root, entity_map
    
    # Otherwise marked facets of the mesh need to be mapped to marked cells of
    # bmesh
    assert tdim - 1 == mesh_facet_f.dim()

    cell_f = CellFunction('size_t', root, 0)
    tags = set([])
    # Unique identifiers are collected in case boundaries need to be marked
    for cell in cells(root):
        tag = mesh_facet_f[c2f[cell]]
        cell_f[cell] = tag
        tags.add(tag)

    # We remove marked cells from the submesh keeping the map from new cells to 
    # old cell. In serial we rely on dolfin's SubMesh if there is only one
    # marker
    if comm.size == 1 and len(marker) == 1:
        bmesh = SubMesh(root, cell_f, marker[0])
        c2c = bmesh.data().array('parent_cell_indices', tdim-1)
        vb2v = bmesh.data().array('parent_vertex_indices', 0)
    # In parallel we mirror cbcpost. 
    else:
        tic = Timer('py_SubMesh')
        bmesh, entity_map = py_SubMesh(root, cell_f, marker)
        dt = tic.stop()
        dt = max(comm.allgather(dt))
        if comm.rank == 0: info(RED % ('py_SubMesh took %g s' % dt))

        c2c = entity_map[tdim-1]
        vb2v = entity_map[0]
    assert len(c2c) and len(vb2v)

    # It then remains to update c2f, v2v so that it has only the used cells, vertices
    # Vertices of bmesh -> vertices of mesh, Cells of bmesh -> facet of mesh
    c2f = np.array([c2f[int(old_c_index)] for old_c_index in c2c], dtype='uint')
    v2v = np.array([v2v[int(v)] for v in vb2v], dtype='uint')
    entity_map = {0: v2v, tdim-1: c2f}
    
    # Nomore work left if boundaries not requested
    if not boundaries: return bmesh, entity_map
   
    boundaries = extract_boundary_mesh_boundaries(root_mesh=root,
                                                  bdry_mesh=bmesh,
                                                  cell_2_cell=c2c,
                                                  cell_f=cell_f,
                                                  tags=tags)

    return bmesh, entity_map, boundaries
                                                  

def py_SubMesh(mesh, markers, marker):
    '''
    Create new mesh from cells of mesh marked by marker. This is essentially
    cbcpost code with an additional map from cells of new mesh to cells of old
    mesh.

    NOTE: To construct the map we rely on cells staying on the same process.
    '''
    if isinstance(marker, int): marker = [marker]
    assert all(isinstance(v, int) for v in marker), str(marker)
    # Collect the iterators over marker
    iterators = chain(*[SubsetIterator(markers, m) for m in marker])

    base_cell_indices = np.array([cell.index() for cell in iterators])
    # Get the marked cell counts acrross processes. We require that each process 
    # has at least some cells so that redistributing the submesh is not needed 
    # - this would mean that facet of cell in mesh would be on a different process 
    # then corresponding cell of bmesh
    comm = mesh.mpi_comm().tompi4py()
    global_cell_distribution = comm.allgather(len(base_cell_indices))
    have_marked_cells = sum(global_cell_distribution) 
    assert have_marked_cells

    if all(count > 0 for count in global_cell_distribution):
        return build_mesh_on_all_cpus(mesh, base_cell_indices)
    else:
        # We can try to build the mesh as distributed if all the info is on one CPU
        # Figure out who has the data
        counts = set(global_cell_distribution)
        zero, not_zero = sorted(counts)
        assert zero == 0

        master = global_cell_distribution.index(not_zero)
        is_master = master == comm.rank

        # The owner prepares the data
        global_num_cells, global_num_vertices = 0, 0
        if is_master:
            # Cells of submesh as ntuple of vertices in base mesh
            base_cells = mesh.cells()[base_cell_indices]  
            # Unique vertices that make up submesh in their mesh indexing
            base_vertex_indices = np.unique(base_cells.flatten())
            # NOTE: coordinates are local so make sense only here
            coordinates = mesh.coordinates().flatten()

            global_num_cells = len(base_cell_indices)
            global_num_vertices = len(base_vertex_indices)
            ncoords = np.array([len(coordinates)], dtype=int)
        # Communicate to every what mesh is going to be built
        global_num_cells = comm.bcast(global_num_cells, master)
        global_num_vertices = comm.bcast(global_num_vertices, master)

        # Communicate the data for mesh editor
        if is_master and not comm.rank == 0:
            comm.Send([base_cells.astype(int), pyMPI.INT], dest=0, tag=1)
            comm.Send([base_vertex_indices.astype(int), pyMPI.INT], 0, tag=2)

            comm.Send([ncoords, pyMPI.INT], 0, tag=3)
            comm.Send([coordinates, pyMPI.FLOAT], 0, tag=4)
        # Complete also ...
        if comm.rank == 0:
            if not is_master:
                base_cells = np.zeros(global_num_cells*3, dtype=int)
                comm.Recv([base_cells, pyMPI.INT], source=master, tag=1)

                base_vertex_indices = np.zeros(global_num_vertices, dtype=int)
                comm.Recv([base_vertex_indices, pyMPI.INT], master, tag=2)

                ncoords = np.zeros(1, dtype=int)
                comm.Recv([ncoords, pyMPI.INT], master, tag=3)
                coordinates = np.zeros(ncoords, dtype=float)
                comm.Recv([coordinates, pyMPI.FLOAT], master, tag=4)

            coordinates = coordinates.reshape((-1, 3))
            base_cells = base_cells.reshape((-1, 3))
            # Map base to local vertex
            base_to_sub_indices = {b: l for l, b in enumerate(base_vertex_indices)}
            # Cells of submesh as ntuple of vertices in submesh numbering
            sub_cells = [[base_to_sub_indices[j] for j in c] for c in base_cells]

        # Everybody init
        submesh = Mesh()
        mesh_editor = MeshEditor()
        cell = mesh.ufl_cell().cellname()
        tdim = mesh.topology().dim()
        gdim = mesh.geometry().dim()
        mesh_editor.open(submesh, cell, tdim, gdim)

        mesh_editor.init_cells_global(global_num_cells, global_num_cells)
        mesh_editor.init_vertices_global(global_num_vertices, global_num_vertices)

        # Only root fills
        if comm.rank == 0:
            for index, cell in enumerate(sub_cells): 
                mesh_editor.add_cell(index, *cell)
            
            for base_vertex, sub_vertex in base_to_sub_indices.iteritems():
                mesh_editor.add_vertex_global(sub_vertex, sub_vertex, coordinates[base_vertex])
        
        mesh_editor.close()

        local_mesh_data = LocalMeshData(submesh)
        ghost_mode = parameters['ghost_mode']
        MeshPartitioning.build_distributed_mesh(submesh, local_mesh_data, ghost_mode);

        return submesh


def build_mesh_on_master_cpu(submesh, mesh, base_cell_indices):
    '''TODO'''
    comm = mesh.mpi_comm().tompi4py()

    # Cells of submesh as ntuple of vertices in base mesh
    base_cells = mesh.cells()[base_cell_indices]  
    # Unique vertices that make up submesh in their mesh indexing
    base_vertex_indices = np.unique(base_cells.flatten())
    # Map base to local vertex
    base_to_sub_indices = {b: l for l, b in enumerate(base_vertex_indices)}
    # Cells of submesh as ntuple of vertices in submesh numbering
    sub_cells = [[base_to_sub_indices[j] for j in c] for c in base_cells]

    # Store vertices as sub_vertices[local_index] = (global_index, coordinates)
    coordinates = mesh.coordinates()
    # Done creating done. Let's write the mesh: global cell and vertex cound
    global_num_cells = len(sub_cells)
    global_num_vertices = len(base_to_sub_indices)

    tdim, gdim = mesh.topology().dim(), mesh.geometry().dim()

    submesh = Mesh()
    mesh_editor = MeshEditor()
    mesh_editor.open(submesh, mesh.ufl_cell().cellname(), tdim, gdim)

    mesh_editor.init_cells_global(global_num_cells, global_num_cells)
    mesh_editor.init_vertices_global(global_num_vertices, global_num_vertices)

    for index, cell in enumerate(sub_cells): 
        mesh_editor.add_cell(index, *cell)
    
    for base_vertex, sub_vertex in base_to_sub_indices.iteritems():
        mesh_editor.add_vertex_global(sub_vertex, sub_vertex, coordinates[base_vertex])

    mesh_editor.close(False)

    return submesh


def build_mesh_on_all_cpus(mesh_, base_cell_indices_):
    '''
    Build the mesh on each process separately, while maintaining some sort
    of notion of how the global mesh is.
    '''
    comm = mesh_.mpi_comm().tompi4py()
    # Get local
    topology = mesh_.topology()
    # Cells of base mesh to use by local coordinates of their vertices
    base_cells = mesh.cells()[base_cell_indices_]  
    base_vertex_indices = np.unique(base_cells.flatten())
    
    # Need global mapping to talk about uniqueness of shared quantities
    vertex_l2g = topology.global_indices(0)
    base_global_vertex_indices = [vertex_l2g[li] for li in base_vertex_indices]

    shared_local_indices = set(base_vertex_indices) & set(topology.shared_entities(0).keys())
    shared_global_indices = [vertex_l2g[li] for li in shared_local_indices]

    unshared_global_indices = list(set(base_global_vertex_indices)-set(shared_global_indices))
    # Communicate counts of unshared_global_indices accross processes
    unshared_vertices_dist = comm.allgather(len(unshared_global_indices))

    # Number unshared vertices on separate process - use the distribution to get offset
    offset = sum(unshared_vertices_dist[:comm.rank])
    base_to_sub_global_indices = {gi: index for index, gi in enumerate(unshared_global_indices, offset)}

    # Gather all shared process on process 0
    all_shared_global_indices = comm.gather(shared_global_indices)
    shared_count = 0
    if comm.rank == 0:
        all_shared_global_indices = np.unique(np.hstack(all_shared_global_indices))
        shared_count = len(all_shared_global_indices)
    shared_count = comm.bcast(shared_count, 0)

    # Root gets to assign global index
    offset = comm.allreduce(max(base_to_sub_global_indices.values()), op=pyMPI.MAX) + 1
    shared_base_to_sub_global_indices = {}
    if comm.rank == 0:
        for index, gi in enumerate(all_shared_global_indices, offset):
            shared_base_to_sub_global_indices[int(gi)] = index

    # Broadcast global numbering of all shared vertices
    shared_base_to_sub_global_indices = dict(zip(comm.bcast(shared_base_to_sub_global_indices.keys()),
                                                 comm.bcast(shared_base_to_sub_global_indices.values())))

    # Join shared and unshared numbering in one dict
    base_to_sub_global_indices = dict(chain(base_to_sub_global_indices.iteritems(),
                                            shared_base_to_sub_global_indices.iteritems()))

    # Create mapping of local indices
    base_to_sub_local_indices = {b: l for l, b in enumerate(base_vertex_indices)}

    # Define sub-cells of submesh
    sub_cells = [[base_to_sub_local_indices[j] for j in c] for c in base_cells]

    # Store vertices as sub_vertices[local_index] = (global_index, coordinates)
    coordinates = mesh.coordinates()
    sub_vertices = {}
    for base_local, sub_local in base_to_sub_local_indices.iteritems():
        sub_vertices[sub_local] = (base_to_sub_global_indices[vertex_l2g[base_local]], coordinates[base_local])

    # Done creating done. Let's write the mesh: global cell and vertex cound
    global_num_cells = comm.allreduce(len(sub_cells))
    global_num_vertices = sum(unshared_vertices_dist) + shared_count

    tdim, gdim = mesh.topology().dim(), mesh.geometry().dim()

    submesh = Mesh()
    mesh_editor = MeshEditor()
    mesh_editor.open(submesh, mesh.ufl_cell().cellname(), tdim, gdim)

    mesh_editor.init_vertices(len(sub_vertices))
    mesh_editor.init_cells_global(len(sub_cells), global_num_cells)

    # Add cell, only local indexing needed
    for index, cell in enumerate(sub_cells): mesh_editor.add_cell(index, *cell)
    # For vertices also global
    
    for local_index, (global_index, coordinates) in sub_vertices.items():
        mesh_editor.add_vertex_global(int(local_index), int(global_index), coordinates)

    mesh_editor.close(False)
    submesh.order()

    from dolfin import compile_extension_module
    cpp_code = """
    void set_shared_entities(Mesh& mesh, std::size_t idx, const Array<std::size_t>& other_processes)
    {
        std::set<unsigned int> set_other_processes;
        for (std::size_t i=0; i<other_processes.size(); i++)
        {
            set_other_processes.insert(other_processes[i]);
            //std::cout << idx << " --> " << other_processes[i] << std::endl;
        }
        //std::cout << idx << " --> " << set_other_processes[0] << std::endl;
        mesh.topology().shared_entities(0)[idx] = set_other_processes;
    }
    """
    set_shared_entities = compile_extension_module(cpp_code).set_shared_entities
    base_se = mesh.topology().shared_entities(0)

    for li in shared_local_indices:
        arr = np.array(base_se[li], dtype=np.uintp)
        sub_li = base_to_sub_local_indices[li]
        set_shared_entities(submesh, base_to_sub_local_indices[li], arr)
    submesh.topology().init(0, len(sub_vertices), global_num_vertices)
    submesh.topology().init(tdim, len(sub_cells), global_num_cells)

    entity_map = {0: base_vertex_indices, tdim: base_cell_indices_}

    return submesh, entity_map


def extract_boundary_mesh_boundaries(root_mesh, bdry_mesh, cell_2_cell, cell_f, tags):
    '''
    Bdry mesh is a subdomain of root mesh. The injective map from cells of bdry
    mesh to root mesh is cell_2_cell. Root mesh has with it associated
    markers(cell_f) with values tags. We want to mark boundaries(facets) of bdry
    mesh such that they take values of the connected marked cell.
    '''
    assert root_mesh.topology().dim() == bdry_mesh.topology().dim()
    assert root_mesh.geometry().dim() == bdry_mesh.geometry().dim()

    boundaries = FacetFunction('size_t', bdry_mesh, 0)
    tdim = boundaries.dim() + 1
    # These are boundaries, right?
    tag = max(tags) + 100
    DomainBoundary().mark(boundaries, tag)
    # FIXME: py_SubMesh does not init shared facets so DomaiBoundary identifies
    # process boundaries too. So we workaound interprocess boundaries
    shared_vertices = set(bdry_mesh.topology().shared_entities(0))
    bdry_mesh.init(tdim-1, 0)
    # Unset tag for process bdries
    for facet in facets(bdry_mesh):
        if len(set(facet.entities(0)) & shared_vertices) == 2:
            boundaries[facet] = 0
    # We shall mark boundary facet by looking up the tag of the cell connected
    # to it in the old/root mesh.
    # FIXME: MPI safety
    bmesh_cells_in_root = set(cell_2_cell)
    # Look up of cells of root connected to its cell by facet
    root_mesh.init(tdim, tdim-1); root_mesh.init(tdim-1, tdim)
    cell_2_facet = root_mesh.topology()(tdim, tdim-1)
    facet_2_cell = root_mesh.topology()(tdim-1, tdim)
    lookup = lambda cell: set(sum((facet_2_cell(f).tolist() for f in cell_2_facet(cell)), []))

    for facet in SubsetIterator(boundaries, tag):
        bmesh_cell = facet.entities(tdim)[0]
        root_cell = cell_2_cell[bmesh_cell]
        connected_cells = lookup(root_cell) - bmesh_cells_in_root
        tags = set(cell_f[int(c)] for c in connected_cells)
        assert len(tags)  == 1
        if tags:
            tag = tags.pop()
            boundaries[facet] = tag

    return boundaries


def preprocess_bdry_mesh(mesh_name, folder, fsi_tag):
    '''
    This is a convenience function for extracting from $folder/$mesh_3d_name,
    which has mesh and it facets_regions, a BoundaryMesh corresponding to
    facet_regions marked with fsi_tag and BoundaryMesh.facet_regions which are
    marked with remaining tags. In addition entity map is dumped.
    '''
    path = partial(os.path.join, folder)

    mesh_path = path(mesh_name)
    assert os.path.exists(mesh_path)
    # See what files would be generated and if they exist
    base, ext = os.path.splitext(mesh_name)
    bmesh_name = '_'.join([base, 'bmesh']) + ext

    mesh = Mesh()
    comm = mesh.mpi_comm()
    hdf = HDF5File(comm, mesh_path, 'r')
    hdf.read(mesh, '/mesh', False)
    tdim = mesh.topology().dim()

    # 0 and tdim-1 would be stored
    emap_names = [['%s_%d_%d_%d.emap' % (base, dim, rank, MPI.size(comm)) 
                   for rank in range(MPI.size(comm))] for dim in (0, tdim-1)]
                 
    # We have work to do...
    if not(os.path.exists(path(bmesh_name)) and all(map(os.path.exists, map(path, sum(emap_names, []))))):
        boundaries = FacetFunction('size_t', mesh)
        hdf.read(boundaries, '/boundaries')
        # Extract 
        bmesh, emap, bmesh_boundaries = py_BoundaryMesh(mesh, boundaries, fsi_tag, True)
        # Save
        out=HDF5File(mesh.mpi_comm(), path(bmesh_name), 'w')
        out.write(bmesh, '/mesh')
        out.write(bmesh_boundaries, '/boundaries')
        # Emap is specific each process write its map
        for dim in (0, tdim-1):
            rank, size = MPI.rank(comm), MPI.size(comm)
            emap_name = path('%s_%d_%d_%d.emap' % (base, dim, rank, size))
            np.savetxt(emap_name, emap[dim],
                       header='Entity map for entities of dim %d from %s mesh @process %d of %d' % (dim, base, rank, size))

    # At this point everything should exist
    assert os.path.exists(path(bmesh_name)) and all(map(os.path.exists, map(path, sum(emap_names, []))))
    # Transform emap to dictionary, local!
    rank, size = MPI.rank(comm), MPI.size(comm)
    entity_map = {}
    for dim in (0, tdim-1):
        emap_name = path('%s_%d_%d_%d.emap' % (base, dim, rank, size))
        entity_map[dim] = map(long, np.loadtxt(emap_name))

    return entity_map

# ----------------------------------------------------------------------------

if __name__ == '__main__':

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(),
                   '../mesh/HOLLOW-ELLIPSOID-HEALTY/hollow-ellipsoid-healty_0.h5',
                   'r')
    hdf.read(mesh, '/mesh', False)
    boundaries = FacetFunction('size_t', mesh)
    hdf.read(boundaries, '/boundaries')

    bmesh = BoundaryMesh(mesh, 'exterior')
    c2f = bmesh.entity_map(2)

    bmesh_subdomains = CellFunction('size_t', bmesh, 0)
    for cell in cells(bmesh):
        bmesh_subdomains[cell] = boundaries[c2f[int(cell.index())]]

    inlet_mesh = py_SubMesh(bmesh, bmesh_subdomains, 1)

    plot(inlet_mesh, interactive=True)

    if False:
        import sys

        cell_lookup = {10: 800, 20: 3200, 30: 7200}
        vertex_lookup = {10: 440, 20: 1680, 30: 3720}

        for ncells in (10, 20, 30):
            mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), *(ncells, )*3)
            
            boundaries = FacetFunction('size_t', mesh, 0)
            DomainBoundary().mark(boundaries, 1)
            CompiledSubDomain('near(1.+x[0], 0)').mark(boundaries, 3)
            CompiledSubDomain('near(1.-x[0], 0)').mark(boundaries, 4)

            bmesh, entity_map, boundaries = py_BoundaryMesh(mesh, boundaries, 1, True)
            # bmesh = py_BoundaryMesh(mesh)
            # c2f = bmesh.entity_map(bmesh.topology().dim())

            # Check mapping of cell to facet
            c2f = entity_map[bmesh.topology().dim()]
            count = 0
            for cell in cells(bmesh):
                facet = Facet(mesh, c2f[cell.index()])
                assert cell.midpoint().distance(facet.midpoint()) < 1E-15
                count += 1
            info('Checked %d facets/cells' % count)

            # Check mapping of vertices
            v2v = entity_map[0]
            count = 0
            for bvertex in vertices(bmesh):
                count += 1
                vertex = Vertex(mesh, v2v[bvertex.index()])
                assert vertex.midpoint().distance(bvertex.midpoint()) < 1E-14
            info('Checked %d vertices' % count)
            
            # Mesh construction okay?
            info('#cells = %d' % bmesh.num_cells()) 
            info('#vertices = %d' % bmesh.num_vertices()) 
            if ncells in (10, 20, 30):
                assert bmesh.size_global(0) == vertex_lookup[ncells]
                assert bmesh.size_global(2) == cell_lookup[ncells]
            
            # FunctionSpace constuction okay?
            V = FunctionSpace(bmesh, 'CG', 1)
            assert V.dim() == bmesh.size_global(0) 

            # NOTE: Run in parallel and load in paraview to see that we have a bug. There
            # is not such thing with simple BoundaryMesh so we are still missing
            # something. This was a BUG, now fixed -- see # 115
            # f = XDMFFile('bmesh.xdmf')
            # f.write(bmesh)

            # Some check for marking boundaries
            for i in (3, 4):
                dsi = Measure('ds', domain=bmesh, subdomain_data=boundaries, subdomain_id=i)
                e = assemble(Constant(1)*dsi)
                assert near(e, 8., 1E-10)

            # Loading ....
            preprocess_bdry_mesh(mesh_name='cylinder_0.h5', folder='./meshes/CYLINDER', fsi_tag=1)
            assert preprocess_bdry_mesh(mesh_name='cylinder_0.h5', folder='./meshes/CYLINDER', fsi_tag=1)

            # Check multiple markers defining FSI domain
            mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), *(ncells, )*3)
            boundaries = FacetFunction('size_t', mesh, 0)

            tag = 0
            for dim in range(3):
                tag += 1
                CompiledSubDomain('near(1.+x[%d], 0)' % dim).mark(boundaries, tag)
                tag += 1
                CompiledSubDomain('near(1.-x[%d], 0)' % dim).mark(boundaries, tag)

            bmesh, entity_map, boundaries = py_BoundaryMesh(mesh, boundaries, [1, 2, 3, 4], True)
            for i in (5, 6):
                dsi = Measure('ds', domain=bmesh, subdomain_data=boundaries, subdomain_id=i)
                e = assemble(Constant(1)*dsi)
                assert near(e, 8., 1E-10)
            # plot(boundaries)
            # interactive()
