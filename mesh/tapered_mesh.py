import subprocess, os, shutil
from mpi4py import MPI

comm = MPI.COMM_WORLD
assert comm.size == 1, 'GMSH does not work in parallel'


def generate_gmsh_meshes(root, nrefs, write_volumes):
    '''
    Geo file $root.geo is fed to gmsh with -clscale 1/2**i i=[0, 1, .., nrefs).
    '''
    geo_file = '%s.geo' % root
    msh_file = '%s.msh' % root

    # Make sure we have file
    assert os.path.exists(geo_file), 'Missing geo file %s .' % geo_file

    sizes = [1./2**i for i in range(nrefs)]

    for i, size in enumerate(sizes, 0):
        # Make the gmsh file for current size
        subprocess.call(['gmsh -clscale %g -3 -optimize %s' % (size, geo_file)], shell=True)
        assert os.path.exists(msh_file)

        print '-'*40, i+1, '/', len(sizes), '-'*40

        # Convert to xdmf
        xml_file = '%s_%d.xml' % (root, i)              
        xml_facets = '%s_%d_facet_region.xml' % (root, i)
        xml_volumes = '%s_%d_physical_region.xml' % (root, i)

        subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)],
                        shell=True)
        # All 3 xml files should exist
        assert all(os.path.exists(f) for f in (xml_file, xml_facets, xml_volumes))
        
        # Convert
        h5_file = '%s_%d.h5' % (root, i)
        if not write_volumes:
            cmd = '''python -c"from dolfin import Mesh, HDF5File, MeshFunction;\
mesh=Mesh('%s');\
facet_f=MeshFunction('size_t', mesh, '%s');\
out=HDF5File(mesh.mpi_comm(), '%s', 'w');\
out.write(mesh, '/mesh');\
out.write(facet_f, '/boundaries');\
"''' % (xml_file, xml_facets, h5_file)
        else:
            cmd = r'''python -c"from dolfin import Mesh, HDF5File, MeshFunction;\
mesh=Mesh('%s');\
facet_f=MeshFunction('size_t', mesh, '%s');\
cell_f=MeshFunction('size_t', mesh, '%s');
out=HDF5File(mesh.mpi_comm(), '%s', 'w');\
out.write(mesh, '/mesh');\
out.write(facet_f, '/boundaries');\
out.write(cell_f, '/volumes');\
"''' % (xml_file, xml_facets, xml_volumes, h5_file)
        
        subprocess.call([cmd], shell=True)
        # Success?
        assert os.path.exists(h5_file)

        # Cleanup
        [os.remove(f) for f in (xml_file, xml_facets, xml_volumes)]
        os.remove(msh_file)
        # Move the h5 file
        if not os.path.exists(root.upper()):
            os.makedirs(root.upper())
        else:
            assert os.path.isdir(root.upper())
        path = os.path.join(root.upper(), h5_file)
        shutil.move(h5_file, path)

    return 0


def tapered_mesh(data, geometry, mesh_params, name='test', nrefs=1):
    '''
    Data specifies either of ('slayer', 'dlayer', 'hollow') geometry. This is
    meshed according to mesh_params a with size being halved nrefs times. The
    output are meshes named geometry-name.
    '''
    # Consistency
    assert geometry in ('slayer', 'dlayer', 'hollow')
    assert 'x' in data and 'z' in data
    assert all(z > 0 for z in data['z'])
    assert len(data['x']) == len(data['z'])
    n = len(data['x'])
    write_volumes = geometry == 'dlayer'

    single = True
    if geometry != 'slayer':
        assert 'Z' in data
        assert all(Z > 0 for Z in data['Z'])
        assert all(z < Z for z, Z in zip(data['z'], data['Z']))
        assert n == len(data['Z'])
        single = False

    assert 'size' in mesh_params
    assert nrefs >= 1

    # Extract data for meshing
    size = mesh_params['size']
    SIZE = mesh_params.get('SIZE', size)
    nsmooth = mesh_params.get('Smoothing', 1)
    nsmooth_normals = mesh_params.get('SmoothNormals', 1)
    nsplines = mesh_params.get('SplinePoints', 10)   # For rotation extrusion

    # Build the declatations header
    x = 'xs[] = {' + ', '.join(map(str, data['x'])) + '};'
    z = 'zs[] = {' + ', '.join(map(str, data['z'])) + '};'
    Z = '' if single else 'Zs[] = {' + ', '.join(map(str, data['Z'])) + '};'
    nsmooth = 'Mesh.Smoothing = %d;' % nsmooth
    nsmooth_normals = 'Mesh.SmoothNormals = %d;' % nsmooth_normals
    nsplines = 'Geometry.ExtrudeSplinePoints = %d;' % nsplines
    size = 'size = %g;' % size
    SIZE = 'SIZE = %g;' % SIZE
    n = 'n = %d;' % n

    header = '\n'.join([x, z, Z, n, nsmooth, nsmooth_normals, nsplines, size])

    # The implementation from geo
    base = '-'.join([geometry, name])
    geometry = '.'.join([geometry, 'geo'])
    impl = open(geometry, 'r').readlines()
    # Search for where the implementation starts
    for i, line in enumerate(impl):
        if line.startswith('//!'):
            break
    impl = ''.join(impl[i:])
    body = '\n'.join([header, impl])
    
    # Finally dump
    name = '.'.join([base, 'geo'])
    with open(name, 'w') as out: out.write(body)

    status = generate_gmsh_meshes(root=base, nrefs=nrefs, write_volumes=write_volumes)

    if status == 0:
        os.remove(name)
        return 0
    return 1


def tapered_mesh_spline(data, mesh_params, name='test', nrefs=1):
    '''Hollow as above but here it is done via spline.'''
    # Consistency
    assert 'x' in data and 'z' in data and 'Z' in data
    assert all(z > 0 for z in data['z']) and all(Z > 0 for Z in data['Z'])
    assert len(data['x']) == len(data['z']) == len(data['Z'])
    n = len(data['x'])

    assert 'size' in mesh_params
    assert nrefs >= 1

    # Extract data for meshing
    size = mesh_params['size']
    if hasattr(size, '__iter__'): 
        assert len(size) == n
    else:
        size = [size]*n

    SIZE = mesh_params.get('SIZE', size)
    if hasattr(SIZE, '__iter__'): 
        assert len(SIZE) == n
    else:
        size = [SIZE]*n
    
    nsmooth = mesh_params.get('Smoothing', 1)
    nsmooth_normals = mesh_params.get('SmoothNormals', 1)
    nsplines = mesh_params.get('SplinePoints', 10)   # For rotation extrusion

    # Build the declatations header
    x = 'xs[] = {' + ', '.join(map(str, data['x'])) + '};'
    z = 'zs[] = {' + ', '.join(map(str, data['z'])) + '};'
    Z = 'Zs[] = {' + ', '.join(map(str, data['Z'])) + '};'
    size = 'size[] = {' + ', '.join(map(str, size)) + '};'
    SIZE = 'SIZE[] = {' + ', '.join(map(str, SIZE)) + '};'
    nsmooth = 'Mesh.Smoothing = %d;' % nsmooth
    nsmooth_normals = 'Mesh.SmoothNormals = %d;' % nsmooth_normals
    nsplines = 'Geometry.ExtrudeSplinePoints = %d;' % nsplines
    n = 'n = %d;' % n

    header = '\n'.join([x, z, Z, size, SIZE, n, nsmooth, nsmooth_normals, nsplines])

    # The implementation from geo
    base = '-'.join(['hollow-spline', name])
    geometry = 'hollow-spline.geo'
    impl = open(geometry, 'r').readlines()
    # Search for where the implementation starts
    for i, line in enumerate(impl):
        if line.startswith('//!'):
            break
    impl = ''.join(impl[i:])
    body = '\n'.join([header, impl])
    
    # Finally dump
    name = '.'.join([base, 'geo'])
    with open(name, 'w') as out: out.write(body)

    status = generate_gmsh_meshes(root=base, nrefs=nrefs, write_volumes=False)

    if status == 0:
        os.remove(name)
        return 0
    return 1

# ----------------------------------------------------------------------------

def _test(which=''):
    '''Check meshgen.'''
    assert which in ('', 'slayer', 'dlayer', 'hollow')

    if which == '':
        options = [True]*3
    else:
        options = [k == which for k in ['slayer', 'dlayer', 'hollow']]
    slayer, dlayer, hollow = options

    from dolfin import Mesh, HDF5File, FacetFunction, SubsetIterator, CellFunction
    x = [0, 1, 2, 3, 4]
    z = [1, 1.1, 1.1, 1.0, 0.9]
    Z = [1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2, 1.2]
    data = {'x': x, 'z': z, 'Z': Z}

    # SLAYER
    if slayer:
        tapered_mesh(data=data,
                     geometry='slayer',
                     name='test1',
                     mesh_params={'size': 0.4},
                     nrefs=1)
        
        mesh = Mesh()
        h5 = HDF5File(mesh.mpi_comm(), 'SLAYER-TEST1/slayer-test1_0.h5', 'r')
        h5.read(mesh, '/mesh', False)
        facet_f = FacetFunction('size_t', mesh)
        h5.read(facet_f, '/boundaries')
        assert all(any(1 for f in SubsetIterator(facet_f, bdry)) for bdry in (1, 2, 3))
        shutil.rmtree('SLAYER-TEST1')

    if dlayer:
        tapered_mesh(data=data,
                     geometry='dlayer',
                     name='test1',
                     mesh_params={'size': 0.4},
                     nrefs=1)
        
        mesh = Mesh()
        h5 = HDF5File(mesh.mpi_comm(), 'DLAYER-TEST1/dlayer-test1_0.h5', 'r')
        h5.read(mesh, '/mesh', False)
        facet_f = FacetFunction('size_t', mesh)
        h5.read(facet_f, '/boundaries')
        assert all(any(1 for f in SubsetIterator(facet_f, bdry)) for bdry in range(1, 7))
        cell_f = CellFunction('size_t', mesh)
        h5.read(cell_f, '/volumes')
        ones = [1 for f in SubsetIterator(cell_f, 1)]
        twos = [1 for f in SubsetIterator(cell_f, 2)]
        assert len(ones) and len(twos)
        assert (len(ones) + len(twos)) == mesh.num_cells()
        shutil.rmtree('DLAYER-TEST1')

    if hollow:
        tapered_mesh(data=data,
                     geometry='hollow',
                     name='test1',
                     mesh_params={'size': 0.4},
                     nrefs=1)
        
        mesh = Mesh()
        h5 = HDF5File(mesh.mpi_comm(), 'HOLLOW-TEST1/hollow-test1_0.h5', 'r')
        h5.read(mesh, '/mesh', False)
        facet_f = FacetFunction('size_t', mesh)
        h5.read(facet_f, '/boundaries')
        assert all(any(1 for f in SubsetIterator(facet_f, bdry)) for bdry in range(1, 5))
        shutil.rmtree('HOLLOW-TEST1')

    return True


def demo():
    '''Showcase ablilities'''
    from math import sin, pi
    from dolfin import Mesh, HDF5File, plot, FacetFunction

    outer = lambda x: 1 + 0.1*sin(2*pi*x)
    inner = lambda x: 0.2*(x-0.5)*(x-0.5) + 0.2
    
    x = [0.1*i for i in range(11)]
    z = map(inner, x)
    Z = map(outer, x)

    data = {'x': x, 'z': z, 'Z': Z}

    tapered_mesh(data=data,
                 geometry='hollow',
                 name='demo',
                 mesh_params={'size': 0.1, 'SIZE': 0.3, 'nsplines': 20},
                 nrefs=1)

    mesh = Mesh()
    h5 = HDF5File(mesh.mpi_comm(), 'HOLLOW-DEMO/hollow-demo_0.h5', 'r')
    h5.read(mesh, '/mesh', False)
    facet_f = FacetFunction('size_t', mesh)
    h5.read(facet_f, '/boundaries')
    plot(facet_f, interactive=True)

    shutil.rmtree('HOLLOW-DEMO')

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # assert _test('slayer')
    # demo()

    from dolfin import Mesh, HDF5File, plot, FacetFunction
    x = [0, 1, 2, 3, 4]
    z = [1, 1.1, 1.1, 1.0, 0.9]
    Z = [1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2, 1.2]
    data = {'x': x, 'z': z, 'Z': Z}

    size = [0.1]*len(x)
    SIZE = [0.4]*len(x)
    mesh_params = {'size': size, 'SIZE': SIZE}
    
    tapered_mesh_spline(data, mesh_params, name='test', nrefs=1)

    mesh = Mesh()
    h5 = HDF5File(mesh.mpi_comm(), 'HOLLOW-SPLINE-TEST/hollow-spline-test_0.h5', 'r')
    h5.read(mesh, '/mesh', False)
    facet_f = FacetFunction('size_t', mesh)
    h5.read(facet_f, '/boundaries')
    plot(facet_f, interactive=True)

    shutil.rmtree('HOLLOW-SPLINE-TEST')
