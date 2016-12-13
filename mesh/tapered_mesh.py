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
            cmd = '''python -c"from dolfin import Mesh, HDF5File, MeshFunction;\
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

    write_volumes = geometry == 'dlayer'
    status = generate_gmsh_meshes(root=base, nrefs=nrefs, write_volumes=write_volumes)

    if status == 0:
        os.remove(name)
        return 0
    return 1

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    x = [0, 1, 2, 3]
    z = [1, 1.1, 1.1, 1.0]
    Z = [1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2]
    data = {'x': x, 'z': z, 'Z': Z}

    tapered_mesh(data=data,
                 geometry='hollow',
                 mesh_params={'size': 0.4},
                 nrefs=2)
