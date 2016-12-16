import subprocess, os, shutil
from mpi4py import MPI

comm = MPI.COMM_WORLD
assert comm.size == 1, 'GMSH does not work in parallel'


def generate_gmsh_meshes(root, nrefs=1):
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
        subprocess.call(['gmsh -clscale %g -3 %s' % (size, geo_file)], shell=True)
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
        cmd = '''python -c"from dolfin import Mesh, HDF5File, MeshFunction;\
                         mesh=Mesh('%s');\
                         facet_f=MeshFunction('size_t', mesh, '%s');\
                         out=HDF5File(mesh.mpi_comm(), '%s', 'w');\
                         out.write(mesh, '/mesh');\
                         out.write(facet_f, '/boundaries');\
                         "''' % (xml_file, xml_facets, h5_file)
        
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

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    
    # Conversion mode
    if len(sys.argv[1:]):
        root = sys.argv[1]
        nrefs = int(sys.argv[2])
        sys.exit(generate_gmsh_meshes(root, nrefs))
