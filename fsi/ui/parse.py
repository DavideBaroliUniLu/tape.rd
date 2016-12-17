import json, os, shutil
import numpy as np
import sys
sys.path.append(os.path.relpath('..', os.path.abspath(__file__)))

MODEL_OBJECTS = ('solver', 'fluid_properties', 'solid_properties', 'geometry',
                 'forcing', 'boundary_conditions', 'output', 'galileo')
SOLVER_OBJECTS = ('time_step', 'tolerance', 'Tfinal')
FP_OBJECTS = ('density', 'viscosity')
SP_OBJECTS = ('E', 'nu', 'epsilon', 'density', 'R', 'alpha0', 'alpha1')
GEOMETRY_OBJECTS = ('r', 'l_i', 'l_c', 'l_e', 'R', 'L_e', 'L_im', 'L_ip', 'size')

RED = '\033[1;37;31m%s\033[0m'

# ----------------------------------------------------------------------------

def has_all_keys(wanted, given, who):
    '''Report differences between wanted and given'''
    missing = set(wanted) - set(given)
    extra = set(given) - set(wanted)
    if len(missing) == 0 and len(extra) == 0:
        msg = ''
    else:
        msg = '%s expecting %s: missing %s, extra %s' % (who, wanted, missing, extra)
    return msg


def verify(model_json, fenics_checks, strict_bcs, as_verify, as_exec):
    '''Validate the JSON input returning a dictionary specifying the model.'''
    # Parsing must go well
    model_json = as_verify(model_json)
    try:
        with open(model_json) as f: model = json.load(f)
    except Exception as e:
        assert False, 'Error parsing %s:' % model_json + str(e)

    # All the subobjects must be present
    msg = has_all_keys(MODEL_OBJECTS, model.keys(), 'model')
    assert not msg, msg

    # Solver object wants all its fields.
    solver = model['solver']
    msg = has_all_keys(SOLVER_OBJECTS, solver.keys(), 'solver')
    assert not msg, msg
    # All the field values are numbers. 
    n_num = filter(lambda (k, v): not isinstance(v, (int, float)), solver.items())
    assert not n_num, 'Solver: Not numbers %s' % n_num
    # Also time stuff positive
    n_pos = filter(lambda (k, v): not v > 0,
                   ((k, solver[k]) for k in ('Tfinal', 'time_step')))
    assert not n_pos, 'Solver: Not positive %s' % n_pos

    # Fluid properties: all must be given
    fp = model['fluid_properties']
    msg = has_all_keys(FP_OBJECTS, fp.keys(), 'fluid_properties')
    assert not msg, msg
    # All the field values are numbers. 
    n_num = filter(lambda (k, v): not isinstance(v, (int, float)), fp.items())
    assert not n_num, 'Fluid properties: Not numbers %s' % n_num
    # All are positive
    n_pos = filter(lambda (k, v): not v > 0, fp.items())
    assert not n_pos, 'Fluid properties: Not positive %s' % n_pos

    # Solid properties, all must be given
    sp = model['solid_properties']
    msg = has_all_keys(SP_OBJECTS, sp.keys(), 'solid_properties')
    assert not msg, msg
    # Numbers
    n_num = filter(lambda (k, v): not isinstance(v, (int, float)), sp.items())
    assert not n_num, 'Solid properties: Not numbers %s' % n_num
    # Positivity for sure on those physical parameters
    n_pos = filter(lambda (k, v): not v > 0,
                   ((k, sp[k]) for k in ('E', 'nu', 'R', 'density', 'epsilon')))
    assert not n_pos, 'Solid properties: Not positive %s' % n_pos

    # Geometry - IDEA if we know where to save the file - overwrite the path entry in
    # model geometry.
    bdries = verify_geometry(model['geometry'], fenics_checks, as_verify, as_exec)
    assert bdries, 'Geometry processing failed'

    # Forcing - substitues 0 for not speced walls
    assert verify_forcing(model['forcing'], bdries['walls'])
    
    # Same as above the input is expanded
    assert verify_boundary_conditions(model['boundary_conditions'],
                                      bdries['inlets'], bdries['outlets'],
                                      strict_bcs,
                                      fenics_checks,
                                      as_verify,
                                      as_exec)

    assert verify_output(model['output'],     
                         bdries['inlets'], bdries['outlets'],
                         fenics_checks)

    # FIXME: galileo?

    return model

# ----------------------------------------------------------------------------

def verify_output(output, inlets, outlets, fenics_checks):
    '''
    Output is valid if it contains simulation mame and one of recognized probes.
    Then we go after probes.
    '''
    keys = set(output.keys())
    assert 'sim_name' in keys, 'Missing sim_name which determines result folder'
    keys.remove('sim_name')
    
    # Handle fields - they must always be present for other probes to work.
    # Present \neq being saved
    required = set(['Velocity', 'Pressure', 'Displacement'])

    fields = output.get('fields', [])
    for f in fields:
        # which has to be supported
        assert f['name'] in required
        dt = f.get('dt', 1)
        assert isinstance(dt, int) and dt >= 1
        f['dt'] = dt          # Assign back in case dt was missing
        required.remove(f['name'])
    keys.remove('fields')
    # Fill in the remainging guys
    if not fields: fields = required
    for f in required:
        output['fields'].append({'name': f, 'dt': 0})

    # The remaing guys must be among probes. Whoever has dt is positive
    # otherwise dt = -1
    point_probes = set(['velocity_probes', 'pressure_probes'])
    crossec_probes = set(['velocity_profiles', 'flow_rates', 'areas'])
    for k in keys:
        assert k in point_probes or k in crossec_probes, 'Unrecognized probe %s' % k
        # A point probe must have x
        if k in point_probes:
            for probe in output[k]:
                assert 'x' in probe
                assert 'name' in probe
                dt = probe.get('dt', 1)
                assert isinstance(dt, int) and dt >= 1
                probe['dt'] = dt

        else:
            # Integral problem is (x, n) or tag
            for probe in output[k]:
                try:
                    assert 'x' in probe and 'normal' in probe
                except AssertionError:
                    assert 'tag' in probe, 'Either (x, normal) or tag is not given'
                    assert probe['tag'] in inlets or probe['tag'] in outlets
                assert 'name' in probe
                # Consistency of time
                dt = probe.get('dt', 1)
                assert isinstance(dt, int) and dt >= 1
                probe['dt'] = dt
    return True

def verify_boundary_conditions(bcs, inlets, outlets, strict, fenics_checks, as_verify, as_exec):
    '''TODO'''
    valid_tags = set(inlets+outlets)
    is_valid_tag = lambda tag: isinstance(tag, int) and tag > 0 and tag in valid_tags

    # Solid part:
    # warn if output Neumann on inlets
    tags = bcs['solid']  # These are dirichlet

    assert len(tags) == len(set(tags)), 'Duplicate tags in solid bcs'
    assert all(map(is_valid_tag, tags)), 'Invalid tag: not size_t or not inlet/outlet domain'
    neumann_bdries = set(inlets + outlets) - set(tags)
    neumann_inlets = neumann_bdries & set(inlets)
    if neumann_inlets:
        msg = 'Setting zero stress on inlets %s' % neumann_inlets
        if strict:
            assert False, msg
        else:
            print RED % 'Warning: ', msg
    # Expand for consistency
    bcs['solid'] = [{'tag': t, 'value': 0} for t in tags]

    # Absorbing bcs:
    if 'outlet' not in bcs: bcs['outlet'] = []
    outlet_bc_tags = [bc['tag'] for bc in bcs['outlet']]
    assert len(outlet_bc_tags) == len(set(outlet_bc_tags)), 'Duplicate tags in outlet bcs'
    outlet_bc_tags = set(outlet_bc_tags)
    assert all(map(is_valid_tag, outlet_bc_tags)), 'Invalid tag: not size_t or not inlet/outlet domain'
    # Only on outlets
    msg = 'Setting outlet bcs on %s which is inlet. Outlets are %s' % (outlet_bc_tags, outlets)
    assert outlet_bc_tags <= set(outlets), msg
    # Exapand not specified outlets - i.e. all outlets are specified
    for tag in (set(outlets) - outlet_bc_tags):
        bcs['outlet'].append({'tag': tag, 'value': 0.0})
    # The added value must be a valid form of specifying pressure
    for bc in bcs['outlet']:
        assert is_valid_pressure(bc['value'], fenics_checks, as_verify, as_exec),\
                'Invalid specifycation of outlet bc for outlet %d' % bc['tag']

    # Fluid
    # In any case must be valid tags
    bcs_fluid = bcs['fluid']
    # More specifically they must live on inlets only
    for bc_list in bcs_fluid.values():
        for bc in bc_list:
            tag = bc['tag']
            assert is_valid_tag(tag), "tag: %d" % tag
    # If both velocity and pressure are specified they must have different domains
    velocity_tags = [bc['tag'] for bc in bcs_fluid['velocity']]\
                     if 'velocity' in bcs_fluid else [] 
    assert len(velocity_tags) == len(set(velocity_tags)), 'Duplicate velocity bcs'

    pressure_tags = [bc['tag'] for bc in bcs_fluid['pressure']]\
                     if 'pressure' in bcs_fluid else [] 
    assert len(pressure_tags) == len(set(pressure_tags)), 'Duplicate pressure bcs'

    vp_tags = set(velocity_tags) & set(pressure_tags)
    assert len(vp_tags) == 0, 'Setting pressure AND velocity on bdries %s' % vp_tags

    # Something must be said about each inlet
    #unspecified = set(inlets) - (set(velocity_tags) | set(pressure_tags))

    # At least on boundary has a pressure of velocity boundary condition
    unspecified = set(inlets + outlets).isdisjoint(set(velocity_tags) | set(pressure_tags))
    assert not unspecified, 'Velocity or pressure must be specified on each inlet. Missing %s' % unspecified

    # Valid pressure 
    if 'pressure' in bcs_fluid:
        for bc in bcs_fluid['pressure']:
            value = bc['value']
            assert is_valid_pressure(value, fenics_checks, as_verify, as_exec)
    # Valid velocity
    if 'velocity' in bcs_fluid:
        for bc in bcs_fluid['velocity']:
            assert 'period' in bc, 'Missing period specification'
            assert isinstance(bc['period'], (int, float)) and bc['period'] > 0, 'Invald period: need positive number'
            assert 'path' in bc, 'Missing path to Womersley'
            assert isinstance(bc['path'], (str, unicode)) and os.path.exists(as_verify(bc['path'])), 'Invalid Womersley path'
            assert len(np.loadtxt(as_verify(bc['path'])).shape) == 1, 'a vector is required'
            bc['path'] = as_exec(bc['path'])

    return True 


def is_valid_pressure(spec, fenics_checks, as_verify, as_exec):
    '''
    Valid pressure is:
        value: number --> maps to Constant pressure
        value: {body: ..., degree: ...}    --> will be compiled
        value: {path: data} --> table t vs value
        value: {path: data, period: number}  -> single column ...
    '''
    if isinstance(spec, (int, float)):
        return True
    else:
        assert isinstance(spec, dict), 'Non-constant function is specified by dictionary'
        assert 'path' in spec or 'body' in spec
        # Tabulated
        if 'path' in spec:
            path = spec['path']

            assert os.path.exists(as_verify(path)), 'The file for tabulated function does not exist'

            table = np.loadtxt(as_verify(path))
            if 'period' not in spec:
                assert len(table.shape) == 2 and table.shape[1] == 2, 'n by 2 matrix is required'
            else:
                assert len(table.shape) == 1, 'a vector is required'

            spec['path'] = as_exec(path)

            return True
        # Compiled
        else:
            body = spec['body']
            assert isinstance(body, (str, unicode))
            body = body.encode('ascii', 'ignore')
            # Need degree
            if not 'degree' in spec: spec['degree'] = 1

            if fenics_checks:
                from dolfin import Expression
                try:
                    Expression(body, t=0., degree=spec['degree'])
                except RuntimeError:
                    assert False, 'Function not given in terms of t'        
            return True


def verify_forcing(forces, walls):
    '''For each wall there is either constant or step forcing.'''
    # Not specified == 0    # Or vector constant?
    zero_walls = set(walls)
    checked_walls = set([])
    for f in forces:
        wall = f['tag']

        assert wall not in checked_walls, 'Duplicit force on wall %d' % wall
        checked_walls.add(wall)

        assert isinstance(wall, int) and wall > 0
        # Now check the value
        value = f['value']
        if isinstance(value, (int, float)):
            pass
        else:
            assert is_step(value, 'forcing %d' % wall)
        zero_walls.remove(wall)
    # Just for consistency = specifying input for each wall
    for w in zero_walls: forces.append({'tag': w, 'value': 0})

    return True


def is_step(value, wall):
    '''Steo function {'v0': 1, 'dt': 1, 'delta': 1}'''
    assert isinstance(value, dict)

    msg = has_all_keys(('dt', 'v0', 'delta'), value.keys(), wall)
    assert not msg, msg
    # And all are numbers
    n_num = filter(lambda (k, v): not isinstance(v, (int, float)), value.items())
    assert not n_num, 'Fluid properties: Not numbers %s' % n_num

    return True
            

def verify_geometry(geometry, fenics_checks, as_verify, as_exec):
    '''
    See that loaded mesh is well defined. Alternatively generate a mesh from the
    correct input.
    '''
    if 'path' in geometry:
        mesh_file = geometry['path']
        assert os.path.exists(as_verify(mesh_file)), 'Mesh %s does not exist' % mesh_file
        # For now support only h5 file
        assert os.path.splitext(mesh_file)[-1] == '.h5', 'Mesh %s is not h5 file' % mesh_file
        assert 'tags' in geometry, 'Tags missing in geometry specification'
        tags = geometry['tags']
        for bdry_type in ('inlets', 'outlets', 'walls'):
            assert bdry_type in tags, 'Missing markers for %s' % bdry_type
            values = tags[bdry_type]
            assert isinstance(values, list) and values, 'Invalid markers for %s' % bdry_type
            assert all(map(lambda v: isinstance(v, int) and v > 0, values)), 'Invalid markers for %s' % bdry_type

    # otherwise the input must make sense for meshing by gmsh
    else:
        assert False

    # Checks:
    # All required bdries are specified - [this also checs if WE made the mesh well]
    bdry_types = ('inlets', 'outlets', 'walls')
    msg = has_all_keys(tags.keys(), bdry_types, 'geometry')
    # No bdry is empty
    assert all(tags[bdry_type] for bdry_type in bdry_types)
    # The boundary domains do not overlap
    for i, btype0 in enumerate(bdry_types):
        assert len(tags[btype0]) == len(set(tags[btype0])), 'Duplicate tag in %s' % btype0
        for btype1 in bdry_types[i+1:]:
            overlap = set(tags[btype0]) & set(tags[btype1])
            assert len(overlap) == 0, 'Same markers %s between %s and %s' % (overlap, tags[btype0], tags[btype1])
    
    if fenics_checks:
        # Fluid domain setup
        mesh = Mesh()
        # NOTE: unicode to string conversion
        hdf = HDF5File(mesh.mpi_comm(), as_verify(mesh_file).encode('ascii', 'ignore'), 'r')
        hdf.read(mesh, '/mesh', False)
        boundaries = FacetFunction('size_t', mesh)
        hdf.read(boundaries, '/boundaries')

        from numpy import array 
        # The markers that user provided as tags for boundary
        user_markers = sum((tags[bdry_type] for bdry_type in bdry_types), [])
        assert all(isinstance(m, int) and m > 0 for m in user_markers), 'Not all size_t markers'

        # The markers that are found on the boundary
        mesh_boundary = FacetFunction('size_t', mesh, 0)
        DomainBoundary().mark(mesh_boundary, 1)
        # Local
        mesh_markers = set(boundaries[f] for f in SubsetIterator(mesh_boundary, 1))
        # Now new tags were found
        assert mesh_markers <= set(user_markers), 'Some tags not given by user %s' %  mesh_markers
        # Now see if ALL the user tags were found
        found = array([any(1 for e in SubsetIterator(boundaries, m))
                       for m in user_markers])
        found = mesh.mpi_comm().tompi4py().allreduce(found)
        assert all(found), 'Missing some markers %s' % filter(lambda v: not v, found)
    # Update the path relative t ...
    geometry['path'] = as_exec(mesh_file)

    return tags
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('model_json', type=str, help='Path JSON file specifying model')
    parser.add_argument('--strict_bcs', type=int,
                        help='Disable checking for consistency between bcs: just warnings ',
                        default=1)
    parser.add_argument('--fenics_checks', type=int,
                        help='Enable possibly expensive fenics checks',
                        default=1)
    parser.add_argument('--verify_dir', type=str, default='.')
    parser.add_argument('--exec_dir', type=str, default='..')
    args = parser.parse_args()

    model_json, fenics_checks, strict_bcs = args.model_json, args.fenics_checks, args.strict_bcs
    # See if FEniCS checks make sense
    if fenics_checks:
        try:
            from dolfin import HDF5File, Mesh, FacetFunction, SubsetIterator, DomainBoundary
            from mpi4py import MPI
        except ImportError:
            print 'Error importing - disabling checks'
            fenics_checks = False

    # NOTE: We assume that all the files path in the user given json are
    # specified by their path relative to w.r.t verify dir. In the parsed json 
    # file the paths are auto rewritten relative to exec directory
    verify_dir = os.path.abspath(args.verify_dir)
    exec_dir = os.path.abspath(args.exec_dir)

    # In general we will check for existence of file by
    as_verify = lambda f: os.path.join(verify_dir, f)
    # And its path will then be set by
    as_exec = lambda f: os.path.join(exec_dir, os.path.relpath(as_verify(f), exec_dir))

    # Sanity here
    assert os.path.exists(as_verify(model_json)), 'Input file %s does not exist' % as_verify(model_json)
    # Let's be strict and require .json
    ext = os.path.splitext(model_json)[-1]
    assert ext == '.json', 'Input file %s is likely not JSON' % model_json
    # Get the parsed and correct model spec to be used for futher generation
    model = verify(model_json, fenics_checks, strict_bcs, as_verify, as_exec)
    # This dictionary should be close to being valid so dumpt it
    model_json = '-'.join(['parsed', model_json])
    parsed_json = os.path.join(exec_dir, model_json)
    with open(parsed_json, 'w') as out: json.dump(model, out)
    sys.exit(0)
