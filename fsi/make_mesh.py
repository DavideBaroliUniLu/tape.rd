import subprocess, os
import numpy as np
from dolfin import *
import json


_TEMPLATE = 'template.geo'


def make_geo(model_json):
    '''Specialize the _TEMPLATE geo file'''
    with open(model_json) as f: model = json.load(f)
    geometry = model["geometry"]
    geometry.update({'q': (1+1./2)*geometry['r']})

    size_params = {'size': geometry['size'], 'alpha': 1./4}
    smoothing_params = {'nsmooth_global': 10,  'nsmooth_normal': 2}
    geometry.pop('size')

    assert set(geometry.keys()) == set(['r', 'R', 'q', 'l_i', 'l_c', 'l_e', 'L_ip', 'L_im', 'L_e'])
    # Total oa
    l = sum(geometry[key] for key in ('l_i', 'l_c', 'l_e'))
    # For construction reason bottom part of ICA must be longer to top part
    L = sum(geometry[key] for key in ('L_im', 'L_e'))
    Lp = geometry['L_ip']
    assert L > Lp
    # Use Lp in symmetries and the bottom is extruded by H
    H = L - Lp
    # Shifts due to radius
    r, R = geometry['r'], geometry['R']
    l, Lp = l+R, Lp+r
    # Now prepare info for template
    params = {'q': geometry['q']}
    params.update(size_params)
    params.update({'r': r, 'R': R, 'l': l, 'L': Lp, 'H': H})
    params.update(smoothing_params)

    header = ['\n'.join('%s = %g;' % kv for kv in params.items())+'\n']
    with open(_TEMPLATE, 'r') as f: template = f.readlines()
    domain = ''.join(header+template)
    with open('mesh.geo', 'w') as f: f.write(domain)

    return 0


def make_h5(model_json, modify_markers=True, debug=False):
    '''This is a conversion from xml mesh to h5.'''
    with open(model_json) as f: model = json.load(f)
    params = model["geometry"]

    xmls = ('%s.xml', '%s_physical_region.xml', '%s_facet_region.xml')
    mesh_xml, cell_xml, facet_xml = (xml % "mesh" for xml in xmls)
    assert all(os.path.exists(f) for f in (mesh_xml, cell_xml, facet_xml))
    # Read in for dolfin and get the markers ready
    mesh = Mesh(mesh_xml)
    facet_f = MeshFunction('size_t', mesh, facet_xml)

    if modify_markers:
        # Collect openings tags
        values, otags = facet_f.array(), (1, 2, 3)
        openings = [np.where(values == tag)[0] for tag in otags]
        # Marking: The painting strategy is so that there are no unmarked facets
        DomainBoundary().mark(facet_f, 13)
        # IA
        # CompiledSubDomain('x[2] < r + DOLFIN_EPS && on_boundary', r=params['r']).mark(facet_f, 16)
        # CompiledSubDomain('x[2] < -r + DOLFIN_EPS && on_boundary', r=params['r']).mark(facet_f, 15)
        # CompiledSubDomain('x[2] < -r + DOLFIN_EPS && on_boundary', r=(params['r']+params['L_im'])).mark(facet_f, 14)
        # OA
        # CompiledSubDomain('x[0] > R-DOLFIN_EPS && on_boundary', R=params['R']).mark(facet_f, 13)
        CompiledSubDomain('x[0] > R-DOLFIN_EPS && on_boundary', R=params['R']+params['l_i']).mark(facet_f, 12)
        CompiledSubDomain('x[0] > R-DOLFIN_EPS && on_boundary', R=params['R']+params['l_i']+params['l_c']).mark(facet_f, 11)
        # Put openings back
        for tag, domain in zip(otags, openings):
            for facet in domain: facet_f[facet] = tag
    # Saving
    mesh_name = '.'.join([model['galileo']['name'], 'h5']).encode('ascii', 'ignore')
    out=HDF5File(mesh.mpi_comm(), mesh_name, 'w')
    out.write(mesh, '/mesh')
    out.write(facet_f, '/boundaries')
    # Cleanup 
    map(os.remove, ("mesh.msh", "mesh.xml", "mesh_physical_region.xml", "mesh_facet_region.xml"))
    if not debug: os.remove("mesh.geo")

    # Finally tags,
    tags = {'inlets': [2], 'outlets': [1, 3], 'walls': [11, 12, 13]}
    return mesh_name, tags

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    json_name, stage = sys.argv[1:]

    if stage == 'geo':
        make_geo(json_name)
    else:
        assert stage == 'h5'
        make_h5(json_name)
