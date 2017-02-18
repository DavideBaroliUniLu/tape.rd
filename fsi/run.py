#
# Driver for problems
#
from dolfin import set_log_level
from cbcpost import PostProcessor, SolutionField, PointEval
from cbcflow import NSSolver
from fsi_decoupled import FSI_Decoupled
from fields import *
import numpy as np
from problems import csf
import os

# How
set_log_level(100)
scheme = FSI_Decoupled(dict(r=1, s=0, u_degree=1))


base = 'hollow-ellipsoid-abnormal'
nref = 2
mesh_file = os.path.join('..', 'mesh', base.upper(), base+'_%d.h5' % nref)
inflow_file = os.path.join('..', 'inflow_profile', base.upper(), base+'_%d.h5' % nref)
flux_file = os.path.join('.', 'input_data', 'Vegards01_VolumetricCSFFlow.txt')

problem = csf.CSF(dict(meshFile=mesh_file,
                       inflowFile=inflow_file,
                       fluxFile=flux_file,
                       T=1,
                       dt=1E-5))

print 'XXXXXXX'
# Saving
_plot = False
name = problem.__class__.__name__
pp = PostProcessor(dict(casedir="Results/%s" % name, clean_casedir=True))
pp.add_field(SolutionField("Velocity", dict(save=True, plot=_plot, save_as="xdmf")))
pp.add_field(SolutionField("Pressure", dict(save=True, plot=_plot, save_as="xdmf")))
# NOTE: Don't really save here - just have a (lambda u: u) available. 
pp.add_field(SolutionField("Displacement", dict(save=False, plot=False)))
# Real deal
pp.add_field(SolidDisplacement(dict(save=True, save_as='xdmf')))

# Run
solver = NSSolver(problem, scheme, pp)
solver.solve()
