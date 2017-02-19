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


base = 'hollow-ellipsoid-healty'
nref = 2
mesh_file = os.path.join('..', 'mesh', base.upper(), base+'_%d.h5' % nref)
inflow_file = os.path.join('..', 'inflow_profile', base.upper(), base+'_%d.h5' % nref)
flux_file = os.path.join('.', 'input_data', 'Vegards01_VolumetricCSFFlow.txt')

problem = csf.CSF(dict(meshFile=mesh_file,
                       inflowFile=inflow_file,
                       fluxFile=flux_file,
                       T=1,
                       dt=1E-5))
info('Let...')
# Saving
_plot = False
save_how = dict(save=True, save_as='xdmf', stride_timestep=10)

name = problem.__class__.__name__ + base
pp = PostProcessor(dict(casedir="Results/%s" % name, clean_casedir=True))
pp.add_field(SolutionField("Velocity", save_how))
pp.add_field(SolutionField("Pressure", save_how))

# NOTE: Don't really save here - just have a (lambda u: u) available. 
pp.add_field(SolutionField("Displacement", dict(save=False, plot=False)))
# Real deal
pp.add_field(SolidDisplacement(save_how))

info('Let us solve')
# Run
solver = NSSolver(problem, scheme, pp)
solver.solve()
