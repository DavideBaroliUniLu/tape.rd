#
# Driver for problems
#
from dolfin import set_log_level
from cbcpost import PostProcessor, SolutionField, PointEval
from cbcflow import NSSolver
from fsi_decoupled import FSI_Decoupled
from fields import *
import numpy as np
# from problems import csf_healthy
from problems import cylinder

# How
set_log_level(100)
scheme = FSI_Decoupled(dict(r=1, s=0, u_degree=1))

# What
dt = 1E-5
N = 2
# problem = csf_healthy.CSF(dict(dt=dt, N=N, T=3e-2, stress_amplitude=8e2, stress_time=5e-3))
problem = cylinder.Cylinder(dict(dt=dt, N=N, T=3e-2, stress_amplitude=8e2, stress_time=5e-3))

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
