#
# Driver for problems
#
from dolfin import set_log_level
from cbcpost import PostProcessor, SolutionField, PointEval
from cbcflow import NSSolver
from fsi_decoupled import FSI_Decoupled
from fields import *
import numpy as np

from problems import cylinder#, tshape

# How
set_log_level(100)
scheme = FSI_Decoupled(dict(r=1, s=0, u_degree=1, tol=1e-5))

# What
dt = 1E-5
N = 2
problem = cylinder.Cylinder(dict(dt=dt, N=N, T=3e-2, stress_amplitude=1e4, stress_time=5e-3))

# Saving
_plot = False
params = dict(save=True, plot=_plot, save_as="xdmf", stride_timestep=10)
name = problem.__class__.__name__

pp = PostProcessor(dict(casedir="Results/%s" % name, clean_casedir=True))
pp.add_field(SolutionField("Velocity", params))
pp.add_field(SolutionField("Pressure", params))
pp.add_field(SolutionField("Displacement", dict(save=False)))

field = SolidDisplacement(params)
pp.add_field(field)

# Run
solver = NSSolver(problem, scheme, pp)
solver.solve()
