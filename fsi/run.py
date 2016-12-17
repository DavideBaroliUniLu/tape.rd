#
# Driver for problems
#
from dolfin import set_log_level
from cbcpost import PostProcessor, SolutionField, PointEval
from cbcflow import NSSolver
from fsi_decoupled import FSI_Decoupled
from fields import *
import numpy as np

from problems import realistic

# How
set_log_level(100)
scheme = FSI_Decoupled(dict(r=1, s=0, u_degree=1, tol=1e-5))

# What
dt = 1E-5
N = 0
nsteps = int(1./dt)*3
problem = realistic.Realistic(dict(dt=dt, N=N, T=nsteps*dt))

# Saving

_plot = False
params = dict(save=True, plot=_plot, save_as="xdmf", stride_timestep=100)

name = 'NewRealistic'
pp = PostProcessor(dict(casedir="Results/%s" % name, clean_casedir=True))
pp.add_field(SolutionField("Velocity", params))
pp.add_field(SolutionField("Pressure", params))

pp.add_field(SolutionField("Displacement", dict(save=False)))
pp.add_field(SolidDisplacement(params))

probe_pts = [np.array([ 215.50300598,  140.26100159,  -37.15499878]),
             np.array([ 214.18899536,  142.85099792,  -36.79090118]),
             np.array([ 213.08299255,  145.51899719,  -37.21089935]),
             np.array([ 213.21000671,  148.38600159,  -36.91189957]),
             np.array([ 213.68600464,  150.99499512,  -35.66939926]),
             np.array([ 212.2250061 ,  153.21800232,  -35.08879852]),
             np.array([ 209.52600098,  153.38299561,  -35.98509979]),
             np.array([ 206.89399719,  154.11700439,  -35.40359879]),
             np.array([ 204.90899658,  156.18299866,  -34.84209824]),
             np.array([ 202.42700195,  157.64700317,  -34.87060165]),
             np.array([ 201.67700195,  159.93600464,  -34.7737999 ])]

params = dict(save=True)
for i, probe in enumerate(probe_pts):
    pp.add_field(PointEval("Velocity", [probe], label=str(i), params=params))
    pp.add_field(PointEval("Pressure", [probe], label=str(i), params=params))


# Run
solver = NSSolver(problem, scheme, pp, dict(timer_frequency=10, check_memory_frequency=10))
solver.solve()
