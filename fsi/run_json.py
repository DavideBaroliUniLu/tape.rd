# prython run_json path-to-validated-json
from dolfin import mpi_comm_world, MPI
from dolfin import set_log_level
from make_json import make
from cbcflow import NSSolver
import argparse, os

set_log_level(100)

parser = argparse.ArgumentParser()
parser.add_argument('model_json', type=str, help='Path JSON file specifying model')
# parser.add_argument('--strict_bcs', type=int,
#                     help='Disable checking for consistency between bcs: just warnings ',
#                     default=0)
# parser.add_argument('--fenics_checks', type=int,
#                     help='Enable possibly expensive fenics checks',
#                     default=0)
# parser.add_argument('--verify_dir', type=str, default='./ui')
args = parser.parse_args()

model_json = args.model_json
# assert 'parsed' not in model_json
# 
# # Do the generation on master
# comm = mpi_comm_world().tompi4py()
# status = 0
# if comm.rank == 0:
#     from ui.parse import verify
#     import json
# 
#     verify_dir = os.path.abspath(args.verify_dir)
#     exec_dir = os.path.abspath('.')
#     as_verify = lambda f: os.path.join(verify_dir, f)
#     as_exec = lambda f: os.path.join(exec_dir, os.path.relpath(as_verify(f), exec_dir))
#     # Sanity here
#     assert os.path.exists(as_verify(model_json)), 'Input file %s does not exist' % as_verify(model_json)
#     # Let's be strict and require .json
#     ext = os.path.splitext(model_json)[-1]
#     assert ext == '.json', 'Input file %s is likely not JSON' % model_json
# 
#     fenics_checks, strict_bcs = 0, 0
#     # Get the parsed and correct model spec to be used for futher generation
#     model = verify(model_json, fenics_checks, strict_bcs, as_verify, as_exec)
#     
#     # This dictionary should be close to being valid so dumpt it
#     parsed_json = '-'.join(['parsed', model_json])
#     parsed_json = os.path.join(exec_dir, parsed_json)
#     with open(parsed_json, 'w') as out: json.dump(model, out)
#     status = 1
# status = comm.bcast(status)
# 
# assert status == 1
# model_json = '-'.join(['parsed', model_json])
assert os.path.exists(model_json), model_json

scheme, problem, postprocessor = make(model_json)
solver = NSSolver(problem, scheme, postprocessor)
# Just to be sure put barrier here so that all processes start solving with
# setup finished
MPI.barrier(mpi_comm_world())
solver.solve()
