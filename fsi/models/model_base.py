from dolfin import Expression

class SolidModelBase(object):
    '''Interface for specifying solid problems'''
    def __init__(self, solution, traction, n, dt, bcs, params):
        '''
        The solid model takes in a vector valued traction and returns a vector which is
        the solution of specific IBVP with bcs. Normal is n, params is the
        material.

        Solution vector specifies initial conditions.
        Note that bcs is a list of tripplets with (value, domain, tag) where the
        value is expected to be vector valued.
        '''
        raise NotImplementedError

    def solve(self):
        '''Perform one step'''
        raise NotImplementedError


def is_tdep_Exr(f, attrs=['t', 'time']):
    '''Is f a time-dependent expression.'''
    return isinstance(f, Exression) and any(hasattr(f, attr) for attr in attrs)
