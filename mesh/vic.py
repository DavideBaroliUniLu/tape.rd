from math import sqrt
import numpy as np
import os

# Here the first list are AP(left-right) and T(up-down) diameters of spinal chord, 
# the second is then data for spinal chanal (which is surrouns the chord. The data
# is in cm

healthy = [[(0.93, 1.13),
            (0.88, 1.24),
            (0.86, 1.32),
            (0.87, 1.40),
            (0.83, 1.39),
            (0.79, 1.32),
            (0.74, 1.14)],  
           [(1.56, 2.77),
            (1.43, 2.55),
            (1.26, 2.24),
            (1.25, 2.23),
            (1.28, 2.27),
            (1.31, 2.33),
            (1.34, 2.39)]]

abnormal = [[(0.93, 1.13),
             (0.88, 1.24),
             (0.86, 1.32),
             (0.87, 1.40),
             (0.83, 1.39),
             (0.79, 1.32),
             (0.74, 1.14)],  
            [(1.56, 2.77),
             (1.43, 2.55),
             (1.26, 2.24),
             (1.25, 2.23),
             (1.55, 2.76),
             (1.85, 3.29),
             (2.15, 3.83)]]


def process_series_rot(data, diameter, fit, lstsq=False):
    '''
    Out of the data above (in general 2 lists of tuples) we build a mesh.
    Since the domain here is defined by rotatiing a curve (crossection is a
    cirlce) the tuple has to be reduced:
      
           ^ t
      <--------->ap
           V

      diameter [ap -> first, t -> last, 'avg' - > fit circle to ellipse]

    The data is equidistant but we need to define where it was measured:
      fit [first -> at the start of C_0, the the last guy is prolongared
           mid   -> the measur. are at mid points and we prolongate first/last]

    One we have x vs data we can optionally do least squares fit
    '''
    chord, canal = data
    assert len(chord) == len(canal)
    assert fit in ('first', 'mid')

    # Choser how to map ap, t values in the diamater to be used for the curve
    assert diameter in ('ap', 't', 'avg')
    
    # NOTE: We are after radii!!!
    if diameter == 'ap':
        select = lambda p: p[0]/2.
    elif diameter == 't':
        select = lambda p: p[1]/2.
    else:
        select = lambda p: sqrt(p[0]*p[1])/2.

    # At this point we have what is refered to in mesh gen as z, Z
    chord = map(select, chord)
    canal = map(select, canal)

    # It remains to get x
    # We do this by either making the last C flat
    if fit == 'first':
        x = [i*1 for i in range(len(chord)+1)]
        chord.append(chord[-1])
        canal.append(canal[-1]) 
    # or flatten first and last
    if fit == 'mid':
        x = [0] + [0.5+i*1 for i in range(len(chord))]
        x.append(x[-1]+0.5)
        chord.append(chord[-1])
        chord = [chord[0]] + chord

        canal.append(canal[-1])
        canal = [canal[0]] + canal
    assert len(x) == len(chord) == len(canal)
    x, chord, canal = map(np.array, (x, chord, canal))

    # Optionally smooth the data by least-squares using.
    # Note, I only use here the data prior to extension
    if lstsq:
        if fit == 'mid':
            for data in chord, canal:
                A = np.vstack([x[1:5], np.ones(len(x[1:5]))]).T
                m0, c0 = np.linalg.lstsq(A, data[1:5])[0]

                A = np.vstack([x[5:9], np.ones(len(x[5:9]))]).T
                m1, c1 = np.linalg.lstsq(A, data[5:9])[0]
              
                data[:] = np.r_[m0*x[:5]+c0, m1*x[5:]+c1]

        if fit == 'first':
            for data in chord, canal:
                A = np.vstack([x[0:4], np.ones(len(x[0:4]))]).T
                m0, c0 = np.linalg.lstsq(A, data[0:4])[0]

                A = np.vstack([x[4:8], np.ones(len(x[4:8]))]).T
                m1, c1 = np.linalg.lstsq(A, data[4:8])[0]
              
                data[:] = np.r_[m0*x[:4]+c0, m1*x[4:]+c1]

    assert len(x) == len(chord) == len(canal)
       
    data = {'x': x,
            'a': chord, 'b': chord,
            'A': canal, 'B': canal}

    return data


def process_series_ellipse(data, fit, lstsq=False):
    '''
    Out of the data above (in general 2 lists of tuples) where the tuple defines
    axis of an ellipse

    The data is equidistant but we need to define where it was measured:
      fit [first -> at the start of C_0, the the last guy is prolongared
           mid   -> the measur. are at mid points and we prolongate first/last]

    One we have x vs data we can optionally do least squares fit
    '''
    chord, canal = data
    assert len(chord) == len(canal)
    assert fit in ('first', 'mid')


    # At this point we have what is refered to in mesh gen as z, Z
    chord0, chord1 = map(lambda p: p[0]/2., chord), map(lambda p: p[1]/2., chord)
    canal0, canal1 = map(lambda p: p[0]/2., canal), map(lambda p: p[1]/2., canal)

    # It remains to get x
    # We do this by either making the last C flat
    if fit == 'first':
        x = [i*1 for i in range(len(chord)+1)]

        chord0.append(chord0[-1])
        chord1.append(chord1[-1])

        canal0.append(canal0[-1])
        canal1.append(canal1[-1])
    # or flatten first and last
    if fit == 'mid':
        x = [0] + [0.5+i*1 for i in range(len(chord))]
        x.append(x[-1]+0.5)

        chord0.append(chord0[-1])
        chord0 = [chord0[0]] + chord0
        chord1.append(chord1[-1])
        chord1 = [chord1[0]] + chord1

        canal0.append(canal0[-1])
        canal0 = [canal0[0]] + canal0
        canal1.append(canal1[-1])
        canal1 = [canal1[0]] + canal1

    assert len(x) == len(chord0) == len(canal0)
    assert len(x) == len(chord1) == len(canal1)

    x = np.array(x)
    chord0, chord1 = map(np.array, (chord0, chord1))
    canal0, canal1 = map(np.array, (canal0, canal1))

    # Optionally smooth the data by least-squares using.
    # Note, I only use here the data prior to extension
    if lstsq:
        if fit == 'mid':
            for data in (chord0, chord1, canal0, canal1):
                A = np.vstack([x[1:5], np.ones(len(x[1:5]))]).T
                m0, c0 = np.linalg.lstsq(A, data[1:5])[0]

                A = np.vstack([x[5:9], np.ones(len(x[5:9]))]).T
                m1, c1 = np.linalg.lstsq(A, data[5:9])[0]
              
                data[:] = np.r_[m0*x[:5]+c0, m1*x[5:]+c1]

        if fit == 'first':
            for data in (chord0, chord1, canal0, canal1):
                A = np.vstack([x[0:4], np.ones(len(x[0:4]))]).T
                m0, c0 = np.linalg.lstsq(A, data[0:4])[0]

                A = np.vstack([x[4:8], np.ones(len(x[4:8]))]).T
                m1, c1 = np.linalg.lstsq(A, data[4:8])[0]
              
                data[:] = np.r_[m0*x[:4]+c0, m1*x[4:]+c1]

    assert len(x) == len(chord0) == len(canal0)
    assert len(x) == len(chord1) == len(canal1)
       
    data = {'x': x,
            'a': chord0, 'b': chord1,
            'A': canal0, 'B': canal1}

    return data

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from tapered_mesh import tapered_mesh, tapered_mesh_spline
    from tapered_mesh import tapered_mesh_ellipse
    from dolfin import Mesh, HDF5File, plot, FacetFunction
    import matplotlib.pyplot as plt
    import shutil

    # Crossection  circle/ellipse/obstructed

    for fit in ('circle', ):
        for kind, data in (('healty', healthy), ('abnormal', abnormal)):
            if fit == 'ellipse':
                data = process_series_ellipse(data=data, fit='mid', lstsq=False)
            elif fit == 'circle':
                data = process_series_rot(data=healthy, diameter='avg', fit='mid', lstsq=False)

            size = [0.2]*len(data['x'])
            SIZE = [0.4]*len(data['x'])
            mesh_params = {'size': size, 
                           'SIZE': SIZE,
                           'nsplines': 30,
                           'nsmooth': 5,
                           'nsmooth_normals': 5}

            tapered_mesh_ellipse(data=data,
                                 name=kind,
                                 mesh_params=mesh_params,
                                 nrefs=3)
