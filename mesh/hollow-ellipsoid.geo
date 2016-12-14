xs[] = {0, 1, 2, 3};
as[] = {1, 1.1, 1.1, 1.0};
bs[] = {1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2};
As[] = {2, 2.1, 2.1, 2.0};
Bs[] = {2+0.5, 2.1+0.5, 2.1+0.5, 2.1+0.2};
size = {0.6, 0.6, 0.6, 0.6};
SIZE = {0.8, 0.8, 0.8, 0.8};
n = 4;
Geometry.ExtrudeSplinePoints = 15;

//!----------------------------------------------------------------------------
// IMPLEMENTATION
//----------------------------------------------------------------------------

// Inner points
// Centers
p = newp;
For i In {0:n-1}
  c[i] = p + i;
  Point(c[i]) = {xs[i], 0, 0, size[i]};
EndFor
// One axis
p = newp;
For i In {0:n-1}
  ca[2*i] = p + 2*i;
  Point(ca[2*i]) = {xs[i], 0, as[i], size[i]};

  ca[2*i+1] = p + 2*i + 1;
  Point(ca[2*i+1]) = {xs[i], 0, -as[i], size[i]};
EndFor
// Other axis
p = newp;
For i In {0:n-1}
  cb[2*i] = p + 2*i;
  Point(cb[2*i]) = {xs[i], bs[i], 0, size[i]};

  cb[2*i+1] = p + 2*i + 1;
  Point(cb[2*i+1]) = {xs[i], -bs[i], 0, size[i]};
EndFor
