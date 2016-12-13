xs[] = {0, 1, 2, 3};
zs[] = {1, 1.1, 1.1, 1.0};
Zs[] = {1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2};
size[] = {0.6, 0.6, 0.6, 0.6};
SIZE[] = {0.8, 0.8, 0.8, 0.8};

n = 4;
Geometry.ExtrudeSplinePoints = 15;

//!----------------------------------------------------------------------------
// IMPLEMENTATION
// Two  curves t -> (x(t), 0, z(t)) and t -> (x(t), 0, Z(t)) are rotated around 
// x-axis to produce the domain. The second curve must majorize the first one. 
// They should not intersect. And z(t), Z(t) > 0
// Here, only the volume bounded by (z, Z) is meshed
//----------------------------------------------------------------------------

// Interior points
p = newp;
For i In {0:n-1}
  points[i] = p+i;
  Point(points[i]) = {xs[i], 0, zs[i], size[i]};
EndFor

// Interior Lines
l = newl;
Spline(l) = {points[]};

// Exterior points
P = newp;
For i In {0:n-1}
  Points[i] = P+i;
  Point(Points[i]) = {xs[i], 0, Zs[i], SIZE[i]};
EndFor

// Exterior Lines
L = newl;
Spline(L) = {Points[]};

// Rotate inner
extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{l};};
extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{extr[0]};};

// and first outer
Extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{L};};
Extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{Extr[0]};};

// The opening for outer
LL = newl;
Line Loop(LL) = {Extr[3], Extr1[3]};
Line Loop(LL+1) = {extr[3], extr1[3]};
Plane Surface(LL+2) = {LL, LL+1};
// Mark opening
Physical Surface(1) = {LL+2};

// Mark outer surface
Physical Surface(2) = {Extr[1], Extr1[1]};

// Mark inner surface
Physical Surface(3) = {extr[1], extr1[1]};

loop = {LL+2, Extr[1], Extr1[1], extr[1], extr1[1]};

// Outflow
LL = newl;
Line Loop(LL) = {Extr[2], Extr1[2]};
Line Loop(LL+1) = {extr[2], extr1[2]};
Plane Surface(LL+2) = {LL, LL+1};
// Mark opening
Physical Surface(4) = {LL+2};

loop += {LL+2};

// Volume
s = newv;
Surface Loop(s) = loop[];
Volume(s+1) = {s};
Physical Volume(1) = {s+1};
