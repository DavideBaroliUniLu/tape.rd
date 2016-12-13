xs[] = {0, 1, 2, 3};
zs[] = {1, 1.1, 1.1, 1.0};
Zs[] = {1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2};
n = 4;
size = 0.6;
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
  Point(p+i) = {xs[i], 0, zs[i], size};
EndFor

// Interior Lines
l = newl;
For i In {0:n-2}
  Line(l+i) = {p+i, p+i+1};
EndFor

// Exterior points
P = newp;
For i In {0:n-1}
  Point(P+i) = {xs[i], 0, Zs[i], size};
EndFor

// Exterior Lines
L = newl;
For i In {0:n-2}
  Line(L+i) = {P+i, P+i+1};
EndFor

// Rotate first inner
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

// Rotate remaining
For i In {1:n-2}
  // Inner
  extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{l+i};};
  extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{extr[0]};};
  Physical Surface(3) += {extr[1], extr1[1]}; 

  // Outer
  Extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{L+i};};
  Extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{Extr[0]};};
  Physical Surface(2) += {Extr[1], Extr1[1]}; 

  loop += {extr[1], extr1[1], Extr[1], Extr1[1]};
EndFor

// The outflow boundary
LL = newl;
Line Loop(LL) = {Extr[2], Extr1[2]};
Line Loop(LL+1) = {extr[2], extr1[2]};
Plane Surface(LL+2) = {LL, LL+1};
Physical Surface(4) = {LL+2};
loop += {LL+2};

// Volume
s = newv;
Surface Loop(s) = loop[];
Volume(s+1) = {s};
Physical Volume(1) = {s+1};
