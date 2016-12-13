xs[] = {0, 1, 2, 3};
zs[] = {1, 1.1, 1.1, 1.0};
Zs[] = {1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2};
n = 4;
size = 0.6;
Geometry.ExtrudeSplinePoints = 15;

/!/!----------------------------------------------------------------------------
// IMPLEMENTATION
// Two  curves t -> (x(t), 0, z(t)) and t -> (x(t), 0, Z(t)) are rotated around 
// x-axis to produce the domain. The second curve must majorize the first one. 
// They should not intersect. And z(t), Z(t) > 0
// Here, volume bounded by z and (z, Z) are meshed such that the interface is
// maintined.
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
// Inner can define opening right away
ll = newl;
Line Loop(ll) = {extr[3], extr1[3]};
Plane Surface(ll+1) = {ll};
// Mark inner opening boundary
Physical Surface(1) = {ll+1};
inner_loop = {ll+1, extr[1], extr1[1]};
// Mark inner interior surface
Physical Surface(3) = {extr[1], extr1[1]};

// and first outer
Extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{L};};
Extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{Extr[0]};};
// The opening for outer is really outer\inner
LL = newl;
Line Loop(LL) = {Extr[3], Extr1[3]};
Plane Surface(LL+1) = {ll, LL};
// Mark outer opening boundary
Physical Surface(2) = {LL+1};
// Mark outer outer surface
Physical Surface(4) = {Extr[1], Extr1[1]};

outer_loop = {LL+1, Extr[1], Extr1[1]};
aux_loop = {extr[1], extr1[1]};

// Rotate remaining
For i In {1:n-2}
  // Inner
  extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{l+i};};
  extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{extr[0]};};
  Physical Surface(3) += {extr[1], extr1[1]}; 
  inner_loop += {extr[1], extr1[1]};

  // Outer
  Extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{L+i};};
  Extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{Extr[0]};};
  outer_loop += {Extr[1], Extr1[1]};
  Physical Surface(4) += {Extr[1], Extr1[1]}; 
  aux_loop += {extr[1], extr1[1]};
EndFor

// Inner can define opening right away
ll = newl;
Line Loop(ll) = {extr[2], extr1[2]};
Plane Surface(ll+1) = {ll};
Physical Surface(5) = {ll+1};
inner_loop += {ll+1};

// The opening for outer is really outer\inner
LL = newl;
Line Loop(LL) = {Extr[2], Extr1[2]};
Plane Surface(LL+1) = {ll, LL};
Physical Surface(6) = {LL+1};
outer_loop += {LL+1};

// Inner Volume
s = newv;
Surface Loop(s) = inner_loop[];
Volume(s+1) = {s};
Physical Volume(1) = {s+1};

// Outer Volume
outer_loop[] += aux_loop[];
s = newv;
Surface Loop(s) = outer_loop[];
Volume(s+1) = {s};
Physical Volume(2) = {s+1};
