xs[] = {0, 1, 2, 3, 4, 5, 6, 7};
zs[] = {1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
n = 7;
size = 0.6;
Geometry.ExtrudeSplinePoints = 15;

/!/!----------------------------------------------------------------------------
// IMPLEMENTATION
// A curve t -> (x(t), 0, z(t)) is roteted around x_axis to produce the domain
// z(t) > 0
//----------------------------------------------------------------------------

// Make points
p = newp;
For i In {0:n-1}
  P[i] = p+i;
  Point(P[i]) = {xs[i], 0, zs[i], size};
EndFor

// Lines
l = newl;
For i In {0:n-2}
  l++;
  L[i] = l;
  Line(L[i]) = {i+1, i+2};
EndFor

count = 1; // Physical surfaces
// Now rotate each curve
extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{l};};
extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{extr[0]};};

// The first suface should define the opening - physical 1
ll = newl;
Line Loop(ll+1) = {extr[2], extr1[2]};
Plane Surface(ll+2) = { ll+1 };
Physical Surface(count) = {ll+2};
loop = {ll+2, extr[1], extr1[1]};

// First rib
count++;
Physical Surface(count) = {extr[1], extr1[1]};

For i In {1:n-2}
  extr[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{l-i};};
  extr1[] = Extrude{ {0,0,0}, {1,0,0}, {0,0,0}, Pi }{ Line{extr[0]};};
  Physical Surface(count) += {extr[1], extr1[1]};
  loop += {extr[1], extr1[1]};
EndFor

// The last surface defines outlet - physical
ll = newl;
Line Loop(ll+1) = {extr[3], extr1[3]};
Plane Surface(ll+2) = { ll+1 };
count++;
Physical Surface(count) = {ll+2};
loop += {ll+2};

// Volume
s = newv;
Surface Loop(s) = loop[];
Volume(s+1) = {s};
Physical Volume(1) = {s+1};
