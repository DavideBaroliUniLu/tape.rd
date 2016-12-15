xs[] = {0, 1, 2, 3, 4};
as[] = {1, 1.1, 1.1, 1.0, 1.2};
bs[] = {1+0.5, 1.1+0.5, 1.1+0.5, 1.1+0.2, 1.4};
As[] = {2, 2.1, 2.1, 2.0, 2.0};
Bs[] = {2+0.5, 2.1+0.5, 2.1+0.5, 2.1+0.2, 2.7};
size = {0.6, 0.6, 0.6, 0.6, 0.4};
SIZE = {0.8, 0.8, 0.8, 0.8, 0.9};
n = 5;
Geometry.ExtrudeSplinePoints = 15;

//!----------------------------------------------------------------------------
// IMPLEMENTATION
//----------------------------------------------------------------------------

// Centers
p = newp;
For i In {0:n-1}
  c[i] = p + i;
  Point(c[i]) = {xs[i], 0, 0, size[i]};
EndFor

// Inner points
// One axis
p = newp;
For i In {0:n-1}
  ca[i] = p + 2*i;
  Point(ca[i]) = {xs[i], 0, as[i], size[i]};

  ac[i] = p + 2*i + 1;
  Point(ac[i]) = {xs[i], 0, -as[i], size[i]};
EndFor
// Other axis
p = newp;
For i In {0:n-1}
  cb[i] = p + 2*i;
  Point(cb[i]) = {xs[i], -bs[i], 0, size[i]};

  bc[i] = p + 2*i + 1;
  Point(bc[i]) = {xs[i], bs[i], 0, size[i]};
EndFor

// Straight connections between points
l = newl;
For i In {0:n-2}
  cas[i] = l+i;
  Line(cas[i]) = {ca[i], ca[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  acs[i] = l+i;
  Line(acs[i]) = {ac[i], ac[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  cbs[i] = l+i;
  Line(cbs[i]) = {cb[i], cb[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  bcs[i] = l+i;
  Line(bcs[i]) = {bc[i], bc[i+1]};
EndFor

// Arches
l = newl;
For i In {0:n-1}
  ur_arch[i] = l+i;
  Ellipse(ur_arch[i]) = {ca[i], c[i], ca[i], cb[i]};
EndFor

l = newl;
For i In {0:n-1}
  lr_arch[i] = l+i;
  Ellipse(lr_arch[i]) = {cb[i], c[i], cb[i], ac[i]};
EndFor

l = newl;
For i In {0:n-1}
  ll_arch[i] = l+i;
  Ellipse(ll_arch[i]) = {ac[i], c[i], ac[i], bc[i]};
EndFor

l = newl;
For i In {0:n-1}
  ul_arch[i] = l+i;
  Ellipse(ul_arch[i]) = {bc[i], c[i], bc[i], ca[i]};
EndFor
// Done with lines for inner

inner[] = {};
// Now surfaces
s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {ur_arch[i], cbs[i], -ur_arch[i+1], -cas[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  inner += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {-lr_arch[i], cbs[i], lr_arch[i+1], -acs[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  inner += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {ll_arch[i], bcs[i], -ll_arch[i+1], -acs[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  inner += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {ul_arch[i], cas[i], -ul_arch[i+1], -bcs[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  inner += {s+2*i+1};
EndFor
// Done with inner

// Outer points
// One axis
p = newp;
For i In {0:n-1}
  CA[i] = p + 2*i;
  Point(CA[i]) = {xs[i], 0, As[i], size[i]};

  AC[i] = p + 2*i + 1;
  Point(AC[i]) = {xs[i], 0, -As[i], size[i]};
EndFor
// Other axis
p = newp;
For i In {0:n-1}
  CB[i] = p + 2*i;
  Point(CB[i]) = {xs[i], -Bs[i], 0, size[i]};

  BC[i] = p + 2*i + 1;
  Point(BC[i]) = {xs[i], Bs[i], 0, size[i]};
EndFor

// Straight connections between points
l = newl;
For i In {0:n-2}
  CAS[i] = l+i;
  Line(CAS[i]) = {CA[i], CA[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  ACS[i] = l+i;
  Line(ACS[i]) = {AC[i], AC[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  CBS[i] = l+i;
  Line(CBS[i]) = {CB[i], CB[i+1]};
EndFor

l = newl;
For i In {0:n-2}
  BCS[i] = l+i;
  Line(BCS[i]) = {BC[i], BC[i+1]};
EndFor

// Arches
l = newl;
For i In {0:n-1}
  UR_ARCH[i] = l+i;
  Ellipse(UR_ARCH[i]) = {CA[i], c[i], CA[i], CB[i]};
EndFor

l = newl;
For i In {0:n-1}
  LR_ARCH[i] = l+i;
  Ellipse(LR_ARCH[i]) = {CB[i], c[i], CB[i], AC[i]};
EndFor

l = newl;
For i In {0:n-1}
  LL_ARCH[i] = l+i;
  Ellipse(LL_ARCH[i]) = {AC[i], c[i], AC[i], BC[i]};
EndFor

l = newl;
For i In {0:n-1}
  UL_ARCH[i] = l+i;
  Ellipse(UL_ARCH[i]) = {BC[i], c[i], BC[i], CA[i]};
EndFor
// Done with lines for outer

outer[] = {};
// Now surfaces
s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {UR_ARCH[i], CBS[i], -UR_ARCH[i+1], -CAS[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  outer += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {-LR_ARCH[i], CBS[i], LR_ARCH[i+1], -ACS[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  outer += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {LL_ARCH[i], BCS[i], -LL_ARCH[i+1], -ACS[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  outer += {s+2*i+1};
EndFor

s = news;
For i In {0:n-2}
  Line Loop(s+2*i) = {UL_ARCH[i], CAS[i], -UL_ARCH[i+1], -BCS[i]};
  Ruled Surface(s+2*i+1) = {s+2*i};
  outer += {s+2*i+1};
EndFor
// Done with outer surfaces
Physical Surface(2) = outer[];
Physical Surface(3) = inner[];

// Opening
s = news;
Line Loop(s) = {ur_arch[0], ul_arch[0], lr_arch[0], ll_arch[0]};
Line Loop(s+1) = {UR_ARCH[0], UL_ARCH[0], LR_ARCH[0], LL_ARCH[0]};
Plane Surface(s+2) = {s+1, s};
Physical Surface(1) = {s+2};
inner += {s+2};

// Close
s = news;
Line Loop(s) = {ur_arch[n-1], ul_arch[n-1], lr_arch[n-1], ll_arch[n-1]};
Line Loop(s+1) = {UR_ARCH[n-1], UL_ARCH[n-1], LR_ARCH[n-1], LL_ARCH[n-1]};
Plane Surface(s+2) = {s+1, s};
Physical Surface(4) = {s+2};
outer += {s+2};

inner += outer[];
v = newv;
Surface Loop(v) = inner[];
Volume(v+1) = {v};
Physical Volume(1) = {v+1};
