// Straight cylindrical pipe with radius and height(length). The left opening
// will be marked by 2, the right one is marked by 3.

radius = 0.5;
height = 6;

size = 0.5;

Point(1) = {-height/2, 0, 0, size};
Point(2) = {-height/2, -radius, 0, size};
Point(3) = {-height/2, radius, 0, size};
Point(4) = {-height/2, 0, -radius, size};
Point(5) = {-height/2, 0, radius, size};

//+
Circle(1) = {5, 1, 2};
//+
Circle(2) = {2, 1, 4};
//+
Circle(3) = {4, 1, 3};
//+
Circle(4) = {3, 1, 5};
//+
Line Loop(5) = {1, 2, 3, 4};
//+
Plane Surface(6) = {5};
//+
Physical Surface(2) = {6};
//+
Extrude {height, 0, 0} {
  Surface{6};
}
//+
Physical Surface(3) = {28};
//+
Physical Surface(1) = {27, 15, 19, 23};
//+
Physical Volume(30) = {1};
