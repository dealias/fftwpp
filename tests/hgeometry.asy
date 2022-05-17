import graph3;

size(100,100);

currentprojection=
  orthographic(camera=(-12.3269389492048,-15.1617329681931,-18.718937229056),
               up=(0.0598485484702487,0.0784484433195963,-0.102952626476146),
               target=(0,3.5527136788005e-15,3.5527136788005e-15),
               zoom=0.957038988760488,
               viewportshift=(-0.0290361296390249,0.00635035245667115));

int X=3;
int Y=3;
int Z=3;
real f=1+1/X;

int c=0;
for(int k=0; k <= Z; ++k) {
  for(int j=k == 0 ? 0 : -Y; j <= Y; ++j) {
    for(int i=k == 0 && j == 0 ? 0 : -X; i <= X; ++i) {
      transform3 t=shift((i,j,k)*f-(0.5,0.5,0.5));
      draw(t*unitcube,green+opacity(0.7));
      ++c;
    }
  }
}

xaxis3("$X$",-X*f-2,X*f+3,blue,Arrow3);
yaxis3("$Y$",-Y*f-2,Y*f+3,blue,Arrow3);
zaxis3(Label("$-Z$",position=0),-Z*f-4,Z*f+2,blue,BeginArrow3);
