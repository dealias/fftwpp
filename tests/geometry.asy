import graph3;

size(100,100);

currentprojection=orthographic(0.75,-1,0.5);

int x=3;
int y=3;
int z=3;

int X=6;
int Y=6;
int Z=6;

real f=1+1/X;

int c=0;
for(int k=0; k < Z; ++k) {
  for(int j=0; j < Y; ++j) {
    for(int i=0; i < X; ++i) {
      transform3 t=shift((i,j,k)*f-(0.5,0.5,0.5));
      if(i < x && j < y && k < z)
        draw(t*unitcube,green+opacity(0.7));
      else
        draw(t*unitcube,red+opacity(0.1));
      ++c;
    }
  }
}

xaxis3("$X$",0,X*f+3,blue,Arrow3);
yaxis3("$Y$",0,Y*f+3,blue,Arrow3);
zaxis3("$Z$",0,Z*f+3,blue,Arrow3);
