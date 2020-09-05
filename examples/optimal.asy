import graph;

size(15cm,10cm,IgnoreAspect);

file in=input("optimal.dat").line();
real[][] a=in;
a=transpose(a);

real[] size=a[0];
real[] mean=a[1];

int[] P={2,3,5,7,11,13};
P=reverse(P);

for(int i=0; i < size.length; ++i) {
  int m=(int) size[i];
  pair z=(m,mean[i]);
  for(int p : P) {
    if(p >= 11 && (m % 121 == 0 || m % 143 == 0 || m % 169 == 0)) continue;
    if(m % p == 0) {
      while(m % p == 0) {
        m #= p;
      }
      frame f;
      fill(f,scale(p^0.6)*unitcircle,Pen(p));
      add(currentpicture,f,z);
    }
  }
  if(m > 1) {
    frame f;
    fill(f,unitcircle,black);
    add(currentpicture,f,z);
  }
}

xaxis("length",BottomTop,LeftTicks);
yaxis("time (s)",LeftRight,RightTicks);
