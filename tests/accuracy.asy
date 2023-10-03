import graph3;
import palette;

size3(200,IgnoreAspect);

file in=input("accuracy.dat").line();
real[] x=in;
real[] y=in;
real[][] z=in;

real log2(real x) {static real log2=log(2); return log(x)/log2;}
real pow2(real x) {return 2^x;}

scaleT Log2=scaleT(log2,pow2,logarithmic=true);
scale(Linear,Linear,Linear);

triple f(pair t) {
  int i=round(t.x);
  int j=round(t.y);
  return (log2(x[i]),log2(y[j]),z[i][j]);
}

surface s=surface(f,(0,0),(x.length-1,y.length-1),x.length-1,y.length-1);
real[] level=uniform(min(z)*(1-sqrtEpsilon),max(z)*(1+sqrtEpsilon),4);

s.colors(palette(s.map(new real(triple v) {return find(level >= v.z);}),
                 Grayscale()));

draw(s,meshpen=thick(),nolight);


/*
for(int i=0; i < x.length; ++i)
for(int j=0; j < y.length; ++j)
  dot(Scale((x[i],y[j],z[i][j])),Pen(i)+linewidth(1mm));
*/

triple m=currentpicture.userMin();
triple M=currentpicture.userMax();
triple target=0.5*(m+M);

xaxis3("$\log_2(L)$",Bounds,InTicks);
yaxis3("$\log_2(m/L)$",Bounds,InTicks);
zaxis3(rotate(90)*"$\hbox{error}(L,m) \times 10^{16}$",Bounds,InTicks);

currentprojection=perspective(camera=target+realmult(dir(68,45),M-m),
                              target=target);
