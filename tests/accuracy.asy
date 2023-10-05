import graph3;
import palette;

size3(200,IgnoreAspect);

currentprojection=orthographic((-46,20,8));

file in=input("accuracy.dat").line();
real[] x=in;
real[] y=in;
real[][] z=in;

real log2(real x) {static real log2=log(2); return log(x)/log2;}
real pow2(real x) {return 2^x;}

scaleT Log2=scaleT(log2,pow2,logarithmic=true);
scale(Log2,Log2,Linear);

surface s=surface(z,x,y,linear,linear);

real[] level=uniform(min(z)*(1-sqrtEpsilon),max(z)*(1+sqrtEpsilon),4);

s.colors(palette(s.map(new real(triple v) {return find(level >= v.z);}),
                 Grayscale()));

draw(s,meshpen=thick(),nolight);

triple m=currentpicture.userMin();
triple M=currentpicture.userMax();
triple target=0.5*(m+M);

xaxis3(Label("$L$",deepgreen),Bounds,InTicks(Label(deepgreen)));
yaxis3(Label("$m/L$",blue),Bounds,InTicks(Label(blue),beginlabel=false));
zaxis3(rotate(90)*"$\hbox{error}(L,m) \times 10^{16}$",Bounds,InTicks);

currentprojection.target=(m+M)/2;
