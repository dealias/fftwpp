include graph;

size(175,200,IgnoreAspect);

barfactor=10;

scale(Log,Log);
real[] me,e,le,he;
real[] mi,i,li,hi;
real[] mp,p,lp,hp;

string pname=getstring("program name");
string dir;
string prunelabel="$y$-pruned";

if(pname == "conv") dir="timings1r";
if(pname == "cconv") dir="timings1c";
if(pname == "biconv") dir="timings1b";
if(pname == "conv2") dir="timings2r";
if(pname == "cconv2") dir="timings2c";
if(pname == "biconv2") dir="timings2b";
if(pname == "cconv3") {
  dir="timings3c"; prunelabel="$xz$-pruned"; legendmargin=8;
}
  
file fin=input(dir+"/explicit").line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
me=a[0]; e=a[1]; le=a[2]; he=a[3];

file fin=input(dir+"/implicit").line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
mi=a[0]; i=a[1]; li=a[2]; hi=a[3];

file fin=input(dir+"/pruned",check=false).line();
bool pruned=!error(fin);
if(pruned) {
  real[][] a=fin.dimension(0,0);
  a=transpose(a);
  mp=a[0]; p=a[1]; lp=a[2]; hp=a[3];
}

colorPen[2]=heavygreen;

guide g0=scale(0.5mm)*unitcircle;
guide g1=scale(0.6mm)*polygon(3);
guide g2=scale(0.6mm)*polygon(4);

marker mark0=marker(g0,Draw(Pen(0)));
marker mark1=marker(g1,Draw(Pen(1)));
marker mark2=marker(g2,Draw(Pen(2)));

pen Lp=fontsize(8pt);

 // error bars:
errorbars(me,e,0*me,he,0*me,le,Pen(0));
draw(graph(me,e,e > 0),Pen(0),Label("explicit",Pen(0)+Lp),mark0);
errorbars(mi,i,0*mi,hi,0*mi,li,Pen(1));
draw(graph(mi,i,i > 0),Pen(1),Label("implicit",Pen(1)+Lp),mark1);

if(pruned) {
  errorbars(mp,p,0*mp,hp,0*mp,lp,Pen(2));
  draw(graph(mp,p,p > 0),Pen(2),Label(prunelabel,Pen(2)+Lp),mark2);
}

// fitting information; requires running rfit under R.
real[] f;
file fin=input(dir+"/implicit.p",check=false).line();
if(!error(fin)) {
  real[][] A=fin.dimension(0,0);
  real fcurve(real m) {
    real val=A[0][0]*m*log(m) +A[1][0]*m + A[2][0]*log(m) + A[3][0];
    return val;
  }

  for(int i=0; i < mi.length; ++i)
    f[i]=fcurve(mi[i]);
  // real a=min(me), b = max(me);
  // draw(graph(fcurve,a,b),Pen(1)+dashed);
  draw(graph(mi,f,f > 0),Pen(1)+dashed);
}

xaxis("$m$",BottomTop,LeftTicks);
yaxis("time (sec)",LeftRight,RightTicks);

legendlinelength=0.5cm;
legendmargin=8;
attach(legend(),point(NW),10SE);
