include graph;

size(175,200,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
//drawerrorbars=false;

scale(Log,Linear);
real[] me,e,le,he;
real[] mi,i,li,hi;
real[] mp,p,lp,hp;

string name;
string base;

usersetting();

if(base == "") base=getstring("base directory",".");
if(name == "") name=getstring("program name","cconv");

string dir;
string prunelabel="$y$-pruned";

bool expl=true;

if(name == "conv") dir="timings1r";
if(name == "cconv") dir="timings1c";
if(name == "tconv") dir="timings1t";
if(name == "conv2") dir="timings2r";
if(name == "cconv2") dir="timings2c";
if(name == "tconv2") dir="timings2t";
if(name == "cconv3") {
  dir="timings3c"; prunelabel="$xz$-pruned"; legendmargin=8;
}
if(name == "conv3") {
  dir="timings3r";
  expl=false;
}
  
real d=1;
if(find(name,"2") >= 0) d=2;
if(find(name,"3") >= 0) d=3;

if(expl) {
  file fin=input(base+"/"+dir+"/explicit").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(a);
  me=a[0]; e=a[1]; le=a[2]; he=a[3];
}
  
file fin=input(base+"/"+dir+"/implicit").line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
mi=a[0]; i=a[1]; li=a[2]; hi=a[3];

file fin=input(base+"/"+dir+"/pruned",check=false).line();
bool pruned=!error(fin);
if(pruned) {
  real[][] a=fin.dimension(0,0);
  a=transpose(a);
  mp=a[0]; p=a[1]; lp=a[2]; hp=a[3];
}

monoPen[0]=dashed;
monoPen[1]=solid;
colorPen[2]=deepgreen;

guide g0=scale(0.5mm)*unitcircle;
guide g1=scale(0.6mm)*polygon(3);
guide g2=scale(0.6mm)*polygon(4);

marker mark0=marker(g0,Draw(Pen(0)+solid));
marker mark1=marker(g1,Draw(Pen(1)+solid));
marker mark2=marker(g2,Draw(Pen(2)+solid));

pen Lp=fontsize(8pt);

real[] f(real[] x) {return 1e-9*x^d*d*log(x)/log(2);}

if(expl) {
  // error bars:
  e /= f(me);
  he /= f(me);
  le /= f(me);
  if(drawerrorbars)
    errorbars(me,e,0*me,he,0*me,le,Pen(0));
  draw(graph(me,e,e > 0),Pentype(0),Label("explicit",Pen(0)+Lp),mark0);

  if(pruned) {
    p /= f(mp);
    hp /= f(mp);
    lp /= f(mp);
    if(drawerrorbars)
      errorbars(mp,p,0*mp,hp,0*mp,lp,Pen(2));
    draw(graph(mp,p,p > 0),Pentype(2)+Dotted,Label(prunelabel,Pen(2)+Lp),mark2);
  }
}

i /= f(mi);
hi /= f(mi);
li /= f(mi);
if(drawerrorbars)
  errorbars(mi,i,0*mi,hi,0*mi,li,Pen(1));
draw(graph(mi,i,i > 0),Pentype(1),Label("implicit",Pen(1)+Lp),mark1);

// fitting information; requires running rfit under R.
real[] f;
file fin=input(base+"/"+dir+"/implicit.p",check=false).line();
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

string D=d > 1 ? "^"+(string) d : "";

xaxis("$N$",BottomTop,LeftTicks);
yaxis("time/($N"+D+"\log_2 N"+D+"$) (ns)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(NW),17SE+2N);

real mean(real[] a){return sum(a)/a.length;};
if(expl) {
  write("speedup="+(string)(mean(e)/mean(i)));
}
