include graph;

size(175,200,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
drawerrorbars=false;

scale(Log,Linear);
real[] me,e,le,he;
real[] mi,i,li,hi;
real[] mh,h,lh,hh;
real[] mp,p,lp,hp;

string name;
string base;

usersetting();

if(base == "") base=getstring("base directory",".");
if(name == "") name=getstring("program name","cconv");

string dir;
string prunelabel="$y$-pruned";

bool expl=true;

if(name == "conv") dir="timings1h";
if(name == "cconv") dir="timings1c";
if(name == "tconv") dir="timings1t";
if(name == "conv2") dir="timings2h";
if(name == "cconv2") dir="timings2c";
if(name == "tconv2") dir="timings2t";
if(name == "cconv3") {
  dir="timings3c"; prunelabel="$xz$-pruned"; legendmargin=8;
}
if(name == "conv3") dir="timings3h";

real d=1;
if(find(name,"2") >= 0) d=2;
if(find(name,"3") >= 0) d=3;

if(expl) {
  file fin=input(base+"/"+dir+"/explicit").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  me=a[0]; e=a[1]; le=a[2]; he=a[3];
}

file fin=input(base+"/"+dir+"/implicit").line();
bool implicit=!error(fin);
if(implicit) {
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  mi=a[0]; i=a[1]; li=a[2]; hi=a[3];
}

file fin=input(base+"/"+dir+"/hybrid").line();
real[][] a=fin.dimension(0,0);
a=transpose(sort(a));
mh=a[0]; h=a[1]; lh=a[2]; hh=a[3];

file fin=input(base+"/"+dir+"/pruned",check=false).line();
bool pruned=!error(fin);
if(pruned) {
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  mp=a[0]; p=a[1]; lp=a[2]; hp=a[3];
}

monoPen[0]=dashed;
monoPen[1]=solid;
colorPen[2]=heavygreen;

guide g0=scale(0.5mm)*unitcircle;
guide g1=scale(0.6mm)*polygon(3);
guide g2=scale(0.6mm)*polygon(4);
guide g3=scale(0.6mm)*polygon(5);

marker mark0=marker(g0,Draw(Pen(0)+solid));
marker mark1=marker(g1,Draw(Pen(1)+solid));
marker mark2=marker(g2,Draw(Pen(2)+solid));
marker mark3=marker(g3,Draw(Pen(3)+solid));

pen Lp=fontsize(8pt);

real log2=log(2);
real[] f(real[] m) {return log2/(1e-9*m^d*d*log(m));}

if(expl) {
  // error bars:
  real[] ne=f(me);
  e *= ne;
  he *= ne;
  le *= ne;
  if(drawerrorbars)
    errorbars(me,e,0*me,he-e,0*me,e-le,Pen(0));
  draw(graph(me,e,e > 0),Pentype(0),Label("explicit",Pen(0)+Lp),mark0);

  if(pruned) {
    real[] np=f(mp);
    p *= np;
    hp *= np;
    lp *= np;
    if(drawerrorbars)
      errorbars(mp,p,0*mp,hp-p,0*mp,p-lp,Pen(3));
    draw(graph(mp,p,p > 0),Pentype(3)+Dotted,Label(prunelabel,Pen(2)+Lp),mark3);
  }
}

if(implicit) {
  real[] ni=f(mi);
  i *= ni;
  hi *= ni;
  li *= ni;
  if(drawerrorbars)
    errorbars(mi,i,0*mi,hi-i,0*mi,i-li,Pen(2));
  draw(graph(mi,i,i > 0),Pentype(2),Label("implicit",Pen(2)+Lp),mark2);
}

real[] nh=f(mh);
h *= nh;
hh *= nh;
lh *= nh;
if(drawerrorbars)
  errorbars(mh,h,0*mh,hh-h,0*mh,h-lh,Pen(1));
draw(graph(mh,h,h > 0),Pentype(1),Label("hybrid",Pen(1)+Lp),mark1);

string D=d > 1 ? "^"+(string) d : "";

xaxis("$L$",BottomTop,LeftTicks);
yaxis("time/($L"+D+"\log_2 L"+D+"$) (ns)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(NW),17SE+2N);

real mean(real[] a){return sum(a)/a.length;};
if(expl)
  write("speedup="+(string)(mean(e)/mean(h)));
