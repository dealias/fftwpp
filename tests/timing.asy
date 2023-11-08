include graph;

barfactor=10;

bool drawerrorbars=true;
drawerrorbars=false;

real[] me,e,le,he;
real[] mb,b;
real[] mo,o,lo,ho;
real[] mi,i,li,hi;
real[] mh,h,lh,hh;
real[] mp,p,lp,hp;

bool explicito=false;
string base,dir;
bool title=true;

usersetting();

if(base == "") base=getstring("base directory",".");
if(dir == "") dir=getstring("directory","timings1-T1");

bool incremental=find(dir,"I1") >= 0;
bool realConv=find(dir,"timingsr") >= 0;

size(incremental || realConv ? 370.4pt : 181.5pt,
     incremental ? 215 : 185,IgnoreAspect);

scaleT yscale;

if(incremental) {
  real log2(real x) {static real log2=log(2); return log(x)/log2;}
  real pow2(real x) {return 2^x;}

  yscale=scaleT(log2,pow2,logarithmic=true);
  scale(Linear,yscale);
} else
  yscale=Linear;

scale(incremental ? Linear : Log,yscale);

string prunelabel="$y$-pruned";

bool expl=true;

real d=1;
if(find(dir,"2-") >= 0) d=2;
if(find(dir,"3-") >= 0) d=3;

if(expl) {
  file fin=input(base+"/"+dir+"/explicit").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  me=a[0]; e=a[1];// le=a[2]; he=a[3];
}

if(incremental) {
  file fin=input(base+"/"+dir+"/explicitbest").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  mb=a[0]; b=a[1];// le=a[2]; he=a[3];
}

if(d == 1) {
  file fin=input(base+"/"+dir+"/explicito").line();
  explicito=!error(fin);
  if(explicito) {
    real[][] a=fin.dimension(0,0);
    a=transpose(sort(a));
    mo=a[0]; o=a[1];// lo=a[2]; ho=a[3];
  }
}
if(!realConv) {
  file fin=input(base+"/"+dir+"/implicit").line();
  bool implicit=!error(fin);
  if(implicit) {
    real[][] a=fin.dimension(0,0);
    a=transpose(sort(a));
    mi=a[0]; i=a[1];// li=a[2]; hi=a[3];
  }
}


file fin=input(base+"/"+dir+"/hybrid").line();
real[][] a=fin.dimension(0,0);
a=transpose(sort(a));
mh=a[0]; h=a[1];// lh=a[2]; hh=a[3];

file fin=input(base+"/"+dir+"/pruned",check=false).line();
bool pruned=!error(fin);
if(pruned) {
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  mp=a[0]; p=a[1];// lp=a[2]; hp=a[3];
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
real[] f(real[] m) {return log2/(1e-9*m*log(m));}
real[] g(real[] x) {return x^(1/d);}

if(expl) {
  // error bars:
  real[] ne=f(me);
  me=g(me);
  e *= ne;
  //he *= ne;
  //le *= ne;
  if(drawerrorbars)
    errorbars(me,e,0*me,he-e,0*me,e-le,Pen(0));
  draw(graph(me,e,e > 0),Pentype(0),Label("explicit"+(explicito ? " (IP)" : ""),Pen(0)+Lp),mark0);


  if(pruned) {
    real[] np=f(mp);
    mp=g(mp);
    p *= np;
    hp *= np;
    lp *= np;
    if(drawerrorbars)
      errorbars(mp,p,0*mp,hp-p,0*mp,p-lp,Pen(3));
    draw(graph(mp,p,p > 0),Pentype(3)+Dotted,Label(prunelabel,Pen(3)+Lp),mark3);
  }
}

if(incremental) {
  // error bars:
  real[] nb=f(mb);
  mb=g(mb);
  b *= nb;
  //he *= ne;
  //le *= ne;
  draw(graph(mb,b,b > 0),Pentype(3),Label("optimal explicit",Pen(3)+Lp),mark3);
}

if(explicito) {
  real[] no=f(mo);
  mo=g(mo);
  o *= no;
  //ho *= no;
  //lo *= no;
  if(drawerrorbars)
    errorbars(mo,o,0*mo,ho-o,0*mo,o-lo,Pen(3));
  draw(graph(mo,o,o > 0),Pentype(3),Label("explicit (OP)",Pen(3)+Lp),mark3);
}
if(!realConv) {
  file fin=input(base+"/"+dir+"/implicit").line();
  bool implicit=!error(fin);
  if(implicit) {
    real[] ni=f(mi);
    mi=g(mi);
    i *= ni;
    //hi *= ni;
    //li *= ni;
    if(drawerrorbars)
      errorbars(mi,i,0*mi,hi-i,0*mi,i-li,Pen(2));
    draw(graph(mi,i,i > 0),Pentype(2),Label("implicit",Pen(2)+Lp),mark2);
  }
}
real[] nh=f(mh);
mh=g(mh);
h *= nh;
//hh *= nh;
//lh *= nh;
if(drawerrorbars)
  errorbars(mh,h,0*mh,hh-h,0*mh,h-lh,Pen(1));
draw(graph(mh,h,h > 0),Pentype(1),Label("hybrid",Pen(1)+Lp),mark1);

if(title)
  label(dir,point(N),N);

string sd=d > 1 ? (string) d : "";
string D=d > 1 ? "^"+sd : "";

xaxis("$L$",BottomTop,LeftTicks);
yaxis("time/($"+sd+"L"+D+"\log_2 L$) (ns)",LeftRight,RightTicks("%#.1f"));

legendlinelength=0.6cm;
legendmargin=4;
attach(legend(),point(NE),15SW+1N);

real mean(real[] a){return sum(a)/a.length;};
if(expl)
  write("speedup="+(string)(mean(e)/mean(h)));

if(!settings.xasy) {
  shipout(dir);
  currentpicture.erase();
}
