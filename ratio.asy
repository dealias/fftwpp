include graph;

size(175,200,IgnoreAspect);

barfactor=10;

scale(Log,Linear);
real[] me,e,le,he;
real[] mi,i,li,hi;
real[] mp,p,lp,hp;

string name;

usersetting();

if(name == "") name=getstring("program name");

int P=getint("numer of cores");

string dir;
string newdir="";
newdir=getstring("new dir");
string newruntype=getstring("implicit or explicit");
string olddir="";
olddir=getstring("old dir");
string oldruntype=getstring("implicit or explicit");
string prunelabel="$y$-pruned";

if(name == "conv") dir="timings1r";
if(name == "cconv") dir="timings1c";
if(name == "tconv") dir="timings1t";
if(name == "conv2") dir="timings2r";
if(name == "cconv2") dir="timings2c";
if(name == "tconv2") dir="timings2t";
if(name == "cconv3") {
  dir="timings3c"; prunelabel="$xz$-pruned"; legendmargin=8;
}
if(name == "conv3") {dir="timings3r";}

real d=1;
if(find(name,"2") >= 0) d=2;
if(find(name,"3") >= 0) d=3;
file fin=input(newdir+dir+"/"+newruntype).line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
me=a[0]; e=a[1]; le=a[2]; he=a[3];

file fin=input(olddir+dir+"/"+oldruntype).line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
mi=a[0]; i=a[1]/P; li=a[2]; hi=a[3];

file fin=input(dir+"/pruned",check=false).line();
bool pruned=!error(fin);
pruned=false;
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

real[] f(real[] x) {return 1e-9*x^d*log(x^d)/log(2);}

e /= f(me);
he /= f(me);
le /= f(me);
i /= f(mi);
hi /= f(mi);
li /= f(mi);

real ratio[];
real mratio[];
for(int a=0; a < i.length; ++a) {
  for(int b=0; b < e.length; ++b) {
    if(mi[a]==me[b]) {
      ratio.push(i[a]/e[b]);
      mratio.push(mi[a]);
    }
  }
}

write(ratio*P);
draw(graph(mratio,ratio),Pentype(0),mark0);



xaxis("$m$",BottomTop,LeftTicks);
yaxis("new/("+(string)P+"$\times$old)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(NW),17SE);
