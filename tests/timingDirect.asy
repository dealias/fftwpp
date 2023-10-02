include graph;

barfactor=10;

bool drawerrorbars=true;
drawerrorbars=false;

real[] me,e;
real[] mo,o;
real[] mD,D;

bool explicito=false;
string base,dir;
bool title=false;

usersetting();

//if(base == "") base=getstring("base directory",".");
//if(dir == "") dir=getstring("directory","timings1-T1");
base=".";
dir="timings1-T1I1";
bool incremental=find(dir,"I1") >= 0;

size(incremental ? 370.4pt : 181.5pt,185,IgnoreAspect);

scale(incremental ? Linear : Log,Linear);

string prunelabel="$y$-pruned";

bool expl=true;

real d=1;
if(find(dir,"2-") >= 0) d=2;
if(find(dir,"3-") >= 0) d=3;


file fin=input(base+"/"+dir+"/direct").line();
real[][] a=fin.dimension(0,0);
a=transpose(sort(a));
mD=a[0]; D=a[1];// le=a[2]; he=a[3];


if(expl) {
  file fin=input(base+"/"+dir+"/explicit").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  me=a[0]; e=a[1];// le=a[2]; he=a[3];
}

/*
file fin=input(base+"/"+dir+"/explicito").line();
explicito=!error(fin);
if(explicito) {
  real[][] a=fin.dimension(0,0);
  a=transpose(sort(a));
  mo=a[0]; o=a[1];// lo=a[2]; ho=a[3];
}
*/


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

real[] nD=f(mD);
mD=g(mD);
D *= nD;
draw(graph(mD,D,D > 0),Pentype(0),Label("direct",Pen(0)+Lp),mark0);

if(expl) {
  real[] ne=f(me);
  me=g(me);
  e *= ne;
  draw(graph(me,e,e > 0),Pentype(1),Label("explicit"+(explicito ? " (IP)" : ""),Pen(1)+Lp),mark1);

}
/*
if(explicito) {
  real[] no=f(mo);
  mo=g(mo);
  o *= no;
  draw(graph(mo,o,o > 0),Pentype(2),Label("explicit (OP)",Pen(2)+Lp),mark2);
}
*/

if(title)
  label(dir,point(N),N);

string sd=d > 1 ? (string) d : "";
string D=d > 1 ? "^"+sd : "";

xaxis("$L$",BottomTop,LeftTicks);
yaxis("time/($"+sd+"L"+D+"\log_2 L$) (ns)",LeftRight,RightTicks("%#.1f"));

legendlinelength=0.6cm;
legendmargin=4;
attach(legend(),point(NW),15SE+1N);

if(!settings.xasy) {
  shipout(dir);
  currentpicture.erase();
}
