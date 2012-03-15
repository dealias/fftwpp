include graph;

size(175,200,IgnoreAspect);

barfactor=10;

scale(Log,Linear);
real[] me,e,le,he;
real[] mi,i,li,hi;
real[] mp,p,lp,hp;

string name, dir;

usersetting();

if(name == "") name=getstring("program name");

if(name == "conv") dir="timings1r";
if(name == "cconv") dir="timings1c";
if(name == "tconv") dir="timings1t";
if(name == "conv2") dir="timings2r";
if(name == "cconv2") dir="timings2c";
if(name == "tconv2") dir="timings2t";
if(name == "cconv3") dir="timings3c"; 

real d=1;
if(find(name,"2") >= 0) d=2;
if(find(name,"3") >= 0) d=3;
real[] f(real[] x) {return 1e-9*x^d*log(x^d)/log(2);}

int N=getint("number compared");


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


for(int n=0; n< N; ++n) {
  string basedir;
  basedir=getstring("new dir");
  
  file fin=input(basedir+dir+"/implicit").line();
  real[][] a=fin.dimension(0,0);
  a=transpose(a);
  // error bars:
  me=a[0]; e=a[1]; le=a[2]; he=a[3];
  e /= f(me);
  he /= f(me);
  le /= f(me);
  
  errorbars(me,e,0*me,he,0*me,le,Pen(n));
  draw(graph(me,e,e > 0),Pentype(n),Label(basedir,Pen(n)+Lp),mark0);

}

string D=d > 1 ? "^"+(string) d : "";

xaxis("$m$",BottomTop,LeftTicks);
yaxis("time/($m"+D+"\log_2 m"+D+"$) (ns)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(E),10E);
//attach(legend());
