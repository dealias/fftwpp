include graph;

size(175,200,IgnoreAspect);

scale(Log,Log(true,true));
real[] mp,p,mu,u,mP,P;

string pname=getstring("program name");
string dir;
if(pname == "conv") dir="timings1r/error.";
else if(pname == "cconv") dir="timings1c/error.";
else abort("error test not implemented for "+pname);

file fin=input(dir+"explicit").line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
mp=a[0]; p=a[1];

file fin=input(dir+"implicit").line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
mu=a[0]; u=a[1];

guide g0=scale(0.5mm)*unitcircle;
guide g1=scale(0.6mm)*polygon(4);

marker mark0=marker(g0,Draw(Pen(0)));
marker mark1=marker(g1,Draw(Pen(1)));

pen lp=fontsize(8pt);
draw(graph(mp,p,p>0),Pen(0),Label("explicit",Pen(0)+lp),mark0);
draw(graph(mu,u,u>0),Pen(1),Label("implicit",Pen(1)+lp),mark1);

xaxis("$m$",BottomTop,LeftTicks);
yaxis("normalized error",LeftRight,RightTicks);

legendlinelength=0.5cm;
legendmargin=8;
attach(legend(),point(NW),10SE);
