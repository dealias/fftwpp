include graph;

size(175,200,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
//drawerrorbars=false;

bool mflops=false;
mflops=(getstring("mflops (y/n)")=="y");
if(mflops)
  scale(Log,Log);
else
  scale(Log,Linear);
real[][] mi,i,li,hi;
string[] runnames;

string name;
usersetting();

if(name == "") name=getstring("program name","cconv2");
int nn;

string dir;
string prunelabel="$y$-pruned";

bool expl=false;

if(name == "conv2") dir="timings2r";
if(name == "cconv2") dir="timings2c";
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

string runs=getstring("subdirs");
string run;
int n=-1;
bool flag=true;
int lastpos;
while(flag) {
  ++n;
  int pos=find(runs,",",lastpos);
  if(lastpos == -1) {run=""; flag=false;}
  run=substr(runs,lastpos,pos-lastpos);
  if(flag) {
    //write(run);
    file fin=input(run).line();
    real[][] a=fin.dimension(0,0);
    a=transpose(a);
    mi[n]=copy(a[0]); i[n]=copy(a[1]); li[n]=copy(a[2]); hi[n]=copy(a[3]);
    runnames[n]=run;
    lastpos=pos > 0 ? pos+1 : -1;
  }
}
nn=n;

monoPen[0]=dashed;
monoPen[1]=solid;
colorPen[2]=deepgreen;

pen Lp=fontsize(8pt);

real[] f(real[] x) {
  return mflops ? 5*1e-6*x^d*d*log(x)/log(2) : 1e-9*x^d*d*log(x)/log(2);
}

for(int p=0; p < nn; ++p) {
  marker mark1=marker(scale(0.6mm)*polygon(2+p),Draw(Pen(p)+solid));
  if(mflops)
    i[p] = f(mi[p])/i[p];
  else
    i[p] /= f(mi[p]);
  hi[p] /= f(mi[p]);
  li[p] /= f(mi[p]);
  if(drawerrorbars && !mflops)
    errorbars(mi[p],i[p],0*mi[p],hi[p],0*mi[p],li[p],Pen(p));
  draw(graph(mi[p],i[p],i[p] > 0),Pentype(p),
       Label(runnames[p],Pen(p)+Lp),mark1);
}

string D=d > 1 ? "^"+(string) d : "";

xaxis("$N$",BottomTop,LeftTicks);
if(mflops)
  yaxis("``mflops\": $5N"+D+"\log_2 N"+D+"$/time (ms)",LeftRight,RightTicks);
else
  yaxis("time/($N"+D+"\log_2 N"+D+"$) (ns)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(E),10E);


label(name+": (MPI procs)$\times{}$(threads/proc)",point(N),5N);
