include graph;

size(175,300,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
//drawerrorbars=false;

string gtype=getstring("speed, mflops, or scaling","mflops");

scale(Linear,Log);
if(gtype == "speed")
  scale(Log,Linear);
if(gtype == "mflops")
  scale(Log,Log);
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
    write(run);
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
  if(gtype == "speed")
    return 1e-9*x^d*d*log(x)/log(2);
  if(gtype == "mflops")
    return 5*1e-6*x^d*d*log(x)/log(2);
  return 1.0+0.0*x; // scaling
}

string D=d > 1 ? "^"+(string) d : "";

if(gtype != "scaling") {
  for(int p=0; p < nn; ++p) {
    marker mark1=marker(scale(0.6mm)*polygon(3+p),Draw(Pen(p)+solid));
    if(gtype == "mflops")
      i[p] = f(mi[p])/i[p];
    if(gtype == "speed")
      i[p] /= f(mi[p]);
    hi[p] /= f(mi[p]);
    li[p] /= f(mi[p]);
    if(drawerrorbars && gtype == "speed")
      errorbars(mi[p],i[p],0*mi[p],hi[p],0*mi[p],li[p],Pen(p));
    draw(graph(mi[p],i[p],i[p] > 0),Pentype(p),
	 Label(runnames[p],Pen(p)+Lp),mark1);
  }
  
  xaxis("$N$",BottomTop,LeftTicks);
  if(gtype=="mflops")
    yaxis("``mflops\": $5N"+D+"\log_2 N"+D+"$/time (ms)",LeftRight,RightTicks);
  if(gtype=="speed")
    yaxis("time/($N"+D+"\log_2 N"+D+"$) (ns)",LeftRight,RightTicks);

  

  label(name+": (MPI procs)$\times{}$(threads/proc)",point(N),5N);
}

if(gtype == "scaling") {
  real[][] s;
  real[] M=mi[0];
  //write(M);
  
  real[] s0;
  for(int b=0; b < M.length; ++b) {
    for(int a=0; a < mi[0].length; ++a) {
      for(int a=0; a < mi[0].length; ++a) {
	if(mi[0][a] == M[b])
	  s0[b]=i[0][a];
      }
    }
  }
  //write(s0);
  
  for(int b=0; b < M.length; ++b) {
    s.push(new real[]);
    for(int p=1; p < nn; ++p) {
      for(int a=0; a < mi[p].length; ++a) {
	if(mi[p][a] == M[b]) {
	  s[b].push(s0[b]/i[p][a]);
	}
      }
    }
  }
  //write(s);

  real[][] A;
  for(int b=0; b < M.length; ++b) {
    A.push(new real[]);
    for(int i=0; i < s[b].length; ++i)
      A[b].push(i);
  }

  for(int b=0; b < M.length; ++b) {
    marker mark1=marker(scale(0.6mm)*polygon(3+b),Draw(Pen(b)+solid));
    draw(graph(A[b],s[b]),Pen(b),Label((string) M[b]),mark1);
  }
  yaxis("speedup",LeftRight,RightTicks);
  xaxis(BottomTop,LeftTicks);
  label("strong scaling: "+name,point(N),5N);

  string lrunnames=runnames[0];
  for(int b=1; b < nn; ++b) {
    lrunnames += "\newline ";
    lrunnames += runnames[b];
  }
  label(minipage(lrunnames),point(S),10S+3W);
}

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(E),10E);
