include graph;

// usage:
// asy -f pdf timing.asy

// or:
// asy -f pdf timing.asy -u "runlegs=\"1k,2k,3k,4k\""
// to specify the legend.

size(175,200,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
//drawerrorbars=false;

string gtype=getstring("time, mflops, scaling, or speedup","mflops");

scale(Linear,Log);
if(gtype == "time" || gtype == "speedup")
  scale(Log,Linear);
if(gtype == "mflops") {
  scale(Log,Log);
  size(300,400,IgnoreAspect);
}
real[][] mi,i,li,hi;
string[] runnames;

string name;
string runs;
string runlegs;
usersetting();


bool myleg=((runlegs== "") ? false: true);
bool flag=true;
int n=-1;
int lastpos=0;
string legends[];
if(myleg) {
  string runleg;
  while(flag) {
    ++n;
    int pos=find(runlegs,",",lastpos);
    if(lastpos == -1) {runleg=""; flag=false;}
    
    runleg=substr(runlegs,lastpos,pos-lastpos);

    lastpos=pos > 0 ? pos+1 : -1;
    if(flag) legends.push(runleg);
  }
}
lastpos=0;

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

if(runs == "") runs=getstring("subdirs");
string run;
n=-1;
flag=true;
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
  if(gtype == "time")
    return 1e-9*x^d*d*log(x)/log(2);
  if(gtype == "mflops")
    return 5*1e-6*x^d*d*log(x)/log(2);
  return 1.0+0.0*x; // scaling
}

string D=d > 1 ? "^"+(string) d : "";

if(gtype == "time" || gtype == "mflops") {
  for(int p=0; p < nn; ++p) {
    marker mark1=marker(scale(0.6mm)*polygon(3+p),Draw(Pen(p)+solid));
    if(gtype == "mflops")
      i[p] = f(mi[p])/i[p];
    if(gtype == "time")
      i[p] /= f(mi[p]);
    hi[p] /= f(mi[p]);
    li[p] /= f(mi[p]);
    if(drawerrorbars && gtype == "time")
      errorbars(mi[p],i[p],0*mi[p],hi[p],0*mi[p],li[p],Pen(p));
    draw(graph(mi[p],i[p],i[p] > 0),Pentype(p),
	 Label(myleg ? legends[p] : runnames[p],Pen(p)+Lp),mark1);
  }
  
  xaxis("$N$",BottomTop,LeftTicks);
  if(gtype=="mflops")
    yaxis("``mflops\": $5N"+D+"\log_2 N"+D+"$/time (ms)",LeftRight,RightTicks);
  if(gtype=="time")
    yaxis("time/($N"+D+"\log_2 N"+D+"$) (ns)",LeftRight,RightTicks);

  //label(name+": (MPI procs)$\times{}$(threads/proc)",point(N),5N);
}

if(gtype == "speedup") {
  for(int p=1; p < nn; ++p) {
    int penp=p-1;
    marker mark1=marker(scale(0.6mm)*polygon(3+p),Draw(Pen(penp)+solid));
    for(int b=0; b < mi[p].length; ++b) {
      bool found=false;
      for(int a=0; a < mi[0].length; ++a) {
	if(mi[0][a] == mi[p][b]) {
	  i[p][b] = i[0][a]/i[p][b];
	  found=true;
	}
      }
      if(!found)
	i[p][b]=0.0;
    }

    draw(graph(mi[p],i[p],i[p] > 0),Pentype(penp),
	 Label(myleg ? legends[p] : runnames[p],Pen(penp)+Lp),mark1);
  }
  
  xaxis("$N$",BottomTop,LeftTicks);

  yaxis("relative speed",LeftRight,RightTicks);

  label(name+": speedup relative to ",point(N),7N);
  label(runnames[0],point(N),3N);
}


if(gtype == "scaling") {
  real[][] s;
  real[] M=mi[0];
  //write(M);

  // first run is the comparison case:
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

  // all other runs are compared with comparison case:
  for(int b=0; b < M.length; ++b) {
    s.push(new real[]);
    for(int p=0; p < nn; ++p) {
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

  marker mark1=marker(scale(0.6mm)*polygon(3+0),Draw(Pen(0)+solid));
  //  draw(graph(A[0],1.0+0*A[0]),invisible);
  for(int b=0; b < M.length; ++b) {
    marker mark1=marker(scale(0.6mm)*polygon(3+b),Draw(Pen(b)+solid));
    draw(graph(A[b],s[b]),Pen(b),Label("$"+(string) M[b]+"^"+(string)d+"$"),mark1);
  }

  int last=M.length-1;
  real[] linearscaling=new real[A[last].length];
  for(int i=0; i < A[last].length; ++i) {
    linearscaling[i]=2^i;
    // this is based on geometric spacing in number of procs
  }
  draw(graph(A[last],linearscaling),black+dashed,"linear");

  /*
  int plin=quotient(linearscaling.length,2);
  label("linear",Scale((A[last][plin],linearscaling[plin])),NW);
  */
  
  /*
  for(int a=0; a < A[A.length-1].length; ++a) {
    if(myleg)
      label(rotate(90)*(myleg ? legends[a] : runnames[a]),
	    (A[A.length-1][a],0),S);
  }
  label("(3,5)",Scale((3,5)));
  */
  
  yaxis("speedup",LeftRight,RightTicks);

  if(myleg)
    xaxis("Number of cores",BottomTop,LeftTicks(new string(real x) {
	  return legends[round(x)];}));
  else
    xaxis(BottomTop);

  //xaxis("Run (see below)",BottomTop,LeftTicks);
  label("Strong scaling: "+name,point(N),3N);

  //label("base: "+runnames[0],point(N),1N);
  
  
  string lrunnames=runnames[0];
  for(int b=1; b < nn; ++b) {
    lrunnames += "\newline ";
    lrunnames += runnames[b];
  }
  //label(minipage(lrunnames),point(S),10S+3W);
}



legendlinelength=0.6cm;
legendmargin=5;

attach(legend(),point(E),10E);
