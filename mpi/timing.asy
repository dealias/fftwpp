include graph;

// usage:
// asy -f pdf timing.asy

// or:
// asy -f pdf timing.asy -u "runlegs=\"1k,2k,4k,8k\""
// to specify the legend.

// Note that the scaling figures assumes subsequent test double the
// number of cores.

size(250,300,IgnoreAspect);

barfactor=10;

bool drawerrorbars=true;
//drawerrorbars=false;

string gtype=getstring("time, mflops, scaling, or speedup","mflops");

scale(Linear,Log);
if(gtype == "time")
  scale(Log,Linear);
if(gtype == "speedup") {
 scale(Log,Log);
}
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

  int runples=getint("how many runs are compared at once?");

  int gnum=-1;
  bool plotme;
  for(int p=0; p < nn; ++p) {
    if(p % runples != 0) {
      ++gnum;
      plotme=true;
    } else {
      plotme=false;
    }
    
    int basep=p - (p % runples);
    if(plotme) {

      // find the matching problem sizes
      for(int b=0; b < mi[p].length; ++b) {
	bool found=false;
	for(int a=0; a < mi[basep].length; ++a) {
	  if(mi[basep][a] == mi[p][b]) {
	    // if we have a matching problem size, determine the speedup
	    i[p][b] = i[basep][a]/i[p][b];
	    found=true;
	  }
	}
	if(!found)
	  i[p][b]=0.0;
      }

      marker mark1=marker(scale(0.6mm)*polygon(3+gnum),Draw(Pen(gnum)+solid));
      draw(graph(mi[p],i[p],i[p] > 0),Pentype(gnum),
	   Label(myleg ? legends[gnum] : runnames[p],Pen(gnum)+Lp),mark1);
    }
    
  }
  
  xaxis("$N$",BottomTop,LeftTicks);

  yaxis("relative speed",LeftRight,RightTicks);

  //  label(name+": speedup relative to ",point(N),7N);
  //  label(runnames[0],point(N),3N);
}


if(gtype == "scaling") {
 
  // find all values of problem size
  real[] thems;
  bool found=false;
  for(int a=0; a < mi.length; ++a) {
    for(int b=0; b < mi[a].length; ++b) {
      real m=mi[a][b];
      found=false;
      for(int c=0; c < thems.length; ++c) {
	if(thems[c]==m)
	  found=true;
      }
      if(!found)
	thems.push(m);
    }
  }
  real[][] y;
  real[][] ym;
  real[][] x;

  for(int c=0; c < thems.length; ++c) {
    real m=thems[c];
    y[c]=new real[];
    x[c]=new real[];
    ym[c]=new real[];
    for(int a=0; a < mi.length; ++a) {
      for(int b=0; b < mi[a].length; ++b) {
	if(m == mi[a][b]) {
	  x[c].push(a);
	  y[c].push(i[a][b]);
	  ym[c].push(m);
	}
      }
    }
  }
  
  real[][] s;
  for(int c=0; c < y.length; ++c) {
    s[c]=new real[];
    for(int d=0; d < y[c].length; ++d) {
      s[c].push((y[c][0]/y[c][d]));
    }
  }
  
  bool[] drawlin=new bool[x.length];
  for(int a=0; a < x.length; ++a) {
    drawlin[a]=true;
  }
  for(int a=0; a < x.length; ++a) {
    for(int b=0; b < a; ++b) {
      if(x[a][0] == x[b][0]) {
	if(x[a].length < x[b].length) {
	  drawlin[a]=false;
	} else {
	  drawlin[b]=false;
	  drawlin[a]=true;
	}
      }
    }
  }
  
  //draw(graph(x[c],2^(x[c]-x[c][0])),black+dashed);

  bool linleg=true;
  for(int c=0; c < y.length; ++c) {
    marker mark1=marker(scale(0.6mm)*polygon(3+c),Draw(Pen(c)+solid));
    draw(graph(x[c],s[c]),Pen(c),Label("$"+(string) thems[c]+"^"+(string)d+"$"),mark1);
    if(drawlin[c]) {
      if(linleg) {
	draw(graph(x[c],2^(x[c]-x[c][0])),black+dashed);
	linleg=false;
      } else {
	draw(graph(x[c],2^(x[c]-x[c][0])),black+dashed);      }
    }
  }
  
  yaxis("speedup",LeftRight,RightTicks);

  if(myleg)
    xaxis("Number of cores",BottomTop,LeftTicks(new string(real x) {
	  return legends[round(x)];}));
  else
    xaxis(BottomTop);

  label("Strong scaling: "+name,point(N),3N);

  yequals(1,grey);
  
}



legendlinelength=0.6cm;
legendmargin=5;

attach(legend(),point(E),10E);
