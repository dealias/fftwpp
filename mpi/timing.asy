include graph;
// usage:
// asy -f pdf timing.asy

// asy -f pdf timing.asy -u "runlegs=\"1k,2k,4k,8k\""
// to specify the legend.

// asy -f pdf timing.asy -u "useN=true"
// makes the legends use N instead of m.

// asy -f pdf timing.asy -u "oldformat=true"
// uses the same pens as in dealias.pdf

// asy timings.asy -u"sscale=\"loglog\""
// forces the scale to be log-log.

// asy timings.asy -u"minm=<float>"
// plots only data with problem size at least minm.

// asy timings.asy -u"skipm=<float>"
// plots one in every skipm values.

// asy timings.asy -u"verbose=<true/false>"

size(250,300,IgnoreAspect);

barfactor=10;
bool verbose=false;

bool drawerrorbars=true;
//drawerrorbars=false;

string gtype=getstring("time, mflops, scaling, peff, or speedup","mflops");

scale(Linear,Log);
if(gtype == "time")
  scale(Log,Linear);
if(gtype == "speedup") {
  //scale(Log,Log);
  scale(Log,Linear);
}
if(gtype == "speeduplog") {
  scale(Log,Log);
  gtype="speedup";
}
if(gtype == "mflops") {
  //scale(Log,Log);
  scale(Log,Linear);
  size(300,400,IgnoreAspect);
}
if(gtype == "scaling") {
  //scale(Linear,Linear);
  scale(Log,Log);
}
if(gtype == "peff") {
  scale(Log,Linear);
}

real[][] mi,i,li,hi;
string[] runnames;

real minm=0;
int skipm=1;
string name;
string runs;
string runlegs;
bool useN=false;
bool oldformat=false;
string sscale="";
string datatype="";

usersetting();

if(sscale != "") {
  if(sscale == "loglog") scale(Log,Log);
  if(sscale == "loglin") scale(Log,Linear);
  if(sscale == "linlog") scale(Linear,Log);
  if(sscale == "linlin") scale(Linear,Linear);
}

string Nm = useN ? "N" : "m";

if(datatype == "") datatype=getstring("datatype", "normal");

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

string prunelabel="$y$-pruned";

bool expl=false;

if(name == "cconv3") {
  prunelabel="$xz$-pruned"; legendmargin=8;
}
if(name == "conv3")
  expl=false;

real d;
bool dknown = false;
if(name == "cconv" || name == "conv" || name == "tconv") {
  d = 1;
  dknown = true;
}
if(name == "cconv2" || name == "conv2" || name == "tconv2") {
  d = 2;
  dknown = true;
}
if(name == "cconv3" || name == "conv3") {
  d = 3;
  dknown = true;
}
if(!dknown)
  d=getreal("dimension of FFT involved",1);

real ymin=infinity, ymax=-infinity;



string[] runnames;

if(runs == "") runs=getstring("files");
string run;

string stats="median90";

triple statspm(real[] data) {
  if(stats == "median90") {
    data = sort(data);
    int N = data.length;
    real median = data[floor(N/2)];
    real p5 = median - data[floor(0.5 * N)];
    real p95 = data[floor(0.95 * N)] - median;
    return (median, p5, p95);
  }
  
  if(stats == "mean") {
    int N = data.length;
    real mean = sum(data) / N;

    real factor=N > 2 ? 2.0/(N-2.0) : 0.0;
    
    real sigmaH=0.0;
    real sigmaL=0.0;
    for(int i=0; i < N; ++i) {
      real v=data[i] - mean;
      if(v < 0)
	sigmaL += v*v;
      if(v > 0)
	sigmaH += v*v;
    }
    sigmaL=sqrt(sigmaL*factor);
    sigmaH=sqrt(sigmaH*factor);

    return (mean, sigmaL, sigmaH);
  }
  
  if(stats == "min") {
    real min = min(data);
    return (min, 0, 0);
  }
  
  return (0, 0, 0);
}

n=-1;
flag=true;
while(flag) {
  ++n;
  int pos=find(runs,",",lastpos);
  if(lastpos == -1) {run=""; flag=false;}
  run=substr(runs,lastpos,pos-lastpos);
  if(flag) {
    write(run);
    runnames.push(run);
    
    if(datatype != "raw") {
      // The input data is in the format: m mean stddevlow stdevhigh
      file fin=input(run).line();
      real[][] a=fin.dimension(0,0);
      a=transpose(a);
      mi[n]=copy(a[0]);
      i[n]=copy(a[1]);
      li[n]=copy(a[2]);
      hi[n]=copy(a[3]);
    } else {
      // The input data is in the format: m N t_0 t_1 ... t_{N-1}

      stats=getstring("stats");
      
      file fin=input(run);
      bool go=true;

      real[] nmi;
      real[] ni;
      real[] nli;
      real[] nhi;
      while(go) {
	int m = fin;
	if(m == 0) {
	  go=false;
	  break;
	}

	int N = fin;
	if(verbose) {
	  write(("m: " + (string) m), (" N: " + (string) N));
	}

	if(m >= minm) { 
	  real times[] = new real[N];
	  for(int i = 0; i < N; ++i)
	    times[i] = fin;
	  triple thestats = statspm(times);
	  nmi.push(m);
	  ni.push(thestats.x);
	  nli.push(thestats.y);
	  nhi.push(thestats.z);
	} else {
	  real dummy;
	  for(int i = 0; i < N; ++i)
	    dummy = fin;
	}
      }
      mi[n] = copy(nmi);
      i[n] = copy(ni);
      li[n] = copy(nli);
      hi[n] = copy(nhi);
    }
    
    runnames[n]=run;
    lastpos=pos > 0 ? pos+1 : -1;
  }
}


nn=n;

monoPen[0]=dashed;
monoPen[1]=solid;
colorPen[2]=deepgreen;

pen Lp=fontsize(8pt);

// Normalization function (based on computational complexity of FFT).
real mscale(real m) {
  return 1e-9 * m^d * d * log(m) / log(2);
}

// Normalization for computing "mflops" ala FFTW.
real mspeed(real m) {
  return 5e-6 *m^d * d * log(m) / log(2);
}

string D=d > 1 ? "^"+(string) d : "";

// FIXME: only for pruned case!
pen barPen(int p) {
  if(oldformat) {
    if(p == 1)
      return Pen(2);
    if(p == 2)
      return Pen(1);
  }
  return Pen(p);
}

pen linePen(int p) {
  if(oldformat) {
    if(p == 0)
      return barPen(p)+dashed;
    if(p == 1)
      return barPen(p)+Dotted;
    return barPen(p);
  }
  if(p == 2)
    return Pentype(p) + Dotted;
  return Pentype(p);
}

string base10(real x) {return "$10^{" + string(x) + "}$";}

if(gtype == "time" || gtype == "mflops") {
  for(int p=0; p < nn; ++p) {
    marker mark1=marker(scale(0.6mm)*polygon(3+p),Draw(barPen(p)+solid));
    if(gtype == "mflops") {
      for(int v = 0; v < i[p].length; ++v) 
	i[p][v] = mspeed(mi[p][v]) / i[p][v];
    }
    if(gtype == "time") {
      for(int v = 0; v < i[p].length; ++v)  {
	i[p][v] /= mscale(mi[p][v]);
	hi[p][v] /= mscale(mi[p][v]);
	li[p][v] /= mscale(mi[p][v]);
      }
    }
    
    for(int q=0; q < i[p].length; ++q) {
      real ii=i[p][q];
      ymin=min(ymin,ii);
      ymax=max(ymax,ii);
    }

    bool[] drawme = i[p] > 0;
    for(int i = 0; i < drawme.length; ++i) {
      drawme[i] = drawme[i] &&  mi[p][i] >= minm;
    }

    if(verbose) {
      write("time:");
      write(i[p]);
    }
    
    if(drawerrorbars && gtype == "time") {
      errorbars(mi[p],i[p],0*mi[p],hi[p],0*mi[p],li[p],drawme,barPen(p));
    }
    
    draw(graph(mi[p],i[p],drawme),linePen(p),
    	 Label(myleg ? legends[p] : texify(runnames[p]),Lp+linePen(p)),mark1);
  }

  xaxis("$"+Nm+"$",BottomTop,LeftTicks);
  if(d > 0) {
    if(gtype=="mflops") {
      if(true || floor(ymax) <= ceil(ymin)) {
	//if(ymax-ymin > 1) {
	yaxis("``mflops\": $5"+Nm+D+"\log_2 "+Nm+D+"$/time (ns)${}^{-1}$",
	      LeftRight, RightTicks);
      } else {
	// write the yticks as 10^{...} equally divided in log-space.
	
	int decpow=floor(log10(ymax-ymin));

	real d=ymax-ymin;
	d=pow10(ceil(log10(d)));
	
	
	real fymin=floor(ymin/d)*d;
	real fymax=ceil(ymax/d)*d;

	// make sure we catch the order of magnitude if it's present.
	int nyticks=floor((fymax-fymin)/d);

	real[] yticks;
	for(int i=0; i < nyticks; ++i)
	  yticks.push((fymin+i*(fymax-fymin)/nyticks));
	//write(yticks);
	//yaxis("``mflops\": $5"+Nm+D+"\log_2 "+Nm+D+"$/time (ns)${}^{-1}$",LeftRight,
	//    RightTicks(new string(real x) {return base10(log10(x));},yticks));
	yaxis("``mflops\": $5"+Nm+D+"\log_2 "+Nm+D+"$/time (ns)${}^{-1}$",
	      LeftRight,
	      RightTicks(defaultformat,yticks));
      }
      
    }
    if(gtype=="time")
      yaxis("time/($"+Nm+D+"\log_2 "+Nm+D+"$) (ns)",LeftRight,RightTicks);
  } else {
    if(gtype=="mflops")
      yaxis("speed: 1/time (ns)${}^{-1}$",LeftRight,RightTicks);
    if(gtype=="time")
      yaxis("time (ns)",LeftRight,RightTicks);
  }
  //label(name+": (MPI procs)$\times{}$(threads/proc)",point(N),5N);
}


if(gtype == "speedup") {
  int runples=getint("how many runs are compared at once");
  string compname="";

  bool drawyzero=false;

  real gmin = realMax;
  real gmax = -realMax;
  
  int gnum=-1;
  bool plotme;
  for(int p=0; p < nn; ++p) {

    for(int v = 0; v < i[p].length; ++v)  {
      i[p][v] /= mscale(mi[p][v]);
      //hi[p][v] /= mscale(mi[p][v]);
      //li[p][v] /= mscale(mi[p][v]);
    }

    if(p % runples == 0) {
      plotme=false;
      compname=runnames[p];
    } else {
      ++gnum;
      plotme=true;
      if(verbose) {
	write("base case: " + compname );
	write("comparison case: " + runnames[p]);
      }
    }
    
    int basep=p - (p % runples);
    if(plotme) {
      real[] speedups;
      real[] goodms;
      
      // find the matching problem sizes
      for(int b = 0; b < mi[p].length; ++b) {
	real m = mi[p][b];
	bool found=false;
	for(int a = 0; a < mi[basep].length; ++a) {
	  real mbase = mi[basep][a]; 
	  if(mbase == m) {
	    // If we have a matching problem size, determine the speedup
	    speedups.push(i[basep][a] / i[p][b]);
	    goodms.push(m);
	    found=true;
	    break;
	  }
	}
	if(!found) {
	  if((min(mi[basep]) < m) && (max(mi[basep]) > m)) {
	    if(verbose)
	      write("We can interpolate!");
	    int v = 0;
	    while(mi[basep][v] < m)
	      ++v;
	    v -= 1;
	    real m0 = mi[basep][v];
	    real m1 = mi[basep][v + 1];
	    real t0 = i[basep][v];
	    real t1 = i[basep][v + 1];
	    real t = t0 + (t1 - t0) * (m - m0) / (m1 - m0);
	    if(verbose) {
	      write("m: ", m);
	      write("m0: ", m0);
	      write("m1: ", m1);
	      write("t0: ", t0);
	      write("t1: ", t1);
	      write("t: ", t);
	    }
	    speedups.push(t / i[p][b]);
	    goodms.push(m);
	  }
	}
      }

      if(goodms.length > 0) {
	marker mark1 = marker(scale(0.6mm)*polygon(3+gnum),
			      Draw(linePen(gnum)+solid));

	if(verbose) {
	  write("m:", "      speedup:");
	  for(int v = 0; v < speedups.length; ++v) {
	    write(mi[p][v],speedups[v]);
	  }
	  write("mean: ", sum(speedups) / speedups.length);
	  write("max: ", max(speedups));
	  write("min: ", min(speedups));
	}

	gmin = min(gmin,min(speedups));
	gmax = max(gmax,max(speedups));
      
	draw(graph(goodms, speedups),
	     Pentype(gnum) + linePen(gnum),
	     Label(myleg ? legends[gnum] :
		   texify(runnames[p]) + " vs " + texify(compname),
		   linePen(gnum) + Lp), mark1);

      }
    }
  }

  if(gmin < 1.0 && gmax > 1.0)
    yequals(1,grey);
  
  xaxis("$" + Nm + "$", BottomTop, LeftTicks);
  yaxis("relative speed",LeftRight,RightTicks);
}

if(gtype == "scaling" || gtype == "peff") {
   // Find all values of problem size
  real[] thems;
  bool found=false;
  for(int a=0; a < mi.length; ++a) {
    for(int b=0; b < mi[a].length; ++b) {
      real m=mi[a][b];
      if(m >= minm) {
	found=false;
	for(int c=0; c < thems.length; ++c) {
	  if(thems[c]==m)
	    found=true;
	}
	if(!found)
	  thems.push(m);
      }
    }
  }

  // if skipm is > 1, then we kick out some values.
  if(skipm > 1) {
    real[] newems;
    for(int c = 0; c < thems.length; c += skipm)
      newems.push(thems[c]);
    thems = newems;
  }
  
  // Get the number of cores for each file.
  real[] allprocs = new real[nn];
  for(int c=0; c < allprocs.length; ++c) {
    allprocs[c] = getint("ncores" + string(c) );
    
    // This isn't dealt with by history properly:
    // allprocs[c] = getint("cores in " + runnames[c] );
  }

  // Collect the runtime for each value of m:
  real[][] runtime;
  real[][] procs;
  for(int c = 0; c < thems.length; ++c) {
    real m = thems[c];
    runtime[c] = new real[];
    procs[c] = new real[];
    for(int a = 0; a < mi.length; ++a) {
      for(int b = 0; b < mi[a].length; ++b) {
	if(m == mi[a][b]) {
	  runtime[c].push(i[a][b]);
	  procs[c].push(allprocs[a]);
	}
      }
    }
  }
  
  // Compute the speedup relative to the first data point:
  real[][] speedup;
  for(int c = 0; c < runtime.length; ++c) {
    speedup[c] = new real[];
    for(int d=0; d < runtime[c].length; ++d) {
      if(gtype == "scaling")
	speedup[c].push((runtime[c][0] / runtime[c][d]));
      if(gtype == "peff")
	speedup[c].push((runtime[c][0] / runtime[c][d] / procs[c][d]));
    }
  }
  
  // Plot the actual data:
  for(int c = 0; c < thems.length; ++c) {
    marker mark1 = marker(scale(0.6mm) * polygon(3 + c),
			  Draw(linePen(c) + solid));
    if(d == 1)
      draw(graph(procs[c], speedup[c]), linePen(c),
	   Label("$" + (string) thems[c] + "$"), mark1);
    else
      draw(graph(procs[c], speedup[c]), linePen(c),
	   Label("$" + (string) thems[c] + "^" + (string)d + "$"), mark1);
  }

  // Plot the ideal cases:
  {
    // Find the unique starting points for the scaling cases
    real[][] procstarts;
    procstarts[0] = new real[];
    procstarts[0].push(allprocs[0]);
    for(int c = 1; c < procs.length; ++c) {
      bool found = false;
      real p = procs[c][0];
      for(int i = 0; i < procstarts.length; ++i) {
	if(procstarts[i][0] == p)
	  found = true;
      }
      if(!found) {
	procstarts.push(new real[]);
	procstarts[procstarts.length - 1][0] = p;
      }
    }

    // For each unique starting point, find all scaling cases which start there
    for(int c = 0; c < procs.length; ++c) {
      for(int d = 0; d < procstarts.length; ++d) {
	if(procs[c][0] == procstarts[d][0]) {
	  for(int i = 1; i < procs[c].length; ++i) {
	    real p = procs[c][i];
	    bool found = false;
	    for(int j = 1; j < procstarts[d].length; ++j) {
	      real pp = procstarts[d][j];
	      if(p == pp)
		found = true;
	    }
	    if(!found) {
	      procstarts[d].push(p);
	    }
	  }
	}
      }
    }
    //write(procstarts);

    if(gtype == "scaling") {
      // Plot the perfect scaling for each unique starting point.
      for(int c = 0; c < procstarts.length; ++c) {
	real[] procsup;
	procsup.push(1.0);
	for(int i = 1; i < procstarts[c].length; ++i) {
	  procsup.push(procstarts[c][i] / procstarts[c][0]);
	}
	draw(graph(procstarts[c], procsup), black+dashed);
      }
    }
  }

  // y-axis points:
  real[] procsup;
  procsup.push(1.0);
  for(int i = 1; i < allprocs.length; ++i) {
    procsup.push(allprocs[i] / allprocs[0]);
  }
  if(gtype == "scaling")
    yaxis("Speedup", LeftRight, LeftTicks(DefaultFormat, procsup));
  if(gtype == "peff")
    yaxis("Efficiency", LeftRight, RightTicks);

  
  if(myleg) {
    xaxis("Number of cores", BottomTop,
	  LeftTicks(new string(real x) {return legends[round(x)];}));
  } else {
    xaxis("Number of cores", BottomTop, LeftTicks(DefaultFormat, allprocs));
    //xaxis("Number of cores", BottomTop, RightTicks(procs) );
  }

  // if(gtype == "scaling") {
  //   label("Strong scaling: "+name,point(N),3N);
  //   yequals(1,grey);
  // }
  
  //if(gtype == "peff")
  //  label("Parallel Efficiency: "+name,point(N),3N);

}

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(E),10E);
