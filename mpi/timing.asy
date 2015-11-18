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


real[][] mi,i,li,hi;
string[] runnames;

string name;
string runs;
string runlegs;
bool useN=true;
bool oldformat=false;
string sscale="";

usersetting();

if(sscale != "") {
  if(sscale == "loglog") scale(Log,Log);
  if(sscale == "loglin") scale(Log,Linear);
  if(sscale == "linlog") scale(Linear,Log);
  if(sscale == "linlin") scale(Linear,Linear);
}


string Nm=useN?"N":"m";


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

real d=getreal("dimension of FFT involved",1);

real ymin=infinity, ymax=-infinity;

string[] runnames;

if(runs == "") runs=getstring("files");
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
    runnames.push(run);
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
  if(d==0) return x; // scaling
  if(gtype == "time")
    return 1e-9*x^d*d*log(x)/log(2);
  if(gtype == "mflops")
    return 5*1e-6*x^d*d*log(x)/log(2);
  return 1.0+0.0*x; // scaling
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
  return Pentype(p);
}

string base10(real x) {
  return "$10^{"+string(x)+"}$";
}
// keep track of y bounds


if(gtype == "time" || gtype == "mflops") {
  for(int p=0; p < nn; ++p) {
    marker mark1=marker(scale(0.6mm)*polygon(3+p),Draw(barPen(p)+solid));
    if(gtype == "mflops")
      i[p] = f(mi[p])/i[p];
    if(gtype == "time")
      i[p] /= f(mi[p]);
    hi[p] /= f(mi[p]);
    li[p] /= f(mi[p]);


    for(int q=0; q < i[p].length; ++q) {
      real ii=i[p][q];
      ymin=min(ymin,ii);
      ymax=max(ymax,ii);
    }

    if(drawerrorbars && gtype == "time")
      errorbars(mi[p],i[p],0*mi[p],hi[p],0*mi[p],li[p],barPen(p));
    guide the_graph=graph(mi[p],i[p]);
    
    { // get the min and max
      //path p=the_graph;
      //if(min(p).y > ymin) ymin=min(p).y;
      //if(max(p).y < ymax) ymax=max(p).y;
    }
      
    draw(the_graph,linePen(p),
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
  
  int gnum=-1;
  bool plotme;
  for(int p=0; p < nn; ++p) {
    if(p % runples != 0) {
      ++gnum;
      plotme=true;
    } else {
      plotme=false;
      compname=runnames[p];
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

      marker mark1=marker(scale(0.6mm)*polygon(3+gnum),
			  Draw(linePen(gnum)+solid));
      draw(graph(mi[p],i[p],i[p] > 0),Pentype(gnum)+linePen(gnum),
	   Label(myleg ? legends[gnum] :
		 texify(runnames[p])+" vs "+texify(compname),
		 linePen(gnum)+Lp),mark1);
    }
    
  }
  
  xaxis("$"+Nm+"$",BottomTop,LeftTicks);

  yaxis("relative speed",LeftRight,RightTicks);

  //  label(name+": speedup relative to ",point(N),7N);
  //  label(runnames[0],point(N),3N);
}


if(gtype == "scaling") {
 
  // Find all values of problem size
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
  
  // Get the number of cores for each file.
  real[] procs = new real[nn];
  for(int c=0; c < procs.length; ++c) {
    procs[c] = getint("ncores" + string(c) );

    // This isn't dealt with by history properly:
    //procs[c] = getint("cores in " + runnames[c] );
  }

  // Collect the runtimes for each value of m:  
  real[][] times; 
  for(int c = 0; c < thems.length; ++c) {
    real m = thems[c];
    times[c] = new real[];
    for(int a = 0; a < mi.length; ++a) {
      for(int b = 0; b < mi[a].length; ++b) {
	if(m == mi[a][b]) {
	  times[c].push(i[a][b]);
	}
      }
    }
  }

  // Compute the speedup relative to the first data point:
  real[][] speedup;
  for(int c = 0; c < times.length; ++c) {
    speedup[c] = new real[];
    for(int d=0; d < times[c].length; ++d) {
      speedup[c].push((times[c][0] / times[c][d]));
    }
  }

  // The ideal case:
  draw(graph(procs, procs), black+dashed);

  // The actual data:
  for(int c = 0; c < thems.length; ++c) {
    marker mark1 = marker(scale(0.6mm) * polygon(3 + c),
			  Draw(linePen(c) + solid));
    draw(graph(procs, speedup[c]), linePen(c),
	 Label("$" + (string) thems[c] + "^" + (string)d + "$"), mark1);
  }
  
  yaxis("speedup", LeftRight, RightTicks);

  if(myleg) {
    xaxis("Number of cores",BottomTop,LeftTicks(new string(real x) {
	  return legends[round(x)];}));
  } else {
    xaxis("Number of cores", BottomTop, RightTicks(procs) );
  }
    
  label("Strong scaling: "+name,point(N),3N);

  yequals(1,grey);
}

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(E),10E);
