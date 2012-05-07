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
if(name == "conv3") dir="timings3r"; 

real d=1;
if(find(name,"2") >= 0) d=2;
if(find(name,"3") >= 0) d=3;
real[] f(real[] x) {return 1e-9*x^d*log(x^d)/log(2);}



monoPen[0]=dashed;
monoPen[1]=solid;
colorPen[2]=deepgreen;

string[] runnames={"serial","4 cores","4 threads"};
//string[] runnames={"explicit","implicit","4 threads"};

guide g0=scale(0.5mm)*unitcircle;
guide g1=scale(0.6mm)*polygon(3);
guide g2=scale(0.6mm)*polygon(4);

marker mark0=marker(g0,Draw(Pen(0)+solid));
marker mark1=marker(g1,Draw(Pen(1)+solid));
marker mark2=marker(g2,Draw(Pen(2)+solid));
marker[] marks={mark0,mark1,mark2};

pen Lp=fontsize(8pt);

string runs=getstring("dirs");
write(runs);
string run;
bool flag=true;
int n=-1;
int lastpos;


while(flag) {
  
  ++n;
  int pos=find(runs,",",lastpos);
  if(lastpos == -1) {run=""; flag=false;}
  run=substr(runs,lastpos,pos-lastpos);
  //write(run);
  lastpos=pos > 0 ? pos+1 : -1;

  if(flag) {
    string runtype=getstring("implicit or explicit");
    file fin=input(run+dir+"/"+runtype).line();
    real[][] a=fin.dimension(0,0);
    a=transpose(a);
    // error bars:
    me=a[0]; e=a[1]; le=a[2]; he=a[3];
    e /= f(me);
    he /= f(me);
    le /= f(me);
    
    errorbars(me,e,0*me,he,0*me,le,Pen(n));
    draw(graph(me,e,e > 0),Pentype(n),Label(run,Pen(n)+Lp),marks[n]);
    //draw(graph(me,e,e > 0),Pentype(n),Label(runnames[n],Pen(n)+Lp),marks[n]);
  }
}


string D=d > 1 ? "^"+(string) d : "";

xaxis("$N$",BottomTop,LeftTicks);
yaxis("time/($N"+D+"\log_2 N"+D+"$) (ns)",LeftRight,RightTicks);

legendlinelength=0.6cm;
legendmargin=5;
attach(legend(),point(S),26S);
/*
if(d ==3)
  attach(legend(),point(E),8W+4N);
else
  attach(legend(),point(E),8W+8S);
//attach(legend());
*/
