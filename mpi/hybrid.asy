import graph;

size(350,300,IgnoreAspect);
// sub 256 1; sub 128 2; sub 64 4; sub 32 8; sub 16 16
// sub 1024 1;sub 512 2; sub 256 4; sub 128 8; sub 64 16

int[] nodes={256,128,64,32,16};
//int[] nodes={1024,512,256,128,64};

int threads(int node) {return quotient(nodes[0],node);}

scale(true);

real[][] mi,i,li,hi;

string text[]={"fftw","hybrid"};

int lastpos=0;
string runs;
int stop=1;
string size;
for(int m=0; m < stop; ++m) {
for(int j=0; j < 2; ++j) {
  string prefix=getstring("prefix"+string(j));

  for(int node : nodes)
    runs += prefix+"/"+string(node)+"x"+string(threads(node))+"/Tout,";

  string run;
  int n=-1;
  bool flag=true;
  while(flag) {
    ++n;
    int pos=find(runs,",",lastpos);
    if(lastpos == -1) {run=""; flag=false;}
    run=substr(runs,lastpos,pos-lastpos);
    if(run == "") break;
    if(flag) {
      write(run);
      file fin=input(run).line();
      real[][] a=fin.dimension(0,0);
      a=transpose(a);
      mi[n]=copy(a[0]); i[n]=copy(a[1]); li[n]=copy(a[2]); hi[n]=copy(a[3]);
      lastpos=pos > 0 ? pos+1 : -1;
    }
  }

  stop=mi[0].length;
  real[][] it=transpose(i);
  real[][] mit=transpose(mi);
  real[][] lit=transpose(li);
  real[][] hit=transpose(hi);
  int[] x=sequence(it[m].length);
  size=string(mi[0][m]);
  draw(graph(x,it[m]),Pentype(j),text[j]+": "+math(size+"^2"),dot);
  //  errorbars(x,it[m],0*mit[m],hit[m],0*mit[m],lit[m],Pen(j));
}

xaxis(BottomTop,LeftTicks(Step=1,new string(real x) {
      int node=nodes[round(x)];
      return math(string(node)+"\times"+string(threads(node)));}));
yaxis("time (s)",LeftRight,RightTicks(4));

add(legend(),point(NE),20SW);
shipout(outprefix()+size);
erase();
}
