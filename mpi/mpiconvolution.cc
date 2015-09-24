#include "mpiconvolution.h"

namespace fftwpp {

void ImplicitConvolution2MPI::convolve(Complex **F, multiplier *pmult,
                                       unsigned int offset)
{
  for(unsigned int a=0; a < A; ++a) {
    Complex *f=F[a]+offset;
    Complex *u=U2[a];
    xfftpad->expand(f,u);
    if(a > 0) T->wait0();
    xfftpad->Backwards->fft(f);
    if(a > 0) T->wait2();
    T->transpose1(f,false,true);
    xfftpad->Backwards->fft(u);
    T->wait1();
    T->transpose2(u,false,true);
  }
      
  unsigned int size=d.x*d.Y;
  subconvolution(F,pmult,offset,size+offset);
  T->wait0();
  T->wait2();
  T->transpose1(F[0]+offset,true,false);
  subconvolution(U2,pmult,0,size);
  T->wait1();
  for(unsigned int b=1; b < B; ++b)
    T->transpose(F[b]+offset,true,false);
    
  for(unsigned int b=0; b < B; ++b) {
    Complex *f=F[b]+offset;
    Complex *u=U2[b];
    T->transpose1(u,true,false);
    xfftpad->Forwards->fft(f);
    T->wait1();
    xfftpad->Forwards->fft(u);
    xfftpad->reduce(f,u);
  }
}
  
void ImplicitHConvolution2MPI::convolve(Complex **F, realmultiplier *pmult,
                                        bool symmetrize, unsigned int offset)
{
  if(d.y0 > 0) symmetrize=false;
    
  backwards(F,U2,d.y,symmetrize,offset);
    
  transpose(T,A,F,false,true,offset);
  transpose(U,A,U2,false,true);
    
  subconvolution(F,pmult,offset,d.x*d.Y+offset,d.Y);
  subconvolution(U2,pmult,0,du.x*du.Y,du.Y);
    
  transpose(T,B,F,true,false,offset);
  transpose(U,B,U2,true,false);
   
  forwards(F,U2,offset);
}

void ImplicitConvolution3MPI::convolve(Complex **F, multiplier *pmult,
                                       unsigned int offset) 
{
  for(unsigned int a=0; a < A; ++a) {
    Complex *f=F[a]+offset;
    Complex *u=U3[a];
    xfftpad->expand(f,u);
    if(a > 0 && d.y < d.Y) T->wait0();
    xfftpad->Backwards->fft(f);
    if(d.y < d.Y) {
      if(a > 0) T->wait2();
      T->transpose1(f,false,true);
    }
    xfftpad->Backwards->fft(u);
    if(d.y < d.Y) {
      T->wait1();
      T->transpose2(u,false,true);
    }
  }
      
  unsigned int stride=d.Y*d.z;
  unsigned int size=d.x*stride;
    
  subconvolution(F,pmult,offset,size+offset,stride);
  if(d.y < d.Y) {
    T->wait0();
    T->wait2();
    T->transpose1(F[0]+offset,true,false);
  }
  subconvolution(U3,pmult,0,size,stride);
  if(d.y < d.Y) {
    T->wait1();
    for(unsigned int b=1; b < B; ++b)
      T->transpose(F[b]+offset,true,false);
  }
    
  for(unsigned int b=0; b < B; ++b) {
    Complex *f=F[b]+offset;
    Complex *u=U3[b];
    if(d.y < d.Y)
      T->transpose1(u,true,false);
    xfftpad->Forwards->fft(f);
    if(d.y < d.Y)
      T->wait1();
    xfftpad->Forwards->fft(u);
    xfftpad->reduce(f,u);
  }
}

// Enforce 3D Hermiticity using given (x,y > 0,z=0) and (x >= 0,y=0,z=0) data.
// u0 is an optional work array of size nu=d.X-!xcompact.
void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
                              splityz& d, bool xcompact, bool ycompact,
                              Complex *f, unsigned int nu, Complex *u0)
{
  if(d.y == d.Y && d.z == d.Z) {
    HermitianSymmetrizeXY(mx,my,d.Z,mx-xcompact,my-ycompact,f);
    return;
  }

  MPI_Status stat;
  int rank,size;
  unsigned int xextra=!xcompact;
  unsigned int yextra=!ycompact;
  unsigned int yorigin=my-ycompact;
  unsigned int nx=d.X-xextra;
  unsigned int y0=d.y0;
  unsigned int dy=d.y;
  unsigned int j0=y0 == 0 ? yextra : 0;
  unsigned int start=(yorigin > y0) ? yorigin-y0 : 0;
  
  if(d.XYplane == NULL) {
    d.XYplane=new MPI_Comm;
    MPI_Comm_split(d.communicator,d.z0 == 0,0,d.XYplane);
    if(d.z0 != 0) return;
    MPI_Comm_rank(*d.XYplane,&rank);
    MPI_Comm_size(*d.XYplane,&size);
    d.reflect=new int[dy];
    range *indices=new range[size];
    indices[rank].n=dy;
    indices[rank].start=y0;
    MPI_Allgather(MPI_IN_PLACE,0,MPI_INT,indices,
                  sizeof(range)/sizeof(MPI_INT),MPI_INT,*d.XYplane);
  
    if(rank == 0) {
      int *process=new int[d.Y];
      for(int p=0; p < size; ++p) {
        unsigned int stop=indices[p].start+indices[p].n;
        for(unsigned int j=indices[p].start; j < stop; ++j)
          process[j]=p;
      }
    
      for(unsigned int j=j0; j < dy; ++j)
        d.reflect[j]=process[2*yorigin-y0-j];
      for(int p=1; p < size; ++p) {
        for(unsigned int j=indices[p].start == 0 ? yextra : 0; j < indices[p].n;
            ++j)
          MPI_Send(process+2*yorigin-indices[p].start-j,1,MPI_INT,p,j,
                   *d.XYplane);
      }
      delete [] process;
    } else {
      for(unsigned int j=0; j < dy; ++j)
        MPI_Recv(d.reflect+j,1,MPI_INT,0,j,*d.XYplane,&stat);
    }
    delete [] indices;
  }
  if(d.z0 != 0) return;
  MPI_Comm_rank(*d.XYplane,&rank);

  Complex *u=(nu < nx) ? ComplexAlign(nx+1) : u0;
  unsigned int stride=dy*d.z;
  for(unsigned int j=start; j < dy; ++j) {
    for(unsigned int i=0; i < nx; ++i)
      u[i]=conj(f[stride*(d.X-1-i)+d.z*j]);
    int J=d.reflect[j];
    if(J != rank)
      MPI_Send(u,2*nx,MPI_DOUBLE,J,0,*d.XYplane);
    else {
      int offset=2*yorigin-y0;
      if(y0+j != yorigin) {
        unsigned int even=1+ycompact-(J % 2);
        for(unsigned int i=0; i < nx; ++i)
          f[stride*(i-even)+d.z*(offset-j)]=u[i];
      } else {
        unsigned int origin=stride*(mx-xcompact)+d.z*j;
        f[origin].im=0.0;
        unsigned int mxstride=mx*stride;
        for(unsigned int i=stride; i < mxstride; i += stride)
          f[origin-i]=conj(f[origin+i]);
      }
    }
  }

  for(unsigned int j=std::min(dy,start); j-- > j0;) {
    int J=d.reflect[j];
    if(J != rank) {
      MPI_Recv(u,2*nx,MPI_DOUBLE,J,0,*d.XYplane,&stat);
      for(unsigned int i=0; i < nx; ++i)
        f[stride*(i+xextra)+d.z*j]=u[i];
    }
  }
  
  if(nu < nx) delete[] u;
}

void ImplicitHConvolution3MPI::convolve(Complex **F, realmultiplier *pmult,
                                        bool symmetrize, unsigned int offset)
{
  backwards(F,U3,symmetrize,offset);

  if(d.y < d.Y) {
    transpose(T,A,F,false,true,offset);
    transpose(U,A,U3,false,true);
  }
    
  unsigned int stride=d.Y*d.z;
  unsigned int ustride=du.Y*du.z;
  subconvolution(F,pmult,offset,d.x*stride+offset,stride);
  subconvolution(U3,pmult,0,du.x*ustride,ustride);
    
  if(d.y < d.Y) {
    transpose(T,B,F,true,false,offset);
    transpose(U,B,U3,true,false);
  }
    
  forwards(F,U3,offset);
}


} // namespace fftwpp
