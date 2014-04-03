#include "mpiconvolution.h"

namespace fftwpp {

void ImplicitConvolution2MPI::convolve(Complex **F, multiplier *pmult,
                                       unsigned int offset)
{
  unsigned int size=d.x*my;
  if(alltoall) {
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
  } else {
    backwards(F,U2,offset);

    transpose(intranspose,A,F,offset);
    transpose(intranspose,A,U2);

    subconvolution(F,pmult,offset,size+offset);
    subconvolution(U2,pmult,0,size);
    
    transpose(outtranspose,B,F,offset);
    transpose(outtranspose,B,U2);
    
    forwards(F,U2,offset);
  }
}
  
void ImplicitHConvolution2MPI::convolve(Complex **F, realmultiplier *pmult,
                                        bool symmetrize, unsigned int offset)
{
  if(d.y0 > 0) symmetrize=false;
    
  backwards(F,U2,d.y,symmetrize,offset);
    
  transpose(intranspose,A,F,offset);
  transpose(uintranspose,A,U2);
    
  unsigned int ny=my+!compact;
  subconvolution(F,pmult,offset,d.x*ny+offset,ny);
  subconvolution(U2,pmult,0,du.x*ny,ny);
    
  transpose(outtranspose,B,F,offset);
  transpose(uouttranspose,B,U2);
    
  forwards(F,U2,offset);
    
}

void ImplicitConvolution3MPI::convolve(Complex **F, multiplier *pmult,
                                       unsigned int offset) 
{
  unsigned int stride=my*d.z;
  unsigned int size=d.x*stride;
    
  if(alltoall) {
    for(unsigned int a=0; a < A; ++a) {
      Complex *f=F[a]+offset;
      Complex *u=U3[a];
      xfftpad->expand(f,u);
      if(a > 0) T->wait0();
      xfftpad->Backwards->fft(f);
      if(a > 0) T->wait2();
      T->transpose1(f,false,true);
      xfftpad->Backwards->fft(u);
      T->wait1();
      T->transpose2(u,false,true);
    }
      
    subconvolution(F,pmult,offset,size+offset,stride);
    T->wait0();
    T->wait2();
    T->transpose1(F[0]+offset,true,false);
    subconvolution(U3,pmult,0,size,stride);
    T->wait1();
    for(unsigned int b=1; b < B; ++b)
      T->transpose(F[b]+offset,true,false);
    
    for(unsigned int b=0; b < B; ++b) {
      Complex *f=F[b]+offset;
      Complex *u=U3[b];
      T->transpose1(u,true,false);
      xfftpad->Forwards->fft(f);
      T->wait1();
      xfftpad->Forwards->fft(u);
      xfftpad->reduce(f,u);
    }
  } else {
    backwards(F,U3,offset);
    
    transpose(intranspose,A,F,offset);
    transpose(intranspose,A,U3);
    
    subconvolution(F,pmult,offset,size+offset,stride);
    subconvolution(U3,pmult,0,size,stride);
  
    transpose(outtranspose,B,F,offset);
    transpose(outtranspose,B,U3);
    
    forwards(F,U3,offset);
  }
}

// Enforce 3D Hermiticity using specified (x,y > 0,z=0) and (x >= 0,y=0,z=0) data.
// u is a work array of size d.nx.
void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
                              dimensions3& d, bool compact, Complex *f,
                              unsigned int nu, Complex *u0)
{
  if(d.y == d.ny && d.z == d.nz) {
    HermitianSymmetrizeXY(mx,my,d.nz,mx-compact,my-compact,f);
    return;
  }

  MPI_Status stat;
  int rank,size;
  unsigned int extra=!compact;
  unsigned int yorigin=my-compact;
  unsigned int nx=d.nx-extra;
  unsigned int y0=d.y0;
  unsigned int dy=d.y;
  unsigned int j0=y0 == 0 ? extra : 0;
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
      int *process=new int[d.ny];
      for(int p=0; p < size; ++p) {
        unsigned int stop=indices[p].start+indices[p].n;
        for(unsigned int j=indices[p].start; j < stop; ++j)
          process[j]=p;
      }
    
      for(unsigned int j=j0; j < dy; ++j)
        d.reflect[j]=process[2*yorigin-y0-j];
      for(int p=1; p < size; ++p) {
        for(unsigned int j=indices[p].start == 0 ? extra : 0; j < indices[p].n;
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
      u[i]=conj(f[stride*(d.nx-1-i)+d.z*j]);
    int J=d.reflect[j];
    if(J != rank)
      MPI_Send(u,2*nx,MPI_DOUBLE,J,0,*d.XYplane);
    else {
      int offset=2*yorigin-y0;
      if(y0+j != yorigin) {
        unsigned int even=1+compact-(J % 2);
        for(unsigned int i=0; i < nx; ++i)
          f[stride*(i-even)+d.z*(offset-j)]=u[i];
      } else {
        unsigned int origin=stride*(mx-compact)+d.z*j;
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
        f[stride*(i+extra)+d.z*j]=u[i];
    }
  }
  
  if(nu < nx) delete[] u;
}

void ImplicitHConvolution3MPI::convolve(Complex **F, realmultiplier *pmult,
                                        bool symmetrize, unsigned int offset)
{
  backwards(F,U3,symmetrize,offset);

  if(d.y < d.ny) {
    transpose(intranspose,A,F,offset);
    transpose(uintranspose,A,U3);
  }
    
  unsigned int stride=d.ny*d.z;
  subconvolution(F,pmult,offset,d.x*stride+offset,stride);
  subconvolution(U3,pmult,0,du.x*stride,stride);
    
  if(d.y < d.ny) {
    transpose(outtranspose,B,F,offset);
    transpose(uouttranspose,B,U3);
  }
    
  forwards(F,U3,offset);
}


} // namespace fftwpp
