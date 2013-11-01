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
      if(a > 0) T->wait0(U2[a-1]);
      xfftpad->Backwards->fft(f);
      if(a > 0) T->wait2(U2[a-1]);
      T->transpose1(f,false,true);
      xfftpad->Backwards->fft(u);
      T->wait1(f);
      T->transpose2(u,false,true);
    }
      
    subconvolution(F,pmult,offset,size+offset);
    Complex *u=U2[A-1];
    T->wait0(u);
    T->wait2(u);
    Complex *f=F[0]+offset;
    T->transpose1(f,true,false);
    subconvolution(U2,pmult,0,size);
    T->wait1(f);
    for(unsigned int b=1; b < B; ++b)
      T->transpose(F[b]+offset,true,false);
    
    for(unsigned int b=0; b < B; ++b) {
      Complex *f=F[b]+offset;
      Complex *u=U2[b];
      T->transpose1(u,true,false);
      xfftpad->Forwards->fft(f);
      T->wait1(u);
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
  
void ImplicitHConvolution2MPI::convolve(Complex **F, Complex **G, 
                                        Complex ***U, Complex **v, 
                                        Complex **w,
                                        Complex **U2, Complex **V2, 
                                        bool symmetrize,  unsigned int offset)
{
    
  Complex *u2=U2[0];
  Complex *v2=V2[0];
    
  if(d.y0 > 0) symmetrize=false;
    
  backwards(F,u2,d.y,du.n,symmetrize,offset);
  backwards(G,v2,d.y,du.n,symmetrize,offset);
    
  pretranspose(F,offset);
  pretranspose(u2);
  pretranspose(G,offset);
  pretranspose(v2);
    
  subconvolution(F,G,U,v,w,offset,d.x*my+offset);
  subconvolution(U2,V2,U,v,w,0,du.x*my);
    
  Complex *f=F[0]+offset;
  posttranspose(outtranspose,f);
  posttranspose(uouttranspose,u2);
    
  forwards(f,u2);
    
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
      if(a > 0) T->wait0(U3[a-1]);
      xfftpad->Backwards->fft(f);
      if(a > 0) T->wait2(U3[a-1]);
      T->transpose1(f,false,true);
      xfftpad->Backwards->fft(u);
      T->wait1(f);
      T->transpose2(u,false,true);
    }
      
    subconvolution(F,pmult,offset,size+offset,stride);
    Complex *u=U3[A-1];
    T->wait0(u);
    T->wait2(u);
    Complex *f=F[0]+offset;
    T->transpose1(f,true,false);
    subconvolution(U3,pmult,0,size,stride);
    T->wait1(f);
    for(unsigned int b=1; b < B; ++b)
      T->transpose(F[b]+offset,true,false);
    
    for(unsigned int b=0; b < B; ++b) {
      Complex *f=F[b]+offset;
      Complex *u=U3[b];
      T->transpose1(u,true,false);
      xfftpad->Forwards->fft(f);
      T->wait1(u);
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

// Enforce 3D Hermiticity using specified (x,y > 0,z=0) and (x >= 0,y=0,z=0)
// data.
// u is a work array of size d.nx.
void HermitianSymmetrizeXYMPI(unsigned int mx, unsigned int my,
                              dimensions3& d, Complex *f, Complex *u)
{
  if(d.y == d.ny && d.z == d.nz) {
    HermitianSymmetrizeXY(mx,my,d.nz,f);
    return;
  }

  MPI_Status stat;
  int rank,size;
  unsigned int yorigin=my-1;
  if(d.XYplane == NULL) {
    d.XYplane=new MPI_Comm;
    MPI_Comm_split(d.communicator,d.z0 == 0,0,d.XYplane);
    if(d.z0 != 0) return;
    MPI_Comm_rank(*d.XYplane,&rank);
    MPI_Comm_size(*d.XYplane,&size);
    d.reflect=new int[d.y];
    range *indices=new range[size];
    indices[rank].n=d.y;
    indices[rank].start=d.y0;
    MPI_Allgather(MPI_IN_PLACE,0,MPI_INT,indices,
                  sizeof(range)/sizeof(MPI_INT),MPI_INT,*d.XYplane);
  
    if(rank == 0) {
      int *process=new int[d.ny];
      for(int p=0; p < size; ++p) {
        unsigned int stop=indices[p].start+indices[p].n;
        for(unsigned int j=indices[p].start; j < stop; ++j)
          process[j]=p;
      }
    
      for(unsigned int j=0; j < indices[0].n; ++j)
        d.reflect[j]=process[2*yorigin-indices[0].start-j];
      for(int p=1; p < size; ++p) {
        for(unsigned int j=0; j < indices[p].n; ++j)
          MPI_Send(process+2*yorigin-indices[p].start-j,1,MPI_INT,p,j,
                   *d.XYplane);
      }
      delete [] process;
    } else {
      for(unsigned int j=0; j < d.y; ++j)
        MPI_Recv(d.reflect+j,1,MPI_INT,0,j,*d.XYplane,&stat);
    }
    delete [] indices;
  }
  if(d.z0 != 0) return;
  MPI_Comm_rank(*d.XYplane,&rank);

  unsigned int stride=d.y*d.z;
  unsigned int offset=2*yorigin-d.y0;
  unsigned int start=(yorigin > d.y0) ? yorigin-d.y0 : 0;
  for(unsigned int j=start; j < d.y; ++j) {
    for(unsigned int i=0; i < d.nx; ++i)
      u[i]=conj(f[stride*(d.nx-1-i)+d.z*j]);
    int J=d.reflect[j];
    if(J != rank)
      MPI_Send(u,2*d.nx,MPI_DOUBLE,J,0,*d.XYplane);
    else {
      if(d.y0+j != yorigin) {
        for(unsigned int i=0; i < d.nx; ++i)
          f[stride*i+d.z*(offset-j)]=u[i];
      } else {
        unsigned int origin=stride*(mx-1)+d.z*j;
        f[origin].im=0.0;
        unsigned int mxstride=mx*stride;
        for(unsigned int i=stride; i < mxstride; i += stride)
          f[origin-i]=conj(f[origin+i]);
      }
    }
  }
  for(int j=std::min((int)d.y,(int)(yorigin-d.y0))-1; j >= 0; --j) {
    int J=d.reflect[j];
    if(J != rank) {
      MPI_Recv(u,2*d.nx,MPI_DOUBLE,J,0,*d.XYplane,&stat);
      for(unsigned int i=0; i < d.nx; ++i)
        f[stride*i+d.z*j]=u[i];
    }
  }
}

void ImplicitHConvolution3MPI::convolve(Complex **F, Complex **G,
                                        Complex ***U, Complex **v, 
                                        Complex **w,
                                        Complex ***U2, Complex ***V2, 
                                        Complex **U3, Complex **V3,
                                        bool symmetrize, unsigned int offset)
{
  Complex *u3=U3[0];
  Complex *v3=V3[0];
    
  backwards(F,u3,du.n,symmetrize,offset);
  backwards(G,v3,du.n,symmetrize,offset);
    
  if(d.y < d.ny) {
    pretranspose(F,offset);
    pretranspose(u3);
    pretranspose(G,offset);
    pretranspose(v3);
  }
    
  unsigned int stride=d.ny*d.z;
  subconvolution(F,G,U,v,w,U2,V2,offset,d.x*stride+offset,stride);
  subconvolution(U3,V3,U,v,w,U2,V2,0,du.x*stride,stride);
    
  Complex *f=F[0]+offset;
  if(d.y < d.ny) {
    posttranspose(outtranspose,f);
    posttranspose(uouttranspose,u3);
  }
    
  forwards(f,u3);
}


} // namespace fftwpp
