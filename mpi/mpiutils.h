#ifndef __mpiutils_h__
#define __mpiutils_h__ 1

#include "mpifftw++.h"

namespace fftwpp {

template<class ftype>
void show(ftype *f, unsigned int, unsigned int ny,
          unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
{
  for(unsigned int i=x0; i < x1; ++i) {
    for(unsigned int j=y0; j < y1; ++j) {
      std::cout << f[ny*i+j]  << "\t";
    }
    std::cout << std::endl;
  }
}

// Copy an MPI-distributed array into an array on rank 0.
template<class ftype>
void accumulate_split(const ftype *part, ftype *whole,
		      const unsigned int X, const unsigned int Y,
		      const unsigned int x0, const unsigned int y0,
		      const unsigned int x, const unsigned int y,
		      const unsigned int Z, const bool transposed, 
		      const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size, rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);

  if(rank == 0) {
    // First copy rank 0's part into the whole
    if(!transposed)
      copyfromblock(part,whole,x*Z,Y,Y);
    else
      copyfromblock(part,whole,X,y*Z,Y*Z);

    for(int p=1; p < size; ++p) {
      unsigned int dims[6];
      MPI_Recv(&dims,6,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int X=dims[0], Y=dims[1];
      unsigned int x0=dims[2], y0=dims[3];
      unsigned int x=dims[4], y=dims[5];
      unsigned int n=Z*(!transposed ? x*Y : X*y);
      if(n > 0) {
        ftype *C=new ftype[n];
        MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
	if(!transposed)
	  copyfromblock(C,whole+Z*x0*Y,x*Z,Y,Y);
	else
	  copyfromblock(C,whole+y0*Z,X,y*Z,Y*Z);
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={X,Y,x0,y0,x,y};
    MPI_Send(&dims,6,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=Z*(!transposed ? x*Y : X*y);
    if(n > 0)
      MPI_Send((ftype *) part,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}

template<class ftype>
void accumulate_split(const ftype *part,
		      ftype *whole,
		      const split splitx,
		      const unsigned int Z,
		      const bool transposed, 
		      const MPI_Comm& communicator)
{
  unsigned int X=splitx.X;
  unsigned int Y=splitx.Y;
  unsigned int x0=splitx.x0;
  unsigned int y0=splitx.y0;
  unsigned int x=splitx.x;
  unsigned int y=splitx.y;

  accumulate_split(part,whole,X,Y,x0,y0,x,y,Z,transposed,communicator);
}
// Copy an MPI-distributed array into an array on rank 0.
template<class ftype>
void accumulate_splityz(const ftype *part,
			ftype *whole,
			const unsigned int X,
			const unsigned int Y,
			const unsigned int Z,
			const unsigned int x0,
			const unsigned int y0,
			const unsigned int z0,
			const unsigned int x,
			const unsigned int y,
			const unsigned int z,
			const int transposed, 
			const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size, rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);

  switch(transposed) {
  case 0:
    //  X * y * z
    if(rank == 0) {
      // First copy rank 0's part into the whole
      const int count=y;
      const int stride=Z;
      const int length=z;
      // std::cout << "(x0,y0,z0): ("  << x0 << "," << y0 << "," << z0 << ")"
      // 		<< std::endl;
      // std::cout << "(x,y,z): ("  << x << "," << y << "," << z << ")"
      // 		<< std::endl;
      std::cout << "count: "  << count << std::endl;
      std::cout << "stride: "  << stride << std::endl;
      std::cout << "length: "  << length << std::endl;
      for(unsigned int i=0; i < X; ++i) {
	const int outoffset=i*Y*Z+y0*Z+z0;
	const int inoffset=i*y*z;
	// std::cout << "outoffset: "  << outoffset << std::endl;
	// std::cout << "inoffset: "  << inoffset << std::endl;
	copyfromblock(part+inoffset,whole+outoffset,
		      count,length,stride);
      }
	  //copyfromblock(part,whole,count,length,stride);
      for(int p=1; p < size; ++p) {
	unsigned int dims[9];
	MPI_Recv(&dims,9,MPI_UNSIGNED,p,0,communicator,&stat);
	unsigned int X=dims[0];
	unsigned int Y=dims[1];
	unsigned int Z=dims[2];
	//unsigned int x0=dims[3];
	unsigned int y0=dims[4];
	unsigned int z0=dims[5];
	//unsigned int x=dims[6];
	unsigned int y=dims[7];
	unsigned int z=dims[8];

	// std::cout << "(x0,y0,z0): ("  << x0 << "," << y0 << "," << z0 << ")"
	// 	  << std::endl;
	// std::cout << "(x,y,z): ("  << x << "," << y << "," << z << ")"
	// 	  << std::endl;

	unsigned int n=X*y*z;
	if(n > 0) {
	  ftype *C=new ftype[n];
	  MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
	  const int count=y;
	  const int stride=Z;
	  const int length=z;
	  // std::cout << "count: "  << count << std::endl;
	  // std::cout << "stride: "  << stride << std::endl;
	  // std::cout << "length: "  << length << std::endl;
	  for(unsigned int i=0; i < X; ++i) {
	    const int outoffset=i*Y*Z+y0*Z+z0;
	    const int inoffset=i*y*z;
	    // std::cout << "outoffset: "  << outoffset << std::endl;
	    copyfromblock(C+inoffset,whole+outoffset,count,length,stride);
	  }
	  delete [] C;
	}
      }
    } else {
      unsigned int dims[9] = {X,Y,Z,x0,y0,z0,x,y,z};
      MPI_Send(&dims,9,MPI_UNSIGNED,0,0,communicator);
      unsigned int n=X*y*z;
      if(n > 0)
	MPI_Send((ftype *) part,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
    }
    break;

  case 1:
    // is x * Y * z
    // FIXME
    break;

  case 2:
    //  is x * yz.x * Z
    // FIXME
    //accumulate_split(part, whole, X, Y, x0, y0, x, y, Z, 0, communicator);
    break;

  default:
    std::cerr << "Invalid transposed choie in accumulate_splityz"
	      << std::cout;
    exit(1);
  }
  // if(rank == 0) {
  //   // First copy rank 0's part into the whole
  //   if(!transposed)
  //     copyfromblock(part,whole,x*Z,Y,Y);
  //   else
  //     copyfromblock(part,whole,X,y*Z,Y*Z);

  //   for(int p=1; p < size; ++p) {
  //     unsigned int dims[6];
  //     MPI_Recv(&dims,6,MPI_UNSIGNED,p,0,communicator,&stat);

  //     unsigned int X=dims[0], Y=dims[1];
  //     unsigned int x0=dims[2], y0=dims[3];
  //     unsigned int x=dims[4], y=dims[5];
  //     unsigned int n=Z*(!transposed ? x*Y : X*y);
  //     if(n > 0) {
  //       ftype *C=new ftype[n];
  //       MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
  // 	if(!transposed)
  // 	  copyfromblock(C,whole+Z*x0*Y,x*Z,Y,Y);
  // 	else
  // 	  copyfromblock(C,whole+y0*Z,X,y*Z,Y*Z);
  //       delete [] C;
  //     }
  //   }
  // } else {
  //   unsigned int dims[]={X,Y,x0,y0,x,y};
  //   MPI_Send(&dims,6,MPI_UNSIGNED,0,0,communicator);
  //   unsigned int n=Z*(!transposed ? x*Y : X*y);
  //   if(n > 0)
  //     MPI_Send((ftype *) part,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  // }
}

// template<class ftype>
// void accumulate_splitxy(const ftype *part, ftype *whole,
// 			const splitxy split,
// 			const unsigned int Z,
// 			const bool transposed, 
// 			const MPI_Comm& communicator)
// {
//   // const unsigned int X=split.X;
//   // unsigned int Y=split.Y;
//   // unsigned int x0=split.x0;
//   // unsigned int y0=split.y0;
//   // unsigned int x=split.x;
//   // unsigned int y=split.y;

//   // accumulate_splitxy(part,whole,X,Y,x0,y0,x,y,Z,transposed,communicator);
// }

// output the contents of a 2D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          unsigned int x0, unsigned int y0,
          unsigned int x1, unsigned int y1, const MPI_Comm& communicator)
          
{  
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,x0,y0,x1,y1);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[6];
      MPI_Recv(&dims,6,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1];
      unsigned int x0=dims[2], y0=dims[3];
      unsigned int x1=dims[4], y1=dims[5];
      unsigned int n=nx*ny;
      std::cout << "process " << p << ":" <<  std::endl;
      if(n > 0) {
        ftype *C=new ftype[n];
        MPI_Recv(C,sizeof(ftype)*n,MPI_BYTE,p,0,communicator,&stat);
      
        show(C,nx,ny,x0,y0,x1,y1);
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={nx,ny,x0,y0,x1,y1};
    MPI_Send(&dims,6,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=nx*ny;
    if(n > 0)
      MPI_Send(f,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}
  
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,0,0,nx,ny,communicator);
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x1, unsigned int y1, unsigned int z1)
{
  for(unsigned int i=x0; i < x1; ++i) {
    for(unsigned int j=y0; j < y1; ++j) {
      for(unsigned int k=z0; k < z1; ++k) {
        std::cout << f[nz*(ny*i+j)+k]  << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// output the contents of a 3D array
template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          unsigned int x0, unsigned int y0, unsigned int z0,
          unsigned int x1, unsigned int y1, unsigned int z1,
          const MPI_Comm& communicator)
{
  MPI_Status stat;
  int size,rank;
  MPI_Comm_size(communicator,&size);
  MPI_Comm_rank(communicator,&rank);
  
  if(rank == 0) {
    std::cout << "process " << 0 << ":" <<  std::endl;
    show(f,nx,ny,nz,x0,y0,z0,x1,y1,z1);
    
    for(int p=1; p < size; p++) {
      unsigned int dims[9];
      MPI_Recv(&dims,9,MPI_UNSIGNED,p,0,communicator,&stat);

      unsigned int nx=dims[0], ny=dims[1], nz=dims[2];
      unsigned int x0=dims[3], y0=dims[4], z0=dims[5];
      unsigned int x1=dims[6], y1=dims[7], z1=dims[8];
      unsigned int n=nx*ny*nz;
      if(n > 0) {
        ftype *C=new ftype[n];
        MPI_Recv(C,n*sizeof(ftype),MPI_BYTE,p,0,communicator,&stat);
      
        std::cout << "process " << p << ":" <<  std::endl;
        show(C,nx,ny,nz,x0,y0,z0,x1,y1,z1);
        delete [] C;
      }
    }
  } else {
    unsigned int dims[]={nx,ny,nz,x0,y0,z0,x1,y1,z1};
    MPI_Send(&dims,9,MPI_UNSIGNED,0,0,communicator);
    unsigned int n=nx*ny*nz;
    if(n > 0)
      MPI_Send(f,n*sizeof(ftype),MPI_BYTE,0,0,communicator);
  }
}

template<class ftype>
void show(ftype *f, unsigned int nx, unsigned int ny, unsigned int nz,
          const MPI_Comm& communicator)
{ 
  show(f,nx,ny,nz,0,0,0,nx,ny,nz,communicator);
}

// hash-check for 2D arrays
int hash(Complex *f, unsigned int nx, unsigned int ny,
         const MPI_Comm& communicator);

// return a hash of the contents of a 3D complex array
int hash(Complex *f, unsigned int nx, unsigned int ny, unsigned int nz,
         MPI_Comm communicator);

} // namespace fftwpp

#endif
  
