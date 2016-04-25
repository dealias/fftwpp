#ifndef __mpigroup_h__
#define __mpigroup_h__ 1

#include "mpitranspose.h"

namespace utils {

extern MPI_Comm Active;
void setMPIplanner();

class MPIgroup {
public:  
  int rank,size;
  MPI_Comm active;                     // active communicator 
  MPI_Comm communicator,communicator2; // 3D transpose communicators
  
  void init(const MPI_Comm& comm) {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  
  void activate(const MPI_Comm& comm) {
    MPI_Comm_split(comm,rank < size,0,&active);
  }
  
// Distribute X.
  MPIgroup(const MPI_Comm& comm, unsigned int X) {
    init(comm);
    unsigned int xblock=ceilquotient(X,size);
    size=ceilquotient(X,xblock);
    activate(comm);
    communicator=communicator2=MPI_COMM_NULL;
  }
  
// Distribute first X, then (if allowpencil=true) Y.
  MPIgroup(const MPI_Comm& comm, unsigned int X, unsigned int Y,
           bool allowPencil=true) {
    init(comm);
    unsigned int x=ceilquotient(X,size);
    unsigned int y=allowPencil ? ceilquotient(Y,size*x/X) : Y;
    size=ceilquotient(X,x)*ceilquotient(Y,y);
    
    activate(comm);
    if(rank < size) {
      int major=ceilquotient(size,X);
      int p=rank % major;
      int q=rank / major;
  
      /* Split nodes into row and columns */ 
      MPI_Comm_split(active,p,q,&communicator);
      MPI_Comm_split(active,q,p,&communicator2);
    }
  }

  ~MPIgroup(){
    int final;
    MPI_Finalized(&final);
    if(final) return;
    if(rank < size && communicator != MPI_COMM_NULL) {
      MPI_Comm_free(&communicator2);
      MPI_Comm_free(&communicator);
    }
    MPI_Comm_free(&active);
  }
};

// Class to compute the local array dimensions and storage requirements for
// distributing the Y dimension among multiple MPI processes and transposing.
// Big letters denote global dimensions; small letters denote local dimensions.
//            local matrix is X * y
// local transposed matrix is x * Y
class split {
public:
  unsigned int X,Y;     // global matrix dimensions
  unsigned int x,y;     // local matrix dimensions
  unsigned int x0,y0;   // local starting values
  unsigned int n;       // total required storage (words)
  MPI_Comm communicator;
  split() {}
  split(unsigned int X, unsigned int Y, MPI_Comm communicator)
    : X(X), Y(Y), communicator(communicator) {
    int size;
    int rank;
      
    MPI_Comm_rank(communicator,&rank);
    MPI_Comm_size(communicator,&size);
    
    localdimension xdim(X,rank,size);
    localdimension ydim(Y,rank,size);
    
    x=xdim.n;
    y=ydim.n;
    
    x0=xdim.start;
    y0=ydim.start;

    n=std::max(X*y,x*Y);
  }

  int Activate() const {
    Active=communicator;
    setMPIplanner();
    return n;
  }

  void Deactivate() const {
    Active=MPI_COMM_NULL;
  }
  
  void show() {
    std::cout << "X=" << X << "\tY=" <<Y << std::endl;
    std::cout << "x=" << x << "\ty=" <<y << std::endl;
    std::cout << "x0=" << x0 << "\ty0=" << y0 << std::endl;
    std::cout << "n=" << n << std::endl;
  }
};

// Class to compute the local array dimensions and storage requirements for
// distributing X and Y among multiple MPI processes and transposing.
//         local matrix is x * y * Z
// yz transposed matrix is x * Y * z allocated n2 words [omit for slab]
// xy transposed matrix is X * xy.y * z allocated n words
//
// If spectral=true, for convenience rename xy.y to y and xy.y0 to y0.
class split3 {
public:
  unsigned int n;             // Total storage (words) for xy transpose
  unsigned int n2;            // Total storage (words) for yz transpose
  unsigned int X,Y,Y2,Z;      // Global dimensions
  unsigned int x,y,z;         // Local dimensions
  unsigned int x0,y0,z0;      // Local offsets
  split yz,xy;
  MPI_Comm communicator;
  MPI_Comm *XYplane;          // Used by HermitianSymmetrizeXYMPI
  int *reflect;               // Used by HermitianSymmetrizeXYMPI
  split3() {}
  void init(const MPIgroup& group, bool spectral) {
    xy=split(X,Y,group.communicator);
    yz=split(Y2,Z,group.communicator2);
    x=xy.x;
    x0=xy.x0;
    if(spectral) {
      y=xy.y;
      y0=xy.y0;
    } else {
      y=yz.x;
      y0=yz.x0;
    }
    z=yz.y;
    z0=yz.y0;
    n2=yz.n;
    n=std::max(xy.n*z,x*n2);
  }
  
  split3(unsigned int X, unsigned int Y, unsigned int Z,
         const MPIgroup& group, bool spectral=false) :
    X(X), Y(Y), Y2(Y), Z(Z), communicator(group.active), XYplane(NULL) {
    init(group,spectral);
  }
    
  split3(unsigned int X, unsigned int Y, unsigned int Y2, unsigned int Z,
         const MPIgroup& group, bool spectral=false) : 
    X(X), Y(Y), Y2(Y2), Z(Z), communicator(group.active), XYplane(NULL) {
    init(group,spectral);
  }
  
  int Activate() const {
    xy.Activate();
    return n;
  }

  void Deactivate() const {
    xy.Deactivate();
  }

  ~split3() {
    if(XYplane && z0 == 0) delete [] reflect;
  }
  
  void show() {
    std::cout << "X=" << X << "\tY=" << Y << "\tZ=" << Z << std::endl;
    std::cout << "x=" << x << "\ty=" << y << "\tz=" << z << std::endl;
    std::cout << "x0=" << x0 << "\ty0=" << y0 << "\tz0=" << z0 << std::endl;
    std::cout << "xy.y=" << xy.y << "\txy.y0=" << xy.y0 << std::endl;
    std::cout << "yz.x=" << yz.x << "\tyz.x0=" << yz.x0 << std::endl;
    std::cout << "n=" << n << std::endl;
    std::cout << "n2=" << n << " Y2=" << Y2 << std::endl;
  }
};

}

#endif
