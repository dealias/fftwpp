#include <math.h>

// Turning off argument checking with -DNDEBUG improves optimization.

#include "Array.h"
using namespace Array;
using namespace std;

int n=2,m=3,p=4;

typedef array1<double>::opt vector;
typedef Array1<double>::opt Vector;

using std::cout;

void f(double *x) {cout << x[0] << endl; return;}

template<class T>
void g(T x) {cout << x << endl; return;}

template<class T>
double h(typename array1<T>::opt& x) {return x[0];}

int main()
{
  array3<double> A(n,m,p);
  double sum=0.0;
        
// Sequential access:
  int size=A.Size();
  for(int i=0; i < size; i++) A(i)=i;
        
// Random access:
  for(int i=0; i < n; i++) {
                
// The following statements are equivalent, but the first one optimizes better.
    array2<double> Ai=A[i];
//    array2<double> Ai(m,p); Ai=A[i]; // This does an extra memory copy.
                
    for(int j=0; j < m; j++) {
//      array1<double> Aij=Ai[j];
// For 1D arrays: many compilers optimize array1<>::opt better than array1<>.
      vector Aij=Ai[j];
      
      for(int k=0; k < p; k++) {
// The following statements are equivalent, but the first one optimizes better.
        sum=sum+Aij[k];
//      sum=sum+A(i,j,k); // This does extra index multiplication.
      }
    }
  }
        
  cout << sum << endl;
        
  f(A); 
  g(A); 
        
  vector x;
  Allocate(x,1);
  x[0]=1.0;
  cout << h<double>(x) << endl;

  cout << endl;
        
// Arrays with offsets:
        
  const int offx=-1;
  const int offy=-1;
        
  Array1<double> B(n,offx); // B(offx)...B(n-1+offx)
  Array2<double> C(n,m,offx,offy);
  Array1<double> D(5);      // Functionally equivalent to array1<double> D(n);
        
  B=1.0;
  C=2.0;
  D=3.0;

  for(int i=offx; i < n+offx; i++) cout << B[i] << endl;
        
  cout << endl;
        
  for(int i=offx; i < n+offx; i++) {
    Vector Ci=C[i];
    for(int j=offy; j < m+offy; j++) 
      cout << Ci[j] << endl;
  }

  cout << D << endl;
  
  return 0;
}
