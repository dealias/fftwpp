#include "chash.h"

extern "C" {
int hash(double __complex__* a, unsigned int m)
{
  int hash=0;
  for(unsigned int i=0; i<m; ++i) {
    hash= (hash+(324723947+(int)(creal(a[i])+0.5)))^93485734985;
    hash= (hash+(324723947+(int)(cimag(a[i])+0.5)))^93485734985;
  }
  return hash;
}

int hasht2(double __complex__ *a, unsigned int mx, unsigned int my)
{
  int hash=0;
  unsigned int i;
  
    int j,pos;
    unsigned int stop=2*mx-1;
    for(i=0; i < stop; i++) {
      int ii=i+1;
      for(j=0; j < my; j++) {
	pos=ii*(my+1)+j;
	hash= (hash+(324723947+(int)(creal(a[pos])+0.5)))^93485734985;
	hash= (hash+(324723947+(int)(cimag(a[pos])+0.5)))^93485734985;
      }
    }
  return hash;
}

}
