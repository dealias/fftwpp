/* Simple hash function for checking convolution output */

#include<complex.h>

#ifndef _CHASH_H_
#define _CHASH_H_

#ifdef  __cplusplus
namespace fftwpp { extern "C" {
#endif
    
int hash(double __complex__* a, unsigned int m);

int hasht2(double __complex__ *a, unsigned int mx, unsigned int my);

#ifdef  __cplusplus
} }
#endif

#endif
