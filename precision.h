#ifndef __precision_h__
#define __precision_h__ 1

#include <cfloat>

#ifndef DOUBLE_PRECISION
#define DOUBLE_PRECISION 1
#endif

#define DBL_STD_MAX 1.0e+308

#if(DOUBLE_PRECISION)
typedef double Real;
#define STD_MAX DBL_STD_MAX
#define REAL_MIN DBL_MIN
#define REAL_MAX DBL_MAX
#define REAL_EPSILON DBL_EPSILON
#define REAL_DIG DBL_DIG
#else
#define STD_MAX 1.0e+38
typedef float Real;
#define REAL_MIN FLT_MIN
#define REAL_MAX FLT_MAX
#define REAL_EPSILON FLT_EPSILON
#define REAL_DIG FLT_DIG
#endif

#endif
