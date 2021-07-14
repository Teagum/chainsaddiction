#ifndef config_h
#define config_h


#ifdef _MSC_VER
    #define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define restrict restrict
#endif


#ifdef LD_MATH
typedef	long double	scalar;
#else
typedef double scalar;
#endif


#endif  /* config_h */
