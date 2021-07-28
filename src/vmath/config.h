#ifndef config_h
#define config_h


#ifdef _MSC_VER
    #define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define restrict restrict
#endif

#ifdef _NO_LD_MATH
typedef double scalar;
#else
typedef	long double	scalar;
#endif

#define CA_NAN nanl ("CA_NAN")

#endif  /* config_h */
