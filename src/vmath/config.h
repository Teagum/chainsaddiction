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

#define SUCCESS 0
#define FAILURE 1

#define RETURN_SUCCESS return SUCCESS
#define RETURN_FAILURE return FAILURE


#endif  /* config_h */
