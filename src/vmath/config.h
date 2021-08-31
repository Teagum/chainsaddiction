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

#define VM_NAN nanl ("CA_NAN")

#define VM_SUCCESS 0
#define VM_FAILURE 1

#define VM_RETURN_SUCCESS return VM_SUCCESS
#define VM_RETURN_FAILURE return VM_FAILURE


#endif  /* config_h */
