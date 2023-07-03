#ifndef SCALAR_H
#define SCALAR_H

#ifdef _NO_LD_MATH
    typedef double scalar;
#else
    typedef	long double	scalar;
#endif

#ifdef _NO_LD_MATH
    #define SF  "%15.5f"
    #define SFN "%15.5f\n"
    #define RSF "%40lf"
#else
    #define SF  "%15.5Lf"
    #define SFN "%15.5Lf\n"
    #define RSF "%40Lf"
#endif


#endif	/* SCALAR_H */
