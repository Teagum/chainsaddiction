#ifndef SCALAR_H
#define SCALAR_H

#ifdef LD_MATH
typedef	long double	scalar;
#else
typedef double scalar;
#endif

#ifdef LD_MATH
#define SF "%10.5Lf"
#define SFN "%10.5Lf\n"
#define RSF "%40Lf"
#else
#define SF "%10.5f"
#define SFN "%10.5f\n"
#define RSF "%40lf"
#endif

#endif	/* SCALAR_H */
