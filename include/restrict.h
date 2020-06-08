#ifndef RESTRICT_H
#define RESTRICT_H

#ifdef _MSC_VER
    #define restrict __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define restrict restrict
#endif

#endif  /* RESTRICT_H */
