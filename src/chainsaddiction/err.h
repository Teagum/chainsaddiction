#ifndef ca_err_h
#define ca_err_h

#include <stdio.h>


#define Ca_SUCCESS 0
#define Ca_FAILURE 1

#define Ca_RETURN_SUCCESS return Ca_SUCCESS
#define Ca_RETURN_FAILURE return Ca_FAILURE

#define Ca_ErrMsg(msg)  \
    fprintf (stderr, "(%s:%d) %s\n", __FILE__, __LINE__, msg)


#endif  /* ca_err_h */
