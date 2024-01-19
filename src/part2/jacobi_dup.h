/* jacobi_alloc.h - Poisson problem
 *
 * $Id: jacobi_alloc.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_DUP_H
#define _JACOBI_DUP_H
#include "util.h"

int jacobi_dup(double ***u_curr, double ***u_prev, double ***f, int N,
        int max_iterations);
           
#endif /* _JACOBI_DUP_H */