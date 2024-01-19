/* jacobi_alloc.h - Poisson problem
 *
 * $Id: jacobi_alloc.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_ALLOC_H
#define _JACOBI_ALLOC_H
#include "util.h"

int jacobi_alloc(double ***u_curr, double ***u_prev, double ***f, int N,
           int max_iterations);
           
#endif /* _JACOBI_ALLOC_H */