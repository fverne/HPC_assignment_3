/* jacobi_expand.h - Poisson problem
 *
 * $Id: jacobi_expand.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_EXPAND_H
#define _JACOBI_EXPAND_H
#include "util.h"

int jacobi_expand(
    double ***u_curr, double ***u_prev, double ***f, 
    int N, int iter_max, double tolerance);

#endif /* _JACOBI_EXPAND_H */