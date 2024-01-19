/* jacobi_dist.h - Poisson problem
 *
 * $Id: jacobi_dist.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_DIST_H
#define _JACOBI_DIST_H
#include "util.h"

int jacobi_dist(double ***u_curr, double ***u_prev, double ***f, int N, int iter_max, double tolerance);

#endif /* _JACOBI_DIST_H */