/* jacobi_dup.h - Poisson problem
 *
 * $Id: jacobi_dup.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_DUP_H
#define _JACOBI_DUP_H
#include "util.h"

int jacobi_dup(double ***u_curr_0, double ***u_prev_0, double ***f_0, double ***u_curr_1, double ***u_prev_1, double ***f_1, int N, int max_iterations);
           
#endif /* _JACOBI_DUP_H */