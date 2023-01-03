/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */

#ifdef SPRAL_SINGLE
#define precision_ float
#else
#define precision_ double
#endif

namespace spral { namespace ssids { namespace cpu {

void cholesky_factor(int m, int n, precision_* a, int lda, precision_ beta, 
   precision_* upd, int ldupd, int blksz, int *info);
void cholesky_solve_fwd(int m, int n, precision_ const* a, int lda, int nrhs, 
   precision_* x, int ldx);
void cholesky_solve_bwd(int m, int n, precision_ const* a, int lda, int nrhs, 
   precision_* x, int ldx);

}}} /* namespaces spral::ssids::cpu */
