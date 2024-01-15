/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  This version 2023-02-01 13:50 GMT
 */

#ifdef SPRAL_SINGLE
#define precision_ float
#define cholesky_factor cholesky_factor_sgl
#define cholesky_solve_fwd cholesky_solve_fwd_sgl
#define cholesky_solve_bwd cholesky_solve_bwd_sgl
#else
#define precision_ double
#define cholesky_factor cholesky_factor_dbl
#define cholesky_solve_fwd cholesky_solve_fwd_dbl
#define cholesky_solve_bwd cholesky_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

void cholesky_factor(int m, int n, precision_* a, int lda, precision_ beta,
   precision_* upd, int ldupd, int blksz, int *info);
void cholesky_solve_fwd(int m, int n, precision_ const* a, int lda, int nrhs,
   precision_* x, int ldx);
void cholesky_solve_bwd(int m, int n, precision_ const* a, int lda, int nrhs,
   precision_* x, int ldx);

}}} /* namespaces spral::ssids::cpu */
