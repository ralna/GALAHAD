/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
namespace spral { namespace ssids { namespace cpu {

void cholesky_factor(int m, int n, double* a, int lda, double beta, double* upd, int ldupd, int blksz, int *info);
void cholesky_solve_fwd(int m, int n, double const* a, int lda, int nrhs, double* x, int ldx);
void cholesky_solve_bwd(int m, int n, double const* a, int lda, int nrhs, double* x, int ldx);

}}} /* namespaces spral::ssids::cpu */
