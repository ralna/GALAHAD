/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

namespace spral { namespace ssids { namespace cpu {

int ldlt_tpp_factor(int m, int n, int* perm, double* a, int lda, double* d,
      double* ld, int ldld, bool action, double u, double small,
      int nleft=0, double *aleft=nullptr, int ldleft=0);
void ldlt_tpp_solve_fwd(int m, int n, double const* l, int ldl, int nrhs, double* x, int ldx);
void ldlt_tpp_solve_diag(int n, double const* d, double* x);
void ldlt_tpp_solve_bwd(int m, int n, double const* l, int ldl, int nrhs, double* x, int ldx);

}}} /* end of namespace spral::ssids::cpu */
