/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#ifdef SPRAL_SINGLE
#define precision_ float
#else
#define precision_ double
#endif

namespace spral { namespace ssids { namespace cpu {

int ldlt_tpp_factor(int m, int n, int* perm, precision_* a, 
      int lda, precision_* d,
      precision_* ld, int ldld, bool action, precision_ u, precision_ small,
      int nleft=0, precision_ *aleft=nullptr, int ldleft=0);
void ldlt_tpp_solve_fwd(int m, int n, precision_ const* l, int ldl, int nrhs, 
      precision_* x, int ldx);
void ldlt_tpp_solve_diag(int n, precision_ const* d, precision_* x);
void ldlt_tpp_solve_bwd(int m, int n, precision_ const* l, int ldl, int nrhs, 
      precision_* x, int ldx);

}}} /* end of namespace spral::ssids::cpu */
