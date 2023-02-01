/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#ifdef SPRAL_SINGLE
#define precision_ float
#define ldlt_nopiv_factor ldlt_nopiv_factor_sgl
#define ldlt_nopiv_solve_fwd ldlt_nopiv_solve_fwd_sgl
#define ldlt_nopiv_solve_diag ldlt_nopiv_solve_diag_sgl
#define ldlt_nopiv_solve_bwd ldlt_nopiv_solve_bwd_sgl
#else
#define precision_ double
#define ldlt_nopiv_factor ldlt_nopiv_factor_dbl
#define ldlt_nopiv_solve_fwd ldlt_nopiv_solve_fwd_dbl
#define ldlt_nopiv_solve_diag ldlt_nopiv_solve_diag_dbl
#define ldlt_nopiv_solve_bwd ldlt_nopiv_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

int ldlt_nopiv_factor(int m, int n, precision_* a, int lda, precision_* work);
void ldlt_nopiv_solve_fwd(int m, int n, precision_ const* a, int lda, 
   precision_ *x);
void ldlt_nopiv_solve_diag(int m, int n, precision_ const* a, int lda, 
   precision_ *x);
void ldlt_nopiv_solve_bwd(int m, int n, precision_ const* a, int lda, 
   precision_ *x);

}}} /* namespaces spral::ssids::cpu */
