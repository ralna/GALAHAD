/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-18 AT 08:40 GMT
 */

#pragma once

#include "ssids_rip.hxx"

#ifdef SINGLE
#define ldlt_nopiv_factor ldlt_nopiv_factor_sgl
#define ldlt_nopiv_solve_fwd ldlt_nopiv_solve_fwd_sgl
#define ldlt_nopiv_solve_diag ldlt_nopiv_solve_diag_sgl
#define ldlt_nopiv_solve_bwd ldlt_nopiv_solve_bwd_sgl
#else
#define ldlt_nopiv_factor ldlt_nopiv_factor_dbl
#define ldlt_nopiv_solve_fwd ldlt_nopiv_solve_fwd_dbl
#define ldlt_nopiv_solve_diag ldlt_nopiv_solve_diag_dbl
#define ldlt_nopiv_solve_bwd ldlt_nopiv_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

ipc_ ldlt_nopiv_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_* work);
void ldlt_nopiv_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_diag(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);

}}} /* namespaces spral::ssids::cpu */
