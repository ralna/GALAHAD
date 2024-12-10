/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-12 AT 11:40 GMT
 */

#pragma once

#include "ssids_rip.hxx"

#ifdef REAL_32
#define ldlt_tpp_factor ldlt_tpp_factor_sgl
#define ldlt_tpp_solve_fwd ldlt_tpp_solve_fwd_sgl
#define ldlt_tpp_solve_diag ldlt_tpp_solve_diag_sgl
#define ldlt_tpp_solve_bwd ldlt_tpp_solve_bwd_sgl
#elif REAL_128
#define ldlt_tpp_factor ldlt_tpp_factor_qul
#define ldlt_tpp_solve_fwd ldlt_tpp_solve_fwd_qul
#define ldlt_tpp_solve_diag ldlt_tpp_solve_diag_qul
#define ldlt_tpp_solve_bwd ldlt_tpp_solve_bwd_qul
#else
#define ldlt_tpp_factor ldlt_tpp_factor_dbl
#define ldlt_tpp_solve_fwd ldlt_tpp_solve_fwd_dbl
#define ldlt_tpp_solve_diag ldlt_tpp_solve_diag_dbl
#define ldlt_tpp_solve_bwd ldlt_tpp_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

ipc_ ldlt_tpp_factor(ipc_ m, ipc_ n, ipc_* perm, rpc_* a, ipc_ lda, rpc_* d,
      rpc_* ld, ipc_ ldld, bool action, rpc_ u, rpc_ small,
      ipc_ nleft=0, rpc_ *aleft=nullptr, ipc_ ldleft=0);
void ldlt_tpp_solve_fwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
      rpc_* x, ipc_ ldx);
void ldlt_tpp_solve_diag(ipc_ n, rpc_ const* d, rpc_* x);
void ldlt_tpp_solve_bwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
      rpc_* x, ipc_ ldx);

}}} /* end of namespace spral::ssids::cpu */
