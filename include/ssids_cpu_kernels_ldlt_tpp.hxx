/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#pragma once

#include "ssids_routines.h"
#include "ssids_rip.hxx"

namespace galahad { namespace ssids { namespace cpu {

ipc_ ldlt_tpp_factor(ipc_ m, ipc_ n, ipc_* perm, rpc_* a, ipc_ lda, rpc_* d,
      rpc_* ld, ipc_ ldld, bool action, rpc_ u, rpc_ small,
      ipc_ nleft=0, rpc_ *aleft=nullptr, ipc_ ldleft=0);
void ldlt_tpp_solve_fwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
      rpc_* x, ipc_ ldx);
void ldlt_tpp_solve_diag(ipc_ n, rpc_ const* d, rpc_* x);
void ldlt_tpp_solve_bwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
      rpc_* x, ipc_ ldx);

}}} /* end of namespace galahad::ssids::cpu */
