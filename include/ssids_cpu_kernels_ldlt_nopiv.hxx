/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 10:40 GMT
 */

#pragma once

#include "ssids_routines.h"
#include "ssids_rip.hxx"

namespace spral { namespace ssids { namespace cpu {

ipc_ ldlt_nopiv_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_* work);
void ldlt_nopiv_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_diag(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);

}}} /* namespaces spral::ssids::cpu */
