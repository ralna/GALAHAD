/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#pragma once

#include "ssids_routines.h"
#include "galahad_precision.h"

namespace galahad { namespace ssids { namespace cpu {

ipc_ ldlt_nopiv_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_* work);
void ldlt_nopiv_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_diag(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);
void ldlt_nopiv_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
   rpc_ *x);

}}} /* namespaces galahad::ssids::cpu */
