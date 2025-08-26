/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 */

#include "ssids_routines.h"
#include "ssids_rit.hxx"

namespace galahad { namespace ssids { namespace cpu {

void cholesky_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_ beta,
   rpc_* upd, ipc_ ldupd, ipc_ blksz, ipc_ *info);
void cholesky_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);
void cholesky_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);

}}} /* namespaces galahad::ssids::cpu */
