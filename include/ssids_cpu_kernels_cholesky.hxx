/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 10:30 GMT
 */

#include "ssids_routines.h"
#include "ssids_rip.hxx"

namespace spral { namespace ssids { namespace cpu {

void cholesky_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_ beta,
   rpc_* upd, ipc_ ldupd, ipc_ blksz, ipc_ *info);
void cholesky_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);
void cholesky_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);

}}} /* namespaces spral::ssids::cpu */
