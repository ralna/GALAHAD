/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-03 AT 14:30 GMT
 */

#include "ssids_rip.hxx"

#ifdef SPRAL_SINGLE
#define cholesky_factor cholesky_factor_sgl
#define cholesky_solve_fwd cholesky_solve_fwd_sgl
#define cholesky_solve_bwd cholesky_solve_bwd_sgl
#else
#define cholesky_factor cholesky_factor_dbl
#define cholesky_solve_fwd cholesky_solve_fwd_dbl
#define cholesky_solve_bwd cholesky_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

void cholesky_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_ beta,
   rpc_* upd, ipc_ ldupd, ipc_ blksz, ipc_ *info);
void cholesky_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);
void cholesky_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda, ipc_ nrhs,
   rpc_* x, ipc_ ldx);

}}} /* namespaces spral::ssids::cpu */
