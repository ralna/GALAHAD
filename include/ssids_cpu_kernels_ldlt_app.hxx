/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-18 AT 08:30 GMT
 */

#pragma once

#include <vector>

#include "ssids_cpu_Workspace.hxx"
#include "ssids_rip.hxx"

#ifdef SINGLE
#define ldlt_app_factor ldlt_app_factor_sgl
#define ldlt_app_solve_fwd ldlt_app_solve_fwd_sgl
#define ldlt_app_solve_diag ldlt_app_solve_diag_sgl
#define ldlt_app_solve_bwd ldlt_app_solve_bwd_sgl
#else
#define ldlt_app_factor ldlt_app_factor_dbl
#define ldlt_app_solve_fwd ldlt_app_solve_fwd_dbl
#define ldlt_app_solve_diag ldlt_app_solve_diag_dbl
#define ldlt_app_solve_bwd ldlt_app_solve_bwd_dbl
#endif

namespace spral { namespace ssids { namespace cpu {

template<typename T, typename Allocator>
ipc_ ldlt_app_factor(ipc_ m, ipc_ n, ipc_ *perm, T *a, ipc_ lda, T *d, T beta,
   T* upd, ipc_ ldupd, struct cpu_factor_options const& options,
   std::vector<Workspace>& work, Allocator const& alloc);

template <typename T>
void ldlt_app_solve_fwd(ipc_ m, ipc_ n, T const* l, ipc_ ldl, ipc_ nrhs, T* x,
   ipc_ ldx);

template <typename T>
void ldlt_app_solve_diag(ipc_ n, T const* d, ipc_ nrhs, T* x, ipc_ ldx);

template <typename T>
void ldlt_app_solve_bwd(ipc_ m, ipc_ n, T const* l, ipc_ ldl, ipc_ nrhs, T* x,
   ipc_ ldx);

}}} /* namespaces spral::ssids::cpu */
