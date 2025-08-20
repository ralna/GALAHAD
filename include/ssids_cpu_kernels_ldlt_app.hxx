/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#pragma once

#include <vector>

#include "ssids_routines.h"
#include "galahad_precision.h"
#include "ssids_cpu_Workspace.hxx"

namespace galahad { namespace ssids { namespace cpu {

template<typename T, typename Allocator>
ipc_ ldlt_app_factor(ipc_ m, ipc_ n, ipc_ *perm, T *a, ipc_ lda, T *d, T beta,
   T* upd, ipc_ ldupd, struct cpu_factor_control const& control,
   std::vector<Workspace>& work, Allocator const& alloc);

template <typename T>
void ldlt_app_solve_fwd(ipc_ m, ipc_ n, T const* l, ipc_ ldl, ipc_ nrhs, T* x,
   ipc_ ldx);

template <typename T>
void ldlt_app_solve_diag(ipc_ n, T const* d, ipc_ nrhs, T* x, ipc_ ldx);

template <typename T>
void ldlt_app_solve_bwd(ipc_ m, ipc_ n, T const* l, ipc_ ldl, ipc_ nrhs, T* x,
   ipc_ ldx);

}}} /* namespaces galahad::ssids::cpu */
