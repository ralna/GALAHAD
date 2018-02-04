/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <vector>

#include "ssids/cpu/Workspace.hxx"

namespace spral { namespace ssids { namespace cpu {

template<typename T, typename Allocator>
int ldlt_app_factor(int m, int n, int *perm, T *a, int lda, T *d, T beta, T* upd, int ldupd, struct cpu_factor_options const& options, std::vector<Workspace>& work, Allocator const& alloc);

template <typename T>
void ldlt_app_solve_fwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx);

template <typename T>
void ldlt_app_solve_diag(int n, T const* d, int nrhs, T* x, int ldx);

template <typename T>
void ldlt_app_solve_bwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx);

}}} /* namespaces spral::ssids::cpu */
