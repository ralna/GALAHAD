/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 *  \brief
 *  Additional support functions and wrappers for OpenMP.
 */
#include "omp.hxx"

#include <cstdio>

/* This file wraps the C interface for OpenMP in C++ for style/safety */
namespace spral { namespace omp {

int get_global_thread_num() {
#ifdef _OPENMP
   int nbelow = 1;
   int thread_num = 0;
   for(int level=omp_get_level(); level>0; --level) {
      thread_num += nbelow * omp_get_ancestor_thread_num(level);
      nbelow *= omp_get_team_size(level);
   }
   return thread_num;
#else
   return 0;
#endif /* _OPENMP */
}

}} /* namepsace spral::omp */
