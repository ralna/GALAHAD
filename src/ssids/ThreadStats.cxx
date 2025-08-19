/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 */
#include "ssids_cpu_ThreadStats.hxx"

#include <algorithm>

namespace galahad { namespace ssids { namespace cpu {

/** \brief Reduce with stats from another thread.
 *
 * This operation is designed so that thread stats can be consolidated
 * sensibly at the end of factorization.
 */
ThreadStats& ThreadStats::operator+=(ThreadStats const& other) {
   flag = (flag<0 || other.flag<0) ? std::min(flag, other.flag) // error
                                   : std::max(flag, other.flag);// warning/pass
   num_delay += other.num_delay;
   num_factor += other.num_factor;
   num_flops += other.num_flops;
   num_neg += other.num_neg;
   num_two += other.num_two;
   num_zero += other.num_zero;
   maxfront = std::max(maxfront, other.maxfront);
   maxsupernode = std::max(maxsupernode, other.maxsupernode);
   not_first_pass += other.not_first_pass;
   not_second_pass += other.not_second_pass;

   return *this;
}

}}} /* namespace galahad::ssids::cpu */
