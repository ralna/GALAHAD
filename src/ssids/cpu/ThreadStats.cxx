/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids/cpu/ThreadStats.hxx"

#include <algorithm>

namespace spral { namespace ssids { namespace cpu {

/** \brief Reduce with stats from another thread.
 *
 * This operation is designed so that thread stats can be consolidated
 * sensibly at the end of factorization.
 */
ThreadStats& ThreadStats::operator+=(ThreadStats const& other) {
   flag = (flag<0 || other.flag<0) ? std::min(flag, other.flag) // error
                                   : std::max(flag, other.flag);// warning/pass
   num_delay += other.num_delay;
   num_neg += other.num_neg;
   num_two += other.num_two;
   num_zero += other.num_zero;
   maxfront = std::max(maxfront, other.maxfront);
   not_first_pass += other.not_first_pass;
   not_second_pass += other.not_second_pass;

   return *this;
}

}}} /* namespace spral::ssids::cpu */
