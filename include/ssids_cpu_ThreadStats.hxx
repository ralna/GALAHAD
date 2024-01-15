/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <cstdint>
#include <stdexcept>

namespace spral { namespace ssids { namespace cpu {

/** \brief SSIDS error/warning flags.
 *
 * Must match Fortran definitions in src/ssids/datatypes.f90
 */
enum Flag : int {
   SUCCESS                 = 0,

   ERROR_SINGULAR          = -5,
   ERROR_NOT_POS_DEF       = -6,
   ERROR_ALLOCATION        = -50,

   WARNING_FACT_SINGULAR   = 7
};

/**
 * \brief Exception class for options.action = false and singular matrix.
 */
class SingularError: public std::runtime_error {
public:
   SingularError(int col)
   : std::runtime_error("Matrix is singular"), col(col)
   {}

   int const col;
};

/**
 * \brief Factorization statistics for a thread.
 *
 * Defines a sensible set of initializations and other useful operations such
 * as "summation" for condensing multiple threads' stats at the end of
 * factorization. Interoperates with Fortran type cpu_factor_stats.
 *
 * \sa spral_ssids_cpu_iface::cpu_factor_stats
 */
struct ThreadStats {
   Flag flag = Flag::SUCCESS; ///< Error flag for thread
   int num_delay = 0;   ///< Number of delays
   int64_t num_factor = 0;    ///< Number of entries in factors
   int64_t num_flops = 0;     ///< Number of floating point operations
   int num_neg = 0;     ///< Number of negative pivots
   int num_two = 0;     ///< Number of 2x2 pivots
   int num_zero = 0;    ///< Number of zero pivots
   int maxfront = 0;    ///< Maximum front size
   int maxsupernode = 0;      ///< Maximum supernode size
   int not_first_pass = 0;    ///< Number of pivots not eliminated in APP
   int not_second_pass = 0;   ///< Number of pivots not eliminated in APP or TPP

   ThreadStats& operator+=(ThreadStats const& other);
};

}}} /* namespaces spral::ssids::cpu */
