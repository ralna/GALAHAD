/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-19 AT 08:30 GMT
 */

#pragma once

#include <cstdint>
#include <stdexcept>

#include "ssids_routines.h"
#include "galahad_precision.h"

namespace galahad { namespace ssids { namespace cpu {

/** \brief SSIDS error/warning flags.
 *
 * Must match Fortran definitions in src/ssids/datatypes.f90
 */
enum Flag : ipc_ {
   SUCCESS                 = 0,

   ERROR_SINGULAR          = -5,
   ERROR_NOT_POS_DEF       = -6,
   ERROR_ALLOCATION        = -50,

   WARNING_FACT_SINGULAR   = 7
};

/**
 * \brief Exception class for control.action = false and singular matrix.
 */
class SingularError: public std::runtime_error {
public:
   SingularError(ipc_ col)
   : std::runtime_error("Matrix is singular"), col(col)
   {}

   ipc_ const col;
};

/**
 * \brief Factorization statistics for a thread.
 *
 * Defines a sensible set of initializations and other useful operations such
 * as "summation" for condensing multiple threads' stats at the end of
 * factorization. Interoperates with Fortran type cpu_factor_stats.
 *
 * \sa galahad_ssids_cpu_iface::cpu_factor_stats
 */
struct ThreadStats {
   Flag flag = Flag::SUCCESS; ///< Error flag for thread
   ipc_ num_delay = 0;   ///< Number of delays
   longc_ num_factor = 0;    ///< Number of entries in factors
   longc_ num_flops = 0;     ///< Number of floating point operations
   ipc_ num_neg = 0;     ///< Number of negative pivots
   ipc_ num_two = 0;     ///< Number of 2x2 pivots
   ipc_ num_zero = 0;    ///< Number of zero pivots
   ipc_ maxfront = 0;    ///< Maximum front size
   ipc_ maxsupernode = 0;     ///< Maximum supernode size
   ipc_ not_first_pass = 0;   ///< Number of pivots not eliminated in APP
   ipc_ not_second_pass = 0;  ///< Number of pivots not eliminated in APP or TPP

   ThreadStats& operator+=(ThreadStats const& other);
};

}}} /* namespaces galahad::ssids::cpu */
