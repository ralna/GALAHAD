//* \file galahad_hash.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-03-13 AT 10:15 GMT.
/*
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_HASH C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 11th 2021
/*
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package hash
 
  \section hash_intro Introduction

  \subsection hash_purpose Purpose

  Set up, insert into, remove from and search a chained scatter table 
  (Williams, CACM 2, 21-24, 1959).

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection hash_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory,
  England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection hash_date Originally released

  December 1990, C interface January 2022.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_HASH_H 
#define GALAHAD_HASH_H

// precision
#include "galahad_precision.h"

/*
 * control derived type as a C struct
 */
struct hash_control_type {

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief
    /// the level of output required. Possible values are:
    /// \li \f$\leq\f$ 0 no output, 
    /// \li \f$\geq\f$ 1 debugging
    int print_level;

    /// \brief
    /// if %space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if %deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
    /// where %prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/*
 * inform derived type as a C struct
 */
struct hash_inform_type {

    /// \brief
    /// return status. 0 is a successful call
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];
};


// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
