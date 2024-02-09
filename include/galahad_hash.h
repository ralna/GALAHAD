//* \file galahad_hash.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-03-13 AT 10:15 GMT.
 *
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_HASH C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 11th 2021
 *
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

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection hash_date Originally released

  December 1990, C interface January 2022.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_HASH_H
#define GALAHAD_HASH_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/*
 * control derived type as a C struct
 */
struct hash_control_type {

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required. Possible values are:
    /// \li \f$\leq\f$ 0 no output,
    /// \li \f$\geq\f$ 1 debugging
    ipc_ print_level;

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
    /// return status. Possible values are:
    /// \li 0 The initialization, insertion or deletion was succesful.
    /// \li -1. An allocation error occurred. A message indicating the
    /// offending array is written on unit control.error, and the
    /// returned allocation status and a string containing the name
    /// of the offending array are held in inform.alloc_status and
    /// inform.bad_alloc respectively.
    /// \li -2. A deallocation error occurred.  A message indicating the
    /// offending array is written on unit control.error and the
    /// returned allocation status and a string containing the
    /// name of the offending array are held in
    /// inform.alloc_status and inform.bad_alloc respectively.
    /// \li -99. The current dictionary is full and should be rebuilt with
    /// more space.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];
};

// *-*-*-*-*-*-*-*-*-    H A S H  _ I N I T I A L I Z E    -*-*--*-*-*-*-

void hash_initialize( ipc_ nchar,
                      ipc_ length,
                      void **data,
                      struct hash_control_type *control,
                      struct hash_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in] nchar is a scalar variable of type int, that holds the
    number of characters permitted in each word in the hash table

  @param[in] length is a scalar variable of type int, that holds the
    maximum number of words that can be held in the dictionary

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see hash_control_type)

  @param[out] inform is a struct containing output information
              (see hash_inform_type)
*/

// *-*-*-*-*-*-*-*-*-   H A S H  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-

void hash_information( void **data,
                       struct hash_inform_type *inform,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see hash_inform_type)

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-    H A S H  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-

void hash_terminate( void **data,
                     struct hash_control_type *control,
                     struct hash_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see hash_control_type)

  @param[out] inform is a struct containing output information
              (see hash_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
