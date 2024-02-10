//* \file galahad_convert.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_CONVERT C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. February 25th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package convert

  \section convert_intro Introduction

  \subsection convert_purpose Purpose

  Given a real matrix \f$A\f$ stored in one format, convert it to another

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection convert_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection convert_date Originally released

  June 2014, C interface February 2022.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_CONVERT_H
#define GALAHAD_CONVERT_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct convert_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    ipc_ error;

    /// \brief
    /// unit for monitor output
    ipc_ out;

    /// \brief
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// obtain the transpose of the input matrix?
    bool transpose;

    /// \brief
    /// add the values of entries in duplicate positions?
    bool sum_duplicates;

    /// \brief
    /// order row or column data by increasing index?
    bool order;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];
};

/**
 * time derived type as a C struct
 */
struct convert_time_type {

    /// \brief
    /// total cpu time spent in the package
    rpc_ total;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;
};

/**
 * inform derived type as a C struct
 */
struct convert_inform_type {

    /// \brief
    /// the return status. Possible values are:
    /// \li 0 a successful conversion.
    /// \li -1. An allocation error occurred. A message indicating the
    ///      offending array is written on unit control.error, and the
    ///      returned allocation status and a string containing the name
    ///      of the offending array are held in inform.alloc_status and
    ///      inform.bad_alloc respectively.
    /// \li -2. A deallocation error occurred.  A message indicating the
    ///      offending array is written on unit control.error and the
    ///      returned allocation status and a string containing the
    ///      name of the offending array are held in
    ///      inform.alloc_status and inform.bad_alloc respectively.
    /// \li -3. The restriction n > 0 or m > 0 or requirement that a type
    ///     contains its relevant string 'coordinate', 'sparse_by_rows',
    ///     'sparse_by_columns', 'dense_by_rows' or 'dense_by_columns'
    ///     has been violated.
    /// \li -32 provided integer workspace is not large enough.
    /// \li -33 provided real workspace is not large enough.
    /// \li -73 an input matrix entry has been repeated.
    /// \li -79 there are missing optional arguments.
    /// \li -90 a requested output format is not recognised.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the number of duplicates found (-ve = not checked).
    ipc_ duplicates;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];

    /// \brief
    /// timings (see above).
    struct convert_time_type time;
};

// *-*-*-*-*-*-*-*-*-    C O N V E R T  _ I N I T I A L I Z E    -*-*--*-*-*-*-

void convert_initialize( void **data,
                     struct convert_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see convert_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-   C O N V E R T  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-

void convert_information( void **data,
                      struct convert_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see convert_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-    C O N V E R T  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-

void convert_terminate( void **data,
                    struct convert_control_type *control,
                    struct convert_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see convert_control_type)

  @param[out] inform is a struct containing output information
              (see convert_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
