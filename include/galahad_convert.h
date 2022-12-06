//* \file galahad_convert.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-02-25 AT 07:13 GMT.
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

/**
 * control derived type as a C struct
 */
struct convert_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    int error;

    /// \brief
    /// unit for monitor output
    int out;

    /// \brief
    /// controls level of diagnostic output
    int print_level;

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
    real_wp_ total;

    /// \brief
    /// total clock time spent in the package
    real_wp_ clock_total;
};

/**
 * inform derived type as a C struct
 */
struct convert_inform_type {

    /// \brief
    /// return status. Possible values are:
    /// \li 0 successful conversion
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
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the number of duplicates found (-ve = not checked)
    int duplicates;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// timings (see above)
    struct convert_time_type time;
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
