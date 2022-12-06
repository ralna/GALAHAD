//* \file galahad_fit.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 16:58 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_FIT C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 28th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package fit

  \section fit_intro Introduction

  \subsection fit_purpose Purpose

  Fit polynomials to function and derivative data.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection fit_authors Authors

  N. I. M. Gould and D. P. Robinson, STFC-Rutherford Appleton Laboratory,
  England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection fit_date Originally released

  March 2010, C interface January 2022.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_FIT_H
#define GALAHAD_FIT_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct fit_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief
    /// the level of output required is specified by print_level
    int print_level;

    /// \brief
    /// if space_critical is true, every effort will be made to use as little
    /// space as possible. This may result in longer computation times
    bool space_critical;

    /// \brief
    /// if deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/**
 * inform derived type as a C struct
 */
struct fit_inform_type {

    /// \brief
    /// return status. Possible values are:
    /// \li 0 Normal termination with the required fit
    /// \li -1 An allocation error occured; the status is given in the component
    ///     .alloc_status
    /// \li -2 A deallocation error occured; the status is given in the
    ///        component alloc_status
    /// \li - 3 the restriction n >= 1 has been violated
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
