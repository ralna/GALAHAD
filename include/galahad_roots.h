//* \file galahad_roots.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 16:59 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_ROOTS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package roots

  \section roots_intro Introduction

  \subsection roots_purpose Purpose

  Use classical formulae together with Newton’s method to find all the real
  roots of a real polynomial.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection roots_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection roots_date Originally released

  April 2005, C interface January 2022.

  \subsection roots_method Method

  Littlewood and Ferrari's algorithms are used to find estimates of the
  real roots of cubic and quartic polynomials, respectively; a
  stabilized version of the well-known formula is used in the quadratic
  case. Newton's method is used to further refine the computed roots if
  necessary. Madsen and Reid's method is used for polynomials whose
  degree exceeds four.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_ROOTS_H
#define GALAHAD_ROOTS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct roots_control_type {

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
    /// the required accuracy of the roots
    real_wp_ tol;

    /// \brief
    /// any coefficient smaller in absolute value than zero_coef will be regarde
    /// to be zero
    real_wp_ zero_coef;

    /// \brief
    /// any value of the polynomial smaller in absolute value than zero_f
    /// will be regarded as giving a root
    real_wp_ zero_f;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
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
struct roots_inform_type {

    /// \brief
    /// return status. Possible values are:
    /// \li 0 the call was successful.
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
    /// \li -3. Either the specified degree of the polynomial in degree 
    /// is less than 0, or the declared dimension of the array roots
    /// is smaller than the specified degree.
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
