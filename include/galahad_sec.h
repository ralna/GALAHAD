//* \file galahad_sec.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 17:01 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SEC C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package sec

  \section sec_intro Introduction

  \subsection sec_purpose Purpose

  Build and update dense BFGS and SR1 secant approximations to a Hessian.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection sec_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sec_date Originally released

  May 2008, C interface January 2022.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SEC_H
#define GALAHAD_SEC_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct sec_control_type {

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
    /// the level of output required. <= 0 gives no output, >= 1 warning message
    int print_level;

    /// \brief
    /// the initial Hessian approximation will be h_initial * \f$I\f$
    real_wp_ h_initial;

    /// \brief
    /// an update is skipped if the resulting matrix would have grown too much
    real_wp_ update_skip_tol;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/**
 * inform derived type as a C struct
 */
struct sec_inform_type {

    /// \brief
    /// return status. Possible valuesa are:
    /// \li 0 successful return
    /// \li -85 an update is inappropriate and has been skipped
    int status;
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
