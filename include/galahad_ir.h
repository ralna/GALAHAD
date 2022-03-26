//* \file galahad_ir.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-06 AT 13:55 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_IR C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.4. January 4th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package ir
 
  \section ir_intro Introduction

  \subsection sls_purpose Purpose

  Given a sparse symmetric \f$n \times n\f$ matrix \f$A = a_{ij}\f$
  and the factorization of \f$A\f$ found by the GALAHAD package SLS,
  this package <b> solves the system of linear equations \f$A x = b\f$ using
  iterative refinement.</b>

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection ir_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection ir_date Originally released

  October 2008, C interface January 2022
 
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_IR_H 
#define GALAHAD_IR_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct ir_control_type {

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
    /// maximum number of iterative refinements allowed
    int itref_max;

    /// \brief
    /// refinement will cease as soon as the residual \f$\|Ax-b\|\f$ falls below
    /// max( acceptable_residual_relative * \f$\|b\|\f$, 
    ///      acceptable_residual_absolute )
    real_wp_ acceptable_residual_relative;
    /// see acceptable_residual_relative
    real_wp_ acceptable_residual_absolute;

    /// \brief
    /// refinement will be judged to have failed if the residual
    /// \f$\|Ax-b\| \geq \f$ required_residual_relative * \f$\|b\|\f$.
    /// No checking if required_residual_relative < 0
    real_wp_ required_residual_relative;

    /// \brief
    /// record the initial and final residual
    bool record_residuals;

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
 * inform derived type as a C struct
 */
struct ir_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0 the solution has been found
    /// \li -1 an array allocation has failed
    /// \li -2 an array deallocation has failed
    int status;

    /// \brief
    /// STAT value after allocate failure
    int alloc_status;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// infinity norm of the initial residual
    real_wp_ norm_initial_residual;

    /// \brief
    /// infinity norm of the final residual
    real_wp_ norm_final_residual;
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
