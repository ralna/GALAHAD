//* \file galahad_ir.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection ir_date Originally released

  October 2008, C interface January 2022

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_IR_H
#define GALAHAD_IR_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct ir_control_type {

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
    /// maximum number of iterative refinements allowed
    ipc_ itref_max;

    /// \brief
    /// refinement will cease as soon as the residual \f$\|Ax-b\|\f$ falls below
    /// max( acceptable_residual_relative * \f$\|b\|\f$,
    ///      acceptable_residual_absolute )
    rpc_ acceptable_residual_relative;
    /// see acceptable_residual_relative
    rpc_ acceptable_residual_absolute;

    /// \brief
    /// refinement will be judged to have failed if the residual
    /// \f$\|Ax-b\| \geq \f$ required_residual_relative * \f$\|b\|\f$.
    /// No checking if required_residual_relative < 0
    rpc_ required_residual_relative;

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
    /// the return status. Possible values are:
    /// \li 0 the solution has been found.
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
    /// \li -11. Iterative refinement has not reduced the
    /// relative residual by more than control.required_relative_residual.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];

    /// \brief
    /// the infinity norm of the initial residual
    rpc_ norm_initial_residual;

    /// \brief
    /// the infinity norm of the final residual
    rpc_ norm_final_residual;
};

// *-*-*-*-*-*-*-*-*-*-    I R  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void ir_initialize( void **data,
                    struct ir_control_type *control,
                    ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see ir_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-*-    I R  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void ir_information( void **data,
                     struct ir_inform_type *inform,
                     ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see ir_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    I R  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void ir_terminate( void **data,
                   struct ir_control_type *control,
                   struct ir_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see ir_control_type)

  @param[out] inform is a struct containing output information
              (see ir_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
