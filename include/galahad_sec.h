//* \file galahad_sec.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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
  so that the approximation B satisfies the secant condition B s = y
  for given vectors s and y.

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
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct sec_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required. <= 0 gives no output, >= 1 warning message
    ipc_ print_level;

    /// \brief
    /// the initial Hessian approximation will be h_initial * \f$I\f$
    rpc_ h_initial;

    /// \brief
    /// an update is skipped if the resulting matrix would have grown too much;
    ///  specifically it is skipped when y^T s / y^T y <= update_skip_tol.
    rpc_ update_skip_tol;

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
    ipc_ status;
};

// *-*-*-*-*-*-*-*-*-*-    S E C  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sec_initialize( struct sec_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[out] control is a struct containing control information
              (see sec_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-*-    S E C  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sec_information( void **data,
                      struct sec_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see sec_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S E C  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void sec_terminate( void **data,
                    struct sec_control_type *control,
                    struct sec_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see sec_control_type)

  @param[out] inform is a struct containing output information
              (see sec_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
