//* \file galahad_lhs.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 13.25 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LHS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 3rd 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package lhs

  \section lhs_intro Introduction

  \subsection lhs_purpose Purpose

  This package <b>computes an array of Latin Hypercube samples.</b>.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection lhs_authors Authors

  J. Burkardt, University of Pittsburgh (LGPL) adapted for GALAHAD by
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection lhs_date Originally released

  June 2016, C interface March 2022.
*/


#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_LHS_H
#define GALAHAD_LHS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/*
 * control derived type as a C struct
 */
struct lhs_control_type {

    /// \brief
    /// error and warning diagnostics occur on stream error.
    ipc_ error;

    /// \brief
    /// general output occurs on stream out.
    ipc_ out;

    /// \brief
    /// the level of output required. Possible values are:
    ///  \li < 1 no output.
    ///  \li > 0 debugging.
    ipc_ print_level;

    /// \brief
    /// the duplication factor. This must be at least 1, a value of 5 is
    /// reasonable.
    ipc_ duplication;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time.
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue.
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(%prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/*
 * inform derived type as a C struct
 */
struct lhs_inform_type {

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
    /// \li -3. The random number seed has not been set.
     ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    ///  occurred.
    char bad_alloc[81];
};

//  *-*-*-*-*-*-*-*-*-*-   L H S _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*

void lhs_initialize( void **data,
                     struct lhs_control_type *control,
                     struct lhs_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see fit_control_type)

  @param[out] inform is a struct containing output information
              (see fit_inform_type)
*/

//  *-*-*-*-*-*-*-*-*-   L H S _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*-*-*

void lhs_read_specfile( struct lhs_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and perform the assignment of
  values associated with given keywords to the corresponding control
  parameters.

  By default, the spcification file will be named RUNLHS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/lhs.pdf for a list of keywords that may be set.

  @param[in] control  a struct containing control information (see above)
  @param[in] specfile a character string containing the name of the specfile
 */

//  *-*-*-*-*-*-*-*-*-*-*-*-*-*-   L H S _ I H S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*

void lhs_ihs( ipc_ n_dimen,
              ipc_ n_points,
              ipc_ *seed,
              ipc_ *X,
              const struct lhs_control_type *control,
              struct lhs_inform_type *inform, void **data );

/*!<
  The improved distributed hyper-cube sampling algorithm.

  Discussion:

  n_points points in an n_dimen dimensional Latin hyper-cube are
  to be selected. Each of the coordinate dimensions is discretized
  to the values 1 through n.  The points are to be chosen in such
  a way that no two points have any coordinate value in common.
  This is a standard Latin hypercube requirement, and there are
  many solutions.

  This algorithm differs in that it tries to pick a solution
  which has the property that the points are "spread out"
  as evenly as possible.  It does this by determining an optimal
  even spacing, and using the DUPLICATION factor to allow it
  to choose the best of the various options available to it.

  Reference:

  Brian Beachkofski, Ramana Grandhi,
  Improved Distributed Hypercube Sampling,
  American Institute of Aeronautics and Astronautics Paper 2002-1274

  @param[in] n_dimen is a scalar variable of type int, that specifies
             the spatial dimension

  @param[in] n_points is a scalar variable of type int, that specifies
             the number of points to be generated

  @param[in,out] seed is a scalar variable of type int, that gives a seed
                 for the random number generator used

  @param[out] X is an array variable of type int, with dimensions
              [n_dimen][n_points] that gives the hyper-cube points

  @param[in,out] control, inform, data - see lhs_initialize

 *  ---------------------------------------------------------------------------
 */

//  *-*-*-*-*-*-*-*-*-*-*-*-   L H S _ G E T _ S E E D  -*-*-*-*-*-*-*-*-*-*-*

void lhs_get_seed( ipc_ *seed );

/*!<
  Get a seed for the random number generator.

  @param[out] seed is a scalar variable of type int, that gives the
              pseudorandom seed value.
*/

// *-*-*-*-*-*-*-*-*-*-    L H S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lhs_information( void **data,
                      struct lhs_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lhs_inform_type)

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

//  *-*-*-*-*-*-*-*-*-*-*-   L H S _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*-*

void lhs_terminate( void **data,
                    struct lhs_control_type *control,
                    struct lhs_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lhs_control_type)

  @param[out] inform is a struct containing output information
              (see lhs_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
