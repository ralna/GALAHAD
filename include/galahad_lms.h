//* \file galahad_lms.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LMS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package lms

  \section lms_intro Introduction

  \subsection lms_purpose Purpose

  Given a sequence of vectors
  \latexonly
$\{s_k\}$ and $\{y_k\}$ \mbox{and scale factors} $\{\delta_k\}$,
  \endlatexonly
  \htmlonly
{s<sub>k</sub>} and {y<sub>k</sub>} and scalars {&#948<sub>k</sub>},
  \endhtmlonly
\manonly
{s_k} and {y_k} and scalars {delta_k},
\endmanonly
<b>obtain the product of a limited-memory secant
  approximation \f$H_k\f$ (or its inverse) with a given vector</b>,
  using one of a variety of well-established formulae.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection lms_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection lms_date Originally released

  July 2014, C interface January 2022.

  \subsection lms_method Method

  Given a sequence of vectors
  \latexonly
$\{s_k\}$ and $\{y_k\}$ \mbox{and scale factors} $\{\delta_k\}$,
  \endlatexonly
  \htmlonly
{s<sub>k</sub>} and {y<sub>k</sub>} and scalars {&#948<sub>k</sub>},
  \endhtmlonly
\manonly
{s_k} and {y_k} and scalars {delta_k},
\endmanonly
a limited-memory secant approximation \f$H_k\f$ is chosen
 so that \f$H_{\max(k-m,0)} = \delta_k I\f$, \f$H_{k-j} s_{k-j} = y_{k-j}\f$
 and \f$\| H_{k-j+1} - H_{k-j}\|\f$ is ``small'' for
\f$j = \min(k-1,m-1), \ldots, 0\f$.
Different ways of quantifying ``small'' distinguish different methods,
but the crucial observation is that it is possible to construct
\f$H_k\f$ quickly from \f${s_k}\f$, \f${y_k}\f$ and \f$\delta_k\f$,
 and to apply it and its inverse  to a given vector \f$v\f$.
 It is also possible to apply similar formulae to the ``shifted'' matrix
\f$H_k + \lambda_k I\f$ that occurs in trust-region methods.

  \subsection lms_references Reference

  The basic methods are those given by

  R. H. Byrd, J. Nocedal and R. B. Schnabel (1994)
  Representations of quasi-Newton matrices and their use in
  limited memory methods.
  Mathenatical Programming, <b>63(2)</b> 129-156,

  with obvious extensions.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_LMS_H
#define GALAHAD_LMS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct lms_control_type {

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
    /// limited memory length
    ipc_ memory_length;

    /// \brief
    /// limited-memory formula required (others may be added in due course):
    /// \li 1 BFGS (default).
    /// \li 2 Symmetric Rank-One (SR1).
    /// \li 3 The inverse of the BFGS formula.
    /// \li 4 The inverse of the shifted BFGS formula. This should be used
    ///       instead of .method = 3 whenever a shift is planned.
    ipc_ method;

    /// \brief
    /// allow space to permit different methods if required (less efficient)
    bool any_method;

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
struct lms_time_type {

    /// \brief
    /// total cpu time spent in the package
    rpc_ total;

    /// \brief
    /// cpu time spent setting up space for the secant approximation
    rpc_ setup;

    /// \brief
    /// cpu time spent updating the secant approximation
    rpc_ form;

    /// \brief
    /// cpu time spent applying the secant approximation
    rpc_ apply;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time spent setting up space for the secant approximation
    rpc_ clock_setup;

    /// \brief
    /// clock time spent updating the secant approximation
    rpc_ clock_form;

    /// \brief
    /// clock time spent applying the secant approximation
    rpc_ clock_apply;
};

/**
 * inform derived type as a C struct
 */
struct lms_inform_type {

    /// \brief
    /// the return status. Possible values are:
    /// \li 0 the update was successful.
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
    /// \li -3. One of the restrictions n > 0, delta > 0, lambda > 0 or
    /// s^T y > 0 has been violated and the update has been skipped.
    /// \li -10. The matrix cannot be built from the current vectors
    /// {s_k} and {y_k} and values delta_k and lambda_k
    /// and the update has been skipped.
    /// \li -31. A call to the function lhs_apply has been made without
    /// a prior call to lhs_form_shift or lhs_form with lambda
    /// specified when control.method = 4, or lhs_form_shift
    /// has been called when control.method = 3, or
    /// lhs_change_method has been called after
    /// control.any_method = false was specified when calling lhs_setup.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the number of pairs (s,y) currently used to represent the limited-memory
    /// matrix.
    ipc_ length;

    /// \brief
    /// have (s,y) pairs been skipped when forming the limited-memory matrix?
    bool updates_skipped;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];

    /// \brief
    /// timings (see above)
    struct lms_time_type time;
};

// *-*-*-*-*-*-*-*-*-*-    L M S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void lms_initialize( void **data,
                     struct lms_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lms_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-*-    L M S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lms_information( void **data,
                      struct lms_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lms_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L M S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void lms_terminate( void **data,
                    struct lms_control_type *control,
                    struct lms_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lms_control_type)

  @param[out] inform is a struct containing output information
              (see lms_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
