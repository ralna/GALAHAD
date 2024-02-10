//* \file galahad_fdc.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_FDC C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 13th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package fdc

  \section fdc_intro Introduction

  \subsection fdc_purpose Purpose

  Given an under-determined set of linear equations/constraints
  \f$a_i^T x = b_i^{}\f$, \f$i = 1, \ldots, m\f$ involving
  \f$n \geq m\f$ unknowns \f$x\f$, this package
  <b>determines whether the constraints are consistent, and if
  so how many of the constraints are dependent</b>; a list of dependent
  constraints, that is, those which may be removed without changing the
  solution set, will be found and the remaining \f$a_i\f$ will be linearly
  independent.  Full advantage is taken of any zero coefficients in the
  vectors \f$a_i\f$.

  \subsection fdc_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection fdc_date Originally released

  August 2006, C interface January 2021

  \subsection fdc_method Method

  A choice of two methods is available. In the first, the matrix
  \f[K = \mat{cc}{ \alpha I & A^T \\ A & 0 }\f]
  is formed and factorized for some small \f$\alpha > 0\f$ using the
  GALAHAD package SLS---the
  factors \f$K = P L D L^T P^T\f$ are used to determine
  whether \f$A\f$ has dependent rows. In particular, in exact arithmetic
  dependencies in \f$A\f$ will correspond to zero pivots in the block
  diagonal matrix \f$D\f$.

  The second choice of method finds factors
  \f$A = P L U Q\f$ of the rectangular matrix \f$A\f$
  using the GALAHAD package ULS.
  In this case, dependencies in \f$A\f$ will be reflected in zero diagonal
  entries in \f$U\f$ in exact arithmetic.

  The factorization in either case may also be used to
  determine whether the system is consistent.

  \subsection fdc_call_order Call order

  To solve a given problem, functions from the fdc package must be called
  in the following order:

  - \link fdc_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link fdc_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link fdc_find_dependent_rows \endlink - find the number of dependent
      rows and, if there are any, whether the constraints are
      independent
  - \link fdc_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \subsection fdc_array_indexing Array indexing

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; add 1 to input integer arrays if fortran-style indexing is
  used, and beware that return integer arrays will adhere to this.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_FDC_H
#define GALAHAD_FDC_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "galahad_uls.h"

/**
 * control derived type as a C struct
 */
struct fdc_control_type {

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
    /// initial estimate of integer workspace for sls (obsolete)
    ipc_ indmin;

    /// \brief
    /// initial estimate of real workspace for sls (obsolete)
    ipc_ valmin;

    /// \brief
    /// the relative pivot tolerance (obsolete)
    rpc_ pivot_tol;

    /// \brief
    /// the absolute pivot tolerance used (obsolete)
    rpc_ zero_pivot;

    /// \brief
    /// the largest permitted residual
    rpc_ max_infeas;

    /// \brief
    /// choose whether SLS or ULS is used to determine dependencies
    bool use_sls;

    /// \brief
    /// should the rows of A be scaled to have unit infinity norm or
    /// should no scaling be applied
    bool scale;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// symmetric (indefinite) linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// unsymmetric linear equation solver
    char unsymmetric_linear_solver[31];

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for SLS
    struct sls_control_type sls_control;

    /// \brief
    /// control parameters for ULS
    struct uls_control_type uls_control;
};

/**
 * time derived type as a C struct
 */
struct fdc_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to
    /// factorization
    rpc_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    rpc_ factorize;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent analysing the required matrices prior to
    /// factorization
    rpc_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    rpc_ clock_factorize;
};

/**
 * inform derived type as a C struct
 */
struct fdc_inform_type {

    /// \brief
    /// return status. See FDC_find_dependent for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the return status from the factorization
    ipc_ factorization_status;

    /// \brief
    /// the total integer workspace required for the factorization
    int64_t factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int64_t factorization_real;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    rpc_ non_negligible_pivot;

    /// \brief
    /// timings (see above)
    struct fdc_time_type time;

    /// \brief
    /// SLS inform type
    struct sls_inform_type sls_inform;

    /// \brief
    /// ULS inform type
    struct uls_inform_type uls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    F D C  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void fdc_initialize( void **data,
                     struct fdc_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see fdc_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    F D C  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void fdc_read_specfile( struct fdc_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNEQP.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/eqp.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see fdc_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-    F D C  _ F I N D _ D E P E N D E N T _ R O W S   -*-*-*-*-*-

void fdc_find_dependent_rows( struct fdc_control_type *control,
                              void **data,
                              struct fdc_inform_type *inform,
                              ipc_ *status,
                              ipc_ m,
                              ipc_ n,
                              ipc_ A_ne,
                              const ipc_ A_col[],
                              const ipc_ A_ptr[],
                              const rpc_ A_val[],
                              const rpc_ b[],
                              ipc_ *n_depen,
                              ipc_ depen[] );

/*!<
 Find dependent rows and, if any, check if \f$A x = b\f$ is consistent

 @param[in] control is a struct containing control information
           (see fdc_control_type)

 @param[in,out] data holds private internal data

 @param[out] inform  is a struct containing output information
              (see fdc_inform_type)

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    Possible exit are:
  \li  0. The run was succesful.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.
  \li -5. The constraints appear to be inconsistent.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    rows of \f$A\f$.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    columns of \f$A\f$.

 @param[in] A_ne is a scalar variable of type ipc_, that holds the number of
    nonzero entries in \f$A\f$.

 @param[in]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of \f$A\f$ in a row-wise storage scheme.
   The nonzeros must be ordered so that those in row i appear directly before
   those in row i+1, the order within each row is unimportant.

 @param[in]  A_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each row of \f$A\f$, as well as the
   total number of entries.

 @param[in] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the \f$A\f$ ordered as in A_col
    and A_ptr.

 @param[in] b is a one-dimensional array of size m and type rpc_, that
    holds the linear term \f$b\f$  in the constraints.
    The i-th component of b, i = 0, ... ,  m-1, contains  \f$b_i\f$.

 @param[out] n_depen is a scalar variable of type ipc_, that holds the number of
    dependent constraints, if any.

 @param[out] depen is a one-dimensional array of size m and type ipc_, whose
    first n_depen components contain the indices of dependent constraints.

*/

// *-*-*-*-*-*-*-*-*-*-    F D C  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void fdc_terminate( void **data,
                    struct fdc_control_type *control,
                    struct fdc_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see fdc_control_type)

  @param[out] inform   is a struct containing output information
              (see fdc_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example fdct.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example fdctf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
