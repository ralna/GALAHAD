//* \file galahad_glrt.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_GLRT C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. December 16th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package glrt

  \section glrt_intro Introduction

  \subsection glrt_purpose Purpose

  Given real \f$n\f$ by \f$n\f$ symmetric matrices \f$H\f$ and \f$M\f$
  (with \f$M\f$ positive definite), real
  \f$n\f$ vector \f$c\f$, and scalars \f$\sigma\geq 0\f$
   and \f$f_0\f$, this package finds an
  <b> approximate minimizer of the regularised quadratic objective function</b>
  \f[\frac{1}{2} x^T H x + c^T x + f_0 +  \frac{1}{p} \sigma \|x\|_M^p,\f]
\manonly
\n
1/2 x^T H x + c^T x + f_0 + 1/p sigma ||x||^p_M,
\n
\endmanonly
  where \f$\|  v \|_M = \sqrt{v^T M v}\f$ is  the \f$M\f$-norm of \f$v\f$.
  This problem commonly occurs as a subproblem in nonlinear optimization
  calculations involving cubic regularisation. The method may be suitable
  for large \f$n\f$ as no factorization of \f$H\f$ is required.
  Reverse communication is used to obtain matrix-vector products of the
  form \f$H z\f$ and \f$M^{-1} z\f$.

  \subsection glrt_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection glrt_date Originally released

  November, 2007, C interface December 2021.

  \subsection glrt_terminology Terminology

  \subsection glrt_method Method

  The required solution \f$x\f$ necessarily satisfies the optimality condition
  \f$H x + \lambda M x + c + \lambda o = 0\f$, where
  \f$\lambda = \sigma [ \|x\|_M^2]^{p/2-1}\f$.
  In addition, the matrix \f$H + \lambda M\f$ will be positive semi-definite.

  The method is iterative. Starting  with the vector \f$M^{-1} c\f$,
  a matrix of Lanczos vectors is built one column at a time
  so that the \f$k\f$-th column is generated during
  iteration \f$k\f$. These columns span a so-called Krylov space.
  The resulting \f$n\f$ by \f$k\f$ matrix \f$Q_k \f$ has the
  property that \f$Q_k^T H Q_k  =  T_k \f$,
  where \f$T_k\f$ is tridiagonal. An approximation to the
  required solution may then be expressed formally as
  \f[x_{k+1}  =  Q_k y_k,\f]
\manonly
\n
   x_{k+1}  =  Q_k y_k
\n
\endmanonly
  where \f$y_k \f$ solves the ``tridiagonal'' subproblem of minimizing
  \f[(1) \;\;\; \frac{1}{2} y^T T_k y  + \| c\|_{M^{-1}} e_1^T y + \frac{1}{p} \sigma \| y \|^p_2,\f]
\manonly
\n
  (1) 1/2  y^T T_k y  + \| c\|_{M^{-1}} e_1^T y + 1/p sigma || y ||^p_2,
\n
\endmanonly
  where \f$e_1\f$ is the first unit vector.

  To minimize (1), the optimality conditions
  \f[(2) \;\;\; ( T_k + \lambda I ) y(\lambda) = - c - \lambda d,\f]
\manonly
\n
  (2)   ( T_k + lambda I ) y(lambda) = - c - lambda d,
\n
\endmanonly
  where \f$\lambda = \sigma \|y(\lambda)+d\|^{p-2}_M\f$
  are used as the basis of an iteration. Specifically, given an estimate
  \f$\lambda\f$ for which \f$ T_k + \lambda I\f$ is positive definite,
  the tridiagonal system (2) may be efficiently solved to give
  \f$y(\lambda)\f$. It is then simply a matter of adjusting \f$\lambda\f$
  (for example by a Newton-like process) to solve the scalar nonlinear equation
  \f[(3) \;\;\; \theta(\lambda) \equiv \|y(\lambda)+d\|^{p-2}_M - \frac{\lambda}{\sigma} = 0.\f]
\manonly
\n
  (3) theta(lambda) =  ||y(\lambda)+d||^{p-2}_M - lambda/sigma = 0.
\n
\endmanonly
  In practice (3) is reformulated, and a more rapidly converging iteration is
  used.

  It is possible to measure the optimality measure
  \f$\|H x  +  \lambda M x  +  c + \lambda o\|_{M^{-1}}\f$
  without computing \f$x_{k+1}\f$, and thus without
  needing \f$Q_k \f$. Once this measure is sufficiently small, a second pass
  is required to obtain the estimate \f$x_{k+1} \f$ from \f$y_k \f$.
  As this second pass is an additional expense, a record is kept of the
  optimal objective function values for each value of \f$k\f$, and the second
  pass is only performed so far as to ensure a given fraction of the
  final optimal objective value. Large savings may be made in the second
  pass by choosing the required fraction to be significantly smaller than one.

  Special code is used in the special case \f$p=2\f$, as in this case a single
  pass suffices.

  \subsection glrt_references Reference

  The method is described in detail in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  Adaptive cubic regularisation methods for unconstrained
  optimization. Part I: motivation, convergence and numerical results.
  Mathematical Programming <b>127(2)</b>, pp.245-295, 2011.

  \subsection glrt_call_order Call order

  To solve a given problem, functions from the glrt package must be called
  in the following order:

  - \link glrt_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link glrt_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link glrt_import_control \endlink - import control parameters prior to
      solution
  - \link glrt_solve_problem \endlink - solve the problem by reverse
      communication, a sequence of calls are made under control of a status
      parameter, each exit either asks the user to provide additional
      informaton and to re-enter, or reports that either the solution has
      been found or that an error has occurred
  - \link glrt_information \endlink (optional) - recover information about
    the solution and solution process
  - \link glrt_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for an example of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for an illustration of use.
  \endhtmlonly
  \manonly
  See the examples section for an illustration of use.
  \endmanonly
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_GLRT_H
#define GALAHAD_GLRT_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct glrt_control_type {

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
    /// the level of output required is specified by print_level
    ipc_ print_level;

    /// \brief
    /// the maximum number of iterations allowed (-ve = no bound)
    ipc_ itmax;

    /// \brief
    /// the stopping rule used (see below). Possible values are:
    /// \li 1 stopping rule = norm of the step.
    /// \li 2 stopping rule is norm of the step / \f$\sigma\f$.
    /// \li other. stopping rule = 1.0.
    ipc_ stopping_rule;

    /// \brief
    /// frequency for solving the reduced tri-diagonal problem
    ipc_ freq;

    /// \brief
    /// the number of extra work vectors of length n used
    ipc_ extra_vectors;

    /// \brief
    /// the unit number for writing debug Ritz values
    ipc_ ritz_printout_device;

    /// \brief
    /// the iteration stops successfully when the gradient in the \f$M^{-1}\f$
    /// norm is smaller than
    /// max( stop_relative * min( 1, stopping_rule ) * norm initial gradient,
    /// stop_absolute )
    rpc_ stop_relative;
    /// see stop_relative
    rpc_ stop_absolute;

    /// \brief
    /// an estimate of the solution that gives at least .fraction_opt times
    /// the optimal objective value will be found
    rpc_ fraction_opt;

    /// \brief
    /// the smallest value that the square of the M norm of the gradient of
    /// the objective may be before it is considered to be zero
    rpc_ rminvr_zero;

    /// \brief
    /// the constant term, f0, in the objective function
    rpc_ f_0;

    /// \brief
    /// is M the identity matrix ?
    bool unitm;

    /// \brief
    /// is descent required i.e., should \f$c^T x < 0\f$ ?
    bool impose_descent;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// should the Ritz values be written to the debug stream?
    bool print_ritz_values;

    /// \brief
    /// name of debug file containing the Ritz values
    char ritz_file_name[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/**
 * inform derived type as a C struct
 */
struct glrt_inform_type {

    /// \brief
    /// return status. See \link glrt_solve_problem \endlink for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations required
    ipc_ iter;

    /// \brief
    /// the total number of pass-2 iterations required
    ipc_ iter_pass2;

    /// \brief
    /// the value of the quadratic function
    rpc_ obj;

    /// \brief
    /// the value of the regularized quadratic function
    rpc_ obj_regularized;

    /// \brief
    /// the multiplier, \f$\sigma \|x\|^{p-2}\f$
    rpc_ multiplier;

    /// \brief
    /// the value of the norm \f$\|x\|_M\f$
    rpc_ xpo_norm;

    /// \brief
    /// an estimate of the leftmost generalized eigenvalue of the pencil
    /// \f$(H,M)\f$
    rpc_ leftmost;

    /// \brief
    /// was negative curvature encountered ?
    bool negative_curvature;

    /// \brief
    /// did the hard case occur ?
    bool hard_case;
};

// *-*-*-*-*-*-*-*-*-*-    G L R T  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void glrt_initialize( void **data,
                     struct glrt_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see glrt_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    G L R T  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void glrt_read_specfile( struct glrt_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNGLRT.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/glrt.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see glrt_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-    G L R T  _ I M P O R T _ C O N T R O L  -*-*-*-*-*-*-

void glrt_import_control( struct glrt_control_type *control,
                          void **data,
                          ipc_ *status );

/*!<
 Import control parameters prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see glrt_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-  G L R T _ S O L V E _ P R O B L E M   -*-*-*-*-*-*-*-*-

void glrt_solve_problem( void **data,
                        ipc_ *status,
                        ipc_ n,
                        const rpc_ power,
                        const rpc_ weight,
                        rpc_ x[],
                        rpc_ r[],
                        rpc_ vector[] );

/*!<
 Solve the regularized-quadratic problem using reverse communication.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n

   This must be set to
   \li  1. on initial entry. Set r (below) to \f$c\f$ for this entry.
   \li  6. the iteration is to be restarted with a larger weight but
         with all other data unchanged. Set r (below) to \f$c\f$ for this entry.

   Possible exit values are:
   \li  0. the solution has been found
   \li  2. the inverse of \f$M\f$ must be applied to
         vector with the result returned in vector and the function
         re-entered with all other data unchanged.
         This will only happen if control.unitm is false
   \li  3. the product \f$H\f$ * vector must be formed, with
         the result returned in vector and the function re-entered
          with all other data unchanged
   \li  4. The iteration must be restarted. Reset r (below) to \f$c\f$ and
         re-enter with all other data unchanged.
   \li -1. an array allocation has failed
   \li -2. an array deallocation has failed
   \li -3. n and/or radius is not positive
   \li -7. the problem is unbounded from below. This can only happen if
         power = 2, and in this case the objective is unbounded along
         the arc x + t vector as t goes to infinity
   \li -15. the matrix \f$M\f$ appears to be indefinite
   \li -18. the iteration limit has been exceeded

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] power is a scalar of type rpc_, that holds the
    egularization power, \f$p \geq 2\f$

 @param[in] weight is a scalar of type rpc_, that holds the positive
   regularization weight, \f$\sigma\f$

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the solution \f$x\f$.
    The j-th component of x, j = 0, ... ,  n-1, contains  \f$x_j \f$.

 @param[in,out] r is a one-dimensional array of size n and type rpc_, that
    that must be set to \f$c\f$ on entry (status = 1) and re-entry
    (status = 4, 5). On exit, r contains the resiual \f$H x + c\f$.

 @param[in,out] vector is a one-dimensional array of size n and type rpc_,
    that should be used and reset appropriately when status = 2 and 3
    as directed.

*/


// *-*-*-*-*-*-*-*-*-*-    G L R T  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void glrt_information( void **data,
                      struct glrt_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see glrt_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    G L R T  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void glrt_terminate( void **data,
                    struct glrt_control_type *control,
                    struct glrt_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see glrt_control_type)

  @param[out] inform   is a struct containing output information
              (see glrt_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example glrtt.c
   This is an example of how to use the package to solve a regularized quadratic
   problem. The use of default and non-default scaling matrices, and restarts
   with a larger regularization weight are illustrated.
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
