//* \file galahad_ugo.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_UGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. July 27th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package ugo

  \section ugo_intro Introduction

  \subsection ugo_purpose Purpose

  The ugo package aims to find the <b>global minimizer of a univariate
  twice-continuously differentiable function \f$f(x)\f$ of a single variable
  over the finite interval \f$x^l \leq x \leq x^u\f$.</b> Function and
  derivative values may be provided either via a subroutine call,
  or by a return to the calling program. Second derivatives may be used
  to advantage if they are available.

  \subsection ugo_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection ugo_date Originally released

  July 2016, C interface August 2021.

  \subsection ugo_method Method

  The algorithm starts by splitting the interval \f$[x^l,x^u]\f$ into a
  specified number of subintervals \f$[x_i,x_{i+1}]\f$ of equal length,
  and evaluating \f$f\f$ and its derivatives at each \f$x_i\f$. A surrogate
  (approximating) lower bound function is constructed on each subinterval
  using the function and derivative values at each end, and an estimate of
  the first- and second-derivative Lipschitz constant. This surrogate is
  minimized, the true objective evaluated at the best predicted point,
  and the corresponding interval split again at this point.
  Any interval whose surrogate lower bound value exceeds an evaluated
  actual value is discarded. The method continues until only one interval
  of a maximum permitted width remains.

  \subsection ugo_references References

  Many ingredients in the algorithm are based on the paper

  D. Lera and Ya. D. Sergeyev (2013),
  ``Acceleration of univariate global optimization algorithms working with
  Lipschitz functions and Lipschitz first derivatives''
  SIAM J. Optimization Vol. 23, No. 1, pp. 508–529,

  but adapted to use second derivatives.

  \section ugo_call_order Call order

  To solve a given problem, functions from the ugo package must be called
  in the following order:

  - \link ugo_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link ugo_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link ugo_import \endlink - set up problem data structures and fixed
      values
  - \link ugo_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of
     - \link ugo_solve_direct \endlink - solve using function calls to
       evaluate function and derivative values, or
     - \link ugo_solve_reverse \endlink - solve returning to the
       calling program to obtain function and derivative values
  - \link ugo_information \endlink (optional) - recover information about
    the solution and solution process
  - \link ugo_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_UGO_H
#define GALAHAD_UGO_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/*
 * control derived type as a C struct
 */
struct ugo_control_type {

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required. Possible values are:
    /// \li \f$\leq\f$ 0 no output,
    /// \li 1 a one-line summary for every improvement
    /// \li 2 a summary of each iteration
    /// \li \f$\geq\f$ 3 increasingly verbose (debugging) output
    ipc_ print_level;

    /// \brief
    /// any printing will start on this iteration
    ipc_ start_print;

    /// \brief
    /// any printing will stop on this iteration
    ipc_ stop_print;

    /// \brief
    /// the number of iterations between printing
    ipc_ print_gap;

    /// \brief
    /// the maximum number of iterations allowed
    ipc_ maxit;

    /// \brief
    /// the number of initial (uniformly-spaced) evaluation points
    /// (<2 reset to 2)
    ipc_ initial_points;

    /// \brief
    /// incremenets of storage allocated (less that 1000 will be reset to 1000)
    ipc_ storage_increment;

    /// \brief
    /// unit for any out-of-core writing when expanding arrays
    ipc_ buffer;

    /// \brief
    /// what sort of Lipschitz constant estimate will be used:
    /// \li 1 = global contant provided
    /// \li 2 = global contant estimated
    /// \li 3 = local costants estimated
    ipc_ lipschitz_estimate_used;

    /// \brief
    /// how is the next interval for examination chosen:
    /// \li 1 = traditional
    /// \li 2 = local_improvement
    ipc_ next_interval_selection;

    /// \brief
    /// try refine_with_newton Newton steps from the vacinity of the global
    /// minimizer to try to improve the estimate
    ipc_ refine_with_newton;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    ipc_ alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief
    /// overall convergence tolerances. The iteration will terminate when
    /// the step is less than .stop_length
    rpc_ stop_length;

    /// \brief
    /// if the absolute value of the gradient is smaller than
    /// small_g_for_newton, the next evaluation point may be at a
    /// Newton estimate of a local minimizer
    rpc_ small_g_for_newton;

    /// \brief
    /// if the absolute value of the gradient at the end of the interval search
    /// is smaller than small_g, no Newton serach is necessary
    rpc_ small_g;

    /// \brief
    /// stop if the objective function is smaller than a specified value
    rpc_ obj_sufficient;

    /// \brief
    /// the global Lipschitz constant for the gradient (-ve means unknown)
    rpc_ global_lipschitz_constant;

    /// \brief
    /// the reliability parameter that is used to boost insufficiently large
    /// estimates of the Lipschitz constant (-ve means that default values
    /// will be chosen depending on whether second derivatives are provided
    /// or not)
    rpc_ reliability_parameter;

    /// \brief
    /// a lower bound on the Lipscitz constant for the gradient
    /// (not zero unless the function is constant)
    rpc_ lipschitz_lower_bound;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// if .second_derivative_available is true, the user must provide them
    /// when requested. The package is generally more effective if second
    /// derivatives are available.
    bool second_derivative_available;

    /// \brief
    /// if .space_critical is true, every effort will be made to use as little
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

/*
 * time derived type as a C struct
 */
struct ugo_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;
};

/*
 * inform derived type as a C struct
 */
struct ugo_inform_type {

    /// \brief
    ///  return status. See UGO_solve for details
    ipc_ status;

    /// \brief
    /// evaluation status for reverse communication interface
    ipc_ eval_status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations performed
    ipc_ iter;

    /// \brief
    /// the total number of evaluations of the objective function
    ipc_ f_eval;

    /// \brief
    /// the total number of evaluations of the gradient of the objective
    /// function
    ipc_ g_eval;

    /// \brief
    /// the total number of evaluations of the Hessian of the objective function
    ipc_ h_eval;

    /// \brief
    /// timings (see above)
    struct ugo_time_type time;
};

/*  *-*-*-*-*-*-*-*-*-*-   U G O _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*
 *
 * Provide default values for UGO controls
 */

void ugo_initialize( void **data,
                     struct ugo_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

 @param[in,out] data  holds private internal data

 @param[out] control is a struct containing control information
              (see ugo_control_type)

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    U G O  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void ugo_read_specfile( struct ugo_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNUGO.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/ugo.pdf for a list of keywords that may be set.

  @param[in,out] control is a struct containing control information
              (see ugo_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

//  *-*-*-*-*-*-*-*-*-*-*-*-   U G O _ I M P O R T    -*-*-*-*-*-*-*-*-*-*-*-*

void ugo_import( struct ugo_control_type *control,
                 void **data,
                 ipc_ *status,
                 const rpc_ *x_l,
                 const rpc_ *x_u );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see ugo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.

 @param[in] x_l is a scalar variable of type rpc_,
    that holds the value \f$x^l\f$ of the lower bound on the optimization
    variable \f$x\f$.

 @param[in] x_u is a scalar variable of type rpc_,
    that holds the value \f$x^u\f$ of the upper bound on the optimization
    variable \f$x\f$.

 */


//  *-*-*-*-*-*-*-*-*-   U G O _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void ugo_reset_control( struct ugo_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see ugo_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
 */

//   *-*-*-*-*-*-*-*-*-*-   U G O _ S O L V E _ D I R E C T   -*-*-*-*-*-*-*-*-*

/*
 *  ugo_solve_direct, a method for finding the global minimizer of a univariate
 *    continuous function with a Lipschitz gradient in an interval
 *
 *   This version is for the case where all function/derivative information
 *   is available by function calls.
 *
 *  Many ingredients in the algorithm are based on the paper
 *
 *   Daniela Lera and Yaroslav D. Sergeyev,
 *   "Acceleration of univariate global optimization algorithms working with
 *    Lipschitz functions and Lipschitz first derivatives"
 *   SIAM J. Optimization Vol. 23, No. 1, pp. 508–529 (2013)
 *
 *  but adapted to use 2nd derivatives
 */

void ugo_solve_direct( void **data,
                       void *userdata,
                       ipc_ *status,
                       rpc_ *x,
                       rpc_ *f,
                       rpc_ *g,
                       rpc_ *h,
                       ipc_ (*eval_fgh)(
                          rpc_, rpc_*, rpc_*, rpc_*,
                          const void * ) );

/*!<
 Find an approximation to the global minimizer of a given univariate
 function with a Lipschitz gradient in an interval.

 This version is for the case where all function/derivative information
 is available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs (see below).

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -7. The objective function appears to be unbounded from below
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

  @param[out] x is a scalar variable of type rpc_, that holds the value of
     the approximate global minimizer \f$x\f$ after a successful (status = 0)
     call.

   @param[out] f is a scalar variable of type rpc_, that holds the the value
     of the objective function \f$f(x)\f$ at the approximate global minimizer
     \f$x\f$ after a successful (status = 0) call.

   @param[out] g is a scalar variable of type rpc_, that holds the the value
     of the gradient of the objective function \f$f^{\prime}(x)\f$
     at the approximate global minimizer
     \f$x\f$ after a successful (status = 0) call.

   @param[out] h is a scalar variable of type rpc_, that holds the the value
     of the second derivative of the objective function
    \f$f^{\prime\prime}(x)\f$ at the approximate global minimizer
    \f$x\f$ after a successful (status = 0) call.

   @param eval_fgh is a user-provided function that must have the following
   signature:
   \code
     ipc_ eval_fgh( rpc_ x,
                   rpc_ *f,
                   rpc_ *g,
                   rpc_ *h,
                   const void *userdata)
   \endcode
   The value of the objective function \f$f(x)\f$ and its first derivative
   \f$f^{\prime}(x)\f$ evaluated at x=\f$x\f$ must be assigned
   to f and g respectively, and the function return value set to 0.
   In addition, if control.second_derivatives_available has been set to true,
   when calling ugo_import, the user must also assign the value of the
   second derivative \f$f^{\prime\prime}(x)\f$ in h; it need not be assigned
   otherwise. If the evaluation is impossible at x, return should be set to a
   nonzero value.

*/

//  *-*-*-*-*-*-*-*-*-*-   U G O _ S O L V E _ R E V E R S E   -*-*-*-*-*-*-*-*

void ugo_solve_reverse( void **data,
                        ipc_ *status,
                        ipc_ *eval_status,
                        rpc_ *x,
                        rpc_ *f,
                        rpc_ *g,
                        rpc_ *h );

/*!<
 Find an approximation to the global minimizer of a given univariate
 function with a Lipschitz gradient in an interval.

 This version is for the case where function/derivative information is only
 available by returning to the calling procedure.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n
    On initial entry, status must be set to 1. \n
    Possible exit are:
  \li  0. The run was succesful

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -7. The objective function appears to be unbounded from below
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file
         named control.alive_file from unit unit control.alive_unit.

  \li  3. The user should compute the objective function value \f$f(x)\f$
        and its first derivative \f$f^{\prime}(x)\f$,  and then
        re-enter the function. The required values should be set in f and g
        respectively, and eval_status (below) should be set to 0.
        If the user is unable to evaluate \f$f(x)\f$ or \f$f^{\prime}(x)\f$ -
        for instance, if the function or its first derivative are
        undefined at x - the user need not set f or g, but should then set
        eval_status to a non-zero value. This value can only occur when
        control.second_derivatives_available = false.

  \li  4. The user should compute the objective function value \f$f(x)\f$
        and its first two derivatives \f$f^{\prime}(x)\f$ and
        \f$f^{\prime\prime}(x)\f$ at x=\f$x\f$, and then
        re-enter the function. The required values should be set in f, g
        and h respectively, and eval_status (below) should be set to 0.
        If the user is unable to evaluate \f$f(x)\f$, \f$f^{\prime}(x)\f$
        or \f$f^{\prime\prime}(x)\f$ - for instance, if
        the function or its derivatives are undefined at x - the user need not
        set f, g or h, but should then set eval_status to a non-zero value.
        This value can only occur when
        control.second_derivatives_available = true.

 @param[in,out] eval_status is a scalar variable of type ipc_, that is used to
    indicate if  objective function and its derivatives can be provided
    (see above).

 @param[out]
   x is a scalar variable of type rpc_, that holds the next value of \f$x\f$
    at which the user is required to evaluate the objective (and its
    derivatives) when status > 0, or the value of the approximate
    global minimizer when status = 0

 @param[in,out]
   f is a scalar variable of type rpc_, that must be set by the user to
    hold the value of \f$f(x)\f$ if required by status > 0 (see above), and
    will return the value of the approximate global minimum when status = 0

 @param[in,out]
   g is a scalar variable of type rpc_, that must be set by the user to
    hold the value of  \f$f^{\prime}(x)\f$ if required by status > 0
    (see above), and will return the value of the first derivative of \f$f\f$
    at the approximate global minimizer when status = 0

 @param[in,out]
   h is a scalar variable of type rpc_, that must be set by the user to
    hold the value of \f$f^{\prime\prime}(x)\f$ if required by status > 0
    (see above), and will return the value of the second derivative of \f$f\f$
    at the approximate global minimizer when status = 0
*/

//  *-*-*-*-*-*-*-*-*-*-   U G O _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void ugo_information( void **data,
                      struct ugo_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
    (see ugo_inform_type)

  @param[out] status   is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

//  *-*-*-*-*-*-*-*-*-*-   U G O _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void ugo_terminate( void **data,
                    struct ugo_control_type *control,
                    struct ugo_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see ugo_control_type)

  @param[out] inform   is a struct containing output information
              (see ugo_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example ugos.c
   This is an example of how to use the package to find an approximation
   to the global minimum of a given univariate function over an interval.

    \example ugot.c
   This is the same example, but now function and derivative information
   is found by reverse communication with the calling program.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
