//* \file galahad_l2rt.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_L2RT C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. December 19th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package l2rt

  \section l2rt_intro Introduction

  \subsection l2rt_purpose Purpose

  Given a real \f$m\f$ by \f$n\f$ matrix \f$A\f$,
  a real \f$m\f$ vector \f$b\f$ and scalars \f$\sigma>0\f$,
  \f$\mu\geq 0\f$ and \f$p \geq 2\f$,  this package finds an
  <b> approximate minimizer of the regularised linear-least-\f$\ell_2\f$-norm
  objective function
  \f[\sqrt{\| A x - b\|_2^2 + \mu \| x \|_2^2} + \frac{1}{p} \sigma \| x \|_2^p.\f]
\manonly
\n
   sqrt{||A x - b||_2^2 + mu ||x||^2_2} + 1/p sigma ||x||^p_2.
\n
\endmanonly
</b>
  This problem commonly occurs as a subproblem in nonlinear
  optimization calculations involving quadratic or cubic regularisation,
  and may be used to regularise the solution
  of under-determined or ill-conditioned linear least-squares problems.
  The method may be suitable for large \f$m\f$ and/or \f$n\f$ as no
  factorization involving \f$A\f$ is required. Reverse communication is
  used to obtain matrix-vector products of the form \f$u + A v\f$ and
  \f$v + A^T u\f$.

  \subsection l2rt_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England and
  M. Porcelli, Universita degli Studi di Firenze.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection l2rt_date Originally released

  January 2008, C interface December 2021.

  \subsection l2rt_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies the optimality condition
  \f$A^T ( A x - b ) + \lambda x = 0\f$, where the multiplier
  \f[\lambda = \mu + \sigma \|x\|_2^{p-2} \sqrt{\| A x - b \|_2^2 + \mu \|x\|_2^2}.\f]
\manonly
\n
    lambda = mu + sigma ||x||_2^{p-2} sqrt{|| A x - b ||_2^2 + mu |\x||_2^2}
\n
\endmanonly

  \subsection l2rt_method Method

  The method is iterative. Starting  with the vector \f$u_1 = b\f$, a
  bi-diagonalisation process is used to generate the vectors \f$v_k\f$ and
  \f$u_k+1\f$ so that the \f$n\f$ by \f$k\f$ matrix
  \f$V_k = ( v_1 \ldots v_k)\f$
  and the \f$m\f$ by \f$(k+1)\f$ matrix \f$U_k = ( u_1 \ldots u_{k+1})\f$
  together satisfy
  \f[A V_k = U_{k+1} B_k \;\mbox{and}\;  b = \|b\|_2 U_{k+1} e_1\f]
\manonly
\n
     A V_k = U_{k+1} B_k and  b = ||b||_2 U_{k+1} e_1
\n
\endmanonly
  where \f$B_k\f$ is \f$(k+1)\f$ by \f$k\f$ and lower bi-diagonal, \f$U_k\f$ and
  \f$V_k\f$ have orthonormal columns and \f$e_1\f$ is the first unit vector.
  The solution sought is of the form \f$x_k = V_k y_k\f$, where \f$y_k\f$
  solves the bi-diagonal regularised least-squares problem
  \f[(1) \;\;\; \min \| B_k y - \|b\| e_1 \|_2  + \frac{1}{p} \mu \|  y \|_2^p.\f]
\manonly
\n
   (1) min || B_k y - ||b|| e_1 ||_2  + 1/p mu || y||^p_2.
\n
\endmanonly

  To minimize (1), the optimality conditions
  \f[( B_k^T ( B_k^{} y(\lambda) - \|b\| e_1^{} ) + \lambda y(\lambda) = 0,\f]
\manonly
\n
     ( B_k^T ( B_k y(lambda) - ||b|| e_1 ) + lambda y(lambda) = 0,
\n
\endmanonly
  where
\f[\lambda = \mu + \sigma \|y(\lambda)\|_2^{p-2}
\sqrt{ \| B_k y(\lambda) - \|b\| e_1 \|_2^2 + \mu \|y(\lambda)\|_2^2},\f]
\manonly
\n
   lambda = mu + sigma ||y(lambda)||^{p-2}_2 sqrt{ || B_k y(lambda) - \|b\| e_1 ||_2^2 + mu ||y(lambda)||_2^2,
\n
\endmanonly
  are used as the basis of an iteration.
  The vector \f$y(\lambda)\f$ is equivalently the solution to the
  regularised least-squares problem
  \f[(2) \;\;\; \min  \left \| \vect{ B_k \\ \lambda^{\frac{1}{2}} I } y- \|b\| e_1^{} \right \|_2.\f]
\manonly
\n
    (2)    min  ||  B_k y - ||b|| e_1 ||
                ||    lambda^{1/2} y  ||
\n
\endmanonly
  Thus, given an estimate \f$\lambda \geq 0\f$, (2) may be efficiently
  solved to give \f$y(\lambda)\f$.
  It is then simply a matter of adjusting \f$\lambda\f$
  (for example by a Newton-like process) to solve the scalar nonlinear equation
  \f[(3) \;\;\; \theta(\lambda) \equiv \sigma \|y(\lambda)\|_2^{p-2}  \sqrt{ \| B_k y(\lambda) - \|b\| e_1 \|_2^2 +  \mu \|y(\lambda)\|_2^2} + \mu - \lambda = 0.\f]
\manonly
\n
   (3)   theta(lambda) = \sigma ||y(lambda)||_2^{p-2} *
             sqrt{||B_k y(lambda) - ||b|| e_1 ||^2_2 +
                     mu ||y(lambda)||^2_2} + mu - lambda = 0.
\n
\endmanonly
  In practice (3) is reformulated, and a more rapidly converging
  iteration is used. Having found  \f$y_k\f$, a second pass in which
  \f$x_k = V_k y_k\f$ is regenerated is needed---this need only be done
  once \f$x_k\f$ has implicitly deemed to be sufficiently close to optimality.
  As this second pass is an additional expense, a record is kept of the
  optimal objective function values for each value of \f$k\f$, and the second
  pass is only performed so far as to ensure a given fraction of the
  final optimal objective value. Large savings may be made in the second
  pass by choosing the required fraction to be significantly smaller than one.

  Special code is used in the special case \f$p=2\f$, as in this case the
  equation (3) significantly simplifies.

  \subsection l2rt_references Reference

  A complete description of the un- and quadratically-regularised
  cases is given by

  C. C. Paige and M. A. Saunders,
  LSQR: an algorithm for sparse linear equations and sparse least  squares.
  ACM Transactions on Mathematical Software, 8(1):43--71, 1982

  and

  C. C. Paige and M. A. Saunders,
  ALGORITHM 583: LSQR: an algorithm for sparse linear equations and
    sparse least squares.
  ACM Transactions on Mathematical Software, 8(2):195--209, 1982.

  Additional details on the Newton-like process needed to determine
  \f$\lambda\f$ and other details are described in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  Trust-region and other regularisation of linear
  least-squares problems.
  BIT 49(1):21-53 (2009).

  \subsection l2rt_call_order Call order

  To solve a given problem, functions from the l2rt package must be called
  in the following order:

  - \link l2rt_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link l2rt_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link l2rt_import_control \endlink - import control parameters prior to
      solution
  - \link l2rt_solve_problem \endlink - solve the problem by reverse
      communication, a sequence of calls are made under control of a status
      parameter, each exit either asks the user to provide additional
      informaton and to re-enter, or reports that either the solution has
      been found or that an error has occurred
  - \link l2rt_information \endlink (optional) - recover information about
    the solution and solution process
  - \link l2rt_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_L2RT_H
#define GALAHAD_L2RT_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct l2rt_control_type {

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
    /// any printing will start on this iteration
    ipc_ start_print;

    /// \brief
    /// any printing will stop on this iteration
    ipc_ stop_print;

    /// \brief
    /// the number of iterations between printing
    ipc_ print_gap;

    /// \brief
    /// the minimum number of iterations allowed (-ve = no bound)
    ipc_ itmin;

    /// \brief
    /// the maximum number of iterations allowed (-ve = no bound)
    ipc_ itmax;

    /// \brief
    /// the maximum number of Newton inner iterations per outer iteration
    /// allowed (-ve = no bound)
    ipc_ bitmax;

    /// \brief
    /// the number of extra work vectors of length n used
    ipc_ extra_vectors;

    /// \brief
    /// the stopping rule used: 0=1.0, 1=norm step, 2=norm step/sigma (NOT USED)
    ipc_ stopping_rule;

    /// \brief
    /// frequency for solving the reduced tri-diagonal problem        (NOT USED)
    ipc_ freq;

    /// \brief
    /// the iteration stops successfully when \f$\|A^Tr\|\f$ is less than
    /// max( stop_relative * \f$\|A^Tr_{initial} \|\f$, stop_absolute )
    rpc_ stop_relative;
    /// see stop_relative
    rpc_ stop_absolute;

    /// \brief
    /// an estimate of the solution that gives at least .fraction_opt times
    /// the optimal objective value will be found
    rpc_ fraction_opt;

    /// \brief
    /// the maximum elapsed time allowed (-ve means infinite)
    rpc_ time_limit;

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
struct l2rt_inform_type {

    /// \brief
    /// return status. See \link l2rt_solve_problem \endlink for details
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
    /// the total number of inner iterations performed
    ipc_ biters;

    /// \brief
    /// the smallest number of inner iterations performed during an
    /// outer iteration
    ipc_ biter_min;

    /// \brief
    /// the largest number of inner iterations performed during an
    /// outer iteration
    ipc_ biter_max;

    /// \brief
    /// the value of the objective function
    rpc_ obj;

    /// \brief
    /// the multiplier, \f$\lambda = \mu + \sigma \|x\|^{p-2} * \sqrt{\|Ax-b\|^2 + \mu \|x\|^2}\f$
    rpc_ multiplier;

    /// \brief
    /// the Euclidean norm of \f$x\f$
    rpc_ x_norm;

    /// \brief
    /// the Euclidean norm of \f$Ax-b\f$
    rpc_ r_norm;

    /// \brief
    /// the Euclidean norm of \f$A^T (Ax-b) + \lambda x\f$
    rpc_ Atr_norm;

    /// \brief
    /// the average number of inner iterations performed during an
    /// outer iteration
    rpc_ biter_mean;
};

// *-*-*-*-*-*-*-*-*-*-    L 2 R T  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void l2rt_initialize( void **data,
                     struct l2rt_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see l2rt_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L 2 R T  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void l2rt_read_specfile( struct l2rt_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNL2RT.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/l2rt.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see l2rt_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-    L 2 R T  _ I M P O R T _ C O N T R O L  -*-*-*-*-*-*-

void l2rt_import_control( struct l2rt_control_type *control,
                          void **data,
                          ipc_ *status );

/*!<
 Import control parameters prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see l2rt_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-  L 2 R T _ S O L V E _ P R O B L E M   -*-*-*-*-*-*-*-*-

void l2rt_solve_problem( void **data,
                         ipc_ *status,
                         ipc_ m,
                         ipc_ n,
                         const rpc_ power,
                         const rpc_ weight,
                         const rpc_ shift,
                         rpc_ x[],
                         rpc_ u[],
                         rpc_ v[] );

/*!<
 Solve the regularized-least-squares problem using reverse communication.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n

   This must be set to
   \li  1. on initial entry. Set u (below) to \f$b\f$ for this entry.

   Possible exit values are:
   \li  0. the solution has been found
   \li  2. The user must perform the operation
          \f[u := u + A v,\f]
\manonly
\n
               u := u + A v,
\n
\endmanonly
           and recall the function. The vectors \f$u\f$ and \f$v\f$ are
           available in the arrays u and v (below)
           respectively, and the result \f$u\f$ must overwrite the content of u.
           No argument except u should be altered before recalling the
           function
   \li  3. The user must perform the operation
          \f[v := v + A^T u,\f]
\manonly
\n
               v := v + A^T u,
\n
\endmanonly
           and recall the function. The vectors \f$u\f$ and \f$v\f$ are
           available in the arrays u and v (below)
           respectively, and the result \f$v\f$ must overwrite the content of v.
           No argument except v should be altered before recalling the
           function
   \li  4. The user must reset u (below) to \f$b\f$ are recall the function.
           No argument except u should be altered before recalling the
           function
   \li -1. an array allocation has failed
   \li -2. an array deallocation has failed
   \li -3. one or more of n, m, weight or shift violates allowed bounds
   \li -18. the iteration limit has been exceeded
   \li -25. status is negative on entry

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    equations (i.e., rows of \f$A\f$), \f$m > 0\f$

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables (i.e., columns of \f$A\f$), \f$n > 0\f$

 @param[in] power is a scalar of type rpc_, that holds the
   regularization power, \f$p \geq 2\f$

 @param[in] weight is a scalar of type rpc_, that holds the
   regularization weight, \f$\sigma > 0\f$

 @param[in] shift is a scalar of type rpc_, that holds the shift,
   \f$\mu\f$

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the solution \f$x\f$.
    The j-th component of x, j = 0, ... ,  n-1, contains  \f$x_j \f$.

 @param[in,out] u is a one-dimensional array of size m and type rpc_,
    that should be used and reset appropriately when status = 1 to 5
    as directed by status.

 @param[in,out] v is a one-dimensional array of size n and type rpc_,
    that should be used and reset appropriately when status = 1 to 5
    as directed by status.

*/

// *-*-*-*-*-*-*-*-*-*-    L 2 R T  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void l2rt_information( void **data,
                      struct l2rt_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see l2rt_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L 2 R T  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void l2rt_terminate( void **data,
                    struct l2rt_control_type *control,
                    struct l2rt_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see l2rt_control_type)

  @param[out] inform   is a struct containing output information
              (see l2rt_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example l2rtt.c
   This is an example of how to use the package to solve a regularized quadratic
   problem. The use of default and non-default scaling matrices, and restarts
   with a larger regularization weight are illustrated.
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
