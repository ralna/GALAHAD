//* \file galahad_lstr.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LSTR C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package lstr

  \section lstr_intro Introduction

  \subsection lstr_purpose Purpose

  Given a real \f$m\f$ by \f$n\f$ matrix \f$A\f$, a real
  \f$m\f$ vector \f$b\f$ and a scalar \f$\Delta>0\f$, this package finds an
  <b> approximate minimizer of \f$\| A x - b\|_2\f$, where the vector
  \f$x\f$ is required to satisfy the ``trust-region''
  constraint \f$\|x\|_2 \leq  \Delta\f$.</b>
  This problem commonly occurs as a trust-region subproblem in nonlinear
  optimization calculations, and may be used to regularize the solution
  of under-determined or ill-conditioned linear least-squares problems.
  The method may be suitable for large \f$m\f$ and/or \f$n\f$ as no
  factorization involving \f$A\f$ is required. Reverse communication is used
  to obtain matrix-vector products of the form \f$u + A v\f$ and
  \f$v + A^T u\f$.

  \subsection lstr_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection lstr_date Originally released

  November 2007, C interface December 2021.

  \subsection lstr_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies the optimality condition
  \f$A^T ( A x - b ) + \lambda x = 0\f$, where \f$\lambda \geq 0\f$
  is a Lagrange multiplier corresponding to the trust-region constraint
  \f$\|x\|_2  \leq  \Delta\f$.

  \subsection lstr_method Method

  The method is iterative. Starting  with the vector \f$u_1 = b\f$, a
  bi-diagonalisation process is used to generate the vectors \f$v_k\f$ and
  \f$u_k+1\f$ so that the \f$n\f$ by \f$k\f$ matrix \f$V_k = ( v_1 \ldots v_k)\f$
  and the \f$m\f$ by \f$(k+1)\f$ matrix \f$U_k = ( u_1 \ldots u_{k+1})\f$
  together satisfy
  \f[A V_k = U_{k+1} B_k \;\mbox{and}\; b = \|b\| U_{k+1} e_1,\f]
\manonly
\n
       A V_k = U_{k+1} B_k and b = ||b|| U_{k+1} e_1,
\n
\endmanonly
  where \f$B_k\f$ is \f$(k+1)\f$ by \f$k\f$ and lower bi-diagonal, \f$U_k\f$ and
  \f$V_k\f$ have orthonormal columns and \f$e_1\f$ is the first unit vector.
  The solution sought is of the form \f$x_k = V_k y_k\f$, where \f$y_k\f$
  solves the bi-diagonal least-squares trust-region problem
  \f[(1) \;\;\; \min \| B_k y - \|b\| e_1 \|_2 \;\mbox{subject to}\; \|y\|_2 \leq \Delta.\f]
\manonly
\n
   (1)    min || B_k y - \|b\| e_1 ||_2 subject to ||y||_2 <= Delta.
\n
\endmanonly

  If the trust-region constraint is inactive, the solution \f$y_k\f$
  may be found, albeit indirectly, via the LSQR algorithm of Paige and Saunders
  which solves the bi-diagonal least-squares problem
  \f[ \min \| B_k y - \|b\| e_1 \|_2\f]
\manonly
\n
     min || B_k y - ||b|| e_1 ||_2
\n
\endmanonly
  using a QR factorization of \f$B_k\f$. Only the most recent
  \f$v_k\f$ and \f$u_{k+1}\f$
  are required, and their predecessors discarded, to compute \f$x_k\f$ from
  \f$x_{k-1}\f$. This method has the important property that the iterates
  \f$y\f$ (and thus \f$x_k\f$) generated increase in norm with \f$k\f$. Thus as
  soon as an LSQR iterate lies outside the trust-region, the required solution
  to (1) and thus to the original problem must lie on the boundary of the
  trust-region.

  If the solution is so constrained, the simplest strategy is to interpolate
  the last interior iterate with the newly discovered exterior one to find the
  boundary point---the so-called Steihaug-Toint point---between them.
  Once the solution is known to lie on the trust-region boundary,
  further improvement may be made by solving
  \f[ \min \| B_k y - \|b\| e_1 \|_2 \;\mbox{subject to}\;  |\|y\|_2 = \Delta,\f]
\manonly
\n
     min || B_k y - ||b|| e_1 ||_2 subject to ||y||_2 = Delta,
\n
\endmanonly
  for which the optimality conditions require that \f$y_k = y(\lambda_k)\f$
  where \f$\lambda_k\f$ is the positive root of
  \f[B_k^T ( B_k^{} y(\lambda) - \|b\| e_1^{} ) + \lambda  y(\lambda) = 0 \;\mbox{and}\;  \|y(\lambda)\|_2 = \Delta\f]
\manonly
\n
      B_k^T ( B_k y(lambda) - ||b|| e_1 ) + lambda y(lambda) = 0
      and ||y(lambda)||_2 = Delta
\n
\endmanonly
  The vector \f$y(\lambda)\f$ is equivalently the solution to the
  regularized least-squares problem
  \f[\min  \left \| \vect{ B_k \\ \lambda^{\frac{1}{2}} I } y - \|b\| e_1^{} \right \|\f]
\manonly
\n
      min  ||  B_k y - ||b|| e_1 ||
           ||    lambda^{1/2} y  ||
\n
\endmanonly
  and may be found efficiently. Given  \f$y(\lambda)\f$, Newton's method
  is then used to find \f$\lambda_k\f$ as the positive root of
  \f$\|y(\lambda)\|_2 = \Delta\f$. Unfortunately, unlike when the solution
  lies in the interior of the trust-region, it is not known how to recur
  \f$x_k\f$ from \f$x_{k-1}\f$ given \f$y_k\f$, and a second pass in which
  \f$x_k = V_k y_k\f$ is regenerated is needed---this need only be done
  once \f$x_k\f$ has implicitly deemed to be sufficiently close to optimality.
  As this second pass is an additional expense, a record is kept of the
  optimal objective function values for each value of \f$k\f$, and the second
  pass is only performed so far as to ensure a given fraction of the
  final optimal objective value. Large savings may be made in the second
  pass by choosing the required fraction to be significantly smaller than one.

  \subsection lstr_references Reference

  A complete description of the unconstrained case is given by

  C. C. Paige and M. A. Saunders,
  LSQR: an algorithm for sparse linear equations and sparse least  squares.
  ACM Transactions on Mathematical Software, 8(1):43--71, 1982

  and

  C. C. Paige and M. A. Saunders,
  ALGORITHM 583: LSQR: an algorithm for sparse linear equations and
    sparse least squares.
  ACM Transactions on Mathematical Software, 8(2):195--209, 1982.

  Additional details on how to proceed once the trust-region constraint is
  encountered are described in detail in

  C. Cartis, N. I. M. Gould and Ph. L. Toint,
  Trust-region and other regularisation of linear
  least-squares problems.
  BIT 49(1):21-53 (2009).

  \subsection lstr_call_order Call order

  To solve a given problem, functions from the lstr package must be called
  in the following order:

  - \link lstr_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link lstr_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link lstr_import_control \endlink - import control parameters prior to
      solution
  - \link lstr_solve_problem \endlink - solve the problem by reverse
      communication, a sequence of calls are made under control of a status
      parameter, each exit either asks the user to provide additional
      informaton and to re-enter, or reports that either the solution has
      been found or that an error has occurred
  - \link lstr_information \endlink (optional) - recover information about
    the solution and solution process
  - \link lstr_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_LSTR_H
#define GALAHAD_LSTR_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct lstr_control_type {

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
    /// the maximum number of iterations allowed once the boundary has been
    /// encountered (-ve = no bound)
    ipc_ itmax_on_boundary;

    /// \brief
    /// the maximum number of Newton inner iterations per outer iteration allowe
    /// (-ve = no bound)
    ipc_ bitmax;

    /// \brief
    /// the number of extra work vectors of length n used
    ipc_ extra_vectors;

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
    /// should the iteration stop when the Trust-region is first encountered?
    bool steihaug_toint;

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
struct lstr_inform_type {

    /// \brief
    /// return status. See \link lstr_solve_problem \endlink for details
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
    /// the total number of pass-2 iterations required if the solution lies on
    /// the trust-region boundary
    ipc_ iter_pass2;

    /// \brief
    /// the total number of inner iterations performed
    ipc_ biters;

    /// \brief
    /// the smallest number of inner iterations performed during an
    /// outer iteration
    ipc_ biter_min;

    /// \brief
    /// the largestt number of inner iterations performed during an
    /// outer iteration
    ipc_ biter_max;

    /// \brief
    /// the Lagrange multiplier, \f$\lambda\f$, corresponding to the
    /// trust-region constraint
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
    /// the average number of inner iterations performed during an outer
   /// iteration
    rpc_ biter_mean;
};

// *-*-*-*-*-*-*-*-*-*-    L S T R  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void lstr_initialize( void **data,
                     struct lstr_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lstr_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
    \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L S T R  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void lstr_read_specfile( struct lstr_control_type *control,
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNLSTR.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/lstr.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see lstr_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-    L S T R  _ I M P O R T _ C O N T R O L  -*-*-*-*-*-*-

void lstr_import_control( struct lstr_control_type *control,
                          void **data,
                          ipc_ *status );

/*!<
 Import control parameters prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lstr_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*-*-*-*-  L S T R _ S O L V E _ P R O B L E M   -*-*-*-*-*-*-*-*-

void lstr_solve_problem( void **data,
                         ipc_ *status,
                         ipc_ m,
                         ipc_ n,
                         const rpc_ radius,
                         rpc_ x[],
                         rpc_ u[],
                         rpc_ v[] );

/*!<
 Solve the trust-region least-squares problem using reverse communication.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the entry and exit status from the package. \n

   This must be set to
   \li  1. on initial entry. Set u (below) to \f$b\f$ for this entry.
   \li  5. the iteration is to be restarted with a smaller radius but
         with all other data unchanged. Set u (below) to \f$b\f$ for this entry.

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
   \li -3. one or more of n, m or weight violates allowed bounds
   \li -18. the iteration limit has been exceeded
   \li -25. status is negative on entry

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    equations (i.e., rows of \f$A\f$), \f$m > 0\f$

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables (i.e., columns of \f$A\f$), \f$n > 0\f$

 @param[in] radius is a scalar of type rpc_, that holds the
   trust-region radius, \f$\Delta > 0\f$

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

// *-*-*-*-*-*-*-*-*-*-    L S T R  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lstr_information( void **data,
                      struct lstr_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lstr_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L S T R  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void lstr_terminate( void **data,
                    struct lstr_control_type *control,
                    struct lstr_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see lstr_control_type)

  @param[out] inform   is a struct containing output information
              (see lstr_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example lstrt.c
   This is an example of how to use the package to solve a trust-region
   problem. The use of default and non-default scaling matrices, and restarts
   with a smaller trust-region radius are illustrated.

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
