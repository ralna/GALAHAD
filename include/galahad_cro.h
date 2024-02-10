//* \file galahad_cro.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_CRO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package cro

  \section cro_intro Introduction

  \subsection cro_purpose Purpose

  Provides a <b>crossover</b> from a solution
  to the <b>convex quadratic programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = \frac{1}{2} x^T H x + g^T x + f \f]
\manonly
  \n
  minimize q(x) := 1/2 x^T H x + g^T x + f
  \n
\endmanonly
  subject to the general linear constraints
  \f[c_i^l  \leq  a_i^Tx  \leq c_i^u, \;\;\; i = 1, \ldots , m,\f]
\manonly
  \n
   c_i^l \[<=] a_i^Tx \[<=] c_i^u, i = 1, ... , m,
  \n
\endmanonly
  and the simple bound constraints
  \f[x_j^l  \leq  x_j \leq x_j^u, \;\;\; j = 1, \ldots , n,\f]
\manonly
  \n
   x_j^l \[<=] x_j \[<=] x_j^u, j = 1, ... , n,
  \n
\endmanonly
  found by an interior-point method to one in which the
  <b>matrix of defining active constraints/variables is of full rank.</b>
  Here, the \f$n\f$ by \f$n\f$ symmetric, positive-semi-definite matrix
  \f$H\f$, the vectors \f$g\f$, \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$, the scalar \f$f\f$ are given. In addition a solution \f$x\f$ along
  with optimal Lagrange multipliers \f$y\f$ for the general constraints
  and dual variables \f$z\f$ for the simple bounds must be provided
  (see Section~\ref{galmethod}).   These will be adjusted as necessary.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$, \f$x_j^l\f$
  and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix \f$H\f$
  or the matrix \f$A\f$ of vectors \f$a_i\f$.

  \subsection cro_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection cro_date Originally released

  August 2010, C interface January 2022.

  \subsection cro_terminology Terminology

  Any required solution \f$x\f$ necessarily satisfies
  the primal optimality conditions
  \f[\mbox{(1a) $\hspace{66mm} A x = c\hspace{66mm}$}\f]
\manonly
  \n
  (1a) A x = c
  \n
\endmanonly
  and
  \f[\mbox{(1b) $\hspace{52mm} c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u,\hspace{52mm}$} \f]
\manonly
  \n
  (1b) c^l \[<=] c \[<=] c^u, x^l \[<=] x \[<=] x^u,
  \n
\endmanonly
  the dual optimality conditions
  \f[\mbox{(2a) $\hspace{58mm} H x + g = A^T y + z\hspace{58mm}$}\f]
\manonly
  \n
  (2a) H x + g = A^T y + z
  \n
\endmanonly
  where
  \f[\mbox{(2b) $\hspace{24mm} y = y^l + y^u, \;\; z = z^l + z^u, \,\,
   y^l \geq 0 , \;\;  y^u \leq 0 , \;\;
   z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,\hspace{24mm}$} \f]
\manonly
  \n
   (2b) y = y^l + y^u, z = z^l + z^u, y^l \[>=] 0, y^u \[<=] 0,
        z^l \[>=] 0 and z^u \[<=] 0,
  \n
\endmanonly
  and the complementary slackness conditions
  \f[\mbox{(3) $\hspace{12mm}
  ( A x - c^l )^T y^l = 0  ,\;\;  ( A x - c^u )^T y^u = 0  ,\;\;
  (x -x^l )^T z^l = 0 \;\;  \mbox{and} \;\; (x -x^u )^T z^u = 0,\hspace{12mm} $}\f]
\manonly
  \n
  (3) (A x - c^l)^T y^l = 0, (A x - c^u)^T y^u = 0,
      (x -x^l)^T z^l = 0 and (x -x^u)^T z^u = 0,
  \n
\endmanonly
  where the vectors \f$y\f$ and \f$z\f$ are known as the Lagrange multipliers
  for the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  \subsection cro_method Method

  Denote the active constraints by \f$A_A x = c_A\f$ and the active bounds by
  \f$I_A x = x_A\f$. Then any optimal solution satisfies the linear system
  \f[\left(\begin{array}{ccc}H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 \end{array}\right) \left(\begin{array}{c}x \\ y_A \\ z_A\end{array}\right) =
\left(\begin{array}{c}- g \\ c_A \\ x_A\end{array}\right).\f]
\manonly
  \n
       ( H   - A_A^T - I_A^T ) (  x  )   ( - g )
       ( A_A     0       0   ) ( y_A ) = ( c_A ),
       ( I_A     0       0   ) ( z_A )   ( x_A )
  \n
\endmanonly
  where \f$y_A\f$ and \f$z_A\f$ are the corresponding active Lagrange
  multipliers and dual variables respectively. Consequently the difference
  between any two solutions \f$(\Delta x, \Delta y, \Delta z)\f$ must satisfy
  \f[\mbox{(4)}\;\; \left(\begin{array}{ccc}H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 \end{array}\right) \left(\begin{array}{c}\Delta x \\ \Delta y_A \\ \Delta z_A\end{array}\right) = 0.\f]
\manonly
  \n
          ( H   - A_A^T - I_A^T ) (  Delta x  )
    (4)   ( A_A     0       0   ) ( Delta y_A ) = 0,
          ( I_A     0       0   ) ( Delta z_A )
  \n
\endmanonly
  Thus there can only be multiple solution if the coefficient matrix \f$K\f$
  of (4) is singular. The algorithm used in CRO
  exploits this. The matrix \f$K\f$ is checked for singularity
  using the GALAHAD package ULS. If \f$K\f$ is
  non singular, the solution is unique and the solution input by the user
  provides a linearly independent active set. Otherwise \f$K\f$ is singular,
  and partitions \f$A_A^T = ( A_B^T \;\; A_N^T)\f$ and
  \f$I_A^T = ( I_B^T \;\; I_N^T)\f$ are found so that
  \f[\left(\begin{array}{ccc}H & - A_B^T & - I_B^T \\ A_B & 0 & 0 \\ I_B & 0 & 0 \end{array}\right)\f]
\manonly
  \n
       ( H   - A_B^T - I_B^T )
       ( A_B     0       0   )
       ( I_B     0       0   )
  \n
\endmanonly
  is non-singular and the <b>non-basic</b> constraints \f$A_N^T\f$
  and \f$I_N^T\f$ are linearly dependent on the <b>basic</b> ones
  \f$( A_B^T \;\; I_B^T)\f$. In this case (4) is equivalent to
  \f[\mbox{(5)}\;\; \left(\begin{array}{ccc}H & - A_B^T & - I_B^T \\ A_B & 0 & 0 \\ I_B & 0 & 0 \end{array}\right) = \left(\begin{array}{c}A_N^T \\ 0 \\ 0\end{array}\right) \Delta y_N + \left(\begin{array}{c}I_N^T \\ 0 \\ 0\end{array}\right) \Delta z_N\f]
\manonly
  \n
          ( H   - A_B^T - I_B^T ) (  Delta x  )
  (5)     ( A_B     0       0   ) ( Delta y_A ) =
          ( I_B     0       0   ) ( Delta z_A )

            ( A_N^T )             ( I_N^T )
            (   0   ) Delta y_N + (   0   ) Delta z_N.
            (   0   )             (   0   )
  \n
\endmanonly
  Thus, starting from the user's \f$(x, y, z)\f$
  and with a factorization of the coefficient matrix of (5)
  found by the GALAHAD package SLS, the alternative solution
  \f$(x + \alpha x, y + \alpha y, z + \alpha z)\f$,
  featuring
  \f$(\Delta x, \Delta y_B, \Delta z_B)\f$
  from (5)  in which successively one of the components of \f$\Delta y_N\f$
  and \f$\Delta z_N\f$ in turn is non zero, is taken.
  The scalar \f$\alpha\f$ at each stage
  is chosen to be the largest possible that guarantees (2.b);
  this may happen when a non-basic multiplier/dual variable reaches zero,
  in which case the corresponding constraint is disregarded, or when this
  happens for a basic multiplier/dual variable, in which case this constraint is
  exchanged with the non-basic one under consideration and disregarded.
  The latter corresponds to changing the basic-non-basic partition
  in (5), and subsequent solutions may be found by updating
  the factorization of the coefficient matrix in (5)
  following the basic-non-basic swap using the GALAHAD package SCU.

  \subsection cro_references Reference

  \subsection cro_call_order Call order

  To solve a given problem, functions from the cro package must be called
  in the following order:

  - \link cro_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link cro_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link cro_crossover_solution \endlink - move from a primal-dual soution
      to a full rank one
  - \link cro_terminate \endlink - deallocate data structures

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
#ifndef GALAHAD_CRO_H
#define GALAHAD_CRO_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "galahad_sbls.h"
#include "galahad_uls.h"
#include "galahad_ir.h"
#include "galahad_scu.h"

/**
 * control derived type as a C struct
 */
struct cro_control_type {

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
    /// the maximum permitted size of the Schur complement before a
    /// refactorization is performed
    ipc_ max_schur_complement;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// feasibility tolerance for KKT violation
    rpc_ feasibility_tolerance;

    /// \brief
    /// if .check_io is true, the input (x,y,z) will be fully tested for
    /// consistency
    bool check_io;

    /// \brief
    /// if .refine solution is true, attempt to satisfy the KKT conditions as
    /// accurately as possible
    bool refine_solution;

    /// \brief
    /// if .space_critical is true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// indefinite linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// unsymmetric linear equation solver
    char unsymmetric_linear_solver[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for SLS
    struct sls_control_type sls_control;

    /// \brief
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;

    /// \brief
    /// control parameters for ULS
    struct uls_control_type uls_control;

    /// \brief
    /// control parameters for iterative refinement
    struct ir_control_type ir_control;
};

/**
 * time derived type as a C struct
 */
struct cro_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the CPU time spent reordering the matrix prior to factorization
    real_sp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_sp_ factorize;

    /// \brief
    /// the CPU time spent computing corrections
    real_sp_ solve;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
    rpc_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    rpc_ clock_factorize;

    /// \brief
    /// the clock time spent computing corrections
    rpc_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct cro_inform_type {

    /// \brief
    /// return status. See CRO_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the number of dependent active constraints
    ipc_ dependent;

    /// \brief
    /// timings (see above)
    struct cro_time_type time;

    /// \brief
    /// information from SLS
    struct sls_inform_type sls_inform;

    /// \brief
    /// information from SBLS
    struct sbls_inform_type sbls_inform;

    /// \brief
    /// information from ULS
    struct uls_inform_type uls_inform;

    /// \brief
    /// information from SCU
    ipc_ scu_status;
    /// see scu_status
    struct scu_inform_type scu_inform;

    /// \brief
    /// information from IR
    struct ir_inform_type ir_inform;
};

// *-*-*-*-*-*-*-*-*-*-    C R O  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void cro_initialize( void **data,
                     struct cro_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see cro_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-    C R O  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void cro_read_specfile( struct cro_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNCRO.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/cro.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see cro_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

void cro_crossover_solution( void **data,
                             struct cro_control_type *control,
                             struct cro_inform_type *inform,
                             ipc_ n,
                             ipc_ m,
                             ipc_ m_equal,
                             ipc_ h_ne,
                             const rpc_ H_val[],
                             const ipc_ H_col[],
                             const ipc_ H_ptr[],
                             ipc_ a_ne,
                             const rpc_ A_val[],
                             const ipc_ A_col[],
                             const ipc_ A_ptr[],
                             const rpc_ g[],
                             const rpc_ c_l[],
                             const rpc_ c_u[],
                             const rpc_ x_l[],
                             const rpc_ x_u[],
                             rpc_ x[],
                             rpc_ c[],
                             rpc_ y[],
                             rpc_ z[],
                             ipc_ x_stat[],
                             ipc_ c_stat[] );

/*!<
 Crosover the solution from a primal-dual to a basic one.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see cro_control_type). The
  parameter .status is as follows:

 @param[in,out] data holds private internal data.

 @param[out] inform is a struct containing output information
    (see cro_inform_type). The component .status gives
    the exit status from the package. Possible values are:
  \li  0. The crossover was succesful.
  \li -1. An allocation error occurred. A message indicating the
       offending array is written on unit control.error, and the
       returned allocation status and a string containing the name
       of the offending array are held in inform.alloc_status and
       inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the
       offending array is written on unit control.error and the
       returned allocation status and a string containing the
       name of the offending array are held in
       inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 or m >= m_equal >= 0 has been violated.
  \li -4 the bound constraints are inconsistent.
  \li -5 the general constraints are likely inconsistent.
  \li -9 an error has occured in SLS_analyse.
  \li -10 an error has occured in SLS_factorize.
  \li -11 an error has occured in SLS_solve.
  \li -12 an error has occured in ULS_factorize.
  \li -14 an error has occured in ULS_solve.
  \li -16 the residuals are large; the factorization may be unsatisfactory.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
   variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
   general linear constraints.

 @param[in] m_equal is a scalar variable of type ipc_, that holds the number of
   general linear equality constraints. Such constraints must occur first in
   \f$A\f$.

 @param[in] h_ne is a scalar variable of type ipc_, that holds the number of
   entries in the <b>lower triangular</b> part of the Hessian matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of type rpc_,
   that holds the values of the entries of the lower triangular part
   of the Hessian matrix \f$H\f$. The entries are stored by consecutive rows,
   the order within each row is unimportant.

 @param[in]  H_col is a one-dimensional array of type ipc_,
   that holds the column indices of the lower triangular part of \f$H\f$,
   in the same order as those in H_val.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$. The n+1-st component holds the total number of
   entries (plus one if fortran indexing is used).

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
   entries in the constraint Jacobian matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of type rpc_,
   that holds the values of the entries of the constraint Jacobian matrix
   \f$A\f$. The entries are stored by consecutive rows,
   the order within each row is unimportant. <b>Equality constraints must be
   ordered first.</b>

 @param[in]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of \f$A\f$ in the same order as those in A_val.

 @param[in]  A_ptr is a one-dimensional array of size m+1 and type ipc_,
   that holds the starting position of each row of \f$A\f$. The
   m+1-st component holds the total number of  entries (plus one if fortran
   indexing is used).

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] c_l is a one-dimensional array of size m and type rpc_, that
    holds the lower bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_l, i = 0, ... ,  m-1, contains  \f$c^l_i\f$.

 @param[in] c_u is a one-dimensional array of size m and type rpc_, that
    holds the upper bounds \f$c^l\f$ on the constraints \f$A x\f$.
    The i-th component of c_u, i = 0, ... ,  m-1, contains  \f$c^u_i\f$.

 @param[in] x_l is a one-dimensional array of size n and type rpc_, that
    holds the lower bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in] x_u is a one-dimensional array of size n and type rpc_, that
    holds the upper bounds \f$x^l\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$x^l_j\f$.

 @param[in,out] x is a one-dimensional array of size n and type rpc_, that
    holds the values \f$x\f$ of the optimization variables. The j-th component
    of x, j = 0, ... , n-1, contains \f$x_j\f$.

 @param[in,out] c is a one-dimensional array of size m and type rpc_, that
    holds the residual \f$c(x) = A x\f$.
    The i-th component of c, j = 0, ... ,  n-1, contains  \f$c_j(x) \f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the general
    linear constraints. The j-th component
    of y, j = 0, ... , n-1, contains \f$y_j\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[in,out] x_stat is a one-dimensional array of size n and type ipc_, that
    must be set on entry to give the status of the problem variables.
    If x_stat(j) is negative, the variable \f$x_j\f$ is active on its lower
    bound, if it is  positive, it is active and lies on its upper bound, and
    if it is zero, it is inactiive and lies between its bounds. On exit, the
    \f$j\f$-th component of x_stat is -1 if the variable is basic and active on
    its lower bound, -2 it is non-basic but active on its lower bound, 1 if it
    is basic and active on its upper bound, 2 it is non-basic but active on its
    upper bound, and 0 if it is inactive.

 @param[in,out] c_stat is a one-dimensional array of size m and type ipc_, that
    must be set on entry to give the status of the general linear constraints.
    If c_stat(i) is negative, the constraint value \f$a_i^Tx\f$ is active on its
    lower  bound, if it is positive, it is active and lies on its upper bound,
    and if it is zero, it is inactiive and lies between its bounds. On exit,
    the \f$i\f$-th component of x_stat is -1 if the constraint is basic and
    active on its lower bound, -2 it is non-basic but active on its lower
    bound, 1 if it is basic and active on its upper bound, 2 it is non-basic
    but active on its upper bound, and 0 if it is inactive.
*/

// *-*-*-*-*-*-*-*-*-*-    C R O  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void cro_terminate( void **data,
                    struct cro_control_type *control,
                    struct cro_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see cro_control_type)

  @param[out] inform   is a struct containing output information
              (see cro_inform_type)
 */

/** \example crot.c
   This is an example of how to use the package.\n
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example crot.c
   This is an example of how to use the package to solve a quadratic program.
   A variety of supported Hessian and constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example crotf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
