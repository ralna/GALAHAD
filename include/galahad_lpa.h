//* \file galahad_lpa.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LPA C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 7th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */


/*! \mainpage GALAHAD C package lpa

  \section lpa_intro Introduction

  \subsection lpa_purpose Purpose

  This package uses the <b> simplex method</b>
  to solve the <b>linear programming problem</b>
  \f[\mbox{minimize}\;\; q(x) = g^T x + f \f]
\manonly
  \n
  minimize q(x) := g^T x + f
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
  where the vectors \f$g\f$, \f$w\f$, \f$x^{0}\f$,
  \f$a_i\f$, \f$c^l\f$, \f$c^u\f$, \f$x^l\f$,
  \f$x^u\f$ and the scalar \f$f\f$ are given.
  Any of the constraint bounds \f$c_i^l\f$, \f$c_i^u\f$,
  \f$x_j^l\f$ and \f$x_j^u\f$ may be infinite.
  Full advantage is taken of any zero coefficients in the matrix
  \f$A\f$ whose rows are the transposes of the vectors \f$a_i\f$.

  <b>N.B.</b> The package is simply a sophisticated interface to the
  HSL package LA04, and requires that a user has obtained the latter.
  <b> LA04 is not included in GALAHAD</b>
  but is available without charge to recognised academics, see
  http://www.hsl.rl.ac.uk/catalogue/la04.html. If LA04
  is unavailable, the GALAHAD interior-point linear programming
  package LPB is an alternative.

  \subsection lpa_authors Authors

  N. I. M. Gould and J. K. Reid, STFC-Rutherford Appleton Laboratory,
  England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection lpa_date Originally released

  October 2018, C interface September 2021.

  \subsection lpa_terminology Terminology

  The required solution \f$x\f$ necessarily satisfies
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
  \f[\mbox{(2a) $\hspace{3mm} g = A^T y + z$}\f]
\manonly
  \n
  (2a) g = A^T y + z
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
  where the vectors \f$y\f$ and \f$z\f$ are
  known as the Lagrange multipliers for
  the general linear constraints, and the dual variables for the bounds,
  respectively, and where the vector inequalities hold component-wise.

  The so-called dual to this problem is another linear program
  \f[- \mbox{minimize} \;\; c^{lT} y^l + c^{uT} y^u + x^{lT} z^l + x^{uT} z^u + f \;\; \mbox{subject to the constraints (2a) and (2b)}\f]
\manonly
  \n
  - minimize c^{lT} y^l + c^{uT} y^u + x^{lT} z^l + x^{uT} z^u + f
  subject to the constraints (2a) and (2b)
  \n
\endmanonly
  that uses the same data. The solution to the two problems, it is exists,
  is the same, but if one is infeasible, the other is unbounded. It can be
  more efficient to solve the dual, particularly if \f$m\f$ is much larger
  than \f$n\f$.

  \subsection lpa_method Method

  The bulk of the work is peformed by the HSL package LA04. The
  main subbroutine from this package requires that the input problem
  be transformed into the ``standard form''
\latexonly
  \[  \begin{array}{rl}\mbox{minimize} & g^{\prime T} x^{\prime} \\(4)  \;\; \mbox{subject to} & A^{\prime} x^{\prime} = b \\ &  l_i \leq x^{\prime}_i \leq u_i, \;\;  (i\leq k) \\ \mbox{and} & x^{\prime}_l \geq 0, \;\; (i \geq l) \end{array} \]
\endlatexonly
\htmlonly
  \f[   \begin{array}{rl}\mbox{minimize} & g^{\prime T} x^{\prime} \\(4)  \;\; \mbox{subject to} & A^{\prime} x^{\prime} = b \\ &  l_i \leq x^{\prime}_i \leq u_i, \;\;  (i\leq k) \\ \mbox{and} & x^{\prime}_l \geq 0, \;\; (i \geq l) \end{array} \f]
\endhtmlonly
\manonly
  \n
         minimize g'^T x'
  (4)  subject to A' x' = b
                  l_i <= x'_i <= u_i, for i <= k
              and x'_l >= 0, for i >= l
  \n
\endmanonly
  by introducing slack an surpulus variables, reordering and
  removing fixed variables and free constraints. The resulting
  problem involves \f$n'\f$ unknowns and \f$m'\f$ general constraints.
  In order to deal with the possibility that the general constraints
  are inconsistent or not of full rank,
  LA04 introduces additional ``artifical'' variables \f$v\f$ and replaces
  the constraints of (4) by
  \f[(5) \;\; A' x' + v = b\f]
  and gradually encourages \f$v\f$ to zero as a first solution phase.

  Once a selection of \f$m'\f$ independent <b>non-basic</b> variables
  is made, the constraints (5) determine the remaining \f$m'\f$
  dependent <b>basic</b> variables. The <b>simplex method</b> is a
  scheme for systematically adjusting the choice of basic and non-basic
  variables until a set which defines an optimal solution of (4) is
  obtained. Each iteration of the simplex method requires the solution
  of a number of sets of linear equations whose coefficient matrix is
  the <b>basis</b> matrix \f$B\f$, made up of the columns of
  \f$[A'\f$  \f$I]\f$ corresponding to the basic variables, or its transpose
  \f$B^T\f$. As the basis matrices for consecutive iterations are
  closely related, it is normally advantageous to update (rather than
  recompute) their factorizations as the computation proceeds.  If an
  initial basis is not provided by the user, a set of basic variables
  which provide a (permuted) triangular basis matrix is found by the
  simple crash algorithm of Gould and Reid (1989), and initial
  steepest-edge weights are calculated.

  Phases one (finding a feasible solution) and two (solving (4)
  of the simplex method are applied, as appropriate, with the choice of
  entering variable as described by Goldfarb and Reid (1977) and the
  choice of leaving variable as proposed by Harris (1973).
  Refactorizations of the basis matrix are performed whenever doing so
  will reduce the average iteration time or there is insufficient memory
  for its factors.  The reduced cost for the entering variable is
  computed afresh. If it is found to be of a different sign from the
  recurred value or more than 10\% different in magnitude, a fresh
  computation of all the reduced costs is performed.  Details of the
  factorization and updating procedures are given by Reid (1982).
  Iterative refinement is encouraged for the basic solution and for the
  reduced costs after each factorization of the basis matrix and when
  they are recomputed at the end of phase 1.

  \subsection lpa_references References

  D. Goldfarb and J. K. Reid (1977).
  A practicable steepest-edge simplex algorithm.
  Mathematical Programming <b>12</b> 361-371.

  N. I. M. Gould and J. K. Reid (1989)
  New crash procedures for large systems of linear constraints.
  Mathematical Programming <b>45</b> 475-501.

  P. M. J. Harris (1973).
  Pivot selection methods of the Devex LP code.
  Mathematical Programming <b>5</b> 1-28.

  J. K. Reid (1982)
  A sparsity-exploiting variant of the Bartels-Golub
  decomposition for linear-programming bases.
  Mathematical Programming <b>24</b> 55-69.

  \subsection lpa_call_order Call order

  To solve a given problem, functions from the lpa package must be called
  in the following order:

  - \link lpa_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link lpa_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link lpa_import \endlink - set up problem data structures and fixed
      values
  - \link lpa_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link lpa_solve_lp \endlink - solve the linear program
  - \link lpa_information \endlink (optional) - recover information about
    the solution and solution process
  - \link lpa_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \subsection main_unsymmetric_matrices Unsymmetric matrix storage formats

  The unsymmetric \f$m\f$ by \f$n\f$ constraint matrix \f$A\f$ may be presented
  and stored in a variety of convenient input formats.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense Dense storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays A_row and
  A_col and real array A_val, respectively, while the number of nonzeros
  is recorded as A_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A\f$ the i-th component of the
  integer array A_ptr holds the position of the first entry in this row,
  while A_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array A_col, and real array A_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_LPA_H
#define GALAHAD_LPA_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_rpd.h"

/**
 * control derived type as a C struct
 */
struct lpa_control_type {

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
    /// (>= 2 turns on LA04 output)
    ipc_ print_level;

    /// \brief
    /// any printing will start on this iteration
    ipc_ start_print;

    /// \brief
    /// any printing will stop on this iteration
    ipc_ stop_print;

    /// \brief
    /// at most maxit inner iterations are allowed
    ipc_ maxit;

    /// \brief
    /// maximum number of iterative refinements allowed
    ipc_ max_iterative_refinements;

    /// \brief
    /// initial size for real array for the factors and other data
    ipc_ min_real_factor_size;

    /// \brief
    /// initial size for integer array for the factors and other data
    ipc_ min_integer_factor_size;

    /// \brief
    /// the initial seed used when generating random numbers
    ipc_ random_number_seed;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    ipc_ sif_file_device;

    /// \brief
    /// specifies the unit number to write generated QPLIB file describing the
    /// current problem
    ipc_ qplib_file_device;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    rpc_ infinity;

    /// \brief
    /// the tolerable relative perturbation of the data (A,g,..) defining
    /// the problem
    rpc_ tol_data;

    /// \brief
    /// any constraint violated by less than feas_tol will be considered to be
    /// satisfied
    rpc_ feas_tol;

    /// \brief
    /// pivot threshold used to control the selection of pivot elements in the
    /// matrix factorization. Any potential pivot which is less than the largest
    /// entry in its row times the threshold is excluded as a candidate
    rpc_ relative_pivot_tolerance;

    /// \brief
    /// limit to control growth in the upated basis factors. A refactorization
    /// occurs if the growth exceeds this limit
    rpc_ growth_limit;

    /// \brief
    /// any entry in the basis smaller than this is considered zero
    rpc_ zero_tolerance;

    /// \brief
    /// any solution component whose change is smaller than a tolerence times
    /// the largest change may be considered to be zero
    rpc_ change_tolerance;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer
    /// than identical_bounds_tol will be reset to the average of their values
    rpc_ identical_bounds_tol;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    rpc_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    rpc_ clock_time_limit;

    /// \brief
    /// if .scale is true, the problem will be automatically scaled prior to
    /// solution. This may improve computation time and accuracy
    bool scale;

    /// \brief
    /// should the dual problem be solved rather than the primal?
    bool dual;

    /// \brief
    /// should a warm start using the data in C_stat and X_stat be attempted?
    bool warm_start;

    /// \brief
    /// should steepest-edge weights be used to detetrmine the variable
    /// leaving the basis?
    bool steepest_edge;

    /// \brief
    /// if .space_critical is true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if .generate_sif_file is .true. if a SIF file describing the current
    /// problem is to be generated
    bool generate_sif_file;

    /// \brief
    /// if .generate_qplib_file is .true. if a QPLIB file describing the current
    /// problem is to be generated
    bool generate_qplib_file;

    /// \brief
    /// name of generated SIF file containing input problem
    char sif_file_name[31];

    /// \brief
    /// name of generated QPLIB file containing input problem
    char qplib_file_name[31];

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/**
 * time derived type as a C struct
 */
struct lpa_time_type {

    /// \brief
    /// the total CPU time spent in the package
    rpc_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    rpc_ preprocess;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    rpc_ clock_preprocess;
};

/**
 * inform derived type as a C struct
 */
struct lpa_inform_type {

    /// \brief
    /// return status. See LPA_solve for details
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
    /// the final value of la04's job argument
    ipc_ la04_job;

    /// \brief
    /// any extra information from an unsuccesfull call to LA04 (LA04's
    /// RINFO(35)
    ipc_ la04_job_info;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by LPA_solve
    rpc_ obj;

    /// \brief
    /// the value of the primal infeasibility
    rpc_ primal_infeasibility;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// the information array from LA04
    rpc_ RINFO[40];

    /// \brief
    /// timings (see above)
    struct lpa_time_type time;

    /// \brief
    /// inform parameters for RPD
    struct rpd_inform_type rpd_inform;
};

// *-*-*-*-*-*-*-*-*-*-    L P A  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void lpa_initialize( void **data,
                     struct lpa_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see lpa_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
    \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L P A  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void lpa_read_specfile( struct lpa_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNLPA.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/lpa.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see lpa_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    L P A  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void lpa_import( struct lpa_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char A_type[],
                 ipc_ A_ne,
                 const ipc_ A_row[],
                 const ipc_ A_col[],
                 const ipc_ A_ptr[] );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lpa_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful
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
  \li -3. The restrictions n > 0 or m > 0 or requirement that A_type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
        has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in]  A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the constraint Jacobian, \f$A\f$. It should be one of 'coordinate',
  'sparse_by_rows' or 'dense; lower or upper case variants are allowed.

 @param[in]  A_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes.

 @param[in]  A_row is a one-dimensional array of size A_ne and type ipc_, that
   holds the row indices of \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes,
   and in this case can be NULL.

 @param[in]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of \f$A\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  A_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each row of \f$A\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.
*/


//  *-*-*-*-*-*-*-*-*-   L P A _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void lpa_reset_control( struct lpa_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lpa_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

//  -*-*-*-*-*-*-*-*-*-*-*-   L P A _ S O L V E _ L P  -*-*-*-*-*-*-*-*-*-*-*-*-

void lpa_solve_lp( void **data,
                   ipc_ *status,
                   ipc_ n,
                   ipc_ m,
                   const rpc_ g[],
                   const rpc_ f,
                   ipc_ a_ne,
                   const rpc_ A_val[],
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
 Solve the linear program.

 @param[in,out] data holds private internal data

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
  \li -3. The restrictions n > 0 and m > 0 or requirement that A_type contains
       its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
        has been violated.
  \li -5. The simple-bound constraints are inconsistent.
  \li -7. The constraints appear to have no feasible point.
  \li -9. The analysis phase of the factorization failed; the return status
         from the factorization package is given in the component
         inform.factor_status
  \li -10. The factorization failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -11. The solution of a set of linear equations using factors from the
         factorization package failed; the return status from the factorization
         package is given in the component inform.factor_status.
  \li -16. The problem is so ill-conditioned that further progress is
           impossible.
  \li -17. The step is too small to make further impact.
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general linear constraints.

 @param[in] g is a one-dimensional array of size n and type rpc_, that
    holds the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

 @param[in] f is a scalar of type rpc_, that
    holds the constant term \f$f\f$ of the objective function.

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the constraint Jacobian matrix
    \f$A\f$ in any of the available storage schemes.

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

 @param[out] c is a one-dimensional array of size m and type rpc_, that
    holds the residual \f$c(x)\f$.
    The i-th component of c, i = 0, ... ,  m-1, contains  \f$c_i(x) \f$.

 @param[in,out] y is a one-dimensional array of size n and type rpc_, that
    holds the values \f$y\f$ of the Lagrange multipliers for the general
    linear constraints. The j-th component
    of y, i = 0, ... , m-1, contains \f$y_i\f$.

 @param[in,out] z is a one-dimensional array of size n and type rpc_, that
    holds the values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

 @param[out] x_stat is a one-dimensional array of size n and type ipc_, that
    gives the optimal status of the problem variables. If x_stat(j) is negative,
    the variable \f$x_j\f$ most likely lies on its lower bound, if it is
    positive, it lies on its upper bound, and if it is zero, it lies
    between its bounds.

 @param[out] c_stat is a one-dimensional array of size m and type ipc_, that
    gives the optimal status of the general linear constraints. If c_stat(i) is
    negative, the constraint value \f$a_i^Tx\f$ most likely lies on its
    lower bound, if it is positive, it lies on its upper bound, and if it
    is zero, it lies  between its bounds.
*/

// *-*-*-*-*-*-*-*-*-*-    L P A  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lpa_information( void **data,
                      struct lpa_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lpa_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L P A  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void lpa_terminate( void **data,
                    struct lpa_control_type *control,
                    struct lpa_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see lpa_control_type)

  @param[out] inform   is a struct containing output information
              (see lpa_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example lpat.c
   This is an example of how to use the package to solve a linear program.
   A variety of supported constraint matrix storage formats are
   shown.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example lpatf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
