/** \file arc.h */

/*
 * THIS VERSION: GALAHAD 3.3 - 10/08/2021 AT 15:30 GMT.
 *
 *-*-*-*-*-*-*-  G A L A H A D _ A R C  C  I N T E R F A C E  -*-*-*-*-*-*-*-
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

/*! 
  \mainpage GALAHAD C package arc  

  \section arc_intro Introduction

  \subsection arc_purpose Purpose

  The arc package uses a <b>regularization method to find a (local)
  unconstrained minimizer of a differentiable objective function 
  \f$\mathbf{f(x)}\f$ of many variables \f$\mathbf{x}\f$.</b> 
  The method offers the choice of 
  direct and iterative solution of the key regularization subproblems, and
  is most suitable for large problems. First derivatives are required,
  and if second derivatives can be calculated, they will be exploited---if
  the product of second derivatives with a vector may be found, but
  not the derivatives themselves, that may also be exploited.

  \subsection arc_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England,
  and M. Porcelli, University of Bologna, Italy.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection arc_date Originally released

  May 2011, C interface August 2021.

  \subsection arc_terminology Terminology

  The \e gradient \f$\nabla_x f(x)\f$ of \f$f(x)\f$ is the vector whose 
  \f$i\f$-th component is \f$\partial f(x)/\partial x_i\f$.
  The \e Hessian \f$\nabla_{xx} f(x)\f$ of \f$f(x)\f$ is the symmetric matrix 
  whose \f$i,j\f$-th entry is \f$\partial^2 f(x)/\partial x_i \partial x_j\f$.
  The Hessian is \e sparse if a significant and useful proportion of the 
  entries are universally zero.

  \subsection arc_method Method

  An adaptive cubic regularization method is used.
  In this, an improvement to a current
  estimate of the required minimizer, \f$x_k\f$ is sought by computing a
  step \f$s_k\f$. The step is chosen to approximately minimize a model 
  \f$m_k(s)\f$  of \f$f(x_k + s)\f$ that includes a weighted term 
  \f$\sigma_k \|s_k\|^3\f$  for some specified positive weight 
  \f$\sigma_k\f$. The quality of the  resulting step \f$s_k\f$ is 
  assessed by computing the "ratio" 
  \f$(f(x_k) - f(x_k + s_k))/ (m_k(0) - m_k(s_k))\f$.
  The step is deemed to have succeeded if the ratio exceeds a given 
  \f$\eta_s > 0\f$,   and in this case \f$x_{k+1} = x_k + s_k\f$. 
  Otherwise   \f$x_{k+1} = x_k\f$, and the weight is increased by 
  powers of a given increase factor up to a given limit. If the ratio is 
  larger than \f$\eta_v \geq \eta_d\f$, the weight will be decreased by 
  powers of a given decrease factor again up to a given limit. The method 
  will terminate as soon as \f$\|\nabla_x f(x_k)\|\f$ is smaller than 
  a specified value.

  Either linear or quadratic models \f$m_k(s)\f$ may be used. 
  The former will be taken as the first two terms 
  \f$f(x_k) + s^T \nabla_x f(x_k)\f$
  of a Taylor series about \f$x_k\f$, while the latter uses an
  approximation to the first three terms
  \f$f(x_k) + s^T \nabla_x f(x_k) + \frac{1}{2} s^T B_k s\f$,
  for which \f$B_k\f$ is a symmetric approximation to the Hessian
  \f$\nabla_{xx}f(x_k)\f$; possible approximations include the 
  true Hessian,  limited-memory secant and sparsity approximations and 
  a scaled identity matrix. Normally a two-norm regularization will be used, 
  but this may change if preconditioning is employed.

  An approximate minimizer of the cubic model
  is found using either a direct approach involving factorization or an
  iterative (conjugate-gradient/Lanczos) approach based on approximations
  to the required solution from a so-called Krlov subspace. The direct
  approach is based on the knowledge that the required solution
  satisfies the linear system of equations 
  \f$(B_k + \lambda_k I) s_k  = - \nabla_x f(x_k)\f$ 
  involving a scalar Lagrange multiplier \f$\lambda_k\f$.
  This multiplier is found by uni-variate root finding, using a safeguarded
  Newton-like process, by the GALAHAD packages RQS or DPS
  (depending on the norm chosen). The iterative approach
  uses  the GALAHAD packag GLRT, and is best accelerated by preconditioning
  with good approximations to \f$B_k\f$ using GALAHAD's PSLS. 
  The iterative approach has the advantage that only matrix-vector products
  \f$B_k v\f$ are required, and thus \f$B_k\f$ is not required 
  explicitly.  However when factorizations of \f$B_k\f$ are possible, 
  the direct approach  is often more efficient.

  \subsection arc_references References

  The generic adaptive cubic regularization method is described in detail in

  C. Cartis,  N. I. M. Gould and Ph. L. Toint,
  ``Adaptive cubic regularisation methods for unconstrained optimization.
  Part I: motivation, convergence and numerical results''
  Mathematical Programming 127(2) (2011) 245-295,

  and uses ``tricks'' as suggested in

  N. I. M. Gould, M. Porcelli and Ph. L. Toint,
  ``Updating the regularization parameter in the adaptive cubic regularization
  algorithm''.
  Computational Optimization and Applications 53(1) (2012) 1-22.

  \section arc_call_order Call order

  To solve a given problem, functions from the arc package must be called 
  in the following order:

  - \link arc_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link arc_read_specfile \endlink (optional) - override control values 
      by reading replacement values from a file
  - \link arc_import \endlink - set up problem data structures and fixed
      values
  - \link arc_reset_control \endlink (optional) - possibly change control 
      parameters if a sequence of problems are being solved
  - solve the problem by calling one of 
     - \link arc_solve_with_mat \endlink - solve using function calls to
       evaluate function, gradient and Hessian values
     - \link arc_solve_without_mat \endlink - solve using function calls to
       evaluate function and gradient values and Hessian-vector products
     - \link arc_solve_reverse_with_mat \endlink - solve returning to the
       calling program to obtain function, gradient and Hessian values, or
     - \link arc_solve_reverse_without_mat \endlink - solve returning to the
       calling prorgram to obtain function and gradient values and 
       Hessian-vector products
  - \link arc_information \endlink (optional) - recover information about
    the solution and solution process
  - \link arc_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \section main_symmetric_matrices Symmetric matrix storage formats

  The symmetric \f$n\f$ by \f$n\f$ matrix \f$H = \nabla_{xx}f\f$ may be 
  presented and stored in a variety of formats. But crucially symmetry is 
  exploited by only storing values from the lower triangular part 
  (i.e, those entries that lie on or below the leading diagonal).

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for 
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing. 

  \subsection symmetric_matrix_dense Dense storage format
  The matrix \f$H\f$ is stored as a compact  dense matrix by rows, that is, 
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$H_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held. 
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array H_val
  will hold the value \f$H_{ij}\f$ (and, by symmetry, \f$H_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsection symmetric_matrix_coordinate Sparse co-ordinate storage format
  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j 
  and value \f$H_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays H_row and
  H_col and real array H_val, respectively, while the number of nonzeros
  is recorded as H_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsection symmetric_matrix_row_wise Sparse row-wise storage format
  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$H\f$ the i-th component of the
  integer array H_ptr holds the position of the first entry in this row,
  while H_ptr(n) holds the total number of entries plus one.
  The column indices j, \f$0 \leq j \leq i\f$, and values 
  \f$H_{ij}\f$ of the  entries in the i-th row are stored in components
  l = H_ptr(i), \f$\ldots\f$, H_ptr(i+1)-1 of the
  integer array H_col, and real array H_val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_ARC_H 
#define GALAHAD_ARC_H

// precision
#include "galahad_precision.h"

// required packages
//#include "rqs.h"
//#include "dps.h"
//#include "glrt.h"
//#include "psls.h"
//#include "lms.h"
//#include "sha.h"

/**
 * control derived type as a C struct
 */
struct arc_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief the level of output required. 
    /// \li \f$\leq\f$ 0 gives no output, 
    /// \li  = 1 gives a one-line summary for every iteration, 
    /// \li  = 2 gives a summary of the inner iteration for each iteration, 
    /// \li \f$\geq\f$ 3 gives increasingly verbose (debugging) output
    int print_level;

    /// \brief
    /// any printing will start on this iteration
    int start_print;

    /// \brief
    /// any printing will stop on this iteration
    int stop_print;

    /// \brief
    /// the number of iterations between printing
    int print_gap;

    /// \brief
    /// the maximum number of iterations performed
    int maxit;

    /// \brief
    /// removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    /// see alive_unit
    char alive_file[31];

    /// \brief the descent strategy used. 
    ///
    /// Possible values are
    /// \li <= 0 a monotone strategy is used.
    /// \li anything else, a non-monotone strategy with history length 
    /// .non_monotine is used.
    int non_monotone;

    ///\brief  the model used. 
    ///
    /// Possible values are
    /// \li 0  dynamic (*not yet implemented*)
    /// \li 1  first-order (no Hessian)
    /// \li 2  second-order (exact Hessian)
    /// \li 3  barely second-order (identity Hessian)
    /// \li 4  secant second-order (limited-memory BFGS, with .lbfgs_vectors 
    ///        history) (*not yet implemented*)
    /// \li 5  secant second-order (limited-memory SR1, with .lbfgs_vectors 
    ///        history) (*not yet implemented*)
    int model;

    /// \brief the regularization norm used. 

    /// The norm is defined via \f$\|v\|^2 = v^T P v\f$,
    /// and will define the preconditioner used for iterative methods.
    /// Possible values for \f$P\f$ are
    /// \li -3  users own preconditioner
    /// \li -2  \f$P =\f$ limited-memory BFGS matrix 
    ///         (with .lbfgs_vectors history)
    /// \li -1  identity (= Euclidan two-norm)
    /// \li  0  automatic (*not yet implemented*)
    /// \li  1  diagonal, \f$P =\f$ diag( max( Hessian, .min_diagonal ) )
    /// \li  2  banded, \f$P =\f$ band( Hessian ) with semi-bandwidth 
    ///         .semi_bandwidth
    /// \li  3  re-ordered band, P=band(order(A)) with semi-bandwidth 
    ///         .semi_bandwidth
    /// \li  4  full factorization, \f$P =\f$ Hessian, 
    ///         Schnabel-Eskow modification
    /// \li  5  full factorization, \f$P =\f$ Hessian, GMPS modification 
    ///         (*not yet implemented*)
    /// \li  6  incomplete factorization of Hessian, Lin-More'
    /// \li  7  incomplete factorization of Hessian, HSL_MI28
    /// \li  8  incomplete factorization of Hessian, Munskgaard
    ///         (*not yet implemented*)
    /// \li  9  expanding band of Hessian (*not yet implemented*)
    /// \li 10  diagonalizing norm from GALAHAD_DPS (*subproblem_direct only*)
    int norm;

    /// \brief
    /// specify the semi-bandwidth of the band matrix P if required
    int semi_bandwidth;

    /// \brief
    /// number of vectors used by the L-BFGS matrix P if required
    int lbfgs_vectors;

    /// \brief
    /// number of vectors used by the sparsity-based secant Hessian if required
    int max_dxg;

    /// \brief
    /// number of vectors used by the Lin-More' incomplete factorization
    /// matrix P if required
    int icfs_vectors;

    /// \brief
    /// the maximum number of fill entries within each column of the incomplete
    /// factor L computed by HSL_MI28. In general, increasing .mi28_lsize 
    /// improve the quality of the preconditioner but increases the time to 
    /// compute and then apply the preconditioner. Values less than 0 are 
    /// treated as 0
    int mi28_lsize;

    /// \brief
    /// the maximum number of entries within each column of the strictly lower
    /// triangular matrix \f$R\f$ used in the computation of the preconditioner
    /// by HSL_MI28.  Rank-1 arrays of size .mi28_rsize * n are allocated 
    /// internally to hold \f$R\f$. Thus the amount of memory used, as well 
    /// as the amount of work involved in computing the preconditioner, 
    /// depends on .mi28_rsize. Setting .mi28_rsize > 0 generally leads to 
    /// a higher quality preconditioner than using .mi28_rsize = 0, and 
    /// choosing .mi28_rsize >= .mi28_lsize is generally recommended
    int mi28_rsize;

    /// \brief
    /// try to pick a good initial regularization weight using .advanced_start
    /// iterates of a variant on the strategy of Sartenaer SISC 18(6)
    /// 1990:1788-1803
    int advanced_start;

    /// \brief
    /// overall convergence tolerances. The iteration will terminate when the
    /// norm of the gradient of the objective function is smaller than
    /// MAX( .stop_g_absolute, .stop_g_relative * norm of the initial gradient
    /// or if the step is less than .stop_s
    real_wp_ stop_g_absolute;
    /// see stop_g_absolute
    real_wp_ stop_g_relative;
    /// see stop_g_absolute
    real_wp_ stop_s;

    /// \brief
    /// Initial value for the regularisation weight  (-ve => 1/||g_0||)
    real_wp_ initial_weight;

    /// \brief
    /// minimum permitted regularisation weight
    real_wp_ minimum_weight;

    /// \brief
    /// expert parameters as suggested in Gould, Porcelli & Toint, "Updating the
    /// regularization parameter in the adaptive cubic regularization algorithm"
    /// RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011),
    /// http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
    /// (these are denoted beta, epsilon_chi and alpha_max in the paper)
    real_wp_ reduce_gap;
    /// see reduce_gap
    real_wp_ tiny_gap;
    /// see reduce_gap
    real_wp_ large_root;

    /// \brief
    /// a potential iterate will only be accepted if the actual decrease
    /// f - f(x_new) is larger than .eta_successful times that predicted
    /// by a quadratic model of the decrease. The regularization weight will be
    /// decreased if this relative decrease is greater than .eta_very_successful
    /// but smaller than .eta_too_successful (the first is eta in Gould, Porcell
    /// and Toint, 2011)
    real_wp_ eta_successful;
    /// see eta_successful
    real_wp_ eta_very_successful;
    /// see eta_successful
    real_wp_ eta_too_successful;

    /// \brief
    /// on very successful iterations, the regularization weight will be reduced
    /// by the factor .weight_decrease but no more than .weight_decrease_min
    /// while if the iteration is unsuccessful, the weight will be increased by
    /// a factor .weight_increase but no more than .weight_increase_max
    /// (these are delta_1, delta_2, delta3 and delta_max in Gould, Porcelli
    /// and Toint, 2011)
    real_wp_ weight_decrease_min;
    /// see weight_decrease_min
    real_wp_ weight_decrease;
    /// see weight_decrease_min
    real_wp_ weight_increase;
    /// see weight_decrease_min
    real_wp_ weight_increase_max;

    /// \brief
    /// the smallest value the onjective function may take before the problem
    /// is marked as unbounded
    real_wp_ obj_unbounded;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    /// \brief
    /// is the Hessian matrix of second derivatives available or is access only
    /// via matrix-vector products?
    bool hessian_available;

    /// \brief
    /// use a direct (factorization) or (preconditioned) iterative method to
    /// find the search direction
    bool subproblem_direct;

    /// \brief
    /// should the weight be renormalized to account for a change in
    /// preconditioner?
    bool renormalize_weight;

    /// \brief
    /// should the test for acceptance involve the quadratic model or the cubic?
    bool quadratic_ratio_test;

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

    /// \brief
    /// control parameters for RQS
    ///struct rqs_control_type rqs_control;

    /// \brief
    /// control parameters for DPS
    ///struct dps_control_type dps_control;

    /// \brief
    /// control parameters for GLRT
    ///struct glrt_control_type glrt_control;

    /// \brief
    /// control parameters for PSLS
    ///struct psls_control_type psls_control;

    /// \brief
    /// control parameters for LMS
    ///struct lms_control_type lms_control;
    /// see LMS_control
    ///struct lms_control_type lms_control;

    /// \brief
    /// control parameters for SHA
    ///struct sha_control_type sha_control;
};

/**
 * time derived type as a C struct
 */
struct arc_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_sp_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    real_sp_ preprocess;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    real_sp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_sp_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    real_sp_ solve;

    /// \brief
    /// the total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    real_wp_ clock_preprocess;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
    real_wp_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    real_wp_ clock_factorize;

    /// \brief
    /// the clock time spent computing the search direction
    real_wp_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct arc_inform_type {

    /// \brief
    /// return status. See ARC_solve for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations performed
    int iter;

    /// \brief
    /// the total number of CG iterations performed
    int cg_iter;

    /// \brief
    /// the total number of evaluations of the objection function
    int f_eval;

    /// \brief
    /// the total number of evaluations of the gradient of the objection functio
    int g_eval;

    /// \brief
    /// the total number of evaluations of the Hessian of the objection function
    int h_eval;

    /// \brief
    /// the return status from the factorization
    int factorization_status;

    /// \brief
    /// the maximum number of factorizations in a sub-problem solve
    int factorization_max;

    /// \brief
    /// the maximum number of entries in the factors
    int max_entries_factors;

    /// \brief
    /// the total integer workspace required for the factorization
    int factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int factorization_real;

    /// \brief
    /// the average number of factorizations per sub-problem solve
    real_wp_ factorization_average;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by the package.
    real_wp_ obj;

    /// \brief
    /// the norm of the gradient of the objective function at the best estimate
    /// of the solution determined by the package.
    real_wp_ norm_g;

    /// \brief
    /// the current value of the regularization weight
    real_wp_ weight;

    /// \brief
    /// timings (see above)
    struct arc_time_type time;

    /// \brief
    /// inform parameters for RQS
    ///struct rqs_inform_type rqs_inform;

    /// \brief
    /// inform parameters for DPS
    ///struct dps_inform_type dps_inform;

    /// \brief
    /// inform parameters for GLRT
    ///struct glrt_inform_type glrt_inform;

    /// \brief
    /// inform parameters for PSLS
    ///struct psls_inform_type psls_inform;

    /// \brief
    /// inform parameters for LMS
    ///struct lms_inform_type lms_inform;
    /// see LMS_inform
    ///struct lms_inform_type lms_inform;

    /// \brief
    /// inform parameters for SHA
    ///struct sha_inform_type sha_inform;
};

//  *-*-*-*-*-*-*-*-*-*-   A R C _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*

void arc_initialize( void **data, 
                     struct arc_control_type *control,
                     struct arc_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information 
              (see arc_control_type)
  @param[out] inform   is a struct containing output information
              (see arc_inform_type) 
*/

//  *-*-*-*-*-*-*-*-*-   A R C _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*-*-*

void arc_read_specfile( struct arc_control_type *control, 
                        const char specfile[]) ;

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control  is a struct containing control information 
              (see arc_control_type)
  @param[in]  specfile  is a character string containing the name of the 
              specification file
*/


//  *-*-*-*-*-*-*-*-*-*-*-*-   A R C _ I M P O R T    -*-*-*-*-*-*-*-*-*-*-*-*

void arc_import( struct arc_control_type *control,
                 void **data,
                 int *status, 
                 int n, 
                 const char H_type[], 
                 int ne, 
                 const int H_row[],
                 const int H_col[], 
                 const int H_ptr[] );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see arc_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.

 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink 
   used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows',
  'dense', 'diagonal' or 'absent', the latter if access to the Hessian is
  via matrix-vector products; lower or upper case variants are allowed

 @param[in]  ne is a scalar variable of type int, that holds the number of
   entries in the  lower triangular part of H in the sparse co-ordinate
   storage scheme. It need not be set for any of the other three schemes.

 @param[in]  H_row is a one-dimensional array of size ne and type int, that 
   holds the row indices of the lower triangular part of H in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL

 @param[in]  H_col is a one-dimensional array of size ne and type int,
   that holds the column indices of the  lower triangular part of H in either
   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
   be set when the dense or diagonal storage schemes are used, and in this 
   case can be NULL

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type int,
   that holds the starting position of  each row of the lower
   triangular part of H, as well as the total number of entries plus one,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL
 */

//  *-*-*-*-*-*-*-*-*-   A R C _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void arc_reset_control( struct arc_control_type *control,
                        void **data,
                        int *status, );

/*!< 
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see arc_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
 */

//  *-*-*-*-*-*-*-*-*-   A R C _ S O L V E _ W I T H _ M A T   -*-*-*-*-*-*-*-*

void arc_solve_with_mat( void **data,
                         void *userdata, 
                         int *status, 
                         int n, 
                         real_wp_ x[], 
                         real_wp_ g[],
                         int ne, 
                         int (*eval_f)(
                           int, const real_wp_[], real_wp_*, const void * ), 
                         int (*eval_g)(
                           int, const real_wp_[], real_wp_[], const void * ),
                         int (*eval_h)(
                           int, int, const real_wp_[], real_wp_[], 
                           const void * ),
                         int (*eval_prec)(
                           int, const real_wp_[], real_wp_[], const real_wp_[], 
                           const void * ) );

/*!<
 Find a local minimizer of a given function using a regularization method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is 
 provided specifically, and all function/derivative information is 
 available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
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
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file 
         named control.alive_file from unit unit control.alive_unit.
 
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param[in] ne is a scalar variable of type int, that holds the number of 
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param eval_f is a user-supplied function that must have the following 
   signature:
   \code
        int eval_f( int n, const double x[], double *f, const void *userdata ) 
   \endcode
   The value of the objective function \f$f(x)\f$ evaluated at x=\f$x\f$ must 
   be assigned to f, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_f via the structure \c userdata. 

 @param eval_g is a user-supplied function that must have the following 
   signature:
   \code
      int eval_g( int n, const double x[], double g[], const void *userdata )
   \endcode
   The components of the gradient \f$g = \nabla_x f(x\f$) of the objective 
   function evaluated at x=\f$x\f$ must be assigned to g, and the function 
   return value set to 0. If the evaluation is impossible at x, return 
   should be set to a nonzero value.
   Data may be passed into \c eval_g via the structure \c userdata. 
 
 @param eval_h is a user-supplied function that must have the following 
   signature:
   \code
        int eval_h( int n, int ne, const double x[], double h[],
                    const void *userdata )
   \endcode
   The nonzeros of the Hessian \f$H = \nabla_{xx}f(x)\f$ of the objective 
   function evaluated at x=\f$x\f$ must be assigned to h in the same order 
   as presented to arc_import, and the function return value set to 0. 
   If the evaluation is impossible at x, return should be set to a 
   nonzero value.
   Data may be passed into \c eval_h via the structure \c userdata. 
 
 @param  eval_prec is an optional user-supplied function that may be NULL. 
   If non-NULL, it must have the following signature:
   \code
       int eval_prec( int n, const double x[], double u[], const double v[],
                      const void *userdata )
   \endcode
   The product \f$u = P(x) v\f$ of the user's preconditioner \f$P(x)\f$ 
   evaluated at \f$x\f$ with the vector v = \f$v\f$, the result \f$u\f$ 
   must be retured in u, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata. 
 */ 

//  *-*-*-*-*-*-*-*-   A R C _ S O L V E _ W I T H O U T _ M A T   -*-*-*-*-*-*

void arc_solve_without_mat( void **data,
                            void *userdata, 
                            int *status, 
                            int n, 
                            real_wp_ x[], 
                            real_wp_ g[], 
                            int (*eval_f)(
                              int, const real_wp_[], real_wp_*, const void * ), 
                            int (*eval_g)(
                              int, const real_wp_[], real_wp_[], const void * ),
                            int (*eval_hprod)(
                              int, const real_wp_[], real_wp_[], 
                              const real_wp_[], bool, const void * ), 
                            int (*eval_prec)(
                              int, const real_wp_[], real_wp_[], 
                               const real_wp_[], const void * ) );

/*!<
 Find a local minimizer of a given function using a regularization method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is 
 provided by Hessian-vector products, and all function/derivative 
 information is available by function calls.

 @param[in,out] data holds private internal data

 @param[in] userdata is a structure that allows data to be passed into
    the function and derivative evaluation programs.

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
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
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file 
         named control.alive_file from unit unit control.alive_unit.
 
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param eval_f is a user-supplied function that must have the following 
   signature:
   \code
        int eval_f( int n, const double x[], double *f, const void *userdata ) 
   \endcode
   The value of the objective function \f$f(x)\f$ evaluated at x=\f$x\f$ must 
   be assigned to f, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_f via the structure \c userdata. 

 @param eval_g is a user-supplied function that must have the following 
   signature:
   \code
      int eval_g( int n, const double x[], double g[], const void *userdata )
   \endcode
   The components of the gradient \f$g = \nabla_x f(x\f$) of the objective 
   function evaluated at x=\f$x\f$ must be assigned to g, and the function 
   return value set to 0. If the evaluation is impossible at x, return 
   should be set to a nonzero value.
   Data may be passed into \c eval_g via the structure \c userdata. 
 
 @param eval_hprod is a user-supplied function that must have the following 
   signature:
   \code
        int eval_hprod( int n, const double x[], double u[], const double v[], 
                        bool got_h, const void *userdata )
   \endcode
   The sum \f$u + \nabla_{xx}f(x) v\f$ of the product of the Hessian 
   \f$\nabla_{xx}f(x)\f$ of the objective function evaluated at x=\f$x\f$
   with the vector v=\f$v\f$ and the vector $\f$u\f$ must be returned in u, 
   and the function return value set to 0. If the
   evaluation is impossible at x, return should be set to a nonzero value.
   The Hessian has already been evaluated or used at x if got_h is true.
   Data may be passed into \c eval_hprod via the structure \c userdata. 

 @param  eval_prec is an optional user-supplied function that may be NULL. 
   If non-NULL, it must have the following signature:
   \code
       int eval_prec( int n, const double x[], double u[], const double v[],
                      const void *userdata )
   \endcode
   The product \f$u = P(x) v\f$ of the user's preconditioner \f$P(x)\f$ 
   evaluated at \f$x\f$ with the vector v = \f$v\f$, the result \f$u\f$ 
   must be retured in u, and the function return value set to 0. If the 
   evaluation is impossible at x, return should be set to a nonzero value.
   Data may be passed into \c eval_prec via the structure \c userdata. 
 */ 

//  *-*-*-*-*-   A R C _ S O L V E _ R E V E R S E _ W I T H _ M A T   -*-*-*-*

void arc_solve_reverse_with_mat( void **data,
                                 int *status, 
                                 int *eval_status, 
                                 int n, 
                                 real_wp_ x[], 
                                 real_wp_ f, 
                                 real_wp_ g[], 
                                 int ne, 
                                 real_wp_ H_val[], 
                                 const real_wp_ u[], 
                                 real_wp_ v[] );

/*!<
 Find a local minimizer of a given function using a regularization method.

 This call is for the case where \f$H = \nabla_{xx}f(x)\f$ is 
 provided specifically, but function/derivative information is only 
 available by returning to the calling procedure

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
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
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file 
         named control.alive_file from unit unit control.alive_unit.

 @param status (continued)
  \li  2. The user should compute the objective function value \f$f(x)\f$ at 
         the point \f$x\f$ indicated in x and then re-enter the function. 
         The required value should be set in f, and eval_status should be 
         set to 0. If the user is unable to evaluate \f$f(x)\f$--- for 
         instance, if the function is undefined at \f$x\f$--- the user need 
         not set f, but should then set eval_status to a non-zero value.
  \li   3. The user should compute the gradient of the objective function
         \f$\nabla_x f(x)\f$ at the point \f$x\f$ indicated in x and then 
         re-enter the function. The value of the i-th component of the g
         radient should be set in g[i], for i = 0, ..., n-1 and eval_status 
         should be set to 0. If the user is unable to evaluate a component 
         of \f$\nabla_x f(x)\f$ --- for instance if a component of the gradient 
         is undefined at \f$x\f$ -the user need not set g, but should then set 
         eval_status to a non-zero value.
  \li   4. The user should compute the Hessian of the objective function
         \f$\nabla_{xx} f(x)\f$ at the point x indicated in \f$x\f$ and then 
         re-enter the function. The value l-th component of the Hessian stored 
         according to the scheme input in the remainder of \f$H\f$ should be 
         set in H_val[l], for l = 0, ..., ne-1 and eval_status should be set 
         to 0. If the user is unable to evaluate a component of 
         \f$\nabla_{xx}f(x)\f$ --- for instance, if a component of the Hessian 
         is undefined at \f$x\f$ --- the user need not set H_val, but should 
         then set eval_status to a non-zero value.
  \li   6. The user should compute the product \f$u = P(x)v\f$ of their 
         preconditioner \f$P(x)\f$ at the point x indicated in \f$x\f$ with 
         the vector \f$v\f$ and then re-enter the function. The vector \f$v\f$ 
         is given in v, the resulting vector \f$u = P(x)v\f$ should be set in 
         u and eval_status should be set to 0. If the user is unable to 
         evaluate the product--- for instance, if a component of the 
         preconditioner is undefined at \f$x\f$ --- the user need not set u, 
         but should then set eval_status to a non-zero value.
 
 @param[in,out] eval_status is a scalar variable of type int, that is used to 
    indicate if  objective function/gradient/Hessian values can be provided 
    (see above) 
  
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in]
   f is a scalar variable pointer of type double, that holds the value of the
    objective function.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param[in] ne is a scalar variable of type int, that holds the number of 
    entries in the lower triangular part of the Hessian matrix \f$H\f$.
 
 @param[in] H_val is a one-dimensional array of size ne and type double, 
    that holds the values of the entries of the lower triangular part of the 
    Hessian matrix \f$H\f$ in any of the available storage schemes.

 @param[in] u is a one-dimensional array of size n and type double, that is 
    used for reverse communication (see above for details)
  
 @param[in,out] v is a one-dimensional array of size n and type double, that is 
    used for reverse communication (see above for details)
*/  


//  *-*-*-   A R C _ S O L V E _ R E V E R S E _ W I T H O U T _ M A T   -*-*-*

void arc_solve_reverse_without_mat( void **data,
                                    int *status, 
                                    int *eval_status, 
                                    int n, 
                                    real_wp_ x[], 
                                    real_wp_ f, 
                                    real_wp_ g[], 
                                    real_wp_ u[], 
                                    real_wp_ v[] );

/*!<
 Find a local minimizer of a given function using a regularization method.

 This call is for the case where access to \f$H = \nabla_{xx}f(x)\f$ is 
 provided by Hessian-vector products, but function/derivative information 
 is only available by returning to the calling procedure.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
  \li -7. The objective function appears to be unbounded from below
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
  \li -18. Too many iterations have been performed. This may happen if
         control.maxit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -19. The CPU time limit has been reached. This may happen if
         control.cpu_time_limit is too small, but may also be symptomatic of
         a badly scaled problem.
  \li -40. The user has forced termination of solver by removing the file 
         named control.alive_file from unit unit control.alive_unit.

 @param status (continued)
  \li  2. The user should compute the objective function value \f$f(x)\f$ at 
         the point \f$x\f$ indicated in x and then re-enter the function. 
         The required value should be set in f, and eval_status should be 
         set to 0. If the user is unable to evaluate \f$f(x)\f$ --- for 
         instance, if the function is undefined at \f$x\f$ --- the user need 
         not set f, but should then set eval_status to a non-zero value.
  \li   3. The user should compute the gradient of the objective function
         \f$\nabla_x f(x)\f$ at the point \f$x\f$ indicated in x and then 
         re-enter the function. The value of the i-th component of the g
         radient should be set in g[i], for i = 0, ..., n-1 and eval_status 
         should be set to 0. If the user is unable to evaluate a component 
         of \f$\nabla_x f(x)\f$ --- for instance if a component of the gradient 
         is undefined at \f$x\f$ -the user need not set g, but should then set 
         eval_status to a non-zero value.
  \li   5. The user should compute the product \f$\nabla_{xx} f(x)v\f$ of the 
         Hessian of the objective function \f$\nabla_{xx} f(x)\f$ at the point
         \f$x\f$ indicated in x with the vector \f$v\f$, add the result to 
         the vector \f$u\f$ and then re-enter the function. The vectors 
         \f$u\f$ and \f$v\f$ are given in u and v respectively, the resulting 
         vector \f$u + \nabla_{xx} f(x)v\f$ should be set
         in u and eval_status should be set to 0. If the user is unable to
         evaluate the product--- for instance, if a component of the Hessian is
         undefined at \f$x\f$ --- the user need not alter u, but should then set
         eval_status to a non-zero value.
  \li   6. The user should compute the product \f$u = P(x)v\f$ of their 
         preconditioner \f$P(x)\f$ at the point x indicated in \f$x\f$ with 
         the vector \f$v\f$ and then re-enter the function. The vector \f$v\f$ 
         is given in v, the resulting vector \f$u = P(x)v\f$ should be set in 
         u and eval_status should be set to 0. If the user is unable to 
         evaluate the product--- for instance, if a component of the 
         preconditioner is undefined at \f$x\f$ --- the user need not set u, 
         but should then set eval_status to a non-zero value.
 
 @param[in,out] eval_status is a scalar variable of type int, that is used to 
    indicate if  objective function/gradient/Hessian values can be provided 
    (see above) 
  
 @param[in] n is a scalar variable of type int, that holds the number of
    variables

 @param[in,out] x is a one-dimensional array of size n and type double, that 
    holds the values \f$x\f$ of the optimization variables. The j-th component 
    of x, j = 0, ... , n-1, contains \f$x_j\f$.
  
 @param[in]
   f is a scalar variable pointer of type double, that holds the value of the
    objective function.
  
 @param[in,out] g is a one-dimensional array of size n and type double, that 
    holds the gradient \f$g = \nabla_xf(x)\f$ of the objective function. 
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.
  
 @param[in,out] u is a one-dimensional array of size n and type double, that is 
    used for reverse communication (see above for details)
  
 @param[in,out] v is a one-dimensional array of size n and type double, that is 
    used for reverse communication (see above for details)
*/  

//  *-*-*-*-*-*-*-*-*-*-   A R C _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void arc_information( void **data,
                      struct arc_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
    (see arc_inform_type) 

  @param[out] status   is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

//  *-*-*-*-*-*-*-*-*-*-   A R C _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void arc_terminate( void **data, 
                    struct arc_control_type *control, 
                    struct arc_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see arc_control_type)

  @param[out] inform   is a struct containing output information
              (see arc_inform_type) 
 */

/** 
   \anchor examples
   \f$\label{examples}\f$
   \example arct.c
   This is an example of how to use the package both when the Hessian 
   is directly available and when its product with vectors may be found.
   Both function call evaluations and returns to the calling program
   to find the required values are illustrated. A variety of supported 
   Hessian storage formats are shown.
  
   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false. In addition, see how 
   parameters may be passed into the evaluation functions via \c userdata.\n

   \example arctf.c
   This is the same example, but now fortran-style indexing is used.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
