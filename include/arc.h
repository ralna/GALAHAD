/*
 * THIS VERSION: GALAHAD 3.3 - 27/07/2021 AT 08:30 GMT.
 *
 *-*-*-*-*-*-*-  G A L A H A D _ A R C  C  I N T E R F A C E  -*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   currently in development
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
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

/*
 * control derived type as a C struct
 */
struct arc_control_type { 

    // use C or Fortran sparse matrix indexing
    bool f_indexing;

    // error and warning diagnostics occur on stream error
    int error;

    // general output occurs on stream out    
    int out;

    // the level of output required. <= 0 gives no output, = 1 gives a one-line
    // summary for every iteration, = 2 gives a summary of the inner iteration
    // for each iteration, >= 3 gives increasingly verbose (debugging) output
    int print_level; 

    // any printing will start on this iteration
    int start_print;

    // any printing will stop on this iteration
    int stop_print;

    // the number of iterations between printing
    int print_gap;

    // the maximum number of iterations allowed
    int maxit;

    // removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    char alive_file[31];

    // non-monotone <= 0 monotone strategy used, anything else non-monotone
    // strategy with this history length used
    int non_monotone;

    // specify the model used. Possible values are
    //
    //   0  dynamic (*not yet implemented*)
    //   1  first-order (no Hessian)
    //   2  second-order (exact Hessian)
    //   3  barely second-order (identity Hessian)
    //   4  secant second-order (sparsity-based)
    //   5  secant second-order (limited-memory BFGS, with .lbfgs_vectors 
    //      history)
    //   6  secant second-order (limited-memory SR1, with .lbfgs_vectors 
    //      history)
    int model;

    // specify the norm used. The norm is defined via ||v||^2 = v^T P v,
    // and will define the preconditioner used for iterative methods.
    // Possible values for P are
    //
    //   -3  users own preconditioner
    //   -2  P = limited-memory BFGS matrix (with .lbfgs_vectors history)
    //   -1  identity (= Euclidan two-norm)
    //    0  automatic (*not yet implemented*)
    //    1  diagonal, P = diag( max( Hessian, .min_diagonal ) )
    //    2  banded, P = band( Hessian ) with semi-bandwidth .semi_bandwidth
    //    3  re-ordered band, P=band(order(A)) with semi-bandwidth 
    //       .semi_bandwidth
    //    4  full factorization, P = Hessian, Schnabel-Eskow modification
    //    5  full factorization, P = Hessian, GMPS modification (*not yet *)
    //    6  incomplete factorization of Hessian, Lin-More'
    //    7  incomplete factorization of Hessian, HSL_MI28
    //    8  incomplete factorization of Hessian, Munskgaard (*not yet *)
    //    9  expanding band of Hessian (*not yet implemented*)
    //   10  diagonalizing norm from GALAHAD_DPS (*subproblem_direct only*)
    int norm;

    // specify the semi-bandwidth of the band matrix P if required
    int semi_bandwidth;

    // number of vectors used by the L-BFGS matrix P if required
    int lbfgs_vectors;

    // number of vectors used by the sparsity-based secant Hessian if required
    int max_dxg;

    // number of vectors used by the Lin-More' incomplete factorization
    // matrix P if required
    int icfs_vectors;

    // the maximum number of fill entries within each column of the incomplete
    // factor L computed by HSL_MI28. In general, increasing mi28_lsize improves
    // the quality of the preconditioner but increases the time to compute
    // and then apply the preconditioner. Values less than 0 are treated as 0
    int mi28_lsize;

    // the maximum number of entries within each column of the strictly lower
    // triangular matrix R used in the computation of the preconditioner by
    // HSL_MI28.  Rank-1 arrays of size mi28_rsize * n are allocated internally
    // to hold R. Thus the amount of memory used, as well as the amount of work
    // involved in computing the preconditioner, depends on mi28_rsize. Setting
    // mi28_rsize > 0 generally leads to a higher quality preconditioner than
    // using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally
    // recommended
    int mi28_rsize;

    // overall convergence tolerances. The iteration will terminate when the
    // norm of the gradient of the objective function is smaller than
    // MAX( .stop_g_absolute, .stop_g_relative * norm of the initial gradient
    // or if the step is less than .stop_s
    real_wp_ stop_g_absolute;
    real_wp_ stop_g_relative;
    real_wp_ stop_s;

    // try to pick a good initial regularization weight using .advanced_start
    // iterates of a variant on the strategy of Sartenaer SISC 18(6)
    // 1990:1788-1803
    int advanced_start;

    // initial value for the regularisation weight  (-ve => 1/||g_0||)
    real_wp_ initial_weight;

    // minimum permitted regularization weight
    real_wp_ minimum_weight;

    // expert parameters as suggested in Gould, Porcelli & Toint, "Updating the
    // regularization parameter in the adaptive cubic regularization algorithm",
    //  RAL-TR-2011-007, Rutherford Appleton Laboratory, England (2011),
    //     http://epubs.stfc.ac.uk/bitstream/6181/RAL-TR-2011-007.pdf
    // (these are denoted beta, epsilon_chi and alpha_max in the paper)
    real_wp_ reduce_gap;
    real_wp_ tiny_gap;
    real_wp_ large_root;

    // a potential iterate will only be accepted if the actual decrease
    // f - f(x_new) is larger than .eta_successful times that predicted
    // by a quadratic model of the decrease. The egularization weight will be
    // decreased if this relative decrease is greater than .eta_very_successful
    // but smaller than .eta_too_successful
    real_wp_ eta_successful;
    real_wp_ eta_very_successful;
    real_wp_ eta_too_successful;

    // on very successful iterations, the regularization weight will be reduced
    // by the factor %weight_decrease but no more than %weight_decrease_min
    // while if the iteration is unsucceful, the weight will be increased by a
    // factor %weight_increase but no more than %weight_increase_max (these are 
    // delta_1, delta_2, delta3 and delta_max in Gould, Porcelli & Toint, 2011)
    real_wp_ weight_decrease_min;
    real_wp_ weight_decrease;    
    real_wp_ weight_increase;    
    real_wp_ weight_increase_max;

    // the smallest value the objective function may take before the problem
    // is marked as unbounded
    real_wp_ obj_unbounded;

    // the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    // the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    // is the Hessian matrix of second derivatives available or is access only
    // via matrix-vector products?
    bool hessian_available;

    // use a direct (factorization) or (preconditioned) iterative method to
    // find the search direction
    bool subproblem_direct;

    // should the weight be renormalized to account for a change in 
    // preconditioner?
    bool renormalize_weight;

    // should the test for acceptance involve the quadratic model or the cubic?
    bool quadratic_ratio_test;

    // if .space_critical true, every effort will be made to use as little
    // space as possible. This may result in longer computation time
    bool space_critical;

    // if .deallocate_error_fatal is true, any array/pointer deallocation error
    // will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    // all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    // where .prefix contains the required string enclosed in
    // quotes, e.g. "string" or 'string'
    char prefix[31];

    // control parameters for RQS
    //struct rqs_control_type rqs_control;

    // control parameters for DPS
    //struct trs_control_type dps_control;

    // control parameters for GLRT
    //struct glrt_control_type glrt_control;

    // control parameters for PSLS
    //struct psls_control_type psls_control;

    // control parameters for LMS
    //struct lms_control_type lms_control;
    //struct lms_control_type lms_control_prec;

    // control parameters for SHA
    //struct sha_control_type sha_control;
};

/* 
 * time derived type as a C struct 
 */
struct arc_time_type {

    // the total CPU time spent in the package
    real_sp_ total;

    // the CPU time spent preprocessing the problem
    real_sp_ preprocess;

    // the CPU time spent analysing the required matrices prior to 
    // factorization
    real_sp_ analyse;

    // the CPU time spent factorizing the required matrices
    real_sp_ factorize;

    // the CPU time spent computing the search direction
    real_sp_ solve;

    // the total clock time spent in the package
    real_wp_ clock_total;

    // the clock time spent preprocessing the problem
    real_wp_ clock_preprocess;

    // the clock time spent analysing the required matrices prior to 
    // factorization
    real_wp_ clock_analyse;

    // the clock time spent factorizing the required matrices
    real_wp_ clock_factorize;

    // the clock time spent computing the search direction
    real_wp_ clock_solve;
};

/*
 * inform derived type as a C struct
 */
struct arc_inform_type {

    // return status. See ARC_solve for details
    int status;

    // the status of the last attempted allocation/deallocation
    int alloc_status ;

    // the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    // the total number of iterations performed
    int iter;

    // the total number of CG iterations performed
    int cg_iter;

    // the total number of evaluations of the objection function
    int f_eval;

    // the total number of evaluations of the gradient of the objection function
    int g_eval;

    // the total number of evaluations of the Hessian of the objection function
    int h_eval;

    // the maximum number of factorizations in a sub-problem solve
    int factorization_max;

    // the return status from the factorization
    int factorization_status;

    // the maximum number of entries in the factors
    long int max_entries_factors;

    // the total integer workspace required for the factorization
    int factorization_integer;

    // the total real workspace required for the factorization
    int factorization_real;

    // the average number of factorizations per sub-problem solve
    real_wp_ factorization_average;

    // the value of the objective function at the best estimate of the solution
    // determined by ARC_solve
    real_wp_ obj;

    // the norm of the gradient of the objective function at the best estimate
    // of the solution determined by ARC_solve
    real_wp_ norm_g;

    // the current value of the regularization weight
    real_wp_ weight;

    // timings (see above)
    struct arc_time_type time;

    // inform parameters for TRS
    //struct trs_inform_type trs_inform;

    // inform parameters for GLTR
    //struct gltr_info_type gltr_inform;

    // inform parameters for PSLS
    //struct psls_inform_type psls_inform;

    // inform parameters for LMS
    //struct lms_inform_type lms_inform;
    //struct lms_inform_type lms_inform_prec;

    // inform parameters for SHA
    //struct sha_inform_type sha_inform;
};

/*  *-*-*-*-*-*-*-*-*-*-   A R C _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*
 *
 * Provide default values for ARC controls
 */

void arc_initialize(void **data, 
                    struct arc_control_type *control,
                    struct arc_inform_type *inform);

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *   data     private internal data
 *   control  a struct containing control information
 *   inform   a struct containing output information
 */

/*  *-*-*-*-*-*-*-*-*-   A R C _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*-*-*
 *
 * Read the content of a specification file, and perform the assignment of
 * values associated with given keywords to the corresponding control parameters
 */

void arc_read_specfile(struct arc_control_type *control, 
                       const char specfile[]);

/*  *-*-*-*-*-*-*-*-*-*-*-*-   A R C _ I M P O R T    -*-*-*-*-*-*-*-*-*-*-*
 *
 * Import problem data into internal storage prior to solution. 
 */

void arc_import(struct arc_control_type *control,
                void **data,
                int *status, 
                int n, 
                const char H_type[], 
                int ne, 
                const int H_row[],
                const int H_col[], 
                const int H_ptr[]);

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  control is a struct whose members are described in 
 *   the leading comments to arc_solve
 *
 *  data is used for internal data
 *
 *  status is a scalar variable of type int, that gives
 *   the exit status from the package. Possible values are:
 *
 *     0. The import was succesful
 *
 *    -1. An allocation error occurred. A message indicating the offending
 *        array is written on unit control.error, and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -2. A deallocation error occurred.  A message indicating the offending
 *        array is written on unit control.error and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -3. The restriction n > 0 or requirement that type contains
 *        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
 *        'DIAGONAL' or 'ABSENT' has been violated.
 *
 *  n is a scalar variable of type int, that holds the number of
 *   variables
 *
 *  H_type is a one-dimensional array of type char that specifies the Hessian
 *   storage scheme used. It should be one of 'coordinate', 'sparse_by_rows',
 *  'dense', 'diagonal' or 'absent', the latter if access to the Hessian is
 *  via matrix-vector products; lower or upper case variants are allowed
 *
 *  ne is a scalar variable of type int, that holds the number of
 *   entries in the  lower triangular part of H in the sparse co-ordinate
 *   storage scheme. It need not be set for any of the other three schemes.
 *
 *  H_row is a one-dimensional array of size ne and type int, that holds
 *   the row indices of the lower triangular part of H in the sparse
 *   co-ordinate storage scheme. It need not be set for any of the other
 *   three schemes, and in this case can be NULL
 *
 *  H_col is a one-dimensional array of size ne and type int,
 *   that holds the column indices of the  lower triangular part of H in either
 *   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
 *   be set when the dense or diagonal storage schemes are used, and in this 
 *   case can be NULL
 *
 *  H_ptr is a one-dimensional array of size n+1 and type int,
 *   that holds the starting position of  each row of the lower
 *   triangular part of H, as well as the total number of entries plus one,
 *   in the sparse row-wise storage scheme. It need not be set when the
 *   other schemes are used, and in this case can be NULL
 */

/*  *-*-*-*-*-*-*-*-*-*-   A R C _ S O L V E _ W I T H _ H   -*-*-*-*-*-*-*-*-*
 *
 * arc_solve_with_h, an adaptive-regularization method for finding
 *   a local minimizer of a given function
 *
 *   This call is for the case where H is provided specifically, and all
 *   function/derivative information is available by function calls
 */

void arc_solve_with_h(void **data,
                      void *userdata, 
                      int *status, 
                      int n, 
                      real_wp_ x[], 
                      real_wp_ g[],
                      int ne, 
                      int (*eval_f)(
                        int, const real_wp_[], real_wp_*, const void *), 
                      int (*eval_g)(
                        int, const real_wp_[], real_wp_[], const void *),
                      int (*eval_h)(
                        int, int, const real_wp_[], real_wp_[], const void *), 
                      int (*eval_prec)(
                        int, const real_wp_[], real_wp_[], const real_wp_[], 
                        const void *));

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_ARC.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_ARC_double
 *
 *  data is used for internal data.
 *
 *  userdata is an optional user-defined struct that may be NULL. If non-NULL, 
 *   it is used to pass user data to the eval_* functions (see below).
 *
 *   status is a scalar variable of type int, that gives the entry and exit 
  *   status for the package
 *
 *    On initial entry, status must be set to 1
 *
 *    Possible exit values are:
 *
 *     0. The run was succesful
 *
 *    -1. An allocation error occurred. A message indicating the offending
 *        array is written on unit control.error, and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -2. A deallocation error occurred.  A message indicating the offending
 *        array is written on unit control.error and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -3. The restriction n > 0 or requirement that type contains
 *        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
 *          or 'DIAGONAL' has been violated.
 *    -7. The objective function appears to be unbounded from below
 *    -9. The analysis phase of the factorization failed; the return status
 *        from the factorization package is given in the component
 *        inform.factor_status
 *   -10. The factorization failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -11. The solution of a set of linear equations using factors from the
 *        factorization package failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -16. The problem is so ill-conditioned that further progress is impossible.
 *   -18. Too many iterations have been performed. This may happen if
 *        control.maxit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -19. The CPU time limit has been reached. This may happen if
 *        control.cpu_time_limit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -40. The user has forced termination of solver by removing the file named
 *        control.alive_file from unit unit control.alive_unit.
 * 
 *  n is a scalar variable of type int, that holds the number of variables
 * 
 *  x is a one-dimensional array of size n and type double, that holds the 
 *   values x of the optimization variables. The j-th component of x, 
 *   j = 0, ... , n-1, contains x_j.
 * 
 *  g is a one-dimensional array of size n and type double, that holds the 
 *   gradient g of the objective function. The j-th component of g, 
 *   j = 0, ... ,  n-1, contains g_j.
 *
 *  ne is a scalar variable of type int, that holds the number of entries in 
 *   the lower triangular part of the Hessian matrix H.
 * 
 *  eval_f is a function that must have the signature given below:
 * 
 *     int eval_f(int n, const double x[], double *f, const void *userdata) 
 * 
 *   The value of the objective function f(x) evaluated at x must be assigned
 *   to f, and the function return value set to 0. If the evaluation is
 *   impossible at x, return should be set to a nonzero value.
 *
 *  eval_g is a function that must have the signature given below: 
 *  
 *     int eval_g(int n, const double x[], double g[], const void *userdata)
 * 
 *   The components of the gradient nabla_x f(x) of the objective function
 *   evaluated at x must be assigned to g, and the function return value set to
 *   0. If the evaluation is impossible at x, return should be set to a nonzero
 *   value.
 *
 *  eval_h is a function that must have the signature given below: 
 * 
 *     int eval_h(int n, int ne, const double x[], double h[],
 *               const void *userdata)
 * 
 *   The nonzeros of the Hessian nabla_xx f(x) of the objective function
 *   evaluated at x must be assigned to h in the same order as presented in H,
 *   and the function return value set to 0. If the evaluation is impossible at
 *   x, return should be set to a nonzero value.
 *
 *  eval_prec is an optional function that may be NULL. If non-NULL, it must 
 *  have the signature given below:
 *
 *     int eval_prec(int n, const double x[], double u[], const double v[],
 *                   const void *userdata)
 * 
 *   The product u = P(x) v of the user's preconditioner P(x) evaluated at x
 *   with the vector v, the result u must be retured in u, and the function
 *   return value set to 0. If the evaluation is impossible at x, return should
 *   be set to a nonzero value.
 */ 

/*  *-*-*-*-*-*-*-*-*-   A R C _ S O L V E _ W I T H O U T _ H   -*-*-*-*-*-*-*
 *
 * arc_solve_without_h, an adaptive-regularization method for finding
 *   a local minimizer of a given function
 *
 *   This call is for the case where access to H is provided by Hessian-vector
 *   products, and all function/derivative information is available by 
 *   function calls
 */

void arc_solve_without_h(void **data,
                         void *userdata, 
                         int *status, 
                         int n, 
                         real_wp_ x[], 
                         real_wp_ g[], 
                         int (*eval_f)(
                           int, const real_wp_[], real_wp_*, const void *), 
                         int (*eval_g)(
                           int, const real_wp_[], real_wp_[], const void *), 
                         int (*eval_hprod)(
                           int, const real_wp_[], real_wp_[], const real_wp_[], 
                           bool, const void *), 
                         int (*eval_prec)(
                           int, const real_wp_[], real_wp_[], const real_wp_[], 
                           const void *));

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_ARC.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_ARC_double
 *
 *  data is used for internal data.
 *
 *  userdata is an optional user-defined struct that may be NULL. If non-NULL,
 *   it is used to pass user data to the eval_* functions (see below).
 * 
 *  status is a scalar variable of type int, that gives the entry and exit 
*     status for the package. 
 *
 *    On initial entry, status must be set to 1.
 *
 *    Possible exit values are:
 *
 *     0. The run was succesful
 *
 *    -1. An allocation error occurred. A message indicating the offending
 *        array is written on unit control.error, and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -2. A deallocation error occurred.  A message indicating the offending
 *        array is written on unit control.error and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -3. The restriction n > 0 or requirement that type contains
 *        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
 *          or 'DIAGONAL' has been violated.
 *    -7. The objective function appears to be unbounded from below
 *    -9. The analysis phase of the factorization failed; the return status
 *        from the factorization package is given in the component
 *        inform.factor_status
 *   -10. The factorization failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -11. The solution of a set of linear equations using factors from the
 *        factorization package failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -16. The problem is so ill-conditioned that further progress is impossible.
 *   -18. Too many iterations have been performed. This may happen if
 *        control.maxit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -19. The CPU time limit has been reached. This may happen if
 *        control.cpu_time_limit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -40. The user has forced termination of solver by removing the file named
 *        control.alive_file from unit unit control.alive_unit.
 *
 *  n is a scalar variable of type int, that holds the number of variables
 * 
 *  x is a one-dimensional array of size n and type double, that holds the 
 *   values x of the optimization variables. The j-th component of x, 
 *   j = 0, ... , n-1, contains x_j.
 * 
 *  g is a one-dimensional array of size n and type double, that holds the 
 *   gradient g of the objective function. The j-th component of g, 
 *   j = 0, ... ,  n-1, contains g_j.
 * 
 *  eval_f is a function that must have the signature given below:
 * 
 *     int eval_f(int n, const double x[], double *f, const void *userdata) 
 * 
 *   The value of the objective function f(x) evaluated at x must be assigned
 *   to f, and the function return value set to 0. If the evaluation is
 *   impossible at x, return should be set to a nonzero value.
 *
 *  eval_g is a function that must have the signature given below: 
 *  
 *     int eval_g(int n, const double x[], double g[], const void *userdata)
 * 
 *   The components of the gradient nabla_x f(x) of the objective function
 *   evaluated at x must be assigned to g, and the function return value set to
 *   0. If the evaluation is impossible at x, return should be set to a nonzero
 *   value.
 *
 *  eval_hprod is a function that must have the signature given below:
 * 
 *     int eval_hprod(int n, const double x[], double u[], const double v[], 
 *                    bool got_h, const void *userdata)
 * 
 *   The sum u + nabla_xx f(x) v of the product of the Hessian nabla_xx f(x) of
 *   the objective function evaluated at x with the vector v and the vector u
 *   must be returned in u, and the function return value set to 0. If the
 *   evaluation is impossible at x, return should be set to a nonzero value.
 *   The Hessian has already been evaluated or used at x if got_h is true.
 * 
 *  eval_prec is an optional function that may be NULL. If non-NULL, it must
 *  have the signature given below:
 *
 *     int eval_prec(int n, const double x[], double u[], const double v[],
 *                   const void *userdata)
 * 
 *   The product u = P(x) v of the user's preconditioner P(x) evaluated at x
 *   with the vector v, the result u must be retured in u, and the function
 *   return value set to 0. If the evaluation is impossible at x, return should
 *   be set to a nonzero value.
 */  

/*  *-*-*-*-*-*-   A R C _ S O L V E _ R E V E R S E _ W I T H _ H   -*-*-*-*-*
 *
 * arc_solve_reverse_with_h, an adaptive-regularization method for finding 
 *   a local minimizer of a given function
 *
 *   This call is for the case where H is provided specifically, but
 *   function/derivative information is only available by returning to the 
 *   calling procedure
 */

void arc_solve_reverse_with_h(void **data,
                              int *status, 
                              int *eval_status, 
                              int n, 
                              real_wp_ x[], 
                              real_wp_ f, 
                              real_wp_ g[], 
                              int ne, 
                              real_wp_ H_val[], 
                              const real_wp_ u[], 
                              real_wp_ v[]);

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_ARC.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_ARC_double
 *
 *  data is used for internal data.
 *
 *  status is a scalar variable of type int, that gives the entry and exit 
*     status for the package. 
 *
 *    On initial entry, status must be set to 1.
 *
 *    Possible exit values are:
 *
 *     0. The run was succesful
 *
 *    -1. An allocation error occurred. A message indicating the offending
 *        array is written on unit control.error, and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -2. A deallocation error occurred.  A message indicating the offending
 *        array is written on unit control.error and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -3. The restriction n > 0 or requirement that type contains
 *        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
 *          or 'DIAGONAL' has been violated.
 *    -7. The objective function appears to be unbounded from below
 *    -9. The analysis phase of the factorization failed; the return status
 *        from the factorization package is given in the component
 *        inform.factor_status
 *   -10. The factorization failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -11. The solution of a set of linear equations using factors from the
 *        factorization package failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -16. The problem is so ill-conditioned that further progress is impossible.
 *   -18. Too many iterations have been performed. This may happen if
 *        control.maxit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -19. The CPU time limit has been reached. This may happen if
 *        control.cpu_time_limit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -40. The user has forced termination of solver by removing the file named
 *        control.alive_file from unit unit control.alive_unit.
 *
 *     2. The user should compute the objective function value f(x) at the point
 *        x indicated in x and then re-enter the function. The required value
 *        should be set in f, and eval_status should be set to 0. If the user
 *        is unable to evaluate f(x) - for instance, if the function is
 *        undefined at x - the user need not set f, but should then set
 *        eval_status to a non-zero value.
 *     3. The user should compute the gradient of the objective function
 *        nabla_x f(x) at the point x indicated in x and then re-enter the
 *        function. The value of the i-th component of the gradient should be
 *        set in g[i], for i = 0, ..., n-1 and eval_status should be set to 0.
 *        If the user is unable to evaluate a component of nabla_x f(x) - for
 *        instance if a component of the gradient is undefined at x - the user
 *        need not set g, but should then set eval_status to a non-zero value.
 *     4. The user should compute the Hessian of the objective function
 *        nabla_xx f(x) at the point x indicated in x and then re-enter the
 *        function. The value l-th component of the Hessian stored according to
 *        the scheme input in the remainder of H should be set in H_val[l], for
 *        l = 0, ..., ne-1 and eval_status should be set to 0. If the user is
 *        unable to evaluate a component of nabla_xx f(x) - for instance, if a
 *        component of the Hessian is undefined at x - the user need not set
 *        H_val, but should then set eval_status to a non-zero value.
 *     6. The user should compute the product u = P(x)v of their preconditioner
 *        P(x) at the point x indicated in x with the vector v and then
 *        re-enter the function. The vectors v is given in v, the resulting
 *        vector u = P(x)v should be set in u and eval_status should be set to
 *        0. If the user is unable to evaluate the product - for instance, if
 *        a component of the preconditioner is undefined at x - the user need
 *        not set u, but should then set eval_status to a non-zero value.
 * 
 *  eval_status is a scalar variable of type int, that is used to indicate if
 *   objective function/gradient/Hessian values can be provided (see above) 
 * 
 *  n is a scalar variable of type int, that holds the number of variables
 *
 *  x is a one-dimensional array of size n and type double, that holds the 
 *   values x of the optimization variables. The j-th component of x, 
 *   j = 0, ... , n-1, contains x_j.
 * 
 *  f is a scalar variable pointer of type double, that holds the value of the
 *   objective function.
 * 
 *  g is a one-dimensional array of size n and type double, that holds the 
 *   gradient g of the objective function. The j-th component of g, 
 *   j = 0, ... ,  n-1, contains g_j.
 * 
 *  ne is a scalar variable of type int, that holds the number of entries in 
 *   the lower triangular part of the Hessian matrix H.
 *
 *  H_val is a one-dimensional array of size ne and type double, that holds the 
 *   values of the entries of the lower triangular part of the Hessian matrix 
 *   H in any of the available storage schemes.
 * 
 *  u is a one-dimensional array of size n and type double, that holds the
 *   product u = P(x)v of the preconditioner P(x) at the point x indicated in
 *   x with the vector v (see above for details)
 * 
 *  v is a one-dimensional array of size n and type double, that holds the
 *   the vector v for which the product u = P(x)v of the preconditioner P(x)
 *   at the point x indicated in x is computed (see above for details)
 */ 

/*  *-*-*-*-   A R C _ S O L V E _ R E V E R S E _ W I T H O U T _ H   -*-*-*-*
 *
 * arc_solve_reverse_without_h, an adaptive-regularization method for finding 
 *   a local minimizer of a given function
 *
 *   This call is for the case where access to H is provided by Hessian-vector
 *   products, but function/derivative information is only available by 
 *   returning to the calling procedure
 */

void arc_solve_reverse_without_h(void **data,
                                 int *status, 
                                 int *eval_status, 
                                 int n, 
                                 real_wp_ x[], 
                                 real_wp_ f, 
                                 real_wp_ g[], 
                                 real_wp_ u[], 
                                 real_wp_ v[]);

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_ARC.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_ARC_double
 *
 *  data is used for internal data.
 * 
 *  status is a scalar variable of type int, that gives the entry and exit 
 *    status for the package. 
 *
 *    On initial entry, status must be set to 1.
 *
 *    Possible exit values are:
 *
 *     0. The run was succesful
 *
 *    -1. An allocation error occurred. A message indicating the offending
 *        array is written on unit control.error, and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -2. A deallocation error occurred.  A message indicating the offending
 *        array is written on unit control.error and the returned allocation
 *        status and a string containing the name of the offending array
 *        are held in inform.alloc_status and inform.bad_alloc respectively.
 *    -3. The restriction n > 0 or requirement that type contains
 *        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
 *          or 'DIAGONAL' has been violated.
 *    -7. The objective function appears to be unbounded from below
 *    -9. The analysis phase of the factorization failed; the return status
 *        from the factorization package is given in the component
 *        inform.factor_status
 *   -10. The factorization failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -11. The solution of a set of linear equations using factors from the
 *        factorization package failed; the return status from the factorization
 *        package is given in the component inform.factor_status.
 *   -16. The problem is so ill-conditioned that further progress is impossible.
 *   -18. Too many iterations have been performed. This may happen if
 *        control.maxit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -19. The CPU time limit has been reached. This may happen if
 *        control.cpu_time_limit is too small, but may also be symptomatic of
 *        a badly scaled problem.
 *   -40. The user has forced termination of solver by removing the file named
 *        control.alive_file from unit unit control.alive_unit.
 *
 *     2. The user should compute the objective function value f(x) at the point
 *        x indicated in x and then re-enter the function. The required value
 *        should be set in f, and eval_status should be set to 0. If the user
 *        is unable to evaluate f(x) - for instance, if the function is
 *        undefined at x - the user need not set f, but should then set
 *        eval_status to a non-zero value.
 *     3. The user should compute the gradient of the objective function
 *        nabla_x f(x) at the point x indicated in x and then re-enter the
 *        function. The value of the i-th component of the gradient should be
 *        set in g[i], for i = 0, ..., n-1 and eval_status should be set to 0.
 *        If the user is unable to evaluate a component of nabla_x f(x) - for
 *        instance if a component of the gradient is undefined at x - the user
 *        need not set g, but should then set eval_status to a non-zero value.
 *     5. The user should compute the product nabla_xx f(x)v of the Hessian of
 *        the objective function nabla_xx f(x) at the point x indicated in x
 *        with the vector v and add the result to the vector u and then
 *        re-enter the function. The vectors u and v are given in u and v
 *        respectively, the resulting vector u + nabla_xx f(x)v should be set
 *        in u and eval_status should be set to 0. If the user is unable to
 *        evaluate the product - for instance, if a component of the Hessian is
 *        undefined at x - the user need not alter u, but should then set
 *        eval_status to a non-zero value.
 *     6. The user should compute the product u = P(x)v of their preconditioner
 *        P(x) at the point x indicated in x with the vector v and then
 *        re-enter the function. The vectors v is given in v, the resulting
 *        vector u = P(x)v should be set in u and eval_status should be set to
 *        0. If the user is unable to evaluate the product - for instance, if
 *        a component of the preconditioner is undefined at x - the user need
 *        not set u, but should then set eval_status to a non-zero value.
 *
 *  eval_status is a scalar variable of type int, that is used to indicate if
 *   objective function/gradient/Hessian values can be provided (see above) 
 * 
 *  n is a scalar variable of type int, that holds the number of variables
 *
 *  x is a one-dimensional array of size n and type double, that holds the 
 *   values x of the optimization variables. The j-th component of x, 
 *   j = 0, ... , n-1, contains x_j.
 * 
 *  f is a scalar variable pointer of type double, that holds the value of the
 *   objective function.
 * 
 *  g is a one-dimensional array of size n and type double, that holds the 
 *   gradient g of the objective function. The j-th component of g, 
 *   j = 0, ... ,  n-1, contains g_j.
 * 
 *  u is a one-dimensional array of size n and type double, that is used for
 *   reverse communication (see above for details)
 * 
 *  v is a one-dimensional array of size n and type double, that is used for
 *   reverse communication (see above for details)
 */  

/*  *-*-*-*-*-*-*-*-*-*-   A R C _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*
 *
 * arc_information fills the output information structure inform 
 * (see arc_inform_type above)
 */

void arc_information(void **data,
                     struct arc_inform_type *inform,
                     int *status);

/*
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_ARC.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_ARC_double
 *
 *  data is used for internal data.
 *
 *  inform is a struct containing output information
 *
 *  status is a scalar variable of type int, that gives the exit 
 *    status for the package. 
 *
 *    Possible exit values are (currently):
 *
 *     0. The information was retrieved succesfully
 *
 */

/*  *-*-*-*-*-*-*-*-*-*-   A R C _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*
 *
 * Deallocate all private storage
 */

void arc_terminate(void **data, 
                   struct arc_control_type *control, 
                   struct arc_inform_type *inform);

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
