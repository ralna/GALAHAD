/*
 * THIS VERSION: GALAHAD 3.3 - 03/08/2021 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 3rd 2021
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
#ifndef GALAHAD_BGO_H 
#define GALAHAD_BGO_H

// precision
#include "galahad_precision.h"

// required packages
#include "trb.h"
#include "ugo.h"
#include "lhs.h"

/* 
 * time derived type as a C struct 
 */
struct bgo_time_type {

    // the total CPU time spent in the package
    real_sp_ total;

    // the CPU time spent performing univariate global optimization
    real_sp_ univariate_global;

    // the CPU time spent performing multivariate local optimization
    real_sp_ multivariate_local;

    // the total clock time spent in the package
    real_wp_ clock_total;

    // the clock time spent performing univariate global optimization
    real_wp_ clock_univariate_global;

    // the clock time spent performing multivariate local optimization
    real_wp_ clock_multivariate_local;
};

/*
 * inform derived type as a C struct
 */
struct bgo_inform_type {

    // return status. See BGO_solve for details
    int status;

    // the status of the last attempted allocation/deallocation
    int alloc_status ;

    // the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    // the total number of evaluations of the objection function
    int f_eval;

    // the total number of evaluations of the gradient of the objection function
    int g_eval;

    // the total number of evaluations of the Hessian of the objection function
    int h_eval;

    // the value of the objective function at the best estimate of the solution
    // determined by BGO_solve
    real_wp_ obj;

    // the norm of the projected gradient of the objective function at the best
    // estimate of the solution determined by BGO_solve
    real_wp_ norm_pg;

    // timings (see above)
    struct bgo_time_type time;

    // inform parameters for TRB
    struct trb_inform_type trb_inform;

    // inform parameters for UGO
    struct ugo_inform_type ugo_inform;

    // inform parameters for LHS
    struct lhs_inform_type lhs_inform;
};

/*
 * control derived type as a C struct
 */
struct bgo_control_type { 

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

    // the maximum number of random searches from the best point found so far
    int attempts_max;

    // the maximum number of function evaluations made
    int max_evals;

    // sampling strategy used, 1=uniform,2=Latin hyper-cube,3=2+1
    int sampling_strategy;

    // hyper-cube discretization (for sampling stategies 2 and 3)
    int hypercube_discretization;

    // removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    char alive_file[31];

    // any bound larger than infinity in modulus will be regarded as infinite
    real_wp_ infinity;

    // the smallest value the objective function may take before the problem
    // is marked as unbounded
    real_wp_ obj_unbounded;

    // the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    // the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    // perform random-multistart as opposed to local minimize and probe
    bool random_multistart;

    // is the Hessian matrix of second derivatives available or is access only
    // via matrix-vector products?
    bool hessian_available;

    // if %space_critical true, every effort will be made to use as little
    // space as possible. This may result in longer computation time
    bool space_critical;

    // if %deallocate_error_fatal is true, any array/pointer deallocation error
    // will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    // all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
    // where %prefix contains the required string enclosed in
    // quotes, e.g. "string" or 'string'
    char prefix[31];

    // control parameters for TRB
    struct trb_control_type trb_control;

    // control parameters for UGO
    struct ugo_control_type ugo_control;

    // control parameters for LHS
    struct lhs_control_type lhs_control;
};

/*  *-*-*-*-*-*-*-*-*-*-   B G O _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*
 *
 * Provide default values for BGO controls
 */

void bgo_initialize( void **data, 
                     struct bgo_control_type *control,
                     struct bgo_inform_type *inform );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *   data     private internal data
 *   control  a struct containing default control information (see above)
 *   inform   a struct containing output information (see above)
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-   B G O _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*-*-*
 *
 * Read the content of a specification file, and perform the assignment of
 * values associated with given keywords to the corresponding control parameters
 */

void bgo_read_specfile( struct bgo_control_type *control, 
                        const char specfile[] );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *   control  a struct containing control information (see above)
 *   specfile a character string containing the name of the specfile
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-*-*-*-   B G O _ I M P O R T    -*-*-*-*-*-*-*-*-*-*-*-*
 *
 * Import problem data into internal storage prior to solution. 
 */

void bgo_import( struct bgo_control_type *control,
                 void **data,
                 int *status, 
                 int n, 
                 const real_wp_ x_l[], 
                 const real_wp_ x_u[],
                 const char H_type[], 
                 int ne, 
                 const int H_row[],
                 const int H_col[], 
                 const int H_ptr[] );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  control is a struct whose members are described in 
 *   the leading comments to bgo_solve
 *
 *  data is used for internal data
 *
 *  status is a scalar variable of type int, that gives
 *   the exit status from the package. Possible values are:
 *
 *     1. The import was succesful, and the package is ready for the solve phase
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
 *  n is a scalar variable of type int, that holds the number of variables
 *
 *  x_l is a one-dimensional array of size n and type double, that holds the
 *   values x_l of the lower bounds on the optimization variables x. The j-th 
 *   component of x_l, j = 0, ... , n-1, contains (x_l)j.
 *
 *  x_u is a one-dimensional array of size n and type double, that holds the 
 *   values x_u of the upper bounds on the optimization variables x. The j-th 
 *   component of x_u, j = 0, ... , n-1, contains (x_u)j.
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
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-   B G O _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*
 *
 * Reset control parameters after import if required
 */

void bgo_reset_control( struct bgo_control_type *control,
                        void **data,
                        int *status, );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  control is a struct whose members are described in 
 *   the leading comments to bgo_solve
 *
 *  data is used for internal data
 *
 *  status is a scalar variable of type int, that gives
 *   the exit status from the package. Possible values are:
 *
 *     1. The import was succesful, and the package is ready for the solve phase
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-   B G O _ S O L V E _ W I T H _ M A T   -*-*-*-*-*-*-*-*
 *
 * bgo_solve_with_mat, a stochastic multistart method that aims to find
 *   the global minimizer of a given function where the variables are 
 *   constrained to lie in a "box"
 *
 *   This call is for the case where H is provided specifically, and all
 *   function/derivative information is available by function calls
 */

void bgo_solve_with_mat( void **data,
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
                         int (*eval_hprod)(
                           int, const real_wp_[], real_wp_[], const real_wp_[], 
                           bool, const void * ),
                         int (*eval_prec)(
                           int, const real_wp_[], real_wp_[], const real_wp_[], 
                           const void * ) );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  For full details see the specification sheet for GALAHAD_BGO.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_BGO_double
 *
 *  data is used for internal data.
 *
 *  userdata is an optional user-defined struct that may be NULL. If non-NULL, 
 *   it is used to pass user data to the eval_* functions (see below).
 *
 *   status is a scalar variable of type int, that gives the entry and exit 
 *    status for the package
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
 *  eval_hprod is a function that must have the signature given below:
 * 
 *     int eval_hprod(int n, const double x[], double u[], const double v[], 
                      bool got_h, const void *userdata)
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
 *
 *  ---------------------------------------------------------------------------
 */ 

/*  *-*-*-*-*-*-*-*-   B G O _ S O L V E _ W I T H O U T _ M A T   -*-*-*-*-*-*
 *
 * bgo_solve_without_mat, a stochastic multistart method that aims to find
 *   the global minimizer of a given function where the variables are 
 *   constrained to lie in a "box"
 *
 *   This call is for the case where access to H is provided by Hessian-vector
 *   products, and all function/derivative information is available by 
 *   function calls
 */

void bgo_solve_without_mat( void **data,
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
                            int (*eval_shprod)(int, const real_wp_[], int, 
                              const int[], const real_wp_[], int*, int[], 
                              real_wp_[], bool, const void * ), 
                            int (*eval_prec)(
                              int, const real_wp_[], real_wp_[], 
                              const real_wp_[], const void * ) );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  For full details see the specification sheet for GALAHAD_BGO.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_BGO_double
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
 *  eval_shprod is a function that must have the signature given below:
 * 
 *     int eval_shprod(int n, const double x[], int nnz_v, 
 *                     const int index_nz_v[], const double v[], 
 *                     int *nnz_u, int index_nz_u[], double u[], 
 *                     bool got_h, const void *userdata)
 * 
 *   The product u = nabla_xx f(x) v of the Hessian nabla_xx f(x) of the
 *   objective function evaluated at x with the sparse vector v must be
 *   returned in u, and the function return value set to 0. Only the components
 *   index_nz_v[0:nnz_v-1] of v are nonzero, and the remaining components may
 *   not have been be set. On exit, the user must indicate the nnz_u indices of
 *   u that are nonzero in index_nz_u[0:nnz_u-1], and only these components of
 *   u need be set. If the evaluation is impossible at x, return should be set
 *   to a nonzero value. The Hessian has already been evaluated or used at x if
 *   got_h is true.
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
 *
 *  ---------------------------------------------------------------------------
 */  

/*  *-*-*-*-*-*-   B G O _ S O L V E _ R E V E R S E _ W I T H _ M A T  -*-*-*-*
 *
 * bgo_solve_reverse_with_mat, a stochastic multistart method that aims to find
 *   the global minimizer of a given function where the variables are 
 *   constrained to lie in a "box"
 *
 *   This call is for the case where H is provided specifically, but
 *   function/derivative information is only available by returning to the 
 *   calling procedure
 */

void bgo_solve_reverse_with_mat( void **data,
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

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  For full details see the specification sheet for GALAHAD_BGO.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_BGO_double
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
 *
 *  ---------------------------------------------------------------------------
 */ 

/*  *-*-*-*-   B G O _ S O L V E _ R E V E R S E _ W I T H O U T _ H   -*-*-*-*
 *
 * bgo_solve_reverse_without_mat, a stochastic multistart method that aims to
 *   find the global minimizer of a given function where the variables are 
 *   constrained to lie in a "box"
 *
 *   This call is for the case where access to H is provided by Hessian-vector
 *   products, but function/derivative information is only available by 
 *   returning to the calling procedure
 */

void bgo_solve_reverse_without_mat( void **data,
                                  int *status, 
                                  int *eval_status, 
                                  int n, 
                                  real_wp_ x[], 
                                  real_wp_ f, 
                                  real_wp_ g[], 
                                  real_wp_ u[], 
                                  real_wp_ v[],
                                  int index_nz_v[], 
                                  int *nnz_v, 
                                  const int index_nz_u[], 
                                  int nnz_u );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  For full details see the specification sheet for GALAHAD_BGO.
 *
 *  ** NB. default real/complex means double precision real/complex in
 *  ** GALAHAD_BGO_double
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
 *     7. The user should compute the product u = nabla_xx f(x)v of the Hessian
 *        of the objective function nabla_xx f(x) at the point x indicated in
 *        x with the *sparse* vector v and then re-enter the function.
 *        The nonzeros of v are stored in
 *          v[index_nz_v[0:nnz_v-1]]
 *        while the nonzeros of u should be returned in
 *          u[index_nz_u[0:nnz_u-1]];
 *        the user must set nnz_u and index_nz_u accordingly, and set
 *        eval_status to 0. If the user is unable to evaluate the product - for
 *        instance, if a component of the Hessian is undefined at x - the user
 *        need not alter u, but should then set eval_status to a non-zero value.
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
 *
 *  index_nz_v is a one-dimensional array of size n and type int, that is used
 *   for reverse communication (see above for details)
 * 
 *  nnz_v is a scalar variable of type int, that is used for reverse
 *   communication (see above for details)
 * 
 *  index_nz_u s a one-dimensional array of size n and type int, that is used
 *   for reverse communication (see above for details)
 * 
 *  nnz_u is a scalar variable of type int, that is used for reverse
 *   communication (see above for details)
 *
 *  ---------------------------------------------------------------------------
 */  

/*  *-*-*-*-*-*-*-*-*-*-*-   B G O _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*-*
 *
 * bgo_information fills the output information structure inform 
 * (see bgo_inform_type above)
 */

void bgo_information( void **data,
                      struct bgo_inform_type *inform,
                      int *status );

/*  ------------------------  A R G U M E N T S  ------------------------------
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
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-*-*-   B G O _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*-*
 *
 * Deallocate all private storage
 */

void bgo_terminate( void **data, 
                    struct bgo_control_type *control, 
                    struct bgo_inform_type *inform );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  see bgo_initialize above
 *
 *  ---------------------------------------------------------------------------
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
