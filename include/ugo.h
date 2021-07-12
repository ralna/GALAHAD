/*
 * THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_UGO C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes
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
#ifndef GALAHAD_UGO_H 
#define GALAHAD_UGO_H

/* 
 * time derived type as a C struct
 */
struct ugo_time_type {

    // the total CPU time spent in the package
    float total;

    // the total clock time spent in the package
    double clock_total;
};

/* 
 * inform derived type as a C struct
 */
struct ugo_inform_type {

    //  return status. See UGO_solve for details
    int status;

    // evaluation status for reverse communication interface
    int eval_status;

    // the status of the last attempted allocation/deallocation
    int alloc_status ;

    // the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    // the total number of iterations performed
    int iter;

    // the total number of evaluations of the objection function
    int f_eval;

    // the total number of evaluations of the gradient of the objection function
    int g_eval;

    // the total number of evaluations of the Hessian of the objection function
    int h_eval;

    // timings (see above)
    struct ugo_time_type time;
};

/* 
 * control derived type as a C struct
 */
struct ugo_control_type { 

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

    // the number of initial (uniformly-spaced) evaluation points (<2 reset to 2)
    int initial_points;

    // incremenets of storage allocated (less that 1000 will be reset to 1000)
    int storage_increment;

    // unit for any out-of-core writing when expanding arrays
    int buffer;

    // what sort of Lipschitz constant estimate will be used:
    // 1 = global contant provided, 2 = global contant estimated,
    // 3 = local costants estimated
    int lipschitz_estimate_used;

    // how is the next interval for examination chosen:
    // 1 = traditional, 2 = local_improvement
    int next_interval_selection;

    // try refine_with_newton Newton steps from the vacinity of the global
    // minimizer to try to improve the estimate
    int refine_with_newton;

    // removal of the file alive_file from unit alive_unit terminates execution
    int alive_unit;
    char alive_file[31];

    // overall convergence tolerances. The iteration will terminate when
    // the step is less than %stop_length
    double stop_length;

    // if the absolute value of the gradient is smaller than small_g_for_newton, the
    // next evaluation point may be at a Newton estimate of a local minimizer
    double small_g_for_newton;

    // if the absolute value of the gradient at the end of the interval search is
    // smaller than small_g, no Newton serach is necessary
    double small_g;

    // stop if the objective function is smaller than a specified value
    double obj_sufficient;

    // the global Lipschitz constant for the gradient (-ve => unknown)
    double global_lipschitz_constant;

    // the reliability parameter that is used to boost insufficiently large
    // estimates of the Lipschitz constant
    double reliability_parameter;

    // a lower bound on the Lipscitz constant for the gradient (not zero unless
    // the function is constant)
    double lipschitz_lower_bound;

    // the maximum CPU time allowed (-ve means infinite)
    double cpu_time_limit;

    // the maximum elapsed clock time allowed (-ve means infinite)
    double clock_time_limit;

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
};

/*
 * Provide default values for UGO controls
 *
 *   Arguments:
 *
 *   data     private internal data
 *   control  a struct containing control information
 *   inform   a struct containing output information
 */
void ugo_initialize(void **data, struct ugo_control_type *control, struct ugo_inform_type *inform);

/*
 * Read the content of a specification file, and perform the assignment of
 * values associated with given keywords to the corresponding control parameters
 */ 
void ugo_read_specfile(struct ugo_control_type *control, const char specfile[]);

/* 
 *  ugo_solve, a method for finding the global minimizer of a univariate
 *    continuous function with a Lipschitz gradient in an interval
 *
 *  Many ingredients in the algorithm are based on the paper
 *
 *   Daniela Lera and Yaroslav D. Sergeyev,
 *   "Acceleration of univariate global optimization algorithms working with
 *    Lipschitz functions and Lipschitz first derivatives"
 *   SIAM J. Optimization Vol. 23, No. 1, pp. 508–529 (2013)
 *
 *  but adapted to use 2nd derivatives
 * 
 *  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 *  For full details see the specification sheet for GALAHAD_UGO.
 *
 *   x_l and x_u are scalars of type double, that hold the bound on the
 *    search interval [x_l,x_u]
 *
 *   x is a scalar pointer of type double, that holds the next value of x at
 *    which the user is required to evaluate the objective (and its derivatives)
 *    when inform.status > 0, or the value of the approximate global minimizer
 *    when inform.status = 0
 *
 *   f is a scalar pointer of type double, that must be set by the user to hold
 *    the value of f(x) if required by inform.status > 0 (see below), and will
 *    return the value of the approximate global minimum when inform.status = 0
 *
 *   g is a scalar pointer of type double, that must be set by the user to hold
 *    the value of f'(x) if required by inform.status > 0 (see below), and will
 *    return the value of the first derivative of f at the approximate global
 *    minimizer when inform.status = 0
 *
 *   h is a scalar pointer of type double, that must be set by the user to hold
 *    the value of f''(x) if required by inform.status > 0 (see below), and will
 *    return the value of the second derivative of f at the approximate global
 *    minimizer when inform.status = 0
 *
 * control is a struct of type ugo_control_type. See above for details
 *
 * inform is a struct of type ugo_inform_type. See above 
 *  for further details. On initial entry, inform.status should be set to 1.
 *  The user must check the value of inform.status on exit, and if required
 *  provide further information and re-enter the function. Possible values
 *  are:
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
 *    -7. The objective function appears to be unbounded from below
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
 *        x and then re-enter the function. The required value should be set
 *        in f, and inform.eval_status should be set to 0.
 *        If the user is unable to evaluate f(x) - for instance, if the function
 *        is undefined at x - the user need not set f, but should then set
 *        inform.eval_status to a non-zero value.
 *     3. The user should compute the objective function value f(x) and its
 *        derivative f'(x) at the point x and then re-enter the function.
 *        The required values should be set in f and g respectively,
 *        and inform.eval_status should be set to 0.
 *        If the user is unable to evaluate f(x) or f'(x) - for instance, if the
 *        function or its derivative is undefined at x - the user need not
 *        set f or g, but should then set inform.eval_status to a non-zero value.
 *     4. The user should compute the objective function value f(x) and its
 *        first two derivatives f'(x) and f''(x) at the point x and then
 *        re-enter the function. The required values should be set in f, g
 *        and h respectively, and inform.eval_status should be set to 0. If the
 *        user is unable to evaluate f(x), f'(x) or f''(x) - for instance, if
 *        the function or its derivatives are undefined at x - the user need not
 *        set f, g or h, but should then set inform.eval_status to a non-zero value
 *
 *  data is used for internal data.
 *
 *  userdata is an optional user-defined struct that may be NULL. If non-NULL, it is
 *   used to pass user data to the eval_fgh function (see below).
 * 
 *  eval_fgh is an optional function that may be NULL. If non-NULL, it must have the
 *   signature given below: 
 * 
 *     int eval_fgh(double x, double *f, double *g, double *h, const void *userdata)
 * 
 *   The value of the objective
 *   function f(x) and its first two derivative f'(x) and f''(x) evaluated at
 *   x=x must be assigned to f, g and h respectively, and the function return value
 *   set to 0. If the evaluation is impossible at x, return should be set to a
 *   nonzero value. If eval_fgh is NULL, ugo_solve will return to the
 *   user with inform.status = 4 each time an evaluation is required.
 */
void ugo_solve(double x_l, double x_u, double *x, double *f, double *g, double *h,
               const struct ugo_control_type *control, struct ugo_inform_type *inform, void **data,
               void *userdata, int (*eval_fgh)(double, double*, double*, double*, const void *));

/*
 * Deallocate all private storage
 */
void ugo_terminate(void **data, struct ugo_control_type *control, struct ugo_inform_type *inform);

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
