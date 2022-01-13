//* \file wcp.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-13 AT 16:05 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_WCP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 13th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package wcp
 
  \section wcp_intro Introduction

  \subsection wcp_purpose Purpose

  \subsection wcp_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection wcp_date Originally released

  \subsection wcp_terminology Terminology

  \subsection wcp_method Method

  \subsection wcp_references Reference

  \subsection wcp_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_WCP_H 
#define GALAHAD_WCP_H

// precision
#include "galahad_precision.h"

// required packages
#include "fdc.h"
#include "sbls.h"

/**
 * control derived type as a C struct
 */
struct wcp_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief
    /// the level of output required is specified by print_level
    int print_level;

    /// \brief
    /// any printing will start on this iteration
    int start_print;

    /// \brief
    /// any printing will stop on this iteration
    int stop_print;

    /// \brief
    /// at most maxit inner iterations are allowed
    int maxit;

    /// \brief
    /// how to choose the initial point. Possible values are
    /// 0  the values input in X, shifted to be at least prfeas from
    /// their nearest bound, will be used
    /// 1  the nearest point to the "bound average" 0.5(X_l+X_u) that satisfies
    /// the linear constraints will be used
    int initial_point;

    /// \brief
    /// the factorization to be used. Possible values are
    /// 0  automatic
    /// 1  Schur-complement factorization
    /// 2  augmented-system factorization                              (OBSOLETE
    int factor;

    /// \brief
    /// the maximum number of nonzeros in a column of A which is permitted
    /// with the Schur-complement factorization                          (OBSOLE
    int max_col;

    /// \brief
    /// an initial guess as to the integer workspace required by SBLS     (OBSOL
    int indmin;

    /// \brief
    /// an initial guess as to the real workspace required by SBLS        (OBSOL
    int valmin;

    /// \brief
    /// the maximum number of iterative refinements allowed               (OBSOL
    int itref_max;

    /// \brief
    /// the number of iterations for which the overall infeasibility of the
    /// problem is not reduced by at least a factor .required_infeas_reduction
    /// before the problem is flagged as infeasible (see required_infeas_reducti
    int infeas_max;

    /// \brief
    /// the strategy used to reduce relaxed constraint bounds. Possible values a
    /// 0 do not perturb the constraints
    /// 1 reduce all perturbations by the same amount with linear reduction
    /// 2 reduce each perturbation as much as possible with linear reduction
    /// 3 reduce all perturbations by the same amount with superlinear reduction
    /// 4 reduce each perturbation as much as possible with superlinear reductio
    int perturbation_strategy;

    /// \brief
    /// indicate whether and how much of the input problem should be restored
    /// on output. Possible values are
    /// 0 nothing restored
    /// 1 scalar and vector parameters
    /// 2 all parameters
    int restore_problem;

    /// \brief
    /// any bound larger than infinity in modulus will be regarded as infinite
    real_wp_ infinity;

    /// \brief
    /// the required accuracy for the primal infeasibility
    real_wp_ stop_p;

    /// \brief
    /// the required accuracy for the dual infeasibility
    real_wp_ stop_d;

    /// \brief
    /// the required accuracy for the complementarity
    real_wp_ stop_c;

    /// \brief
    /// initial primal variables will not be closer than prfeas from their bound
    real_wp_ prfeas;

    /// \brief
    /// initial dual variables will not be closer than dufeas from their bounds
    ///
    real_wp_ dufeas;

    /// \brief
    /// the target value of the barrier parameter. If mu_target is not positive,
    /// it will be reset to an appropriate value
    real_wp_ mu_target;

    /// \brief
    /// the complemtary slackness x_i.z_i will be judged to lie within an
    /// acceptable margin around its target value mu as soon as
    /// mu_accept_fraction * mu <= x_i.z_i <= ( 1 / mu_accept_fraction ) * mu;
    /// the perturbations will be reduced as soon as all of the complemtary
    /// slacknesses x_i.z_i lie within acceptable bounds. mu_accept_fraction
    /// will be reset to ensure that it lies in the interval (0,1]
    real_wp_ mu_accept_fraction;

    /// \brief
    /// the target value of the barrier parameter will be increased by
    /// mu_increase_factor for infeasible constraints every time the
    /// perturbations are adjusted
    ///
    real_wp_ mu_increase_factor;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least
    /// a factor required_infeas_reduction over .infeas_max iterations, the
    /// problem is flagged as infeasible (see infeas_max)
    real_wp_ required_infeas_reduction;

    /// \brief
    /// any primal or dual variable that is less feasible than implicit_tol will
    /// be regarded as defining an implicit constraint
    real_wp_ implicit_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization.
    /// See the documentation for SBLS for details                       (OBSOLE
    real_wp_ pivot_tol;

    /// \brief
    /// the threshold pivot used by the matrix factorization when attempting to
    /// detect linearly dependent constraints.
    /// See the documentation for SBLS for details                       (OBSOLE
    real_wp_ pivot_tol_for_dependencies;

    /// \brief
    /// any pivots smaller than zero_pivot in absolute value will be regarded to
    /// zero when attempting to detect linearly dependent constraints    (OBSOLE
    real_wp_ zero_pivot;

    /// \brief
    /// the constraint bounds will initially be relaxed by .perturb_start;
    /// this perturbation will subsequently be reduced to zero.
    /// If perturb_start < 0, the amount by which the bounds are relaxed will
    /// be computed automatically
    real_wp_ perturb_start;

    /// \brief
    /// the test for rank defficiency will be to factorize
    /// ( alpha_scale I  A^T )
    /// (       A          0 )
    real_wp_ alpha_scale;

    /// \brief
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha
    /// identical_bounds_tol will be reset to the average of their values
    real_wp_ identical_bounds_tol;

    /// \brief
    /// the constraint perturbation will be reduced as follows:
    ///
    /// - if the variable lies outside a bound, the corresponding perturbation
    /// will be reduced to
    /// reduce_perturb_factor * current pertubation
    /// + ( 1 - reduce_perturb_factor ) * violation
    /// - otherwise, if the variable lies within insufficiently_feasible of its
    /// bound the pertubation will be reduced to
    /// reduce_perturb_multiplier * current pertubation
    /// - otherwise if will be set to zero
    real_wp_ reduce_perturb_factor;
    /// see reduce_perturb_factor
    real_wp_ reduce_perturb_multiplier;
    /// see reduce_perturb_factor
    real_wp_ insufficiently_feasible;

    /// \brief
    /// if the maximum constraint pertubation is smaller than
    /// perturbation_small and the violation is smaller than implicit_tol, the
    /// method will deduce that there is a feasible point but no interior
    real_wp_ perturbation_small;

    /// \brief
    /// the maximum CPU time allowed (-ve means infinite)
    real_wp_ cpu_time_limit;

    /// \brief
    /// the maximum elapsed clock time allowed (-ve means infinite)
    real_wp_ clock_time_limit;

    /// \brief
    /// the equality constraints will be preprocessed to remove any linear
    /// dependencies if true
    bool remove_dependencies;

    /// \brief
    /// any problem bound with the value zero will be treated as if it were a
    /// general value if true
    bool treat_zero_bounds_as_general;

    /// \brief
    /// if .just_feasible is true, the algorithm will stop as soon as a feasible
    /// point is found. Otherwise, the optimal solution to the problem will be
    /// found
    bool just_feasible;

    /// \brief
    /// if .balance_initial_complementarity is .true. the initial complemetarity
    /// will be balanced
    bool balance_initial_complementarity;

    /// \brief
    /// if .use_corrector, a corrector step will be used
    bool use_corrector;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// if .record_x_status is true, the array inform.X_status will be allocated
    /// and the status of the bound constraints will be reported on exit.
    bool record_x_status;

    /// \brief
    /// if .record_c_status is true, the array inform.C_status will be allocated
    /// and the status of the general constraints will be reported on exit.
    bool record_c_status;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for FDC
    struct fdc_control_type fdc_control;

    /// \brief
    /// control parameters for SBLS
    struct sbls_control_type sbls_control;
};

/**
 * time derived type as a C struct
 */
struct wcp_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_wp_ total;

    /// \brief
    /// the CPU time spent preprocessing the problem
    real_wp_ preprocess;

    /// \brief
    /// the CPU time spent detecting linear dependencies
    real_wp_ find_dependent;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    real_wp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_wp_ factorize;

    /// \brief
    /// the CPU time spent computing the search direction
    real_wp_ solve;

    /// \brief
    /// the total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// the clock time spent preprocessing the problem
    real_wp_ clock_preprocess;

    /// \brief
    /// the clock time spent detecting linear dependencies
    real_wp_ clock_find_dependent;

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
struct wcp_inform_type {

    /// \brief
    /// return status. See LSQP_solve for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// the total number of iterations required
    int iter;

    /// \brief
    /// the return status from the factorization
    int factorization_status;

    /// \brief
    /// the total integer workspace required for the factorization
    int factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int factorization_real;

    /// \brief
    /// the total number of factorizations performed
    int nfacts;

    /// \brief
    /// the number of general constraints that lie on (one) of their bounds for
    /// feasible solutions
    int c_implicit;

    /// \brief
    /// the number of variables that lie on (one) of their bounds for all
    /// feasible solutions
    int x_implicit;

    /// \brief
    /// the number of Lagrange multipliers for general constraints that lie on
    /// (one) of their bounds for all feasible solutions
    int y_implicit;

    /// \brief
    /// the number of dual variables that lie on (one) of their bounds for all
    /// feasible solutions
    int z_implicit;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by LSQP_solve
    real_wp_ obj;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    real_wp_ non_negligible_pivot;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// if control.record_x_status is true, .X_status will be allocated
    /// and the status of the bound constraints will be reported on exit.
    /// In this case, possible values of .X_status(i) are as follows:
    /// 0  the variable lies between its bounds
    /// -1  the variable lies on its lower bound for all feasible points
    /// 1  the variable lies on its upper bound for all feasible points
    /// -2  the variable never lies on its lower bound at any feasible point
    /// 2  the variable never lies on its upper bound at any feasible point
    /// 3  the bounds are equal, and the variable takes this value for
    /// all feasible points
    /// -3  the variable never lies on either bound at any feasible point
    /// INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status
    /// if control.record_c_status is true, .C_status will be allocated
    /// and the status of the general constraints will be reported on exit.
    /// In this case, possible values of inform.C_status(i) are as follows:
    /// 0  the constraint lies between its bounds
    /// -1  the constraint lies on its lower bound for all feasible points
    /// and may be fixed at this value and removed from the problem
    /// 1  the constraint lies on its upper bound for all feasible points
    /// and may be fixed at this value and removed from the problem
    /// -2  the constraint never lies on its lower bound at any feasible point
    /// and the bound may be removed from the problem
    /// 2  the constraint never lies on its upper bound at any feasible point
    /// and the bound may be removed from the problem
    /// 3  the bounds are equal, and the constraint takes this value for
    /// all feasible points
    /// -3  the constraint never lies on either bound at any feasible point
    /// and the constraint may be removed from the problem
    /// 4  the constraint is implied by the others and may be removed
    /// from the problem
    /// INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status
    /// timings (see above)
    struct wcp_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    W C P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void wcp_initialize( void **data, 
                     struct wcp_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see wcp_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    W C P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void wcp_read_specfile( struct wcp_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information 
              (see wcp_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    W C P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void wcp_import( struct wcp_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see wcp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
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
  \li -3. The restriction n > 0 or requirement that type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal' or 'absent' has been violated.
*/

// *-*-*-*-*-*-*-    W C P  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void wcp_reset_control( struct wcp_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see wcp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    W C P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void wcp_information( void **data,
                      struct wcp_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see wcp_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    W C P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void wcp_terminate( void **data, 
                    struct wcp_control_type *control, 
                    struct wcp_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see wcp_control_type)

  @param[out] inform   is a struct containing output information
              (see wcp_inform_type)
 */

/** \example wcpt.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
