//* \file lsqp.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-13 AT 15:54 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LSQP C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package lsqp
 
  \section lsqp_intro Introduction

  \subsection lsqp_purpose Purpose

  \subsection lsqp_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection lsqp_date Originally released

  \subsection lsqp_terminology Terminology

  \subsection lsqp_method Method

  \subsection lsqp_references Reference

  \subsection lsqp_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_LSQP_H 
#define GALAHAD_LSQP_H

// precision
#include "galahad_precision.h"

// required packages
#include "fdc.h"
#include "sbls.h"

/**
 * control derived type as a C struct
 */
struct lsqp_control_type {

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
    /// the number of iterations for which the overall infeasibility
    /// of the problem is not reduced by at least a factor .reduce_infeas
    /// before the problem is flagged as infeasible (see reduce_infeas)
    int infeas_max;

    /// \brief
    /// the initial value of the barrier parameter will not be changed for the
    /// first muzero_fixed iterations
    ///
    int muzero_fixed;

    /// \brief
    /// indicate whether and how much of the input problem
    /// should be restored on output. Possible values are
    /// 0 nothing restored
    /// 1 scalar and vector parameters
    /// 2 all parameters
    int restore_problem;

    /// \brief
    /// specifies the type of indicator function used. Pssible values are
    /// 1 primal indicator: constraint active <=> distance to nearest bound
    /// <= .indicator_p_tol
    /// 2 primal-dual indicator: constraint active <=> distance to nearest bound
    /// <= .indicator_tol_pd * size of corresponding multiplier
    /// 3 primal-dual indicator: constraint active <=> distance to nearest bound
    /// <= .indicator_tol_tapia * distance to same bound at previous iteration
    int indicator_type;

    /// \brief
    /// should extrapolation be used to track the central path? Possible values
    /// 0 never
    /// 1 after the final major iteration
    /// 2 at each major iteration
    int extrapolate;

    /// \brief
    /// the maximum number of previous path points to use when fitting the data
    int path_history;

    /// \brief
    /// the maximum order of path derivative to use
    ///
    int path_derivatives;

    /// \brief
    /// the order of (Puiseux) series to fit to the path data: <=0 to fit all da
    int fit_order;

    /// \brief
    /// specifies the unit number to write generated SIF file describing the
    /// current problem
    int sif_file_device;

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
    /// the initial value of the barrier parameter. If muzero is not positive,
    /// it will be reset to an appropriate value
    real_wp_ muzero;

    /// \brief
    /// if the overall infeasibility of the problem is not reduced by at least a
    /// factor reduce_infeas over .infeas_max iterations, the problem is flagged
    /// as infeasible (see infeas_max)
    real_wp_ reduce_infeas;

    /// \brief
    /// if W=0 and the potential function value is smaller than
    /// potential_unbounded * number of one-sided bounds,
    /// the analytic center will be flagged as unbounded
    real_wp_ potential_unbounded;

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
    /// any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha
    /// identical_bounds_tol will be reset to the average of their values
    real_wp_ identical_bounds_tol;

    /// \brief
    /// start terminal extrapolation when mu reaches mu_min
    real_wp_ mu_min;

    /// \brief
    /// if .indicator_type = 1, a constraint/bound will be
    /// deemed to be active <=> distance to nearest bound <= .indicator_p_tol
    real_wp_ indicator_tol_p;

    /// \brief
    /// if .indicator_type = 2, a constraint/bound will be deemed to be active
    /// <=> distance to nearest bound
    /// <= .indicator_tol_pd * size of corresponding multiplier
    real_wp_ indicator_tol_pd;

    /// \brief
    /// if .indicator_type = 3, a constraint/bound will be deemed to be active
    /// <=> distance to nearest bound
    /// <= .indicator_tol_tapia * distance to same bound at previous iteration
    real_wp_ indicator_tol_tapia;

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
    /// if .getdua, is true, advanced initial values are obtained for the
    /// dual variables
    bool getdua;

    /// \brief
    /// If extrapolation is to be used, decide between Puiseux and Taylor series
    bool puiseux;

    /// \brief
    /// if .feasol is true, the final solution obtained will be perturbed so tha
    /// variables close to their bounds are moved onto these bounds
    bool feasol;

    /// \brief
    /// if .balance_initial_complentarity is true, the initial complemetarity
    /// is required to be balanced
    ///
    bool balance_initial_complentarity;

    /// \brief
    /// if .use_corrector, a corrector step will be used
    bool use_corrector;

    /// \brief
    /// if .array_syntax_worse_than_do_loop is true, f77-style do loops will be
    /// used rather than f90-style array syntax for vector operations   (OBSOLET
    bool array_syntax_worse_than_do_loop;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
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
    /// name of generated SIF file containing input problem
    char sif_file_name[31];

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
struct lsqp_time_type {

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
struct lsqp_inform_type {

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
    /// the total number of "wasted" function evaluations during the linesearch
    int nbacts;

    /// \brief
    /// the value of the objective function at the best estimate of the solution
    /// determined by LSQP_solve
    real_wp_ obj;

    /// \brief
    /// the value of the logarithmic potential function
    /// sum -log(distance to constraint boundary)
    real_wp_ potential;

    /// \brief
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    real_wp_ non_negligible_pivot;

    /// \brief
    /// is the returned "solution" feasible?
    bool feasible;

    /// \brief
    /// timings (see above)
    struct lsqp_time_type time;

    /// \brief
    /// inform parameters for FDC
    struct fdc_inform_type fdc_inform;

    /// \brief
    /// inform parameters for SBLS
    struct sbls_inform_type sbls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    L S Q P  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void lsqp_initialize( void **data, 
                     struct lsqp_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see lsqp_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    L S Q P  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void lsqp_read_specfile( struct lsqp_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information 
              (see lsqp_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    L S Q P  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void lsqp_import( struct lsqp_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lsqp_control_type)

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

// *-*-*-*-*-*-*-    L S Q P  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void lsqp_reset_control( struct lsqp_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see lsqp_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    L S Q P  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void lsqp_information( void **data,
                      struct lsqp_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see lsqp_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    L S Q P  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void lsqp_terminate( void **data, 
                    struct lsqp_control_type *control, 
                    struct lsqp_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see lsqp_control_type)

  @param[out] inform   is a struct containing output information
              (see lsqp_inform_type)
 */

/** \example lsqpt.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
