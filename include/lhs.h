/*
 * THIS VERSION: GALAHAD 3.3 - 03/08/2021 AT 07:50 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_LHS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes
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
#ifndef GALAHAD_LHS_H 
#define GALAHAD_LHS_H

// precision
#include "galahad_precision.h"

/* 
 * inform derived type as a C struct
 */
struct lhs_inform_type {

    //  return status. See LHS_solve for details
    int status;

    // the status of the last attempted allocation/deallocation
    int alloc_status ;

    // the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];
};

/* 
 * control derived type as a C struct
 */
struct lhs_control_type { 

    // error and warning diagnostics occur on stream error
    int error;

    // general output occurs on stream out    
    int out;

    // the level of output required. <= 0 gives no output, = 1 gives a one-line
    // summary for every iteration, = 2 gives a summary of the inner iteration
    // for each iteration, >= 3 gives increasingly verbose (debugging) output 
    int print_level; 

    // the duplication factor. This must be at least 1, a value of 5 is 
    // reasonable
    int duplication;

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

/*  *-*-*-*-*-*-*-*-*-*-   L H S _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*-*
 *
 * Provide default values for LHS controls
 */

void lhs_initialize( void **data, 
                     struct lhs_control_type *control, 
                     struct lhs_inform_type *inform );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *   data     private internal data
 *   control  a struct containing default control information (see above)
 *   inform   a struct containing output information (see above)
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-   L H S _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*-*-*
 *
 * Read the content of a specification file, and perform the assignment of
 * values associated with given keywords to the corresponding control parameters
 */ 

void lhs_read_specfile( struct lhs_control_type *control, 
                        const char specfile[] );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *   control  a struct containing control information (see above)
 *   specfile a character string containing the name of the specfile
 *
 *  ---------------------------------------------------------------------------
 */

/*  *-*-*-*-*-*-*-*-*-*-*-*-*-*-   L H S _ I H S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*
 *
 * lhs_ihs implements the improved distributed hyper-cube sampling algorithm.
 *
 *  Discussion:
 *
 *    n_points points in an n_dimen dimensional Latin hyper-cube are
 *    to be selected. Each of the coordinate dimensions is discretized
 *    to the values 1 through n.  The points are to be chosen in such
 *    a way that no two points have any coordinate value in common.
 *    This is a standard Latin hypercube requirement, and there are
 *    many solutions.
 *
 *    This algorithm differs in that it tries to pick a solution
 *    which has the property that the points are "spread out"
 *    as evenly as possible.  It does this by determining an optimal
 *    even spacing, and using the DUPLICATION factor to allow it
 *    to choose the best of the various options available to it.
 *
 *  Reference:
 *
 *    Brian Beachkofski, Ramana Grandhi,
 *    Improved Distributed Hypercube Sampling,
 *    American Institute of Aeronautics and Astronautics Paper 2002-1274
 */

void lhs_ihs( int n_dimen, 
              int n_points, 
              int *seed, 
              int X[n_dimen][n_points], 
              const struct lhs_control_type *control, 
              struct lhs_inform_type *inform, void **data );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *    Input, int n_dimen, the spatial dimension
 *
 *    Input, int n_points, the number of points to be generated
 *
 *    Input/output, int *seed, a seed for the random number generator
 *
 *    Output, int X[n_dimen][n_points], the points
 *
 *    control, inform, data - see lhs_initialize
 *
 *  ---------------------------------------------------------------------------
 */ 

/*  *-*-*-*-*-*-*-*-*-*-*-*-   L H S _ G E T _ S E E D  -*-*-*-*-*-*-*-*-*-*-*
 *
 * lhs_get_seed gets a seed for the random number generator.
 *
 *  Discussion:
 *
 *    The seed depends on the current time, and ought to be (slightly)
 *    different every millisecond.  Once the seed is obtained, a random
 *    number generator should be called a few times to further process
 *    the seed.
 */ 

void lhs_get_seed( int *seed );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *    Output, int* seed, a pseudorandom seed value.
 *
 *  ---------------------------------------------------------------------------
 */ 

/*  *-*-*-*-*-*-*-*-*-*-*-   L H S _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*-*
 *
 * Deallocate all private storage
 */

void lhs_terminate( void **data, 
                    struct lhs_control_type *control, 
                    struct lhs_inform_type *inform );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 * see lhs_initialize above
 *
 *  ---------------------------------------------------------------------------
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
