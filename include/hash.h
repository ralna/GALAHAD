/*
 * THIS VERSION: GALAHAD 3.3 - 11/08/2021 AT 15:39 GMT.
/*
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_HASH C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. August 11th 2021
/*
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
#ifndef GALAHAD_HASH_H 
#define GALAHAD_HASH_H

// precision
#include "galahad_precision.h"

/*
 * control derived type as a C struct
 */
struct hash_control_type {

    // error and warning diagnostics occur on stream error
    int error;

    // general output occurs on stream out
    int out;

    // the level of output required. <= 0 gives no output, >= 1 enables debuggin
    int print_level;

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
 * inform derived type as a C struct
 */
struct hash_inform_type {

    // return status. See DGO_solve for details
    int status;

    // the status of the last attempted allocation/deallocation
    int alloc_status;

    // the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];
};

/* *-*-*-*-*-*-*-*-*-*-    H A S H  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*
 *
 * Provide default values for HASH controls
 */

void hash_initialize( int nchar, 
                      int length, 
                      void **data, 
                      struct hash_control_type *control,
                      struct hash_inform_type *inform );

/*  ------------------------  A R G U M E N T S  --------------------------
 *
 *   data     private internal data
 *   control  a struct containing default control information (see above)
 *   inform   a struct containing output information (see above)
 *
 *  -----------------------------------------------------------------------
 */

/* *-*-*-*-*-*-*-*-*-    H A S H  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*
 *
 * Read the content of a specification file, and perform the assignment of
 * values associated with given keywords to the corresponding control parameters
 */

void hash_read_specfile( struct hash_control_type *control, 
                         const char specfile[] );

/*  ------------------------  A R G U M E N T S  --------------------------
 *
 *   control  a struct containing control information (see above)
 *   specfile a character string containing the name of the specfile
 *
 *  -----------------------------------------------------------------------
 */

/* *-*-*-*-*-*-*-*-*-*-*-*-    H A S H  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*
 *
 * Import problem data into internal storage prior to solution. 
 */

void hash_import( struct hash_control_type *control,
                  void **data,
                  int *status );

/*  ------------------------  A R G U M E N T S  ------------------------------
 *
 *  control is a struct whose members are described in 
 *   the leading comments to hash_solve
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
 *    -3. An input restriction has been violated.
 *
 *  ---------------------------------------------------------------------------
 */

/* *-*-*-*-*-*-*-*-*-*-    H A S H  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*
 *
 * hash_information fills the output information structure inform 
 * (see hash_inform_type above)
 */

void hash_information( void **data,
                       struct hash_inform_type *inform,
                       int *status );

/*  ------------------------  A R G U M E N T S  --------------------------
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
 *  -----------------------------------------------------------------------
 */
/* *-*-*-*-*-*-*-*-*-*-    H A S H  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

 *
 * Deallocate all private storage
 */

void hash_terminate( void **data, 
                     struct hash_control_type *control, 
                     struct hash_inform_type *inform );

/*  ------------------------  A R G U M E N T S  --------------------------
 *
 *  see hash_initialize above
 *
 *  -----------------------------------------------------------------------
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
