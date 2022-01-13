//* \file fdc.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-13 AT 16:09 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_FDC C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package fdc
 
  \section fdc_intro Introduction

  \subsection fdc_purpose Purpose

  \subsection fdc_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection fdc_date Originally released

  \subsection fdc_terminology Terminology

  \subsection fdc_method Method

  \subsection fdc_references Reference

  \subsection fdc_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_FDC_H 
#define GALAHAD_FDC_H

// precision
#include "galahad_precision.h"

// required packages
#include "sls.h"
#include "uls.h"

/**
 * control derived type as a C struct
 */
struct fdc_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    int error;

    /// \brief
    /// unit for monitor output
    int out;

    /// \brief
    /// controls level of diagnostic output
    int print_level;

    /// \brief
    /// initial estimate of integer workspace for sls (obsolete)
    int indmin;

    /// \brief
    /// initial estimate of real workspace for sls (obsolete)
    int valmin;

    /// \brief
    /// the relative pivot tolerance (obsolete)
    real_wp_ pivot_tol;

    /// \brief
    /// the absolute pivot tolerance used (obsolete)
    /// REAL ( KIND = wp ) :: zero_pivot = epsmch ** 0.75_wp
    real_wp_ zero_pivot;

    /// \brief
    /// the largest permitted residual
    /// REAL ( KIND = wp ) :: max_infeas = epsmch ** 0.33_wp
    real_wp_ max_infeas;

    /// \brief
    /// chose whether SLS or ULS is used to determine dependencies
    bool use_sls;

    /// \brief
    /// should the rows of A be scaled to have unit infinity norm or
    /// should no scaling be applied
    bool scale;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// symmetric (indefinite) linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// unsymmetric linear equation solver
    char unsymmetric_linear_solver[31];

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];

    /// \brief
    /// control parameters for SLS
    struct sls_control_type sls_control;

    /// \brief
    /// control parameters for ULS
    struct uls_control_type uls_control;
};

/**
 * time derived type as a C struct
 */
struct fdc_time_type {

    /// \brief
    /// the total CPU time spent in the package
    real_wp_ total;

    /// \brief
    /// the CPU time spent analysing the required matrices prior to factorizatio
    real_wp_ analyse;

    /// \brief
    /// the CPU time spent factorizing the required matrices
    real_wp_ factorize;

    /// \brief
    /// the total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// the clock time spent analysing the required matrices prior to factorizat
    real_wp_ clock_analyse;

    /// \brief
    /// the clock time spent factorizing the required matrices
    real_wp_ clock_factorize;
};

/**
 * inform derived type as a C struct
 */
struct fdc_inform_type {

    /// \brief
    /// return status. See FDC_find_dependent for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

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
    /// the smallest pivot which was not judged to be zero when detecting linear
    /// dependent constraints
    real_wp_ non_negligible_pivot;

    /// \brief
    /// timings (see above)
    struct fdc_time_type time;

    /// \brief
    /// SLS inform type
    struct sls_inform_type sls_inform;

    /// \brief
    /// ULS inform type
    struct uls_inform_type uls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    F D C  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void fdc_initialize( void **data, 
                     struct fdc_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see fdc_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    F D C  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void fdc_read_specfile( struct fdc_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information 
              (see fdc_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    F D C  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void fdc_import( struct fdc_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see fdc_control_type)

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

// *-*-*-*-*-*-*-    F D C  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void fdc_reset_control( struct fdc_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see fdc_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    F D C  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void fdc_information( void **data,
                      struct fdc_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see fdc_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    F D C  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void fdc_terminate( void **data, 
                    struct fdc_control_type *control, 
                    struct fdc_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see fdc_control_type)

  @param[out] inform   is a struct containing output information
              (see fdc_inform_type)
 */

/** \example fdct.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
