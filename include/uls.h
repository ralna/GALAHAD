//* \file uls.h */

/*
 * THIS VERSION: GALAHAD 3.3 - 30/11/2021 AT 08:48 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_ULS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 30th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package uls
 
  \section uls_intro Introduction

  \subsection uls_purpose Purpose

  \subsection uls_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection uls_date Originally released

  \subsection uls_terminology Terminology

  \subsection uls_method Method

  \subsection uls_references Reference

  \subsection uls_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_ULS_H 
#define GALAHAD_ULS_H

// precision
#include "galahad_precision.h"

// required packages
#include "gls.h"
#include "ma48.h"

/**
 * control derived type as a C struct
 */
struct uls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    int error;

    /// \brief
    /// unit for warning messages
    int warning;

    /// \brief
    /// unit for monitor output
    int out;

    /// \brief
    /// controls level of diagnostic output
    int print_level;

    /// \brief
    /// controls level of diagnostic output from external solver
    int print_level_solver;

    /// \brief
    /// prediction of factor by which the fill-in will exceed the initial
    /// number of nonzeros in A
    int initial_fill_in_factor;

    /// \brief
    /// initial size for real array for the factors and other data
    int min_real_factor_size;

    /// \brief
    /// initial size for integer array for the factors and other data
    int min_integer_factor_size;

    /// \brief
    /// maximum size for real array for the factors and other data
    int max_factor_size;

    /// \brief
    /// level 3 blocking in factorize
    int blas_block_size_factorize;

    /// \brief
    /// level 2 and 3 blocking in solve
    int blas_block_size_solve;

    /// \brief
    /// pivot control:
    /// 1  Threshold Partial Pivoting is desired
    /// 2  Threshold Rook Pivoting is desired
    /// 3  Threshold Complete Pivoting is desired
    /// 4  Threshold Symmetric Pivoting is desired
    /// 5  Threshold Diagonal Pivoting is desired
    int pivot_control;

    /// \brief
    /// number of rows/columns pivot selection restricted to (0 = no restriction
    int pivot_search_limit;

    /// \brief
    /// the minimum permitted size of blocks within the block-triangular form
    int minimum_size_for_btf;

    /// \brief
    /// maximum number of iterative refinements allowed
    int max_iterative_refinements;

    /// \brief
    /// stop if the matrix is found to be structurally singular
    bool stop_if_singular;

    /// \brief
    /// factor by which arrays sizes are to be increased if they are too small
    real_wp_ array_increase_factor;

    /// \brief
    /// switch to full code when the density exceeds this factor
    real_wp_ switch_to_full_code_density;

    /// \brief
    /// if previously allocated internal workspace arrays are greater than
    /// array_decrease_factor times the currently required sizes, they are reset
    /// to current requirements
    real_wp_ array_decrease_factor;

    /// \brief
    /// pivot threshold
    real_wp_ relative_pivot_tolerance;

    /// \brief
    /// any pivot small than this is considered zero
    real_wp_ absolute_pivot_tolerance;

    /// \brief
    /// any entry smaller than this in modulus is reset to zero
    real_wp_ zero_tolerance;

    /// \brief
    /// refinement will cease as soon as the residual ||Ax-b|| falls below
    /// max( acceptable_residual_relative * ||b||, acceptable_residual_absolute
    real_wp_ acceptable_residual_relative;
    /// see acceptable_residual_relative
    real_wp_ acceptable_residual_absolute;

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];
};

/**
 * inform derived type as a C struct
 */
struct uls_inform_type {

    /// \brief
    /// reported return status:
    /// 0  success
    /// -1  allocation error
    /// -2  deallocation error
    /// -3  matrix data faulty (.n < 1, .ne < 0)
    /// -29  unavailable option
    /// -31  input order is not a permutation or is faulty in some other way
    /// -32  error with integer workspace
    /// -33  error with real workspace
    /// -34  error from PARDISO
    /// -50  solver-specific error; see the solver's info parameter
    /// -101  unknown solver
    int status;

    /// \brief
    /// STAT value after allocate failure
    int alloc_status;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// further information on failure
    int more_info;

    /// \brief
    /// number of indices out-of-range
    int out_of_range;

    /// \brief
    /// number of duplicates
    int duplicates;

    /// \brief
    /// number of entries dropped during the factorization
    int entries_dropped;

    /// \brief
    /// predicted or actual number of reals and integers to hold factors
    int workspace_factors;

    /// \brief
    /// number of compresses of data required
    int compresses;

    /// \brief
    /// number of entries in factors
    int entries_in_factors;

    /// \brief
    /// estimated rank of the matrix
    int rank;

    /// \brief
    /// structural rank of the matrix
    int structural_rank;

    /// \brief
    /// pivot control:
    /// 1  Threshold Partial Pivoting has been used
    /// 2  Threshold Rook Pivoting has been used
    /// 3  Threshold Complete Pivoting has been desired
    /// 4  Threshold Symmetric Pivoting has been desired
    /// 5  Threshold Diagonal Pivoting has been desired
    int pivot_control;

    /// \brief
    /// number of iterative refinements performed
    int iterative_refinements;

    /// \brief
    /// has an "alternative" y: A^T y = 0 and yT b > 0 been found when trying to
    /// solve A x = b ?
    bool alternative;

    /// \brief
    /// the output arrays from GLS
    struct gls_ainfo_type gls_ainfo;
    /// see gls_ainfo
    struct gls_finfo_type gls_finfo;
    /// see gls_ainfo
    struct gls_sinfo_type gls_sinfo;

    /// \brief
    /// the output arrays from MA48
    struct ma48_ainfo_d ma48_ainfo;
    /// see ma48_ainfo
    struct ma48_finfo_d ma48_finfo;
    /// see ma48_ainfo
    struct ma48_sinfo_d ma48_sinfo;
};

// *-*-*-*-*-*-*-*-*-*-    U L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void uls_initialize( void **data, 
                     struct uls_control_type *control,
                     struct uls_inform_type *inform );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information 
              (see uls_control_type)
  @param[out] inform   is a struct containing output information
              (see uls_inform_type) 
*/

// *-*-*-*-*-*-*-*-*-    U L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void uls_read_specfile( struct uls_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control  is a struct containing control information 
              (see uls_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    U L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void uls_import( struct uls_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see uls_control_type)

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

// *-*-*-*-*-*-*-    U L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void uls_reset_control( struct uls_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see uls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    U L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void uls_information( void **data,
                      struct uls_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see uls_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    U L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void uls_terminate( void **data, 
                    struct uls_control_type *control, 
                    struct uls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see uls_control_type)

  @param[out] inform   is a struct containing output information
              (see uls_inform_type)
 */

/** \example ulst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
