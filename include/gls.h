//* \file gls.h */

/*
 * THIS VERSION: GALAHAD 3.3 - 30/11/2021 AT 09:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_GLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package gls
 
  \section gls_intro Introduction

  \subsection gls_purpose Purpose

  \subsection gls_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection gls_date Originally released

  \subsection gls_terminology Terminology

  \subsection gls_method Method

  \subsection gls_references Reference

  \subsection gls_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_GLS_H 
#define GALAHAD_GLS_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct gls_control {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// Unit for error messages
    int lp;

    /// \brief
    /// Unit for warning messages
    int wp;

    /// \brief
    /// Unit for monitor output
    int mp;

    /// \brief
    /// Controls level of diagnostic output
    int ldiag;

    /// \brief
    /// Minimum block size for block-triangular form (BTF) ... >=n to avoid
    int btf;

    /// \brief
    /// Maximum number of iterations
    int maxit;

    /// \brief
    /// Level 3 blocking in factorize
    int factor_blocking;

    /// \brief
    /// Switch for using Level 1 or 2 BLAS in solve.
    int solve_blas;

    /// \brief
    /// Initial size for real array for the factors.
    int la;

    /// \brief
    /// Initial size for integer array for the factors.
    int la_int;

    /// \brief
    /// Maximum size for real array for the factors.
    int maxla;

    /// \brief
    /// Controls pivoting:  Number of columns searched.  Zero for Markowitz
    int pivoting;

    /// \brief
    /// Initially fill_in * ne space allocated for factors
    int fill_in;

    /// \brief
    /// Factor by which arrays sizes are to be increased if they are too small
    real_wp_ multiplier;

    /// \brief
    /// if previously allocated internal workspace arrays are greater than reduc
    /// times the currently required sizes, they are reset to current requirment
    real_wp_ reduce;

    /// \brief
    /// Pivot threshold
    real_wp_ u;

    /// \brief
    /// Density for switch to full code
    real_wp_ switch_full;

    /// \brief
    /// Drop tolerance
    real_wp_ drop;

    /// \brief
    /// anything < this is considered zero
    real_wp_ tolerance;

    /// \brief
    /// Ratio for required reduction using IR
    real_wp_ cgce;

    /// \brief
    /// Set to 0 for diagonal pivoting
    bool diagonal_pivoting;

    /// \brief
    /// Control to abort if structurally singular
    bool struct_abort;
};

/**
 * ainfo derived type as a C struct
 */
struct gls_ainfo {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// More information on failure
    int more;

    /// \brief
    /// Size for analysis
    int len_analyse;

    /// \brief
    /// Size for factorize
    int len_factorize;

    /// \brief
    /// Number of compresses
    int ncmpa;

    /// \brief
    /// Estimated rank
    int rank;

    /// \brief
    /// Number of entries dropped
    int drop;

    /// \brief
    /// Structural rank of matrix
    int struc_rank;

    /// \brief
    /// Number of indices out-of-range
    int oor;

    /// \brief
    /// Number of duplicates
    int dup;

    /// \brief
    /// STAT value after allocate failure
    int stat;

    /// \brief
    /// Size largest non-triangular block
    int lblock;

    /// \brief
    /// Sum of orders of non-triangular blocks
    int sblock;

    /// \brief
    /// Total entries in all non-tringular blocks
    int tblock;

    /// \brief
    /// Number of operations in elimination
    real_wp_ ops;
};

/**
 * finfo derived type as a C struct
 */
struct gls_finfo {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// More information on failure
    int more;

    /// \brief
    /// Number of words to hold factors
    int size_factor;

    /// \brief
    /// Size for subsequent factorization
    int len_factorize;

    /// \brief
    /// Number of entries dropped
    int drop;

    /// \brief
    /// Estimated rank
    int rank;

    /// \brief
    /// Status value after allocate failure
    int stat;

    /// \brief
    /// Number of operations in elimination
    real_wp_ ops;
};

/**
 * sinfo derived type as a C struct
 */
struct gls_sinfo {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// More information on failure
    int more;

    /// \brief
    /// Status value after allocate failure
    int stat;
};


// *-*-*-*-*-*-*-*-*-*-    G L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void gls_initialize( void **data, 
                     struct gls_control *control );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see gls_control)
*/

// *-*-*-*-*-*-*-*-*-    G L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void gls_read_specfile( struct gls_control *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control  is a struct containing control information 
              (see gls_control)

  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    G L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void gls_import( struct gls_control *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see gls_control)

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

// *-*-*-*-*-*-*-    G L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void gls_reset_control( struct gls_control *control,
                        void **data,
                        int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see gls_control)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    G L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void gls_information( void **data,
                      struct gls_ainfo *ainfo,
                      struct gls_finfo *finfo,
                      struct gls_sinfo *sinfo,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] ainfo   is a struct containing analysis output information
              (see gls_ainfo) 

  @param[out] finfo   is a struct containing factorization output information
              (see gls_finfo) 

  @param[out] sinfo   is a struct containing solver output information
              (see gls_sinfo) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    G L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void gls_finalize( void **data, 
                   struct gls_control *control, 
                   int *status );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see gls_control)

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
 */

/** \example glst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
