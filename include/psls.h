//* \file psls.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-14 AT 14:24 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_PSLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 14th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package psls
 
  \section psls_intro Introduction

  \subsection psls_purpose Purpose

  \subsection psls_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection psls_date Originally released

  \subsection psls_terminology Terminology

  \subsection psls_method Method

  \subsection psls_references Reference

  \subsection psls_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_PSLS_H 
#define GALAHAD_PSLS_H

// precision
#include "galahad_precision.h"

// required packages
#include "sls.h"
#include "mi28.h"

/**
 * control derived type as a C struct
 */
struct psls_control_type {

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
    /// which preconditioner to use:
    /// <0  no preconditioning, P = I
    /// 0  automatic
    /// 1  diagonal, P = diag( max( A, .min_diagonal ) )
    /// 2  banded, P = band( A ) with semi-bandwidth .semi_bandwidth
    /// 3  re-ordered band, P = band(order(A)) with semi-bandwidth .semi_bandwid
    /// 4  full factorization, P = A, Schnabel-Eskow modification
    /// 5  full factorization, P = A, GMPS modification
    /// 6  incomplete factorization, Lin-More'
    /// 7  incomplete factorization, HSL_MI28
    /// 8  incomplete factorization, Munskgaard
    /// 9  expanding band
    int preconditioner;

    /// \brief
    /// the semi-bandwidth for band(H)
    int semi_bandwidth;

    /// \brief
    /// not used at present
    int scaling;
    /// see scaling
    int ordering;

    /// \brief
    /// maximum number of nonzeros in a column of A for Schur-complement
    /// factorization  to accommodate newly fixed variables
    int max_col;

    /// \brief
    /// number of extra vectors of length n required by the Lin-More' incomplete
    /// Cholesky preconditioner
    int icfs_vectors;

    /// \brief
    /// the maximum number of fill entries within each column of the incomplete
    /// factor L computed by HSL_MI28. In general, increasing mi28_lsize improve
    /// the quality of the preconditioner but increases the time to compute
    /// and then apply the preconditioner. Values less than 0 are treated as 0
    int mi28_lsize;

    /// \brief
    /// the maximum number of entries within each column of the strictly lower
    /// triangular matrix R used in the computation of the preconditioner by
    /// HSL_MI28.  Rank-1 arrays of size mi28_rsize *  n are allocated internall
    /// to hold R. Thus the amount of memory used, as well as the amount of work
    /// involved in computing the preconditioner, depends on mi28_rsize. Setting
    /// mi28_rsize > 0 generally leads to a higher quality preconditioner than
    /// using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally
    /// recommended
    int mi28_rsize;

    /// \brief
    /// the minimum permitted diagonal in diag(max(H,min_diag))
    real_wp_ min_diagonal;

    /// \brief
    /// set new_structure true if the storage structure for the input matrix has
    /// changed, and false if only the values have changed
    bool new_structure;

    /// \brief
    /// set get_semi_bandwidth true if the semi-bandwidth of the submatrix is to
    /// calculated
    bool get_semi_bandwidth;

    /// \brief
    /// set get_norm_residual true if the residual when applying the preconditio
    /// are to be calculated
    bool get_norm_residual;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// definite linear equation solver
    char definite_linear_solver[31];

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
    /// control parameters for HSL_MI28
    struct mi28_control_type mi28_control;
};

/**
 * time derived type as a C struct
 */
struct psls_time_type {

    /// \brief
    /// total time
    real_sp_ total;

    /// \brief
    /// time to assemble the preconditioner prior to factorization
    real_sp_ assemble;

    /// \brief
    /// time for the analysis phase
    real_sp_ analyse;

    /// \brief
    /// time for the factorization phase
    real_sp_ factorize;

    /// \brief
    /// time for the linear solution phase
    real_sp_ solve;

    /// \brief
    /// time to update the factorization
    real_sp_ update;

    /// \brief
    /// total clock time spent in the package
    real_wp_ clock_total;

    /// \brief
    /// clock time to assemble the preconditioner prior to factorization
    real_wp_ clock_assemble;

    /// \brief
    /// clock time for the analysis phase
    real_wp_ clock_analyse;

    /// \brief
    /// clock time for the factorization phase
    real_wp_ clock_factorize;

    /// \brief
    /// clock time for the linear solution phase
    real_wp_ clock_solve;

    /// \brief
    /// clock time to update the factorization
    real_wp_ clock_update;
};

/**
 * inform derived type as a C struct
 */
struct psls_inform_type {

    /// \brief
    /// reported return status:
    /// 0  success
    /// -1  allocation error
    /// -2  deallocation error
    /// -3  matrix data faulty (.n < 1, .ne < 0)
    /// -20  alegedly +ve definite matrix is not
    int status;

    /// \brief
    /// STAT value after allocate failure
    int alloc_status;

    /// \brief
    /// status return from factorization
    int analyse_status;

    /// \brief
    /// status return from factorization
    int factorize_status;

    /// \brief
    /// status return from solution phase
    int solve_status;

    /// \brief
    /// number of integer words to hold factors
    int factorization_integer;

    /// \brief
    /// number of real words to hold factors
    int factorization_real;

    /// \brief
    /// code for the actual preconditioner used (see control.preconditioner)
    int preconditioner;

    /// \brief
    /// the actual semi-bandwidth
    int semi_bandwidth;

    /// \brief
    /// the semi-bandwidth following reordering (if any)
    int reordered_semi_bandwidth;

    /// \brief
    /// number of indices out-of-range
    int out_of_range;

    /// \brief
    /// number of duplicates
    int duplicates;

    /// \brief
    /// number of entries from the strict upper triangle
    int upper;

    /// \brief
    /// number of missing diagonal entries for an allegedly-definite matrix
    int missing_diagonals;

    /// \brief
    /// the semi-bandwidth used
    int semi_bandwidth_used;

    /// \brief
    /// number of 1 by 1 pivots in the factorization
    int neg1;

    /// \brief
    /// number of 2 by 2 pivots in the factorization
    int neg2;

    /// \brief
    /// has the preconditioner been perturbed during the fctorization?
    bool perturbed;

    /// \brief
    /// ratio of fill in to original nonzeros
    real_wp_ fill_in_ratio;

    /// \brief
    /// the norm of the solution residual
    real_wp_ norm_residual;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// the integer and real output arrays from mc61
    int mc61_info[10];
    /// see mc61_info
    real_wp_ mc61_rinfo[15];

    /// \brief
    /// times for various stages
    struct psls_time_type time;

    /// \brief
    /// inform values from SLS
    struct sls_inform_type sls_inform;

    /// \brief
    /// the output structure from mi28
    struct mi28_inform_type mi28_inform;
};

// *-*-*-*-*-*-*-*-*-*-    P S L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void psls_initialize( void **data, 
                     struct psls_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see psls_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    P S L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void psls_read_specfile( struct psls_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information 
              (see psls_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    P S L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void psls_import( struct psls_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see psls_control_type)

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

// *-*-*-*-*-*-*-    P S L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void psls_reset_control( struct psls_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see psls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    P S L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void psls_information( void **data,
                      struct psls_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see psls_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    P S L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void psls_terminate( void **data, 
                    struct psls_control_type *control, 
                    struct psls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see psls_control_type)

  @param[out] inform   is a struct containing output information
              (see psls_inform_type)
 */

/** \example pslst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
