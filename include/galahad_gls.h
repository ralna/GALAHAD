//* \file galahad_gls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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

  This package <b>solves sparse unsymmetric system of linear equations.</b>.
  Given an \f$m\f$ by \f$n\f$ sparse matrix \f$A = a_{ij}\f$, the package
  solves the system \f$A x = b\f$ (or optionally \f$A^T x = b\f$).
  The matrix \f$A\f$ can be rectangular.

  <b>N.B.</b> The package is simply a sophisticated interface to the
  HSL package MA33, and requires that a user has obtained the latter.
  <b> MA33 is not included in GALAHAD</b>
  but is available without charge to recognised academics, see
  https://www.hsl.rl.ac.uk/archive/specs/ma33.pdf .
  The package offers additional features to MA33.
  The storage required for the factorization is chosen
  automatically and, if there is insufficient space for the factorization,
  more space is allocated and the factorization is repeated.  The package
  also returns the number of entries in the factors and has facilities
  for identifying the rows and columns that are treated specially
  when the matrix is singular or rectangular.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.
  Extended functionality is available using the GALAHAD package uls.

  \subsection gls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection gls_date Originally released

  March 2006, C interface December 2021.

  \subsection main_unsymmetric_matrices Unsymmetric matrix storage formats

  The unsymmetric \f$m\f$ by \f$n\f$  matrix \f$A\f$ may be presented
  and stored in a variety of convenient input formats.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection unsymmetric_matrix_dense Dense storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$n \ast i + j\f$  of the storage array A_val
  will hold the value \f$A_{ij}\f$ for \f$0 \leq i \leq m-1\f$,
  \f$0 \leq j \leq n-1\f$.

  \subsubsection unsymmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays A_row and
  A_col and real array A_val, respectively, while the number of nonzeros
  is recorded as A_ne = \f$ne\f$.

  \subsubsection unsymmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A\f$ the i-th component of the
  integer array A_ptr holds the position of the first entry in this row,
  while A_ptr(m) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq n-1\f$, and values
  \f$A_{ij}\f$ of the  nonzero entries in the i-th row are stored in components
  l = A_ptr(i), \f$\ldots\f$, A_ptr(i+1)-1,  \f$0 \leq i \leq m-1\f$,
  of the integer array A_col, and real array A_val, respectively.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_GLS_H
#define GALAHAD_GLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct gls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// Unit for error messages
    ipc_ lp;

    /// \brief
    /// Unit for warning messages
    ipc_ wp;

    /// \brief
    /// Unit for monitor output
    ipc_ mp;

    /// \brief
    /// Controls level of diagnostic output
    ipc_ ldiag;

    /// \brief
    /// Minimum block size for block-triangular form (BTF). Set to \f$n\f$
    /// to avoid
    ipc_ btf;

    /// \brief
    /// Maximum number of iterations
    ipc_ maxit;

    /// \brief
    /// Level 3 blocking in factorize
    ipc_ factor_blocking;

    /// \brief
    /// Switch for using Level 1 or 2 BLAS in solve.
    ipc_ solve_blas;

    /// \brief
    /// Initial size for real array for the factors.
    ipc_ la;

    /// \brief
    /// Initial size for integer array for the factors.
    ipc_ la_int;

    /// \brief
    /// Maximum size for real array for the factors.
    ipc_ maxla;

    /// \brief
    /// Controls pivoting:  Number of columns searched.  Zero for Markowitz
    ipc_ pivoting;

    /// \brief
    /// Initially fill_in * ne space allocated for factors
    ipc_ fill_in;

    /// \brief
    /// Factor by which arrays sizes are to be increased if they are too small
    rpc_ multiplier;

    /// \brief
    /// if previously allocated internal workspace arrays are greater than
    /// reduce times the currently required sizes, they are reset to current
    /// requirment
    rpc_ reduce;

    /// \brief
    /// Pivot threshold
    rpc_ u;

    /// \brief
    /// Density for switch to full code
    rpc_ switch_full;

    /// \brief
    /// Drop tolerance
    rpc_ drop;

    /// \brief
    /// anything < this is considered zero
    rpc_ tolerance;

    /// \brief
    /// Ratio for required reduction using IR
    rpc_ cgce;

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
struct gls_ainfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// More information on failure
    ipc_ more;

    /// \brief
    /// Size for analysis
    ipc_ len_analyse;

    /// \brief
    /// Size for factorize
    ipc_ len_factorize;

    /// \brief
    /// Number of compresses
    ipc_ ncmpa;

    /// \brief
    /// Estimated rank
    ipc_ rank;

    /// \brief
    /// Number of entries dropped
    ipc_ drop;

    /// \brief
    /// Structural rank of matrix
    ipc_ struc_rank;

    /// \brief
    /// Number of indices out-of-range
    ipc_ oor;

    /// \brief
    /// Number of duplicates
    ipc_ dup;

    /// \brief
    /// STAT value after allocate failure
    ipc_ stat;

    /// \brief
    /// Size largest non-triangular block
    ipc_ lblock;

    /// \brief
    /// Sum of orders of non-triangular blocks
    ipc_ sblock;

    /// \brief
    /// Total entries in all non-tringular blocks
    ipc_ tblock;

    /// \brief
    /// Number of operations in elimination
    rpc_ ops;
};

/**
 * finfo derived type as a C struct
 */
struct gls_finfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// More information on failure
    ipc_ more;

    /// \brief
    /// Number of words to hold factors
    ipc_ size_factor;

    /// \brief
    /// Size for subsequent factorization
    ipc_ len_factorize;

    /// \brief
    /// Number of entries dropped
    ipc_ drop;

    /// \brief
    /// Estimated rank
    ipc_ rank;

    /// \brief
    /// Status value after allocate failure
    ipc_ stat;

    /// \brief
    /// Number of operations in elimination
    rpc_ ops;
};

/**
 * sinfo derived type as a C struct
 */
struct gls_sinfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// More information on failure
    ipc_ more;

    /// \brief
    /// Status value after allocate failure
    ipc_ stat;
};


// *-*-*-*-*-*-*-*-*-*-    G L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void gls_initialize( void **data,
                     struct gls_control_type *control );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see gls_control_type)
*/

// *-*-*-*-*-*-*-*-*-    G L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void gls_read_specfile( struct gls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNGLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/gls.pdf for a list of keywords that may be set.

  @param[in,out]  control  is a struct containing control information
              (see gls_control_type)

  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    G L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void gls_import( struct gls_control_type *control,
                 void **data,
                 ipc_ *status );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see gls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
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

void gls_reset_control( struct gls_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see gls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    G L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void gls_information( void **data,
                      struct gls_ainfo_type *ainfo,
                      struct gls_finfo_type *finfo,
                      struct gls_sinfo_type *sinfo,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] ainfo   is a struct containing analysis output information
              (see gls_ainfo_type)

  @param[out] finfo   is a struct containing factorization output information
              (see gls_finfo_type)

  @param[out] sinfo   is a struct containing solver output information
              (see gls_sinfo_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    G L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void gls_finalize( void **data,
                   struct gls_control_type *control,
                   ipc_ *status );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see gls_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
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
