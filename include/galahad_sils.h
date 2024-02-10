//* \file galahad_sils.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SILS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 29th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package sils

  \section sils_intro Introduction

  \subsection sils_purpose Purpose

  This package <b>solves sparse symmetric system of linear equations.</b>.
  Given an \f$n\f$ by \f$n\f$ sparse matrix \f$A = {a_{ij}}\f$, and an
  \f$n\f$ vector \f$b\f$, the package solves the system \f$A x = b\f$.
  The matrix \f$A\f$ need not be definite. There is an option for iterative
  refinement.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.
  Extended functionality is available using the GALAHAD package sls.

  \subsection sils_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sils_date Originally released

  April 2001, C interface December 2021.

  \subsection sils_method Method

  The method used is a direct method based on a sparse variant
  of Gaussian elimination and is discussed further by

  I. S. Duff and J. K. Reid (1983),
  ACM Trans. Math. Software <b>9</b> pp.302-325.

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  The symmetric \f$n\f$ by \f$n\f$ coefficient matrix \f$A\f$ may be presented
  and stored in a variety of convenient input formats.  Crucially symmetry
  is exploited  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$A\f$ is symmetric, only the lower triangular part (that is the part
  \f$A_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array val
  will hold the value \f$A_{ij}\f$ (and, by symmetry, \f$A_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays row and
  col and real array val, respectively, while the number of nonzeros
  is recorded as ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$A\f$ the i-th component of the
  integer array ptr holds the position of the first entry in this row,
  while ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$A_{ij}\f$ of the  entries in the i-th row are stored in components
  l = ptr(i), \f$\ldots\f$, ptr(i+1)-1 of the
  integer array col, and real array val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
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
#ifndef GALAHAD_SILS_H
#define GALAHAD_SILS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct sils_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// MA27 internal integer controls
    ipc_ ICNTL[30];

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
    /// Unit for statistical output
    ipc_ sp;

    /// \brief
    /// Controls level of diagnostic output
    ipc_ ldiag;

    /// \brief
    /// Initial size for real array for the factors. If less than nrlnec,
    /// default size used.
    ipc_ la;

    /// \brief
    /// Initial size for integer array for the factors. If less than nirnec,
    /// default size used.
    ipc_ liw;

    /// \brief
    /// Max. size for real array for the factors.
    ipc_ maxla;

    /// \brief
    /// Max. size for integer array for the factors.
    ipc_ maxliw;

    /// \brief
    /// Controls pivoting. Possible values are:
    /// \li 1  Numerical pivoting will be performed.
    /// \li 2  No pivoting will be performed and an error exit will occur
    ///        immediately a pivot sign change is detected.
    /// \li 3  No pivoting will be performed and an error exit will occur if a
    ///        zero pivot is detected.
    /// \li 4  No pivoting is performed but pivots are changed to all be positive.
    ipc_ pivoting;

    /// \brief
    /// Minimum number of eliminations in a step (unused)
    ipc_ nemin;

    /// \brief
    /// Level 3 blocking in factorize (unused)
    ipc_ factorblocking;

    /// \brief
    /// Level 2 and 3 blocking in solve
    ipc_ solveblocking;

    /// \brief
    /// Controls threshold for detecting full rows in  analyse, registered as
    /// percentage of N, 100 Only fully dense rows detected (default)
    ipc_ thresh;

    /// \brief
    /// Controls ordering: Possible values are:
    /// \li 0  AMD using HSL's MC47
    /// \li 1  User defined
    /// \li 2  AMD using HSL's MC50
    /// \li 3  Min deg as in HSL's MA57
    /// \li 4  Metis_nodend ordering
    /// \li 5  Ordering chosen depending on matrix characteristics.
    ///        At the moment choices are HSL's MC50 or Metis_nodend
    /// \li >5  Presently equivalent to 5 but may chnage
    ipc_ ordering;

    /// \brief
    /// Controls scaling: Possible values are:
    /// \li 0  No scaling
    /// \li >0  Scaling using HSL's MC64 but may change for > 1
    ipc_ scaling;

    /// \brief
    /// MA27 internal real controls
    rpc_ CNTL[5];

    /// \brief
    /// Factor by which arrays sizes are to be increased if they are too small
    rpc_ multiplier;

    /// \brief
    /// If previously allocated internal workspace arrays are greater than
    /// reduce times the currently required sizes, they are reset to current
    /// requirment
    rpc_ reduce;

    /// \brief
    /// Pivot threshold
    rpc_ u;

    /// \brief
    /// used for setting static pivot level
    rpc_ static_tolerance;

    /// \brief
    /// used for switch to static
    rpc_ static_level;

    /// \brief
    /// Anything less than this is considered zero
    rpc_ tolerance;

    /// \brief
    /// used to monitor convergence in iterative refinement
    rpc_ convergence;
};

/**
 * ainfo derived type as a C struct
 */
struct sils_ainfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// More information on failure
    ipc_ more;

    /// \brief
    /// Number of elimination steps
    ipc_ nsteps;

    /// \brief
    /// Size for a without compression
    ipc_ nrltot;

    /// \brief
    /// Size for iw without compression
    ipc_ nirtot;

    /// \brief
    /// Size for a with compression
    ipc_ nrlnec;

    /// \brief
    /// Size for iw with compression
    ipc_ nirnec;

    /// \brief
    /// Number of reals to hold factors
    ipc_ nrladu;

    /// \brief
    /// Number of integers to hold factors
    ipc_ niradu;

    /// \brief
    /// Number of compresses
    ipc_ ncmpa;

    /// \brief
    /// Number of indices out-of-range
    ipc_ oor;

    /// \brief
    /// Number of duplicates
    ipc_ dup;

    /// \brief
    /// Forecast maximum front size
    ipc_ maxfrt;

    /// \brief
    /// STAT value after allocate failure
    ipc_ stat;

    /// \brief
    /// legacy component, now not used
    ipc_ faulty;

    /// \brief
    /// Anticipated number of operations in assembly
    rpc_ opsa;

    /// \brief
    /// Anticipated number of operations in elimination
    rpc_ opse;
};

/**
 * finfo derived type as a C struct
 */
struct sils_finfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// More information on failure
    ipc_ more;

    /// \brief
    /// Largest front size
    ipc_ maxfrt;

    /// \brief
    /// Number of entries in factors
    ipc_ nebdu;

    /// \brief
    /// Number of reals that hold factors
    ipc_ nrlbdu;

    /// \brief
    /// Number of integers that hold factors
    ipc_ nirbdu;

    /// \brief
    /// Size for a without compression
    ipc_ nrltot;

    /// \brief
    /// Size for iw without compression
    ipc_ nirtot;

    /// \brief
    /// Size for a with compression
    ipc_ nrlnec;

    /// \brief
    /// Size for iw with compression
    ipc_ nirnec;

    /// \brief
    /// Number of compresses of real data
    ipc_ ncmpbr;

    /// \brief
    /// Number of compresses of integer data
    ipc_ ncmpbi;

    /// \brief
    /// Number of 2x2 pivots
    ipc_ ntwo;

    /// \brief
    /// Number of negative eigenvalues
    ipc_ neig;

    /// \brief
    /// Number of delayed pivots (total)
    ipc_ delay;

    /// \brief
    /// Number of pivot sign changes when control.pivoting=3
    ipc_ signc;

    /// \brief
    /// Number of static pivots chosen
    ipc_ nstatic;

    /// \brief
    /// First pivot modification when control.pivoting=4
    ipc_ modstep;

    /// \brief
    /// Rank of original factorization
    ipc_ rank;

    /// \brief
    /// STAT value after allocate failure
    ipc_ stat;

    /// \brief
    /// legacy component, now not used
    ipc_ faulty;

    /// \brief
    /// legacy component, now not used
    ipc_ step;

    /// \brief
    /// # operations in assembly
    rpc_ opsa;

    /// \brief
    /// number of operations in elimination
    rpc_ opse;

    /// \brief
    /// Additional number of operations for BLAS
    rpc_ opsb;

    /// \brief
    /// Largest control.pivoting=4 modification
    rpc_ maxchange;

    /// \brief
    /// Minimum scaling factor
    rpc_ smin;

    /// \brief
    /// Maximum scaling factor
    rpc_ smax;
};

/**
 * sinfo derived type as a C struct
 */
struct sils_sinfo_type {

    /// \brief
    /// Flags success or failure case
    ipc_ flag;

    /// \brief
    /// STAT value after allocate failure
    ipc_ stat;

    /// \brief
    /// Condition number of matrix (category 1 eqs)
    rpc_ cond;

    /// \brief
    /// Condition number of matrix (category 2 eqs)
    rpc_ cond2;

    /// \brief
    /// Backward error for the system (category 1 eqs)
    rpc_ berr;

    /// \brief
    /// Backward error for the system (category 2 eqs)
    rpc_ berr2;

    /// \brief
    /// Estimate of forward error
    rpc_ error;
};

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sils_initialize( void **data,
                      struct sils_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information
              (see sils_control_type)
  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-    S I L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void sils_read_specfile( struct sils_control_type *control,
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNSILS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/sils.pdf for a list of keywords that may be set.

  @param[in,out]  control  is a struct containing control information
              (see sils_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    S I L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void sils_import( struct sils_control_type *control,
                  void **data,
                  ipc_ *status );

/*!<
 Import problem data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sils_control_type)

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

// *-*-*-*-*-*-*-    S I L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void sils_reset_control( struct sils_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sils_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sils_information( void **data,
                       struct sils_ainfo_type *ainfo,
                       struct sils_finfo_type *finfo,
                       struct sils_sinfo_type *sinfo,
                       ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] ainfo is a struct containing output information
              (see sils_ainfo_type)

  @param[out] finfo is a struct containing output information
              (see sils_finfo_type)

  @param[out] sinfo is a struct containing output information
              (see sils_sinfo_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ F I N A L I Z E   -*-*-*-*-*-*-*-*-*-*

void sils_finalize( void **data,
                     struct sils_control_type *control,
                     ipc_ *status );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see sils_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
  \li \f$\neq\f$ 0. The Fortran STAT value of an allocate or
          deallocate statement that has failed.
 */

/** \example silst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
