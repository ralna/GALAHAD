//* \file galahad_uls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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

  This package
  <b> solves dense or sparse unsymmetric systems of linear equations</b>
  using variants of Gaussian elimination.
  Given a sparse symmetric \f$m \times n\f$ matrix \f$A = a_{ij}\f$, and an
  \f$m\f$-vector \f$b\f$, this subroutine solves the system \f$A x = b\f$. If
  \f$b\f$ is an \f$n\f$-vector, the package may solve instead the system
  \f$A^T x = b\f$. Both square (\f$m=n\f$) and
  rectangular (\f$m \neq n\f$)  matrices are handled; one of an infinite
  class of solutions for consistent systems will be returned
  whenever \f$A\f$ is not of full rank.

  The method provides a common interface to a variety of well-known solvers
  from HSL and elsewhere. Currently supported solvers include \c MA28/GLS
  and \c HSL\_MA48 from HSL, as well as \c GETR from LAPACK.
  Note that, with the exception of he Netlib reference LAPACK code,
  <b> the solvers themselves do not form part of this package
  and must be obtained separately.</b>
  Dummy instances are provided for solvers that are unavailable.
  Also note that additional flexibility may be obtained by calling the
  solvers directly rather that via this package.

  \subsection uls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection uls_date Originally released

  August 2009,  C interface December 2021.

  \subsection uls_terminology Terminology

  The solvers used each produce an \f$P_R L U P_C\f$ factorization
  of \f$A\f$, where \f$L\f$ and \f$U\f$ are lower and upper triangular
  matrices, and \f$P_R\f$ and \f$P_C\f$ are row and column permutation
  matrices respectively.

  \subsection uls_method Method

  Variants of sparse Gaussian elimination are used.

  The solver \c GLS is available as part of GALAHAD and relies on
  the HSL Archive packages \c MA33. To obtain HSL Archive packages, see

    http://hsl.rl.ac.uk/archive/ .

  The solver \c HSL\_MA48 is part of HSL 2007. To obtain HSL 2007 packages, see

    http://hsl.rl.ac.uk/hsl2007/ .

    \subsection uls_references Reference

  The methods used are described in the user-documentation for

    HSL 2007, A collection of {F}ortran codes for large-scale scientific
    computation (2007).  \n
    http://www.cse.clrc.ac.uk/nag/hsl

  The solver \c GETR is available as \c S/DGETRF/S as part of LAPACK.
  Reference versions are provided by GALAHAD, but for good performance
  machined-tuned versions should be used.

  \subsection uls_call_order Call order

  To solve a given problem, functions from the uls package must be called
  in the following order:

  - \link uls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link uls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link uls_factorize_matrix \endlink - set up matrix data structures,
       analyse the structure to choose a suitable order for factorization,
       and then factorize the matrix \f$A\f$
  - \link uls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link uls_solve_system \endlink - solve the linear system of
        equations \f$Ax=b\f$ or \f$A^Tx=b\f$
  - \link uls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link uls_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

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
#ifndef GALAHAD_ULS_H
#define GALAHAD_ULS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_gls.h"
#include "hsl_ma48.h"

/**
 * control derived type as a C struct
 */
struct uls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    ipc_ error;

    /// \brief
    /// unit for warning messages
    ipc_ warning;

    /// \brief
    /// unit for monitor output
    ipc_ out;

    /// \brief
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// controls level of diagnostic output from external solver
    ipc_ print_level_solver;

    /// \brief
    /// prediction of factor by which the fill-in will exceed the initial
    /// number of nonzeros in \f$A\f$
    ipc_ initial_fill_in_factor;

    /// \brief
    /// initial size for real array for the factors and other data
    ipc_ min_real_factor_size;

    /// \brief
    /// initial size for integer array for the factors and other data
    ipc_ min_integer_factor_size;

    /// \brief
    /// maximum size for real array for the factors and other data
    int64_t max_factor_size;

    /// \brief
    /// level 3 blocking in factorize
    ipc_ blas_block_size_factorize;

    /// \brief
    /// level 2 and 3 blocking in solve
    ipc_ blas_block_size_solve;

    /// \brief
    /// pivot control:
    /// \li 1  Threshold Partial Pivoting is desired
    /// \li 2  Threshold Rook Pivoting is desired
    /// \li 3  Threshold Complete Pivoting is desired
    /// \li 4  Threshold Symmetric Pivoting is desired
    /// \li 5  Threshold Diagonal Pivoting is desired
    ipc_ pivot_control;

    /// \brief
    /// number of rows/columns pivot selection restricted to
    /// (0 = no restriction)
    ipc_ pivot_search_limit;

    /// \brief
    /// the minimum permitted size of blocks within the block-triangular form
    ipc_ minimum_size_for_btf;

    /// \brief
    /// maximum number of iterative refinements allowed
    ipc_ max_iterative_refinements;

    /// \brief
    /// stop if the matrix is found to be structurally singular
    bool stop_if_singular;

    /// \brief
    /// factor by which arrays sizes are to be increased if they are too small
    rpc_ array_increase_factor;

    /// \brief
    /// switch to full code when the density exceeds this factor
    rpc_ switch_to_full_code_density;

    /// \brief
    /// if previously allocated internal workspace arrays are greater than
    /// array_decrease_factor times the currently required sizes, they are reset
    /// to current requirements
    rpc_ array_decrease_factor;

    /// \brief
    /// pivot threshold
    rpc_ relative_pivot_tolerance;

    /// \brief
    /// any pivot small than this is considered zero
    rpc_ absolute_pivot_tolerance;

    /// \brief
    /// any entry smaller than this in modulus is reset to zero
    rpc_ zero_tolerance;

    /// \brief
    /// refinement will cease as soon as the residual \f$\|Ax-b\|\f$ falls
    /// below max( acceptable_residual_relative * \f$\|b\|\f$,
    ///            acceptable_residual_absolute )
    rpc_ acceptable_residual_relative;
    /// see acceptable_residual_relative
    rpc_ acceptable_residual_absolute;

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
    /// \li 0  success
    /// \li -1  allocation error
    /// \li -2  deallocation error
    /// \li -3  matrix data faulty (m < 1, n < 1, ne < 0)
    /// \li -26  unknown solver
    /// \li -29  unavailable option
    /// \li -31  input order is not a permutation or is faulty in some other way
    /// \li -32  error with integer workspace
    /// \li -33  error with real workspace
    /// \li -50  solver-specific error; see the solver's info parameter
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// further information on failure
    ipc_ more_info;

    /// \brief
    /// number of indices out-of-range
    int64_t out_of_range;

    /// \brief
    /// number of duplicates
    int64_t duplicates;

    /// \brief
    /// number of entries dropped during the factorization
    int64_t entries_dropped;

    /// \brief
    /// predicted or actual number of reals and integers to hold factors
    int64_t workspace_factors;

    /// \brief
    /// number of compresses of data required
    ipc_ compresses;

    /// \brief
    /// number of entries in factors
    int64_t entries_in_factors;

    /// \brief
    /// estimated rank of the matrix
    ipc_ rank;

    /// \brief
    /// structural rank of the matrix
    ipc_ structural_rank;

    /// \brief
    /// pivot control:
    /// \li 1  Threshold Partial Pivoting has been used
    /// \li 2  Threshold Rook Pivoting has been used
    /// \li 3  Threshold Complete Pivoting has been desired
    /// \li 4  Threshold Symmetric Pivoting has been desired
    /// \li 5  Threshold Diagonal Pivoting has been desired
    ipc_ pivot_control;

    /// \brief
    /// number of iterative refinements performed
    ipc_ iterative_refinements;

    /// \brief
    /// has an "alternative" y: A^T y = 0 and yT b > 0 been found when
    /// trying to solve A x = b ?
    bool alternative;

    /// \brief
    /// name of external solver used to factorize and solve
    char solver[21];

    /// \brief
    /// the output arrays from GLS
    struct gls_ainfo_type gls_ainfo;
    /// see gls_ainfo
    struct gls_finfo_type gls_finfo;
    /// see gls_ainfo
    struct gls_sinfo_type gls_sinfo;

    /// \brief
    /// the output arrays from MA48
    struct ma48_ainfo ma48_ainfo;
    /// see ma48_ainfo
    struct ma48_finfo ma48_finfo;
    /// see ma48_ainfo
    struct ma48_sinfo ma48_sinfo;

    /// \brief
    /// the LAPACK error return code
    ipc_ lapack_error;

};

// *-*-*-*-*-*-*-*-*-*-    U L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void uls_initialize( const char solver[],
                     void **data,
                     struct uls_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

 Select solver, set default control values and initialize private data

 @param[in] solver is a one-dimensional array of type char that specifies
    the \link external solver package \endlink
    that should be used to factorize the matrix \f$A\f$. It should be one of
   'gls', 'ma28', 'ma48 or 'getr'; lower or upper case variants are allowed.

 @param[in,out] data  holds private internal data

 @param[out] control is a struct containing control information
              (see uls_control_type)

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
  \li -26. The requested solver is not available.
*/

// *-*-*-*-*-*-*-*-*-    U L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void uls_read_specfile( struct uls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNULS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/uls.pdf for a list of keywords that may be set.

  @param[in,out]  control  is a struct containing control information
              (see uls_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-    U L S  _ F A C T O R I Z E _ M A T R I X   -*-*-*-*-*-*-

void uls_factorize_matrix( struct uls_control_type *control,
                           void **data,
                           ipc_ *status,
                           ipc_ m,
                           ipc_ n,
                           const char type[],
                           ipc_ ne,
                           const rpc_ val[],
                           const ipc_ row[],
                           const ipc_ col[],
                           const ipc_ ptr[] );

/*!<
 Import matrix data into internal storage prior to solution, analyse
 the sparsity patern, and subsequently factorize the matrix

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see uls_control_type)

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The import, analysis and factorization were conducted succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 and m> 0 or requirement that the matrix type
       must contain the relevant string 'dense', 'coordinate' or 'sparse_by_rows
       has been violated.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.
  \li -32. More than control.max integer factor size words of internal
       integer storage are required for in-core factorization.
  \li -50. A solver-specific error occurred; check the solver-specific
       information component of inform along with the solver’s
       documentation for more details.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    rows in the unsymmetric matrix \f$A\f$.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    columns in the unsymmetric matrix \f$A\f$.

 @param[in] type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices unsymmetric storage scheme \endlink
   used for the matrix \f$A\f$. It should be one of 'coordinate',
   'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the matrix \f$A\f$ in any of
   the supported storage schemes.

 @param[in] row is a one-dimensional array of size ne and type ipc_, that
   holds the row indices of the matrix \f$A\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in] col is a one-dimensional array of size ne and type ipc_,
   that holds the column indices of the matrix \f$A\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense storage schemes is used,  and in this
   case can be NULL.

 @param[in]  ptr is a one-dimensional array of size m+1 and type ipc_,
   that holds the starting position of  each row of the matrix
   \f$A\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

*/

// *-*-*-*-*-*-*-    U L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void uls_reset_control( struct uls_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see uls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
*/

//  *-*-*-*-*-*-*-*-   U L S _ s o l v e _ s y s t e m   -*-*-*-*-*-*-*-*-*-

void uls_solve_system( void **data,
                       ipc_ *status,
                       ipc_ m,
                       ipc_ n,
                       rpc_ sol[],
                       bool trans );

/*!<
 Solve the linear system \f$Ax=b\f$ or \f$A^Tx=b\f$.

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The required solution was obtained.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -34. The package PARDISO failed; check the solver-specific
       information components inform.pardiso iparm and inform.pardiso_dparm
       along with PARDISO’s documentation for more details.
  \li -35. The package WSMP failed; check the solver-specific information
       components inform.wsmp_iparm and inform.wsmp dparm along with WSMP’s
       documentation for more details.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    rows in the unsymmetric matrix \f$A\f$.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    columns in the unsymmetric matrix \f$A\f$.

 @param[in,out] sol is a one-dimensional array of size n and type double.
    On entry, it must hold the vector \f$b\f$. On a successful exit,
    its contains the solution \f$x\f$.

 @param[in] trans is a scalar variable of type bool, that specifies
    whether to solve the equation \f$A^Tx=b\f$ (trans=true) or
    \f$Ax=b\f$ (trans=false).
*/

// *-*-*-*-*-*-*-*-*-*-    U L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void uls_information( void **data,
                      struct uls_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see uls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
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

/** \anchor examples
   \f$\label{examples}\f$
   \example ulst.c
   This is an example of how to use the package in conjunction with the
   sparse linear solver \c sils.
   A variety of supported matrix storage formats are illustrated.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example ulstf.c
   This is the same example, but now fortran-style indexing is used.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
