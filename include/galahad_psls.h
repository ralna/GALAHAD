//* \file galahad_psls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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

  Given an \f$n\f$ by \f$n\f$  sparse symmetric matrix \f$A =a_{ij}\f$,
  this package <b>builds a suitable symmetric, positive definite (or
  diagonally dominant)-preconditioner \f$P\f$ of \f$A\f$ or a symmetric
  sub-matrix thereof</b>. The matrix \f$A\f$ need not be definite. Facilities
  are provided to apply the preconditioner to a given vector, and to
  remove rows and columns (symmetrically) from the initial
  preconditioner without a full re-factorization.

  \subsection psls_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection psls_date Originally released

  April 2008, C interface January 2022.

  \subsection psls_method Method and references

  The basic preconditioners are described in detail in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint (1992).
  LANCELOT. A fortran package for large-scale nonlinear optimization
  (release A). Springer Verlag Series in Computational Mathematics 17,
  Berlin, Section 3.3.10,

  along with the more modern versions implements in ICFS due to

  C.-J. Lin and J. J. More' (1999).
  Incomplete Cholesky factorizations with limited memory.
  SIAM Journal on Scientific Computing <b>21</b> 21-45,

  and in HSL_MI28 described by

  J. A. Scott and M. Tuma (2013). HSL MI28: an efficient and robust
  limited-memory incomplete Cholesky factorization code.
  ACM Transactions on Mathematical Software <b>40(4)</b> (2014), Article 24.

  The factorization methods used by the GALAHAD package SLS in conjunction
  with some preconditioners are described in the documentation to that
  package. The key features of the external solvers supported by SLS are
  given in the following table.

\manonly
(ignore next paragraph - doxygen bug!)
\endmanonly

<table>
<caption>External solver characteristics</caption>
<tr><th> solver <th> factorization <th> indefinite \f$A\f$
    <th> out-of-core <th> parallelised
<tr><td> \c SILS/MA27 <td> multifrontal <td> yes <td> no <td> no
<tr><td> \c HSL_MA57 <td> multifrontal <td> yes <td> no <td> no
<tr><td> \c HSL_MA77 <td> multifrontal <td> yes <td> yes <td> OpenMP core
<tr><td> \c HSL_MA86 <td> left-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c HSL_MA87 <td> left-looking <td> no <td> no <td> OpenMP fully
<tr><td> \c HSL_MA97 <td> multifrontal <td> yes <td> no <td> OpenMP core
<tr><td> \c SSIDS <td> multifrontal <td> yes <td> no <td> CUDA core
<tr><td> \c MUMPS <td> multifrontal <td> yes <td> optionally <td> MPI
<tr><td> \c PARDISO <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c MKL_PARDISO <td> left-right-looking <td> yes <td> optionally
     <td> OpenMP fully
<tr><td> \c PaStix <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c WSMP <td> left-right-looking <td> yes <td> no <td> OpenMP fully
<tr><td> \c POTR <td> dense <td> no <td> no <td> with parallel LAPACK
<tr><td> \c SYTR <td> dense <td> yes <td> no <td> with parallel LAPACK
<tr><td> \c PBTR <td> dense band <td> no <td> no <td> with parallel LAPACK
</table>

\manonly
External solver characteristics (ooc = out-of-core factorization)

 solver     factorization indefinite A  ooc    parallelised
 SILS/MA27   multifrontal          yes   no    no
 HSL_MA57    multifrontal          yes   no    no
 HSL_MA77    multifrontal          yes  yes    OpenMP core
 HSL_MA86    left-looking          yes   no    OpenMP fully
 HSL_MA87    left-looking           no   no    OpenMP fully
 HSL_MA97    multifrontal          yes   no    OpenMP core
 SSIDS       multifrontal          yes   no    CUDA core
 MUMPS       multifrontal          yes  optionally  MPI
 PARDISO     left-right-looking    yes   no    OpenMP fully
 MKL_PARDISO left-right-looking    yes  optionally  OpenMP fully
 PaStix      left-right-looking    yes   no    OpenMP fully
 WSMP        left-right-looking    yes   no    OpenMP fully
 POTR        dense                  no   no    with parallel LAPACK
 SYTR        dense                 yes   no    with parallel LAPACK
 PBTR        dense band             no   no    with parallel LAPACK
\endmanonly

  Note that <b> the solvers themselves do not form part of this package and
  must be obtained separately.</b>
  Dummy instances are provided for solvers that are unavailable.

  Orderings to reduce the bandwidth, as implemented in HSL's MC61, are due to

  J. K. Reid and J. A. Scott (1999)
  Ordering symmetric sparse matrices for small profile and wavefront
  International Journal for Numerical Methods in Engineering
  <b>45</b> 1737-1755.

  If a subset of the rows and columns are specified, the remaining rows/columns
  are removed before processing. Any subsequent removal of rows and columns
  is achieved using the GALAHAD Schur-complement updating package SCU
  unless a complete re-factorization is likely more efficient.

  \subsection psls_call_order Call order

  To solve a given problem, functions from the psls package must be called
  in the following order:

  - \link psls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link psls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link psls_import \endlink - set up matrix data structures for \f$A\f$
      prior to solution
  - \link psls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - one of
    - \link psls_form_preconditioner \endlink - form and factorize a
      preconditioner \f$P\f$ of the matrix \f$A\f$
    - \link psls_form_subset_preconditioner \endlink - form and factorize a
      preconditioner \f$P\f$ of a symmetric submatrix of the matrix \f$A\f$
  - \link psls_update_preconditioner \endlink (optional) - update the
       preconditioner \f$P\f$ when rows (amd columns) are removed
  - \link psls_apply_preconditioner \endlink - solve the linear system of
        equations \f$Px=b\f$
  - \link psls_information \endlink (optional) - recover information about
    the preconditioner and solution process
  - \link psls_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

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
#ifndef GALAHAD_PSLS_H
#define GALAHAD_PSLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "hsl_mi28.h"

/**
 * control derived type as a C struct
 */
struct psls_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    ipc_ error;

    /// \brief
    /// unit for monitor output
    ipc_ out;

    /// \brief
    /// controls level of diagnostic output
    ipc_ print_level;

    /// \brief
    /// which preconditioner to use:
    /// \li <0 no preconditioning occurs, \f$P = I\f$
    /// \li  0 the preconditioner is chosen automatically
    ///         (forthcoming, and currently defaults to 1).
    /// \li  1 \f$A\f$ is replaced by the diagonal,
    ///         \f$P\f$ = diag( max( \f$A\f$, .min_diagonal ) ).
    /// \li  2 \f$A\f$ is replaced by the band
    ///         \f$P\f$ = band( \f$A\f$ ) with semi-bandwidth .semi_bandwidth.
    /// \li  3 \f$A\f$ is replaced by the reordered band
    ///         \f$P\f$ = band( order( \f$A\f$ ) ) with semi-bandwidth
    ///         .semi_bandwidth, where order is chosen by the HSL package
    ///         MC61 to move entries closer to the diagonal.
    /// \li  4 \f$P\f$ is a full factorization of \f$A\f$ using Schnabel-Eskow
    ///         modifications, in which small or negative diagonals are
    ///         made sensibly positive during the factorization.
    /// \li  5 \f$P\f$ is a full factorization of \f$A\f$  due to Gill, Murray,
    ///         Ponceleon and Saunders, in which an indefinite factorization
    ///         is altered to give a positive definite one.
    /// \li  6 \f$P\f$ is an incomplete Cholesky factorization of \f$A\f$ using
    ///        the  package ICFS due to Lin and More'.
    /// \li 7  \f$P\f$ is an incomplete factorization of \f$A\f$ implemented
    ///         as HSL_MI28 from HSL.
    /// \li 8  \f$P\f$ is an incomplete factorization of \f$A\f$ due
    ///         to Munskgaard (forthcoming).
    /// \li >8 treated as 0.
    ///
    /// <b>N.B.</b> Options 3-8 may require additional external software that
    /// is not part of the package, and that must be obtained separately.
    ipc_ preconditioner;

    /// \brief
    /// the semi-bandwidth for band(H) when .preconditioner = 2,3
    ipc_ semi_bandwidth;

    /// \brief
    /// not used at present
    ipc_ scaling;
    /// see scaling
    ipc_ ordering;

    /// \brief
    /// maximum number of nonzeros in a column of \f$A\f$ for Schur-complement
    /// factorization to accommodate newly deleted rpws and columns
    ipc_ max_col;

    /// \brief
    /// number of extra vectors of length n required by the Lin-More'
    /// incomplete Cholesky preconditioner when .preconditioner = 6
    ipc_ icfs_vectors;

    /// \brief
    /// the maximum number of fill entries within each column of the incomplete
    /// factor L computed by HSL_MI28 when .preconditioner = 7. In general,
    /// increasing mi28_lsize improve
    /// the quality of the preconditioner but increases the time to compute
    /// and then apply the preconditioner. Values less than 0 are treated as 0
    ipc_ mi28_lsize;

    /// \brief
    /// the maximum number of entries within each column of the strictly lower
    /// triangular matrix \f$R\f$ used in the computation of the preconditioner
    /// by HSL_MI28 when .preconditioner = 7. Rank-1 arrays of size
    /// .mi28_rsize *  n are allocated internally to hold \f$R\f$. Thus the
    /// amount of memory used, as well as the amount of work involved
    /// in computing the preconditioner, depends on mi28_rsize. Setting
    /// mi28_rsize > 0 generally leads to a higher quality preconditioner than
    /// using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally
    /// recommended
    ipc_ mi28_rsize;

    /// \brief
    /// the minimum permitted diagonal in diag(max(H,.min_diagonal))
    rpc_ min_diagonal;

    /// \brief
    /// set new_structure true if the storage structure for the input matrix
    /// has changed, and false if only the values have changed
    bool new_structure;

    /// \brief
    /// set get_semi_bandwidth true if the semi-bandwidth of the submatrix is
    /// to be calculated
    bool get_semi_bandwidth;

    /// \brief
    /// set get_norm_residual true if the residual when applying the
    /// preconditioner are to be calculated
    bool get_norm_residual;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// the definite linear equation solver used when .preconditioner = 3,4.
    /// Possible choices are currently:
    /// sils, ma27, ma57, ma77, ma86, ma87, ma97, ssids, mumps, pardiso,
    /// mkl_pardiso,pastix, wsmp, potr and pbtr, although only sils, potr,
    /// pbtr and,
    /// for OMP 4.0-compliant compilers, ssids are installed by default.
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
    struct mi28_control mi28_control;
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
    rpc_ clock_total;

    /// \brief
    /// clock time to assemble the preconditioner prior to factorization
    rpc_ clock_assemble;

    /// \brief
    /// clock time for the analysis phase
    rpc_ clock_analyse;

    /// \brief
    /// clock time for the factorization phase
    rpc_ clock_factorize;

    /// \brief
    /// clock time for the linear solution phase
    rpc_ clock_solve;

    /// \brief
    /// clock time to update the factorization
    rpc_ clock_update;
};

/**
 * inform derived type as a C struct
 */
struct psls_inform_type {

    /// \brief
    /// reported return status:
    /// \li 0  success
    /// \li -1  allocation error
    /// \li -2  deallocation error
    /// \li -3  matrix data faulty (.n < 1, .ne < 0)
    ipc_ status;

    /// \brief
    /// STAT value after allocate failure
    ipc_ alloc_status;

    /// \brief
    /// status return from factorization
    ipc_ analyse_status;

    /// \brief
    /// status return from factorization
    ipc_ factorize_status;

    /// \brief
    /// status return from solution phase
    ipc_ solve_status;

    /// \brief
    /// number of integer words to hold factors
    int64_t factorization_integer;

    /// \brief
    /// number of real words to hold factors
    int64_t factorization_real;

    /// \brief
    /// code for the actual preconditioner used (see control.preconditioner)
    ipc_ preconditioner;

    /// \brief
    /// the actual semi-bandwidth
    ipc_ semi_bandwidth;

    /// \brief
    /// the semi-bandwidth following reordering (if any)
    ipc_ reordered_semi_bandwidth;

    /// \brief
    /// number of indices out-of-range
    ipc_ out_of_range;

    /// \brief
    /// number of duplicates
    ipc_ duplicates;

    /// \brief
    /// number of entries from the strict upper triangle
    ipc_ upper;

    /// \brief
    /// number of missing diagonal entries for an allegedly-definite matrix
    ipc_ missing_diagonals;

    /// \brief
    /// the semi-bandwidth used
    ipc_ semi_bandwidth_used;

    /// \brief
    /// number of 1 by 1 pivots in the factorization
    ipc_ neg1;

    /// \brief
    /// number of 2 by 2 pivots in the factorization
    ipc_ neg2;

    /// \brief
    /// has the preconditioner been perturbed during the fctorization?
    bool perturbed;

    /// \brief
    /// ratio of fill in to original nonzeros
    rpc_ fill_in_ratio;

    /// \brief
    /// the norm of the solution residual
    rpc_ norm_residual;

    /// \brief
    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// the integer and real output arrays from mc61
    ipc_ mc61_info[10];
    /// see mc61_info
    rpc_ mc61_rinfo[15];

    /// \brief
    /// times for various stages
    struct psls_time_type time;

    /// \brief
    /// inform values from SLS
    struct sls_inform_type sls_inform;

    /// \brief
    /// the output structure from mi28
    struct mi28_info mi28_info;
};

// *-*-*-*-*-*-*-*-*-*-    P S L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void psls_initialize( void **data,
                     struct psls_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see psls_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    P S L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void psls_read_specfile( struct psls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNPSLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/psls.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see psls_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    P S L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void psls_import( struct psls_control_type *control,
                  void **data,
                  ipc_ *status,
                  ipc_ n,
                  const char type[],
                  ipc_ ne,
                  const ipc_ row[],
                  const ipc_ col[],
                  const ipc_ ptr[] );

/*!<
 Import structural matrix data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see psls_control_type)

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
       its relevant string 'dense', 'coordinate', 'sparse_by_rows' or
       'diagonal' has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$A\f$.

 @param[in] type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the matrix \f$A\f$. It should be one of 'coordinate',
   'sparse_by_rows' or 'dense'; lower or upper case variants are allowed.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$A\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in] row is a one-dimensional array of size ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$A\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in] col is a one-dimensional array of size ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$A\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense storage scheme is used, and in this case
   can be NULL.

 @param[in]  ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$A\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

*/

// *-*-*-*-*-*-*-    P S L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void psls_reset_control( struct psls_control_type *control,
                 void **data,
                 ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see psls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

//  *-*-*-*-*-*- P S L S _ f o r m _ p r e c o n d i t i o n e r  -*-*-*-*-*-*-

void psls_form_preconditioner( void **data,
                               ipc_ *status,
                               ipc_ ne,
                               const rpc_ val[] );

/*!<
 Form and factorize a preconditioner \f$P\f$ of the matrix \f$A\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The factors were generated succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$A\f$.

 @param[in] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$A\f$ in any of the supported storage schemes.
*/

//  *-*-*- P S L S _ f o r m _ s u b s e t _ p r e c o n d i t i o n e r  -*-*-

void psls_form_subset_preconditioner( void **data,
                                      ipc_ *status,
                                      ipc_ ne,
                                      const rpc_ val[],
                                      ipc_ n_sub,
                                      const ipc_ sub[] );

/*!<
 Form and factorize a \f$P\f$ preconditioner of a symmetric submatrix of
 the matrix \f$A\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The factors were generated succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$A\f$.

 @param[in] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$A\f$ in any of the supported storage schemes.

 @param[in] n_sub is a scalar variable of type ipc_, that holds the number of
    rows (and columns) of the required submatrix of \f$A\f$.

 @param[in] sub is a one-dimensional array of size n_sub and type ipc_,
    that holds the indices of the rows of required submatrix.
*/

//  *-*-*-*-*- P S L S _ u p d a t e _ p r e c o n d i t i o n e r  -*-*-*-*-

void psls_update_preconditioner( void **data,
                                 ipc_ *status,
                                 ipc_ ne,
                                 const rpc_ val[],
                                 ipc_ n_del,
                                 const ipc_ del[] );

/*!<
 Update the preconditioner \f$P\f$ when rows (amd columns) are removed.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The factors were generated succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -26. The requested solver is not available.
  \li -29. This option is not available with this solver.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$A\f$.

 @param[in] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$A\f$ in any of the supported storage schemes.

 @param[in] n_del is a scalar variable of type ipc_, that holds the number of
    rows (and columns) of (sub) matrix that are to be deleted.

 @param[in] del is a one-dimensional array of size n_fix and type ipc_,
    that holds the indices of the rows that are to be deleted.
*/

//  *-*-*-*-*- P S L S _ a p p l y _ p r e c o n d i t i o n e r  -*-*-*-*-

void psls_apply_preconditioner( void **data,
                                ipc_ *status,
                                ipc_ n,
                                rpc_ sol[] );

/*!<
 Solve the linear system \f$Px=b\f$.

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

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    entries in the vectors \f$b\f$ and \f$x\f$.

 @param[in,out] sol is a one-dimensional array of size n and type double.
    On entry, it must hold the vector \f$b\f$. On a successful exit,
    its contains the solution \f$x\f$. Any component corresponding to
    rows/columns not in the initial subset recorded by
    psls_form_subset_preconditioner, or in those subsequently deleted by
    psls_update_preconditioner, will not be altered.
*/

// *-*-*-*-*-*-*-*-*-*-    P S L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void psls_information( void **data,
                      struct psls_inform_type *inform,
                      ipc_ *status );

/*!<
  Provide output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see psls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
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
