//* \file galahad_sbls.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SBLS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 24th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package sbls

  \section sbls_intro Introduction

  \subsection sbls_purpose Purpose

  Given a <b>block, symmetric matrix</b>
  \f[K_H = \mat{cc}{ H & A^T \\ A & - C },\f]
\manonly
  \n
  K_H = ( H  A^T )
        ( A  - C )
  \n
\endmanonly
  this package constructs a variety of <b>preconditioners</b> of the form
  \f[K_{G} = \mat{cc}{ G & A^T \\ A & - C }.\f]
\manonly
  \n
  K_G = ( G  A^T ).
        ( A  - C )
  \n
\endmanonly
  Here, the leading-block matrix \f$G\f$ is a suitably-chosen
  approximation to \f$H\f$; it may either be prescribed <b>explicitly</b>, in
  which case a symmetric indefinite factorization of \f$K_G\f$
  will be formed using the GALAHAD symmetric matrix factorization package SLS,
  or <b>implicitly</b>, by requiring certain sub-blocks of \f$G\f$
  be zero. In the latter case, a factorization of \f$K_G\f$ will be
  obtained implicitly (and more efficiently) without recourse to SLS.
  In particular, for implicit preconditioners, a reordering
  \f[K_G = P
  \mat{ccc}{ G_{11}^{} & G_{21}^T & A_1^T \\  G_{21}^{} & G_{22}^{} & A_2^T \\
  A_{1}^{} & A_{2}^{} & - C} P^T
  \f]
\manonly
  \n
          ( G_11  G_21^T  A_1^T )
  K_G = P ( G_21   G_22   A_2^T ) P^T
          (  A_1   A_2     - C  )
  \n
\endmanonly
  involving a suitable permutation \f$P\f$ will be found, for some
  invertible sub-block (``basis'') \f$A_1\f$ of the columns of \f$A\f$;
  the selection and factorization of \f$A_1\f$ uses
  the GALAHAD unsymmetric matrix factorization package ULS.
  Once the preconditioner has been constructed,
  solutions to the preconditioning system
  \f[\mat{cc}{ G & A^T \\ A  & - C } \vect{ x \\ y }
   = \vect{a \\ b}
  \f]
\manonly
\n
  ( G  A^T ) ( x ) = ( a )
  ( A  - C ) ( y )   ( b )
\n
\endmanonly
may be obtained by the package.
  Full advantage is taken of any zero coefficients in the matrices \f$H\f$,
  \f$A\f$ and \f$C\f$.

  \subsection sbls_authors Authors

  H. S. Dollar and N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sbls_date Originally released

  April 2006, C interface November 2021.

  \subsection sbls_method Method

  The method used depends on whether an explicit or implicit
  factorization is required. In the explicit case, the
  package is really little more than a wrapper for the GALAHAD
  symmetric, indefinite linear solver SLS in
  which the system matrix \f$K_G\f$ is assembled from its constituents
  \f$A\f$, \f$C\f$ and whichever \f$G\f$ is requested by the user.
  Implicit-factorization preconditioners are more involved,
  and there is a large variety of different possibilities. The
  essential ideas are described in detail in

  H. S. Dollar, N. I. M. Gould and A. J. Wathen.
  ``On implicit-factorization constraint preconditioners''.
  In Large Scale Nonlinear Optimization (G. Di Pillo and M. Roma, eds.)
  Springer Series on Nonconvex Optimization and Its Applications, Vol. 83,
  Springer Verlag (2006) 61--82

  and

  H. S. Dollar, N. I. M. Gould, W. H. A. Schilders and A. J. Wathen
  ``On iterative methods and implicit-factorization preconditioners for
  regularized saddle-point systems''.
  SIAM Journal on Matrix Analysis and Applications, 28(1) (2006) 170--189.

  The range-space factorization is based upon the decomposition
  \f[K_{G} = \mat{cc}{ G & 0 \\ A & I}
  \mat{cc}{ G^{-1} & 0 \\ 0 & - S}  \mat{cc}{ G & A^T \\ 0 & I},
  \f]
\manonly
  \n
  K_G = ( G  0 ) ( G^{-1} 0 ) ( G A^T )
        ( A  I ) (   0   -S ) ( 0  I  )
  \n
\endmanonly
  where the ``Schur complement'' \f$S = C + A G^{-1} A^T\f$.
  Such a method requires that \f$S\f$ is easily invertible. This is often the
  case when \f$G\f$ is a diagonal matrix, in which case \f$S\f$ is frequently
  sparse, or when \f$m \ll n\f$ in which case \f$S\f$
  is small and a dense Cholesky factorization may be used.

  When \f$C = 0\f$, the null-space factorization is based upon the decomposition
  \f[K_{G} = P  \mat{ccc}{  G_{11}^{} & 0 & I \\
  G_{21}^{} & I & A_{2}^{T} A_{1}^{-T} \\  A_{1}^{} & 0 & 0 }
  \mat{ccc}{0 & 0 & I \\ \;\;\; 0 \;\; & \;\; R \;\; & 0 \\ I & 0 & - G_{11}^{}}
  \mat{ccc}{  G_{11}^{} & G_{21}^T & A_{1}^T \\  0 & I & 0 \\
  I & A_{1}^{-1} A_{2}^{} & 0} P^T,
  \f]
\manonly
  \n
          ( G_11  0      I          ) ( 0  0   I   )
  K_G = P ( G_21  I  A_2^T A_1^{-T} ) ( 0  R   0   )
          ( A_1   0      0          ) ( I  0 -G_11 )

            ( G_11   G_21^T   A_1^T )
          . (  0        I        0  ) P^T,
            (  I    A_1^{-1} A_2  0 )
  \n
\endmanonly
  where the ``reduced Hessian''
  \f[R = ( - A_{2}^{T} A_1^{-T} \;\; I )
  \mat{cc}{G_{11}^{} & G_{21}^{T} \\ G_{21}^{} & G_{22}^{}}
  \vect{ - A_1^{-1} A_2^{} \\ I}
  \f]
\manonly
  \n
   R = ( -A_2^T A_1^{-T}  I )( G_11  G_21^T ) ( -A_1^{-1} A_2 )
                             ( G_21   G_22  ) (       I       )
  \n
\endmanonly
  and \f$P\f$ is a suitably-chosen permutation for which \f$A_1\f$ is
  invertible. The method is most useful when \f$m \approx n\f$ as then the
  dimension of \f$R\f$ is small and a dense Cholesky factorization may be used.

  \subsection sbls_call_order Call order

  To solve a given problem, functions from the sbls package must be called
  in the following order:

  - \link sbls_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link sbls_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link sbls_import \endlink - set up matrix data structures
  - \link sbls_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link sbls_factorize_matrix \endlink - form and factorize the block
      matrix from its components
  - \link sbls_solve_system \endlink - solve the block linear system of
      equations
  - \link sbls_information \endlink (optional) - recover information about
    the solution and solution process
  - \link sbls_terminate \endlink - deallocate data structures

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

  The unsymmetric \f$m\f$ by \f$n\f$ constraint matrix \f$A\f$ may be presented
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

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  Likewise, the symmetric \f$n\f$ by \f$n\f$ matrix \f$H\f$, as well as
  the \f$m\f$ by \f$m\f$ matrix \f$C\f$,  may be presented
  and stored in a variety of formats. But crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal). We focus
  on \f$H\f$, but everything we say applies equally to \f$C\f$.

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$H\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$h_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array H_val
  will hold the value \f$h_{ij}\f$ (and, by symmetry, \f$h_{ji}\f$)
  for \f$0 \leq j \leq i \leq n-1\f$.

  \subsubsection symmetric_matrix_coordinate Sparse co-ordinate storage format

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j
  and value \f$h_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays H_row and
  H_col and real array H_val, respectively, while the number of nonzeros
  is recorded as H_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsubsection symmetric_matrix_row_wise Sparse row-wise storage format

  Again only the nonzero entries are stored, but this time
  they are ordered so that those in row i appear directly before those
  in row i+1. For the i-th row of \f$H\f$ the i-th component of the
  integer array H_ptr holds the position of the first entry in this row,
  while H_ptr(n) holds the total number of entries.
  The column indices j, \f$0 \leq j \leq i\f$, and values
  \f$h_{ij}\f$ of the  entries in the i-th row are stored in components
  l = H_ptr(i), \f$\ldots\f$, H_ptr(i+1)-1 of the
  integer array H_col, and real array H_val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsubsection symmetric_matrix_diagonal Diagonal storage format

  If \f$H\f$ is diagonal (i.e., \f$H_{ij} = 0\f$ for all
  \f$0 \leq i \neq j \leq n-1\f$) only the diagonals entries
  \f$H_{ii}\f$, \f$0 \leq i \leq n-1\f$ need
  be stored, and the first n components of the array H_val may be
  used for the purpose.

  \subsubsection symmetric_matrix_scaled_identity Multiples of the identity storage format

  If \f$H\f$ is a multiple of the identity matrix, (i.e., \f$H = \alpha I\f$
  where \f$I\f$ is the n by n identity matrix and \f$\alpha\f$ is a scalar),
  it suffices to store \f$\alpha\f$ as the first component of H_val.

  \subsubsection symmetric_matrix_identity The identity matrix format

  If \f$H\f$ is the identity matrix, no values need be stored.

  \subsubsection symmetric_matrix_zero The zero matrix format

  The same is true if \f$H\f$ is the zero matrix.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SBLS_H
#define GALAHAD_SBLS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_sls.h"
#include "galahad_uls.h"


/**
 * control derived type as a C struct
 */
struct sbls_control_type {

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
    /// initial estimate of integer workspace for SLS (obsolete)
    ipc_ indmin;

    /// \brief
    /// initial estimate of real workspace for SLS (obsolete)
    ipc_ valmin;

    /// \brief
    /// initial estimate of workspace for ULS (obsolete)
    ipc_ len_ulsmin;

    /// \brief
    /// maximum number of iterative refinements with preconditioner allowed
    ipc_ itref_max;

    /// \brief
    /// maximum number of projected CG iterations allowed
    ipc_ maxit_pcg;

    /// \brief
    /// how much has \f$A\f$ changed since last factorization:
    /// 0 = not changed, 1 = values changed, 2 = structure changed
    ipc_ new_a;

    /// \brief
    /// how much has \f$H\f$ changed since last factorization:
    /// 0 = not changed, 1 = values changed, 2 = structure changed
    ipc_ new_h;

    /// \brief
    /// how much has \f$C\f$ changed since last factorization:
    /// 0 = not changed, 1 = values changed, 2 = structure changed
    ipc_ new_c;

    /// \brief
    /// which preconditioner to use:
    /// \li 0 selected automatically
    /// \li 1 explicit with \f$G = I\f$
    /// \li 2 explicit with \f$G = H\f$
    /// \li 3 explicit with \f$G = \f$ diag(max(\f$H\f$,min_diag))
    /// \li 4 explicit with \f$G =\f$ band\f$(H)\f$
    /// \li 5 explicit with \f$G =\f$ (optional, diagonal) \f$D\f$
    /// \li 11 explicit with \f$G_{11} = 0\f$, \f$G_{21} = 0\f$,
    ///      \f$G_{22} = H_{22}\f$
    /// \li 12 explicit with \f$G_{11} = 0\f$, \f$G_{21} = H_{21}\f$,
    /// \f$G_{22} = H_{22}\f$
    /// \li -1 implicit with \f$G_{11} = 0\f$, \f$G_{21} = 0\f$,
    /// \f$G_{22} = I\f$
    /// \li -2 implicit with \f$G_{11} = 0\f$, \f$G_{21} = 0\f$,
    /// \f$G_{22} = H_{22}\f$
    ipc_ preconditioner;

    /// \brief
    /// the semi-bandwidth for band(H)
    ipc_ semi_bandwidth;

    /// \brief
    /// the explicit factorization used:
    /// \li 0 selected automatically
    /// \li 1 Schur-complement if \f$G\f$ is diagonal and successful
    ///       otherwise augmented system
    /// \li 2 augmented system
    /// \li 3 null-space
    /// \li 4 Schur-complement if \f$G\f$ is diagonal and successful
    ///       otherwise failure
    /// \li 5 Schur-complement with pivoting if \f$G\f$ is diagonal and
    ///       successful otherwise failure
    ipc_ factorization;

    /// \brief
    /// maximum number of nonzeros in a column of \f$A\f$ for Schur-complement
    /// factorization
    ipc_ max_col;

    /// \brief
    /// not used at present
    ipc_ scaling;
    /// see scaling
    ipc_ ordering;

    /// \brief
    /// the relative pivot tolerance used by ULS (obsolete)
    rpc_ pivot_tol;

    /// \brief
    /// the relative pivot tolerance used by ULS when determining
    /// the basis matrix
    rpc_ pivot_tol_for_basis;

    /// \brief
    /// the absolute pivot tolerance used by ULS (obsolete)
    rpc_ zero_pivot;

    /// \brief
    /// not used at present
    rpc_ static_tolerance;
    /// see static_tolerance
    rpc_ static_level;

    /// \brief
    /// the minimum permitted diagonal in diag(max(\f$H\f$,min_diag))
    rpc_ min_diagonal;

    /// \brief
    /// the required absolute and relative accuracies
    rpc_ stop_absolute;
    /// see stop_absolute
    rpc_ stop_relative;

    /// \brief
    /// preprocess equality constraints to remove linear dependencies
    bool remove_dependencies;

    /// \brief
    /// determine implicit factorization preconditioners using a
    /// basis of A found by examining A's transpose
    bool find_basis_by_transpose;

    /// \brief
    /// can the right-hand side \f$c\f$ be assumed to be zero?
    bool affine;

    /// \brief
    /// do we tolerate "singular" preconditioners?
    bool allow_singular;

    /// \brief
    /// if the initial attempt at finding a preconditioner is unsuccessful,
    /// should the diagonal be perturbed so that a second attempt succeeds?
    ///
    bool perturb_to_make_definite;

    /// \brief
    /// compute the residual when applying the preconditioner?
    bool get_norm_residual;

    /// \brief
    /// if an implicit or null-space preconditioner is used, assess and
    /// correct for ill conditioned basis matrices
    bool check_basis;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// indefinite linear equation solver
    char symmetric_linear_solver[31];

    /// \brief
    /// definite linear equation solver
    char definite_linear_solver[31];

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
struct sbls_time_type {

    /// \brief
    /// total cpu time spent in the package
    rpc_ total;

    /// \brief
    /// cpu time spent forming the preconditioner \f$K_G\f$
    rpc_ form;

    /// \brief
    /// cpu time spent factorizing \f$K_G\f$
    rpc_ factorize;

    /// \brief
    /// cpu time spent solving linear systems inolving \f$K_G\f$
    rpc_ apply;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time spent forming the preconditioner \f$K_G\f$
    rpc_ clock_form;

    /// \brief
    /// clock time spent factorizing \f$K_G\f$
    rpc_ clock_factorize;

    /// \brief
    /// clock time spent solving linear systems inolving \f$K_G\f$
    rpc_ clock_apply;
};

/**
 * inform derived type as a C struct
 */
struct sbls_inform_type {

    /// \brief
    /// return status. See SBLS_form_and_factorize for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred
    char bad_alloc[81];

    /// \brief
    /// the return status from the sorting routines
    ipc_ sort_status;

    /// \brief
    /// the total integer workspace required for the factorization
    int64_t factorization_integer;

    /// \brief
    /// the total real workspace required for the factorization
    int64_t factorization_real;

    /// \brief
    /// the preconditioner used
    ipc_ preconditioner;

    /// \brief
    /// the factorization used
    ipc_ factorization;

    /// \brief
    /// how many of the diagonals in the factorization are positive
    ipc_ d_plus;

    /// \brief
    /// the computed rank of \f$A\f$
    ipc_ rank;

    /// \brief
    /// is the matrix A rank defficient?
    bool rank_def;

    /// \brief
    /// has the used preconditioner been perturbed to guarantee correct inertia?
    bool perturbed;

    /// \brief
    /// the total number of projected CG iterations required
    ipc_ iter_pcg;

    /// \brief
    /// the norm of the residual
    rpc_ norm_residual;

    /// \brief
    /// has an "alternative" \f$y\f$: \f$K y = 0\f$ and \f$y^T c > 0\f$
    /// been found when trying to solve \f$K y = c\f$ for generic \f$K\f$?
    bool alternative;

    /// \brief
    /// timings (see above)
    struct sbls_time_type time;

    /// \brief
    /// inform parameters from the GALAHAD package SLS used
    struct sls_inform_type sls_inform;

    /// \brief
    /// inform parameters from the GALAHAD package ULS used
    struct uls_inform_type uls_inform;
};

// *-*-*-*-*-*-*-*-*-*-    S B L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sbls_initialize( void **data,
                     struct sbls_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

 @param[in,out] data  holds private internal data

 @param[out] control  is a struct containing control information
              (see sbls_control_type)

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    S B L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void sbls_read_specfile( struct sbls_control_type *control,
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNSBLS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/sbls.pdf for a list of keywords that may be set.

  @param[in,out]  control  is a struct containing control information
              (see sbls_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    S B L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void sbls_import( struct sbls_control_type *control,
                 void **data,
                 ipc_ *status,
                 ipc_ n,
                 ipc_ m,
                 const char H_type[],
                 ipc_ H_ne,
                 const ipc_ H_row[],
                 const ipc_ H_col[],
                 const ipc_ H_ptr[],
                 const char A_type[],
                 ipc_ A_ne,
                 const ipc_ A_row[],
                 const ipc_ A_col[],
                 const ipc_ A_ptr[],
                 const char C_type[],
                 ipc_ C_ne,
                 const ipc_ C_row[],
                 const ipc_ C_col[],
                 const ipc_ C_ptr[] );

/*!<
 Import structural matrix data into internal storage prior to solution.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sbls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
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
  \li -3. The restrictions n > 0 or m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$H\f$.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$C\f$.

 @param[in]  H_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the matrix \f$H\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity',
   'zero' or 'none', the latter pair if \f$H=0\f$; lower or upper
   case variants are allowed.

 @param[in]  H_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$H\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  H_row is a one-dimensional array of size H_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$H\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  H_col is a one-dimensional array of size H_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$H\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense, diagonal or (scaled) identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  H_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$H\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.

 @param[in]  A_type is a one-dimensional array of type char that specifies the
   \link main_unsymmetric_matrices symmetric storage scheme \endlink
   used for the matrix \f$A\f$. It should be one of 'coordinate',
  'sparse_by_rows', 'dense' or 'absent', the latter if access to the Jacobian
  is via matrix-vector products; lower or upper case variants are allowed.

 @param[in]  A_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes.

 @param[in]  A_row is a one-dimensional array of size A_ne and type ipc_, that
   holds the row indices of \f$A\f$ in the sparse co-ordinate storage scheme.
   It need not be set for any of the other schemes,
   and in this case can be NULL.

 @param[in]  A_col is a one-dimensional array of size A_ne and type ipc_,
   that holds the column indices of \f$A\f$ in either the sparse co-ordinate,
   or the sparse row-wise storage scheme. It need not be set when the
   dense or diagonal storage schemes are used, and in this case can be NULL.

 @param[in]  A_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each row of \f$A\f$, as well as the
   total number of entries, in the sparse row-wise storage scheme.
   It need not be set when the other schemes are used,
   and in this case can be NULL.

 @param[in]  C_type is a one-dimensional array of type char that specifies the
   \link main_symmetric_matrices symmetric storage scheme \endlink
   used for the matrix \f$C\f$. It should be one of 'coordinate',
   'sparse_by_rows', 'dense', 'diagonal', 'scaled_identity', 'identity',
   'zero' or 'none', the latter pair if \f$C=0\f$; lower or upper
   case variants are allowed.

 @param[in]  C_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$C\f$ in the sparse co-ordinate
   storage scheme. It need not be set for any of the other schemes.

 @param[in]  C_row is a one-dimensional array of size C_ne and type ipc_, that
   holds the row indices of the lower triangular part of \f$C\f$ in the sparse
   co-ordinate storage scheme. It need not be set for any of the other
   three schemes, and in this case can be NULL.

 @param[in]  C_col is a one-dimensional array of size C_ne and type ipc_,
   that holds the column indices of the lower triangular part of \f$C\f$ in
   either the sparse co-ordinate, or the sparse row-wise storage scheme. It
   need not be set when the dense, diagonal or (scaled) identity storage
   schemes are used,  and in this case can be NULL.

 @param[in]  C_ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of  each row of the lower
   triangular part of \f$C\f$, as well as the total number of entries,
   in the sparse row-wise storage scheme. It need not be set when the
   other schemes are used, and in this case can be NULL.
*/

// *-*-*-*-*-*-*-    S B L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void sbls_reset_control( struct sbls_control_type *control,
                 void **data,
                 ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sbls_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
*/

//  *-*-*-*-*-*-   S B L S _ f a c t o r i z e _ m a t r i x   -*-*-*-*-*-*-*-

void sbls_factorize_matrix( void **data,
                            ipc_ *status,
                            ipc_ n,
                            ipc_ h_ne,
                            const rpc_ H_val[],
                            ipc_ a_ne,
                            const rpc_ A_val[],
                            ipc_ c_ne,
                            const rpc_ C_val[],
                            const rpc_ D[] );

/*!<
 Form and factorize the block matrix
  \f[K_{G} = \mat{cc}{ G & A^T \\ A  & - C }\f]
\manonly
  \n
  K_G = ( G  A^T )
        ( A  - C )
  \n
\endmanonly
for some appropriate matrix \f$G\f$.

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
  \li -3. The restrictions n > 0 and m > 0 or requirement that a type contains
       its relevant string 'dense', 'coordinate', 'sparse_by_rows',
       'diagonal', 'scaled_identity', 'identity', 'zero' or 'none'
        has been violated.

  \li -9. An error was reported by SLS analyse. The return status from SLS
       analyse is given in inform.sls_inform.status.  See the
       documentation for the GALAHAD package SLS for further details.

  \li -10. An error was reported by SLS_factorize. The return status from SLS
       factorize is given in inform.sls_inform.status.  See the
       documentation for the GALAHAD package SLS for further details.

  \li -13. An error was reported by ULS_factorize. The return status from
       ULS_factorize is given in inform.uls_factorize_status. See the
       documentation for the GALAHAD package ULS for further details.

  \li -15. The computed preconditioner \f$K_G\f$ is singular and is thus
       unsuitable.

  \li -20. The computed preconditioner \f$K_G\f$ has the wrong inertia
       and is thus unsuitable.

  \li -24. An error was reported by the GALAHAD package SORT_reorder_by_rows.
    The return status from SORT_reorder_by_rows is given in inform.sort_status.
    See the documentation for the GALAHAD package SORT for further details.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$H\f$.

 @param[in] h_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$H\f$.

 @param[in] H_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$H\f$ in any of the available storage schemes

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
    entries in the unsymmetric matrix \f$A\f$.

 @param[in] A_val is a one-dimensional array of size a_ne and type rpc_,
    that holds the values of the entries of the unsymmetric matrix
    \f$A\f$ in any of the available storage schemes.

 @param[in] c_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the symmetric matrix \f$C\f$.

 @param[in] C_val is a one-dimensional array of size c_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    symmetric matrix \f$C\f$ in any of the available storage schemes

 @param[in] D is a one-dimensional array of size n and type rpc_,
    that holds the values of the entries of the diagonal matrix
    \f$D\f$ that is required if the user has specified
    control.preconditioner = 5. It need not be set otherwise.
*/

//  *-*-*-*-*-*-*-*-   S B L S _ s o l v e _ s y s t e m   -*-*-*-*-*-*-*-*-*-

void sbls_solve_system( void **data,
                        ipc_ *status,
                        ipc_ n,
                        ipc_ m,
                        rpc_ sol[] );

/*!<
 Solve the block linear system
  \f[\mat{cc}{ G & A^T \\ A  & - C } \vect{ x \\ y }
   = \vect{a \\ b}.
  \f]
\manonly
\n
  ( G  A^T ) ( x ) = ( a ).
  ( A  - C ) ( y )   ( b )
\n
\endmanonly

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

  \li -11. An error was reported by SLS_solve. The return status from SLS
       solve is given in inform.sls_inform.status.  See the documentation
       for the GALAHAD package SLS for further details.

  \li -14. An error was reported by ULS_solve. The return status from
       ULS_solve is given in inform.uls_solve_status.  See the documentation
       for the GALAHAD package ULS for further details.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    entries in the vector \f$a\f$.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    entries in the vector \f$b\f$.

 @param[in,out] sol is a one-dimensional array of size n + m and type double.
    on entry, its first n entries must hold the vector \f$a\f$, and the
    following entries must hold the vector \f$b\f$. On a successful exit,
    its first n entries contain the solution components \f$x\f$, and the
    following entries contain the components \f$y\f$.
*/

// *-*-*-*-*-*-*-*-*-*-    S B L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sbls_information( void **data,
                      struct sbls_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see sbls_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S B L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void sbls_terminate( void **data,
                    struct sbls_control_type *control,
                    struct sbls_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control is a struct containing control information
              (see sbls_control_type)

  @param[out] inform is a struct containing output information
              (see sbls_inform_type)
 */

/** \anchor examples
   \f$\label{examples}\f$
   \example sblst.c
   This is an example of how to use the package.\n
   A variety of supported matrix storage formats are illustrated.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example sblstf.c
   This is the same example, but now fortran-style indexing is used.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
