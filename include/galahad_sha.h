//* \file galahad_sha.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SHA C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. January 28th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package sha

  \section sha_intro Introduction

  \subsection sha_purpose Purpose

  Find a <b>component-wise secant approximation to the Hessian matrix</b>
  \f$H(x)\f$, for which \f$(H(x))_{i,j} = \partial f^2 (x) / \partial x_i \partial x_j\f$,
  \f$1 \leq i, j \leq n\f$, using values of the gradient \f$g(x) = \nabla_x f(x)\f$
  of the function \f$f(x)\f$ of \f$n\f$ unknowns \f$x = (x_1, \ldots, x_n)^T\f$
  at a sequence of given distinct \f${x^{(k)}}\f$, \f$k \geq 0\f$.
  More specifically, given <b>differences</b>
  \f[s^{(k)} = x^{(k+1)} - x^{(k)}\f]
\manonly
  \n
  s^(k) = x^(k+1) - x^(k)
  \n
\endmanonly
  and
  \f[y^{(k)} = g(x^{(k+1)}) - g(x^{(k)}) \f]
\manonly
  \n
  y^(k) = g(x^(k+1)) - g(x^(k))
  \n
\endmanonly
  the package aims to find a good estimate \f$B\f$ of \f$H(x)\f$ for
  which the secant conditions \f$B s^{(k)} = y^{(k)}\f$ hold
  approximately for a chosen set of values \f$k\f$.
  The methods provided take advantage of the entries in the Hessian that
  are known to be zero.

  The package is particularly intended to allow gradient-based
  optimization methods, that generate iterates
  \f$x^{(k+1)} = x^{(k)} + s^{(k)}\f$ based upon the values \f$g( x^{(k)})\f$
  for \f$k \geq 0\f$, to build a suitable approximation to the Hessian
  \f$H(x^{(k+1)})\f$. This then gives the method an opportunity to
  accelerate the iteration using the Hessian approximation.

  \subsection sha_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sha_date Originally released

  April 2013, C interface January 2022.

  \subsection sha_method Method

  The package computes the entries in the each row of \f$B\f$ one at a time.
  The entries \f$b_{ij}\f$ in row \f$i\f$ may be chosen to
  \f[(1) \;\;\; \minin{b_{i,j}} \;\; \sum_{k \in {\cal I}_i}
  \left[ \sum_{{\scriptscriptstyle \mbox{nonzeros}}\; j}
  b_{i,j} s_j^{(k)} - y_i^{(k)} \right]^2,
  \f]
\manonly
  \n
  (1)  min_{b_{i,j}} sum_{k \in I_i}
                   [ sum_{nonzeros j} b_{i,j} s_j^(k) - y_i^(k) ]^2
  \n
\endmanonly
  where \f$I_i\f$ is ideally chosen to be sufficiently large so that
  (1) has a unique minimizer. Since this requires that there are at least
  as many \f$(s^{(k)}, y^{(k)})\f$ pairs as the maximum number of nonzeros
  in any row, this may be prohibitive in some cases. We might then be content
  with a minimum-norm (under-determined) least-squares solution. Or, we may
  take advantage of the symmetry of the Hessian, and note that if we
  have already found the values in row \f$j\f$, then the value
  \f$b_{i,j} = b_{j,i}\f$
  in (1) is known before we process row \f$i\f$. Thus by ordering the rows
  and exploiting symmetry we may reduce the numbers of unknowns in
  future unprocessed rows.

  In the analysis phase, we order the rows by constructing the connectivity
  graph---a graph comprising nodes \f$1\f$ to \f$n\f$ and edges connecting
  nodes \f$i\f$ and \f$j\f$ if \f$h_{i,j}\f$ is everywhere nonzero---of
  \f$H(x)\f$.
  The nodes are ordered by increasing degree (that is, the number of edges
  emanating from the node) using a bucket sort. The row chosen to be
  ordered next corresponds to a node of minimum degree, the node
  is removed from the graph, the degrees updated efficiently, and the
  process repeated until all rows have been ordered. This often leads
  to a significant reduction in the numbers of unknown values in each
  row as it is processed in turn, but numerical rounding can lead to
  inaccurate values in some cases. A useful remedy is to process all
  rows for which there are sufficient \f$(s^{(k)}, y^{(k)})\f$ as before,
  and then process the remaining rows taking into account the symmetry.
  That is, the rows and columns are rearranged so that the matrix
  is in block form
  \f[B = \mat{cc}{ B_{11} & B_{12} \\ B^T_{12} & B_{22}},\f]
\manonly
  \n
  B = (  B_11  B_12 ),
      ( B_12^T B_22 )
  \n
\endmanonly
  the \f$( B_{11} \;\; B_{12})\f$ rows are processed without regard
  for symmetry but give the \f$2,1\f$ block \f$B^T_{12}\f$, and finally
  the \f$2,2\f$ block \f$B_{22}\f$ is processed either
  with the option of exploiting
  symmetry. More details of the precise algorithms (Algorithms 2.1--2.5)
  are given in the reference below. The linear least-squares problems (1)
  themselves are solved by a choice of LAPACK packages.

  \subsection sha_references Reference

  The method employed is described in detail in

  J. M. Fowkes, N. I. M. Gould and J. A. Scott,
   Approximating large-scale Hessians using secant equations.
   Technical Report TR-2023, Rutherford Appleton Laboratory.

  \subsection sha_call_order Call order

  To find the Hessian approximation, functions from the sha package
  must be called in the following order:

  - \link sha_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link sha_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link sha_analyse_matrix \endlink - set up structures needed to
      construct the Hessian approximation
  - \link sha_recover_matrix \endlink - construct the Hessian approximation
  - \link sha_information \endlink (optional) - recover information about
    the solution and solution process
  - \link sha_terminate \endlink - deallocate data structures

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SHA_H
#define GALAHAD_SHA_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct sha_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required. <= 0 gives no output, = 1 gives a one-line
    /// summary for every iteration, = 2 gives a summary of the inner iteration
    /// for each iteration, >= 3 gives increasingly verbose (debugging) output
    ipc_ print_level;

    /// \brief
    /// which approximation algorithm should be used?
    /// \li 1 : unsymmetric (alg 2.1 in paper)
    /// \li 2 : symmetric (alg 2.2 in paper)
    /// \li 3 : composite (alg 2.3 in paper)
    /// \li 4 : composite 2 (alg 2.4 in paper)
    /// \li 5 : cautious (alg 2.5 in paper)
    ipc_ approximation_algorithm;

    /// \brief
    /// which dense linear equation solver should be used?
    /// \li 1 : Gaussian elimination
    /// \li 2 : QR factorization
    /// \li 3 : singular-value decomposition
    /// \li 4 : singular-value decomposition with divide-and-conquer
    ipc_ dense_linear_solver;

    /// \brief
    /// if available use an addition extra_differences differences
    ipc_ extra_differences;

    /// \brief
    /// rows with no more that sparse_row entries are considered sparse
    ipc_ sparse_row;

    /// \brief
    /// if a recursive algorithm is used (Alg 2.4), limit on the maximum number 
    /// of levels of recursion
    ipc_ recursion_max;

    /// \brief
    /// if a recursive algorithm is used (Alg 2.4), recursion can only 
    /// occur for a (reduced) row if it has at least .recursion_allowed entries
    ipc_ recursion_entries_required;

    /// \brief
    /// select if pairs of off-diagonal Hessian estimates are to be averaged 
    /// on return. Otherwise pick the value from the upper triangle
    bool average_off_diagonals;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];
};

/**
 * inform derived type as a C struct
 */
struct sha_inform_type {

    /// \brief
    /// return status. See SHA_solve for details
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the maximum degree in the adgacency graph.
    ipc_ max_degree;

    /// \brief
    /// the number of differences that will be needed.
    ipc_ differences_needed;

    /// \brief
    /// the maximum reduced degree in the adgacency graph.
    ipc_ max_reduced_degree;

    /// \brief
    /// which approximation algorithm has been used
    ipc_ approximation_algorithm_used;

    /// \brief
    /// a failure occured when forming the bad_row-th row (0 = no failure).
    ipc_ bad_row;

    /// \brief
    /// the maximum difference between estimated Hessian off-diagonal 
    /// pairs if approximation algorithm 1, 3 or 4 has been employed.
    rpc_ max_off_diagonal_difference;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];
};

// *-*-*-*-*-*-*-*-*-*-    S H A  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sha_initialize( void **data,
                     struct sha_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see sha_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

//  *-*-*-*-*-*-*-*-*-   S H A _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*-*

void sha_reset_control( struct sha_control_type *control,
                        void **data,
                        ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sha_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The import was succesful.
 */

// *-*-*-*-*-*-*-*-*-    S H A  _ A N A L Y S E _ M A T R I X   -*-*-*-*-*-*-*-

void sha_analyse_matrix( struct sha_control_type *control,
                         void **data,
                         ipc_ *status,
                         ipc_ n,
                         ipc_ ne,
                         const ipc_ row[],
                         const ipc_ col[],
                         ipc_ *m );

/*!<
 Import structural matrix data into internal storage prior to solution

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sha_control_type)

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. \n
    Possible values are:
  \li  0. The import and analysis were conducted succesfully.

  \li -1. An allocation error occurred. A message indicating the offending
       array is written on unit control.error, and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -2. A deallocation error occurred.  A message indicating the offending
       array is written on unit control.error and the returned allocation
       status and a string containing the name of the offending array
       are held in inform.alloc_status and inform.bad_alloc respectively.
  \li -3. The restrictions n > 0 or ne >= 0 has been violated.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$H\f$.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
   entries in the upper triangular part of \f$H\f$ in the sparse co-ordinate
   storage scheme

 @param[in] row is a one-dimensional array of size ne and type ipc_, that
   holds the row indices of the upper triangular part of \f$H\f$ in the sparse
   co-ordinate storage scheme

 @param[in] col is a one-dimensional array of size ne and type ipc_,
   that holds the column indices of the upper triangular part of \f$H\f$ in
   sparse row-wise storage scheme.

 @param[out] m is a scalar variable of type ipc_, that holds the minimum number
  of \f$(s^(k),y^(k))\f$ pairs that will be needed to recover a good
  Hessian approximation

*/

//  *-*-*-*-*-*-*-   S H A _ R E C O V E R _ m a t r i x   -*-*-*-*-*-*-*-

void sha_recover_matrix( void **data,
                         ipc_ *status,
                         ipc_ ne,
                         ipc_ m,
                         ipc_ ls1,
                         ipc_ ls2,
                         const rpc_ strans[][ls2],
                         ipc_ ly1,
                         ipc_ ly2,
                         const rpc_ ytrans[][ly2],
                         rpc_ val[],
                         const ipc_ precedence[] );

/*!<
 Form and factorize the symmetric matrix \f$A\f$.

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
  \li -3. The restrictions n > 0 or ne >= 0 has been violated.

 @param[in] ne is a scalar variable of type ipc_, that holds the number of
    entries in the upper triangular part of the symmetric matrix \f$H\f$.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    \f$(s,y)\f$ pairs that are available.

 @param[in] ls1 is a scalar variable of type ipc_, that holds the leading
    dimension of the array s.

 @param[in] ls2 is a scalar variable of type ipc_, that holds the trailing
    dimension of the array s.

 @param[in] strans is a two-dimensional array of size [ls1][ls2] and
    type rpc_, that holds the values of the vectors \f$s^{(k) T}\f$.
    Component \f$k,i\f$ holds \f$s_i^{(k)}\f$.

 @param[in] ly1 is a scalar variable of type ipc_, that holds the leading
    dimension of the array y.

 @param[in] ly2 is a scalar variable of type ipc_, that holds the trailing
    dimension of the array y.

 @param[in] ytrans is a two-dimensional array of size [ly1][ly2] and
    type rpc_, that holds the values of the vectors \f$y^{(k) T}\f$.
    Component \f$k,i\f$ holds \f$y_i^{(k)}\f$.

 @param[out] val is a one-dimensional array of size ne and type rpc_,
    that holds the values of the entries of the upper triangular part of the
    symmetric matrix \f$H\f$ in the sparse coordinate scheme.

 @param[in] precedence is a one-dimensional array of size m and type ipc_, that
   holds the preferred order of access for the pairs \f$(s^(k),y^(k))\f$.
   The \f$k\f$-th component of precedence specifies the row number of strans
   and ytrans that will be used as the \f$k\f$-th most favoured. precedence
   need not be set if the natural order, \f$k, k = 1,...,\f$ m, is desired,
   and this case precedence should be NULL.
*/

// *-*-*-*-*-*-*-*-*-*-    S H A  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sha_information( void **data,
                      struct sha_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see sha_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S H A  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void sha_terminate( void **data,
                    struct sha_control_type *control,
                    struct sha_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see sha_control_type)

  @param[out] inform is a struct containing output information
              (see sha_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
