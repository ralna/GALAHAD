//* \file galahad.h */

/*! \mainpage GALAHAD C packages

  \section main_intro Introduction

  GALAHAD is foremost a modern fortran library of packages designed
  to solve continuous optimization problems, with a particular
  emphasis on those that involve a large number of unknowns.
  Since many application programs or applications are written in
  other languages, of late there has been a considerable effort to
  provide interfaces to GALAHAD. Thus there are Matlab interfaces,
  and here we provide details of those to C using the standardized
  ISO C support now provided within fortran.

  \subsection main_authors Main authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England, \n
  D. Orban, Polytechnique Montréal, Canada, \n
  D. P. Robinson, Leheigh University, USA, \n
  Ph. L. Toint, The University of Namur, Belgium, \n
  J. Fowkes, STFC-Rutherford Appleton Laboratory, England, and \n
  A. Montoison, Polytechnique Montréal, Canada.

  GALAHAD provides packages as named for the following problems:

  \li \link fdc \endlink - determine consistency and redundancy of linear
      systems
      \latexonly \href{fdc_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/fdc/fdc.html">(link)</a> \endhtmlonly
  \li \link lpa \endlink - linear programming using an active-set method
      \latexonly \href{lpa_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/lpa/lpa.html">(link)</a> \endhtmlonly
  \li \link lpb \endlink - linear programming using an interior-point method
      \latexonly \href{lpb_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/lpb/lpb.html">(link)</a> \endhtmlonly
  \li \link wcp \endlink - linear feasibility using an interior-point method
      \latexonly \href{wcp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/wcp/wcp.html">(link)</a> \endhtmlonly
  \li \link blls \endlink - bound-constrained linear least-squares problems
      using a gradient-projection method
      \latexonly \href{blls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/blls/blls.html">(link)</a> \endhtmlonly
  \li bllsb - bound-constrained linear-least-squares using
      an interior-point method (in preparation)
  \li \link slls \endlink - simplex-constrained linear least-squares problems
      using a gradient-projection method
      \latexonly \href{slls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/blls/slls.html">(link)</a> \endhtmlonly
  \li \link presolve \endlink - simplify quadratic programs prior to solution
      \latexonly \href{presolve_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/presolve/presolve.html">(link)</a>
      \endhtmlonly
  \li \link bqp \endlink - bound-constrained convex quadratic programming
      using a gradient-projection method
      \latexonly \href{bqp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/bqp/bqp.html">(link)</a> \endhtmlonly
  \li \link bqpb \endlink - bound-constrained convex quadratic programming
      using an interior-point method
      \latexonly \href{bqpb_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/bqpb/bqpb.html">(link)</a> \endhtmlonly
  \li \link lsqp \endlink - linear and separable quadratic programming using
      an interior-point method
      \latexonly \href{lsqp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/lsqp/lsqp.html">(link)</a> \endhtmlonly
  \li \link cqp \endlink - convex quadratic programming using
      an interior-point method
      \latexonly \href{cqp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/cqp/cqp.html">(link)</a> \endhtmlonly
  \li \link dqp \endlink - convex quadratic programming using a
       dual active-set method
      \latexonly \href{dqp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/dqp/dqp.html">(link)</a> \endhtmlonly
  \li \link eqp \endlink - equality-constrained quadratic programming using
      an iterative method
      \latexonly \href{eqp_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/eqp/eqp.html">(link)</a> \endhtmlonly
  \li \link trs \endlink - the trust-region subproblem using
       matrix factorization
      \latexonly \href{trs_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/trs/trs.html">(link)</a> \endhtmlonly
  \li \link gltr  \endlink - the trust-region subproblem using
       matrix-vector products
      \latexonly \href{gltr_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/gltr/gltr.html">(link)</a> \endhtmlonly
  \li \link rqs \endlink - the regularized quadratic subproblem
      using matrix factorization
      \latexonly \href{rqs_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/rqs/rqs.html">(link)</a> \endhtmlonly
  \li \link glrt  \endlink - the regularized quadratic subproblem using
       matrix-vector products
      \latexonly \href{glrt_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/glrt/glrt.html">(link)</a> \endhtmlonly
  \li \link dps \endlink - the trust-region and regularized quadratic
      subproblems in a diagonalising norm
      \latexonly \href{dps_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/dps/dps.html">(link)</a> \endhtmlonly
  \li \link lstr  \endlink - the least-squares trust-region subproblem using
       matrix-vector products
      \latexonly \href{lstr_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/lstr/lstr.html">(link)</a> \endhtmlonly
  \li \link lsrt  \endlink - the regularized least-squares subproblem using
       matrix-vector products
      \latexonly \href{lsrt_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/lsrt/lsrt.html">(link)</a> \endhtmlonly
  \li \link l2rt  \endlink - the regularized linear \f$l_2\f$ norm subproblem
       using matrix-vector products
      \latexonly \href{l2rt_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/l2rt/l2rt.html">(link)</a> \endhtmlonly
  \li \link qpa \endlink - general quadratic programming using an
       active-set method
      \latexonly \href{qpa_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/qpa/qpa.html">(link)</a> \endhtmlonly
  \li \link qpb \endlink - general quadratic programming using an
      interior-point method
      \latexonly \href{qpb_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/qpb/qpb.html">(link)</a> \endhtmlonly
  \li \link tru \endlink - unconstrained optimization using a trust-region
      method
      \latexonly \href{tru_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/tru/tru.html">(link)</a> \endhtmlonly
  \li \link arc \endlink - unconstrained optimization using a regularization
      method
      \latexonly \href{arc_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/arc/arc.html">(link)</a> \endhtmlonly
  \li \link nls \endlink - least-squares optimization using a regularization
      method
      \latexonly \href{nls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/nls/nls.html">(link)</a> \endhtmlonly
  \li \link trb \endlink - bound-constrained optimization using a
      gradient-projection trust-region method
      \latexonly \href{trb_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/trb/trb.html">(link)</a> \endhtmlonly
  \li \link ugo \endlink - univariate global optimization
      \latexonly \href{ugo_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/ugo/ugo.html">(link)</a> \endhtmlonly
  \li \link bgo \endlink - multivariate global optimization in a box using a
      multi-start trust-region method
      \latexonly \href{bgo_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/bgo/bgo.html">(link)</a> \endhtmlonly
  \li \link dgo \endlink - multivariate global optimization in a box using a
      deterministic partition-and-bound method
      \latexonly \href{dgo_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/dgo/dgo.html">(link)</a> \endhtmlonly
  \li nlsb - bound-constrained least-squares optimization
      using a gradient-projection regularization method  (in preparation)
  \li lancelot - general constrained optimization using
      an augmented Lagrangian method (interface in preparation)
  \li fisqp - general constrained optimization using an SQP method
      (in preparation)

  In addition, there are packages for solving a variety of required
  sub tasks, and most specifically interface routines to external
  solvers for solving linear equations:

  \li \link uls \endlink - unsymmetric linear systems
      \latexonly \href{uls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/uls/uls.html">(link)</a> \endhtmlonly
  \li \link sls \endlink - symmetric linear systems
      \latexonly \href{sls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/sls/sls.html">(link)</a> \endhtmlonly
  \li \link sbls \endlink - symmetric block linear systems
      \latexonly \href{sbls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/sbls/sbls.html">(link)</a> \endhtmlonly
  \li \link psls \endlink - preconditioners for symmetric linear systems
      \latexonly \href{psls_c.pdf}{(link)} \endlatexonly
      \htmlonly <a href="../../../html/C/psls/psls.html">(link)</a> \endhtmlonly

  C interfaces to all of these are underway, and each will be released
  once it is ready. If \b you have a particular need, please let us know,
  and we will raise its priority!

  Interface header files are in $GALAHAD/include; that for a package named
  pack will be in the file galahad_pack.h. PDF documentation for pack
  will be in pack_c.pdf in the directory, and there is a man page entry
  in the file pack_c.3 in $GALAHAD/man/man3.

  \section main_topics Further topics

  \subsection main_unsymmetric_matrices Unsymmetric matrix storage formats

  An unsymmetric \f$m\f$ by \f$n\f$ matrix \f$A\f$ may be presented and
  stored in a variety of convenient input formats.

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

  \subsubsection unsymmetric_matrix_dense Dense by columns storage format

  The matrix \f$A\f$ is stored as a compact  dense matrix by columns, that is,
  the values of the entries of each column in turn are
  stored in order within an appropriate real one-dimensional array.
  In this case, component \f$m \ast j + i\f$  of the storage array A_val
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

  \subsubsection unsymmetric_matrix_column_wise Sparse column-wise storage format

  Once again only the nonzero entries are stored, but this time
  they are ordered so that those in column j appear directly before those
  in column j+1. For the j-th column of \f$A\f$ the j-th component of the
  integer array A_ptr holds the position of the first entry in this column,
  while A_ptr(n) holds the total number of entries.
  The row indices i, \f$0 \leq i \leq m-1\f$, and values \f$A_{ij}\f$
  of the  nonzero entries in the j-th columnsare stored in components
  l = A_ptr(j), \f$\ldots\f$, A_ptr(j+1)-1, \f$0 \leq j \leq n-1\f$,
  of the integer array A_row, and real array A_val, respectively.
  As before, for sparse matrices, this scheme almost always requires less
  storage than the co-ordinate format.

  \subsection main_symmetric_matrices Symmetric matrix storage formats

  Likewise, a symmetric \f$n\f$ by \f$n\f$ matrix \f$H\f$ may be presented
  and stored in a variety of formats. But crucially symmetry is exploited
  by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  \subsubsection symmetric_matrix_dense Dense storage format

  The matrix \f$H\f$ is stored as a compact  dense matrix by rows, that is,
  the values of the entries of each row in turn are
  stored in order within an appropriate real one-dimensional array.
  Since \f$H\f$ is symmetric, only the lower triangular part (that is the part
  \f$H_{ij}\f$ for \f$0 \leq j \leq i \leq n-1\f$) need be held.
  In this case the lower triangle should be stored by rows, that is
  component \f$i \ast i / 2 + j\f$  of the storage array H_val
  will hold the value \f$H_{ij}\f$ (and, by symmetry, \f$h_{ji}\f$)
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
  \f$H_{ij}\f$ of the  entries in the i-th row are stored in components
  l = H_ptr(i), \f$\ldots\f$, H_ptr(i+1)-1 of the
  integer array H_col, and real array H_val, respectively.
  Note that as before only the entries in the lower triangle should be stored.
  For sparse matrices, this scheme almost always requires less storage than
  its predecessor.

  \subsubsection symmetric_matrix_diagonal Diagonal storage format

  If \f$H\f$ is diagonal (i.e., \f$h_{ij} = 0\f$ for all
  \f$0 \leq i \neq j \leq n-1\f$) only the diagonals entries
  \f$h_{ii}\f$, \f$0 \leq i \leq n-1\f$ need
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

#include "galahad_arc.h"
#include "galahad_bgo.h"
#include "galahad_blls.h"
#include "galahad_bllsb.h"
#include "galahad_bnls.h"
#include "galahad_bqp.h"
#include "galahad_bqpb.h"
#include "galahad_bsc.h"
#include "galahad_ccqp.h"
#include "galahad_clls.h"
#include "galahad_convert.h"
#include "galahad_cqp.h"
#include "galahad_cro.h"
#include "galahad_dgo.h"
#include "galahad_dps.h"
#include "galahad_dqp.h"
#include "galahad_eqp.h"
#include "galahad_fdc.h"
#include "galahad_fit.h"
#include "galahad_glrt.h"
#include "galahad_gls.h"
#include "galahad_gltr.h"
#include "galahad_hash.h"
#include "galahad_ir.h"
#include "galahad_l2rt.h"
#include "galahad_lhs.h"
#include "galahad_llsr.h"
#include "galahad_llst.h"
#include "galahad_lms.h"
#include "galahad_lpa.h"
#include "galahad_lpb.h"
#include "galahad_lsqp.h"
#include "galahad_lsrt.h"
#include "galahad_lstr.h"
#include "galahad_nls.h"
#include "galahad_presolve.h"
#include "galahad_psls.h"
#include "galahad_qpa.h"
#include "galahad_qpb.h"
#include "galahad_roots.h"
#include "galahad_rpd.h"
#include "galahad_rqs.h"
#include "galahad_sbls.h"
#include "galahad_scu.h"
#include "galahad_sec.h"
#include "galahad_sha.h"
#include "galahad_sils.h"
#include "galahad_slls.h"
#include "galahad_sls.h"
#include "galahad_trb.h"
#include "galahad_trs.h"
#include "galahad_tru.h"
#include "galahad_ugo.h"
#include "galahad_uls.h"
#include "galahad_version.h"
#include "galahad_wcp.h"
