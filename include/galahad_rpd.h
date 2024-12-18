//* \file galahad_rpd.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_RPD C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package rpd

  \section rpd_intro Introduction

  \subsection rpd_purpose Purpose

   Read and write data for the linear program (LP)
  \f[\mbox{minimize}\;\; g^T x + f
  \;\mbox{subject to}\; c_l \leq A x \leq c_u
  \;\mbox{and}\; x_l \leq  x  \leq x_u,
\f]
\manonly
  \n
  minimize     g^T x + f
   subject to  c_l <= A x <= c_u
               x_l <=  x  <= x_u,
  \n
\endmanonly
   the linear program with quadratic constraints (QCP)
  \f[\mbox{minimize}\;\; g^T x + f
  \;\mbox{subject to}\; c_l \leq A x + \frac{1}{2} \mbox{vec}(x.H_c.x) \leq c_u
  \;\mbox{and}\; x_l \leq  x  \leq x_u,
\f]
\manonly
  \n
  minimize     g^T x + f
   subject to  c_l <= A x + 1/2 vec(x.H_c.x) <= c_u
               x_l <=  x  <= x_u,
  \n
\endmanonly
   the bound-constrained quadratic program (BQP)
  \f[\mbox{minimize}\;\; \frac{1}{2} x^T H x + g^T x + f
  \;\mbox{subject to}\; x_l \leq  x  \leq x_u,
\f]
\manonly
  \n
   minimize     1/2 x^T H x + g^T x + f
   subject to   x_l <=  x  <= x_u,
  \n
\endmanonly
   the quadratic program (QP)
  \f[\mbox{minimize}\;\; \frac{1}{2} x^T H x + g^T x + f
  \;\mbox{subject to}\; c_l \leq A x \leq c_u
  \;\mbox{and}\; x_l \leq  x  \leq x_u,
\f]
\manonly
  \n
   minimize    1/2 x^T H x + g^T x + f
   subject to  c_l <= A x <= c_u
               x_l <=  x  <= x_u,
  \n
\endmanonly
   or the quadratic program with quadratic constraints (QCQP)
  \f[\mbox{minimize}\;\; \frac{1}{2} x^T H x + g^T x + f
  \;\mbox{subject to}\; c_l \leq A x + \frac{1}{2} \mbox{vec}(x.H_c.x) \leq c_u
  \;\mbox{and}\; x_l \leq  x  \leq x_u,
\f]
\manonly
  \n
  minimize     1/2 x^T H x + g^T x + f
   subject to  c_l <= A x + 1/2 vec(x.H_c.x) <= c_u
               x_l <=  x  <= x_u,
  \n
\endmanonly
   where vec\f$( x . H_c . x )\f$ is the vector whose
   \f$i\f$-th component is  \f$x^T (H_c)_i x\f$ for the \f$i\f$-th
   constraint, from and to a QPLIB-format data file.
   Variables may be continuous, binary or integer.

  \subsection rpd_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection rpd_date Originally released

  January 2006, C interface January 2022.

  \subsection rpd_references Reference

  The QPBLIB format is defined in

  F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould,
  L. Liberti, A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis,
  S. Vigerske and A. Wiegele  (2019).
  QPLIB: a library of quadratic programming instances,
  Mathematical Programming Computation <b>11</b> 237–265.

  \subsection rpd_call_order Call order

  To decode a given QPLIB file, functions from the rpd package must be called
  in the following order:

  - \link rpd_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link rpd_get_stats \endlink - read a given QPLIB file into internal
     data structures, and report vital statistics
  - (optionally, and in any order, where relevant)
     - \link rpd_get_g \endlink - get the objective gradient term \f$g\f$
     - \link rpd_get_f \endlink - get the objective constant term \f$f\f$
     - \link rpd_get_xlu \endlink - get the variable bounds
           \f$x_l\f$ and \f$x_u\f$
     - \link rpd_get_xlu \endlink - get the constraint bounds
            \f$c_l\f$ and \f$c_u\f$
     - \link rpd_get_h \endlink - get the objective Hessian term \f$H\f$
     - \link rpd_get_a \endlink - get the constrain Jacobian term \f$A\f$
     - \link rpd_get_h_c \endlink - get the constraint Hessian terms \f$H_c\f$
     - \link rpd_get_x_type \endlink - determine the type of each variable
           \f$x\f$
     - \link rpd_get_x \endlink - get initial value of the variable \f$x\f$
     - \link rpd_get_y \endlink - get initial value of Lagrange multipliers
            \f$y\f$
     - \link rpd_get_z \endlink - get initial value of the dual variables
            \f$z\f$
  - \link rpd_terminate \endlink - deallocate data structures

  \latexonly
  See Section~\ref{examples} for examples of use.
  \endlatexonly
  \htmlonly
  See the <a href="examples.html">examples tab</a> for illustrations of use.
  \endhtmlonly
  \manonly
  See the examples section for illustrations of use.
  \endmanonly

  \subsection unsymmetric_matrix_coordinate Sparse unsymmetric co-ordinate storage format

  The unsymmetric \f$m\f$ by \f$n\f$ constraint matrix \f$A\f$ will be
  output in sparse co-ordinate format.

  Both C-style (0 based)  and fortran-style (1-based) indexing is allowed.
  Choose \c control.f_indexing as \c false for C style and \c true for
  fortran style; the discussion below presumes C style, but add 1 to
  indices for the corresponding fortran version.

  Wrappers will automatically convert between 0-based (C) and 1-based
  (fortran) array indexing, so may be used transparently from C. This
  conversion involves both time and memory overheads that may be avoided
  by supplying data that is already stored using 1-based indexing.

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$A\f$,
  its row index i, column index j
  and value \f$A_{ij}\f$,
  \f$0 \leq i \leq m-1\f$,  \f$0 \leq j \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays A_row and
  A_col and real array A_val, respectively, while the number of nonzeros
  is recorded as A_ne = \f$ne\f$.

  \subsection symmetric_matrix_coordinate Sparse symmetric co-ordinate storage format

  Likewise, the symmetric \f$n\f$ by \f$n\f$ objective Hessian matrix
  \f$H\f$ will be returned in a sparse co-ordinate format. But crucially
  symmetry is exploited by only storing values from the lower triangular part
  (i.e, those entries that lie on or below the leading diagonal).

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its row index i, column index j
  and value \f$h_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays H_row and
  H_col and real array H_val, respectively, while the number of nonzeros
  is recorded as H_ne = \f$ne\f$.
  Note that only the entries in the lower triangle should be stored.

  \subsection joint_symmetric_matrix_coordinate Joint sparse symmetric co-ordinate storage format

  The symmetric \f$n\f$ by \f$n\f$ constraint Hessian matrices
  \f$ (H_c)_i\f$ are stored as a whole in a joint symmetric co-ordinate
  storage format.
  In addition to the row and column indices and values of each lower
  triangular matrix, record is also kept of the particular constraint invlved.

  Only the nonzero entries of the matrices are stored.
  For the \f$l\f$-th entry, \f$0 \leq l \leq ne-1\f$, of \f$H\f$,
  its constraint index k, row index i, column index j
  and value \f$(h_k)_{ij}\f$, \f$0 \leq j \leq i \leq n-1\f$,  are stored as
  the \f$l\f$-th components of the integer arrays H_c_ptr, H_c_row and
  H_c_col and real array H_c_val, respectively, while the number of nonzeros
  is recorded as H_c_ne = \f$ne\f$.
  Note as before that only the entries in the lower triangles should be stored.

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_RPD_H
#define GALAHAD_RPD_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct rpd_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// QPLIB file input stream number
    ipc_ qplib;

    /// \brief
    /// error and warning diagnostics occur on stream error
    ipc_ error;

    /// \brief
    /// general output occurs on stream out
    ipc_ out;

    /// \brief
    /// the level of output required is specified by print_level
    /// \li \f$\leq\f$ 0 gives no output,
    /// \li \f$\geq\f$ 1 gives increasingly verbose (debugging) output
    ipc_ print_level;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;
};

/**
 * inform derived type as a C struct
 */
struct rpd_inform_type {
    /// \brief
    ///  return status. Possible values are:
    /// \li  0 The call was successful.
    /// \li -1. An allocation error occurred. A message indicating the
    /// offending array is written on unit control.error, and the
    /// returned allocation status and a string containing the name
    /// of the offending array are held in inform.alloc_status and
    /// inform.bad_alloc respectively.
    /// \li -2. A deallocation error occurred.  A message indicating the
    /// offending array is written on unit control.error and the
    /// returned allocation status and a string containing the
    /// name of the offending array are held in
    /// inform.alloc_status and inform.bad_alloc respectively.
    /// \li -22 An input/outpur error occurred.
    /// \li -25 The end of the input file was reached prematurely.
    /// \li -29 The problem type was not recognised.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation or deallocation
    ipc_ alloc_status;

    /// \brief
    /// status from last read attempt
    ipc_ io_status;

    /// \brief
    /// number of last line read from i/o file
    ipc_ line;

    /// \brief
    /// problem type
    char p_type[4];

    /// \brief
    /// the name of the array for which an allocation or deallocation
    /// error occurred
    char bad_alloc[81];
};

// *-*-*-*-*-*-*-*-*-*-    R P D  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void rpd_initialize( void **data,
                     struct rpd_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see rpd_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ S T A T S   -*-*-*-*-*-*-*-*-*-*

void rpd_get_stats( char qplib_file[],
                    ipc_ qplib_file_len,
                    struct rpd_control_type *control,
                    void **data,
                    ipc_ *status,
                    char p_type[4],
                    ipc_ *n,
                    ipc_ *m,
                    ipc_ *h_ne,
                    ipc_ *a_ne,
                    ipc_ *h_c_ne );

/*!<
 Read the data from a specified QPLIB file into internal storage,
 and report the type of problem encoded, along with problem-specific
 dimensions.

 @param[in] qplib_file is a one-dimensional array of type char that
  specifies the name of the QPLIB file that is to be read.

 @param[in] qplib_file_len is a scalar variable of type ipc_, that gives
    the number of characters in the name encoded in qplib_file.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see rpd_control_type)

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
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

 @param[out] p_type is a one-dimensional array of size 4 and type char that
  specifies the type of quadratic programming problem encoded in the
  QPLIB file.\n\n
The first character indicates the type of objective function used.
   It will be one of the following:
   <ul>
      <li> L a linear objective function.
      <li> D a convex quadratic objective function whose Hessian is a
              diagonal matrix.
      <li> C a convex quadratic objective function.
      <li> Q a quadratic objective function whose Hessian may be indefinite.
   </ul>
   \n
The second character indicates the types of variables that are present.
     It will be one of the following:
   <ul>
      <li> C  all the variables are continuous.
      <li> B  all the variables are binary (0-1).
      <li> M  the variables are a mix of continuous and binary.
      <li> I  all the variables are integer.
      <li> G  the variables are a mix of continuous, binary and integer.
   </ul>
   \n
The third character indicates the type of the (most extreme)
     constraint function used; other constraints may be of a lesser type.
     It will be one of the following:
   <ul>
       <li> N  there are no constraints.
       <li> B  some of the variables lie between lower and upper bounds
              (box constraint).
       <li> L  the constraint functions are linear.
       <li> D  the constraint functions are convex quadratics with diagonal
              Hessians.
       <li> C  the constraint functions are convex quadratics.
       <li> Q  the constraint functions are quadratics whose Hessians
              may be indefinite.
   </ul>
     Thus for continuous problems, we would have
   <ul>
       <li> LCL            a linear program.
       <li> LCC or LCQ     a linear program with quadratic constraints.
       <li> CCB or QCB     a bound-constrained quadratic program.
       <li> CCL or QCL     a quadratic program.
       <li> CCC or CCQ or QCC or QCQ a quadratic program with quadratic
                          constraints.
   </ul>
     For integer problems, the second character would be I rather than C,
     and for mixed integer problems, the second character would by M or G.

 @param[out] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[out] m is a scalar variable of type ipc_, that holds the number of
    general constraints.

 @param[out]  h_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$H\f$ stored in the sparse
   symmetric co-ordinate storage scheme.

 @param[out]  a_ne is a scalar variable of type ipc_, that holds the number of
   entries in \f$A\f$ stored in the sparse co-ordinate storage scheme.

 @param[out]  h_c_ne is a scalar variable of type ipc_, that holds the number of
   entries in the lower triangular part of \f$H_c\f$ stored in the joint
   sparse co-ordinate storage scheme.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ G  -*-*-*-*-*-*-*-*-*-*

void rpd_get_g( void **data,
                ipc_ *status,
                ipc_ n,
                rpc_ g[]
                );

/*!<
 Recover the linear term \f$g\f$ from in objective function

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[out] g is a one-dimensional array of size n and type rpc_, that
    gives the linear term \f$g\f$ of the objective function.
    The j-th component of g, j = 0, ... ,  n-1, contains  \f$g_j \f$.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ F  -*-*-*-*-*-*-*-*-*-*

void rpd_get_f( void **data,
                ipc_ *status,
                rpc_ *f
                );

/*!<
 Recover the constant term \f$f\f$ in the objective function.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[out] f is a scalar of type rpc_, that
    gives the constant term \f$f\f$ from the objective function.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ X L U  -*-*-*-*-*-*-*-*-*-*

void rpd_get_xlu( void **data,
                  ipc_ *status,
                  ipc_ n,
                  rpc_ x_l[],
                  rpc_ x_u[] );

/*!<
 Recover the variable lower and upper bounds \f$x_l\f$ and \f$x_u\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

  @param[out] x_l is a one-dimensional array of size n and type rpc_, that
    gives the lower bounds \f$x_l\f$ on the variables \f$x\f$.
    The j-th component of x_l, j = 0, ... ,  n-1, contains  \f$(x_l)_j\f$.

  @param[out] x_u is a one-dimensional array of size n and type rpc_, that
    gives the upper bounds \f$x_u\f$ on the variables \f$x\f$.
    The j-th component of x_u, j = 0, ... ,  n-1, contains  \f$(x_u)_j\f$.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ C L U  -*-*-*-*-*-*-*-*-*-*

void rpd_get_clu( void **data,
                  ipc_ *status,
                  ipc_ m,
                  rpc_ c_l[],
                  rpc_ c_u[] );

/*!<
 Recover the constraint lower and upper bounds \f$c_l\f$ and \f$c_u\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general constraints.

 @param[out] c_l is a one-dimensional array of size m and type rpc_,
    that gives the lower bounds \f$c_l\f$ on the constraints \f$A x\f$.
    The i-th component of c_l, i = 0, ... ,  m-1, contains  \f$(c_l)_i\f$.

 @param[out] c_u is a one-dimensional array of size m and type rpc_,
    that gives the upper bounds \f$c_u\f$ on the constraints \f$A x\f$.
    The i-th component of c_u, i = 0, ... ,  m-1, contains  \f$(c_u)_i\f$.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ H  -*-*-*-*-*-*-*-*-*-*

void rpd_get_h( void **data,
                ipc_ *status,
                ipc_ h_ne,
                ipc_ h_row[],
                ipc_ h_col[],
                rpc_ h_val[]
                );

/*!<
 Recover the Hessian term \f$H\f$ in the objective function.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

  @param[in] h_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[out] h_row is a one-dimensional array of size h_ne and type ipc_, that
   gives the row indices of the lower triangular part of \f$H\f$ in the
   \ref symmetric_matrix_coordinate "sparse co-ordinate storage scheme".

 @param[out] h_col is a one-dimensional array of size h_ne and type ipc_,
   that gives the column indices of the lower triangular part of \f$H\f$ in
   the sparse co-ordinate storage scheme.

  @param[out] h_val is a one-dimensional array of size h_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H\f$ in the sparse co-ordinate storage scheme.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ A  -*-*-*-*-*-*-*-*-*-*

void rpd_get_a( void **data,
                ipc_ *status,
                ipc_ a_ne,
                ipc_ a_row[],
                ipc_ a_col[],
                rpc_ a_val[]
                );

/*!<
 Recover the Jacobian term \f$A\f$ in the constraints.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] a_ne is a scalar variable of type ipc_, that holds the number of
    entries in the constraint Jacobian matrix \f$A\f$.

 @param[out] a_row is a one-dimensional array of size a_ne and type ipc_, that
   gives the row indices of \f$A\f$ in the
  \ref unsymmetric_matrix_coordinate "sparse co-ordinate storage scheme".

 @param[out] a_col is a one-dimensional array of size a_ne and type ipc_,
   that gives the column indices of \f$A\f$ in the sparse co-ordinate,
   storage scheme.

 @param[out] a_val is a one-dimensional array of size a_ne and type rpc_,
    that gives the values of the entries of the constraint Jacobian matrix
    \f$A\f$ in the sparse co-ordinate scheme.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ H _ C  -*-*-*-*-*-*-*-*-*-*

void rpd_get_h_c( void **data,
                  ipc_ *status,
                  ipc_ h_c_ne,
                  ipc_ h_c_ptr[],
                  ipc_ h_c_row[],
                  ipc_ h_c_col[],
                  rpc_ h_c_val[]
                  );

/*!<
 Recover the Hessian terms \f$H_c\f$ in the constraints.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

  @param[in] h_c_ne is a scalar variable of type ipc_, that holds the number of
    entries in the lower triangular part of the Hessian matrix \f$H\f$.

 @param[out] h_c_ptr is a one-dimensional array of size h_c_ne and type ipc_,
   that gives the constraint indices of the lower triangular part of
   \f$H_c\f$ in the
  \ref joint_symmetric_matrix_coordinate "joint sparse co-ordinate storage scheme".

 @param[out] h_c_row is a one-dimensional array of size h_c_ne and type ipc_,
  that gives the row indices of the lower triangular part of \f$H_c\f$ in the
  joint sparse co-ordinate storage scheme.

 @param[out] h_c_col is a one-dimensional array of size h_c_ne and type ipc_,
   that gives the column indices of the lower triangular part of \f$H_c\f$ in
   the sparse co-ordinate storage scheme.

  @param[out] h_c_val is a one-dimensional array of size h_c_ne and type rpc_,
    that holds the values of the entries of the lower triangular part of the
    Hessian matrix \f$H_c\f$ in the sparse co-ordinate storage scheme.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ X _ T Y P E  -*-*-*-*-*-*-*-*-*-*

void rpd_get_x_type( void **data,
                     ipc_ *status,
                     ipc_ n,
                     ipc_ x_type[] );

/*!<
 Recover the types of the variables \f$x\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

  @param[out] x_type is a one-dimensional array of size n and type ipc_, that
    specifies the type of each variable \f$x\f$. Specifically,
    for j = 0, ... , n-1, x(j) =
    \li 0 variable \f$x_j\f$ is continuous,
    \li 1 variable \f$x_j\f$ is integer, and
    \li 2 variable \f$x_j\f$ is binary (0,1)
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ X  -*-*-*-*-*-*-*-*-*-*

void rpd_get_x( void **data,
                ipc_ *status,
                ipc_ n,
                rpc_ x[]
                );

/*!<
 Recover the initial values of the variables \f$x\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

  @param[out] x is a one-dimensional array of size n and type rpc_, that
    gives the initial values \f$x\f$ of the optimization variables. The
    j-th component of x, j = 0, ... , n-1, contains \f$x_j\f$.
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ Y  -*-*-*-*-*-*-*-*-*-*

void rpd_get_y( void **data,
                ipc_ *status,
                ipc_ m,
                rpc_ y[] );

/*!<
 Recover the initial values of the Lagrange multipliers \f$y\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] m is a scalar variable of type ipc_, that holds the number of
    general constraints.

 @param[out] y is a one-dimensional array of size n and type rpc_, that
    gives the initial values \f$y\f$ of the Lagrange multipliers for the
    general constraints. The j-th component of y, j = 0, ... , n-1,
    contains \f$y_j\f$.

*/

// *-*-*-*-*-*-*-*-*-*-*-*-    R P D _ G E T _ Z  -*-*-*-*-*-*-*-*-*-*

void rpd_get_z( void **data,
                ipc_ *status,
                ipc_ n,
                rpc_ z[] );

/*!<
 Recover the initial values of the dual variables \f$z\f$.

 @param[in,out] data holds private internal data

 @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are:
  \li  0. The statistics have been recovered succesfully.
  \li -93. The QPLIB file did not contain the required data.

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    variables.

 @param[out] z is a one-dimensional array of size n and type rpc_, that
    gives the initial values \f$z\f$ of the dual variables.
    The j-th component of z, j = 0, ... , n-1, contains \f$z_j\f$.

*/

// *-*-*-*-*-*-*-*-*-*-    R P D  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void rpd_information( void **data,
                      struct rpd_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see rpd_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    R P D  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void rpd_terminate( void **data,
                    struct rpd_control_type *control,
                    struct rpd_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see rpd_control_type)

  @param[out] inform   is a struct containing output information
              (see rpd_inform_type)
 */


/** \anchor examples
   \f$\label{examples}\f$
   \example rpdt.c
   This is an example of how to use the package to decode a QPLIB file.

   Notice that C-style indexing is used, and that this is flaggeed by
   setting \c control.f_indexing to \c false.

    \example rpdtf.c
   This is the same example, but now fortran-style indexing is used.\n

 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
