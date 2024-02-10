//* \file galahad_bsc.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_BSC C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package bsc

  \section bsc_intro Introduction

  \subsection bsc_purpose Purpose

  Given matrices \f$A\f$ and (diagonal) \f$D\f$, <b>build the
  "Schur complement"
  \f$S  =  A D A^T\f$</b> in sparse co-ordinate (and optionally sparse column)
  format(s). Full advantage is taken of any zero coefficients in the matrix
  \f$A\f$.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection bsc_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection bsc_date Originally released

  October 2013, C interface January 2022.

 \subsection bsc_call_order Call order

  To solve a given problem, functions from the bsc package must be called
  in the following order:

  - \link bsc_initialize \endlink - provide default control parameters and
      set up initial data structures
  - \link bsc_read_specfile \endlink (optional) - override control values
      by reading replacement values from a file
  - \link bsc_import \endlink - set up matrix data structures for \f$A\f$.
  - \link bsc_reset_control \endlink (optional) - possibly change control
      parameters if a sequence of problems are being solved
  - \link bsc_form \endlink - form the Schur complement \f$S\f$
  - \link bsc_information \endlink (optional) - recover information about
    the process
  - \link bsc_terminate \endlink - deallocate data structures

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

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_BSC_H
#define GALAHAD_BSC_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct bsc_control_type {

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
    /// the level of output required is specified by print_level
    ipc_ print_level;

    /// \brief
    /// maximum permitted number of nonzeros in a column of \f$A\f$;
    /// -ve means unlimit
    ipc_ max_col;

    /// \brief
    /// how much has \f$A\f$ changed since it was last accessed:
    /// \li 0 = not changed,
    /// \li 1 = values changed,
    /// \li 2 = structure changed
    /// \li 3 = structure changed but values not required
    ipc_ new_a;

    /// \brief
    /// how much extra space is to be allocated in \f$S\f$ above that needed to
    /// hold the Schur complement
    ipc_ extra_space_s;

    /// \brief
    /// should s.ptr also be set to indicate the first entry in each column
    /// of \f$S\f$
    bool s_also_by_column;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
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
struct bsc_inform_type {

    /// \brief
    /// the return status from the package. Possible values are:
    /// \li 0. The call was succesful
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
    /// \li -3. The restrictions n > 0 or m > 0 or requirement that a type
    /// contains its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
    /// has been violated.

    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    ///  occurred.
    char bad_alloc[81];

    /// \brief
    /// the maximum number of entries in a column of \f$A\f$
    ipc_ max_col_a;

    /// \brief
    /// the number of columns of \f$A\f$ that have more than control.max_col
    /// entries
    ipc_ exceeds_max_col;

    /// \brief
    /// the total CPU time spent in the package
    rpc_ time;

    /// \brief
    /// the total clock time spent in the package
    rpc_ clock_time;
};

// *-*-*-*-*-*-*-*-*-    B S C  _ I N I T I A L I Z E    -*-*--*-*-*-*-

void bsc_initialize( void **data,
                     struct bsc_control_type *control,
                     ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see bsc_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The initialization was succesful.
*/

// *-*-*-*-*-*-*-*-*-   B S C  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-

void bsc_information( void **data,
                      struct bsc_inform_type *inform,
                      ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see bsc_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-    B S C  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-

void bsc_terminate( void **data,
                    struct bsc_control_type *control,
                    struct bsc_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see bsc_control_type)

  @param[out] inform is a struct containing output information
              (see bsc_inform_type)
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
