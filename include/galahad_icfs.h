//* \file galahad_icfs.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_ICFS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, GALAHAD productions
 *  Principal authors: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. Decemeber 6th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package icfs

  \section icfs_intro Introduction

  \subsection icfs_purpose Purpose

  Given a symmetric matrix \f$\bmA\f$, this package
  <b> computes a symmetric, positive-definite approximation
  \f$\bmL \bmL^T\f$ using an incomplete Cholesky factorization</b>;
  the resulting matrix \f$\bmL\f$ is lower triangular.
  Subsequently, the solution \f$\bmx\f$ to the either of the linear systems
  \f$\bmL \bmx = \bmb\f$ and \f$\bmL^T \bmx = \bmb\f$
  may be found for a given vector \f$\bmb\f$.

  \subsection icfs_authors Authors

  C.-J, Lin and J. J. Moré, Argonne National Laboratory,

  C interface, additionally N. I. M. Gould and J. Fowkes, STFC-Rutherford
  Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

 \subsection icfs_date Originally released

  May 1998, C interface December 2022.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_ICFS_H
#define GALAHAD_ICFS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct icfs_control_type {

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
    /// number of extra vectors of length n required by the incomplete
    /// Cholesky preconditioner
    ipc_ icfs_vectors;

    /// \brief
    /// an initial estimate of the shift $\alpha$ used so that the incomplete
    /// factorization of \f$A + D\f$ is positive definite, where \f$D\f$ is
    /// a diagonal matrix whose entries are no larger than \f$\alpha\f$.
    /// This value \f$\alpha\f$ may subsequently be increased, as necessary,
    /// by the package, see icfs_inform_type.shift.
    rpc_ shift;

    /// \brief
    /// if .space_critical true, every effort will be made to use as little
    /// space as possible. This may result in longer computation time
    bool space_critical;

    /// \brief
    /// if .deallocate_error_fatal is true, any array/pointer deallocation error
    /// will terminate execution. Otherwise, computation will continue
    bool deallocate_error_fatal;

    /// \brief
    /// all output lines will be prefixed by
    /// prefix(2:LEN(TRIM(.prefix))-1)
    /// where prefix contains the required string enclosed in quotes,
    /// e.g. "string" or 'string'
    char prefix[31];
};

/**
 * time derived type as a C struct
 */
struct icfs_time_type {

    /// \brief
    /// total time
    real_sp_ total;

    /// \brief
    /// time for the factorization phase
    real_sp_ factorize;

    /// \brief
    /// time for the triangular solution phase
    real_sp_ solve;

    /// \brief
    /// total clock time spent in the package
    rpc_ clock_total;

    /// \brief
    /// clock time for the factorization phase
    rpc_ clock_factorize;

    /// \brief
    /// clock time for the triangular solution phase
    rpc_ clock_solve;
};

/**
 * inform derived type as a C struct
 */
struct icfs_inform_type {

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

    /// name of array which provoked an allocate failure
    char bad_alloc[81];

    /// \brief
    /// the integer and real output arrays from mc61
    ipc_ mc61_info[10];
    /// see mc61_info
    rpc_ mc61_rinfo[15];

    /// \brief
    /// times for various stages
    struct icfs_time_type time;
};

// *-*-*-*-*-*-*-*-*-*-    I C F S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void icfs_initialize( void **data,
                      struct icfs_control_type *control,
                      ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see icfs_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-*-    I C F S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void icfs_read_specfile( struct icfs_control_type *control,
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNICFS.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/icfs.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see icfs_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-    I C F S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void icfs_reset_control( struct icfs_control_type *control,
                         void **data,
                         ipc_ *status );

/*!<
 Reset control parameters after import if required.

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see icfs_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful
*/

//  *-*-*-*-*-*- I C F S _ f o r m _ p r e c o n d i t i o n e r  -*-*-*-*-*-*-

void icfs_factorize_matrix( void **data,
                            ipc_ *status,
                            ipc_ n,
                            const ipc_ ptr[] );
                            const ipc_ row[],
                            const rpc_ diag[] );
                            const rpc_ val[] );

/*!<
 Form and factorize an incomplete factorization \f$P = L L^T\f$ of the
 matrix \f$A\f$.

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
  \li -3. The restriction n > 0 has been violated

 @param[in] n is a scalar variable of type ipc_, that holds the number of
    rows in the symmetric matrix \f$A\f$.

 @param[in]  ptr is a one-dimensional array of size n+1 and type ipc_,
   that holds the starting position of each column of the strictly
   lower triangular part of \f$A\f$ (i.e., \f$a_{i,j}\f$ for \f$i > j\f$),
   as well as the total number of entries. The entries for
   column i will occur in positions matrix_ptr(i),...,matrix_ptr(i+1)-1 of
   the arrays matrix_row and matrix_val.

 @param[in] row is a one-dimensional array of size matrix_ptr(n+1)-1
   and type ipc_, that holds the row indices of the strictly lower triangular
   part of \f$A\f$ as dictated by matrix_ptr.

 @param[in] diag is a one-dimensional array of size n and type rpc_, that
   holds the values of the diagonals of \f$A\f$., i.e., matrix_diag(i) =
   \f$a_{i,i}\f$.

 @param[in] val is a one-dimensional array of size matrix_ptr(n+1)-1 and type
    rpc_, that holds the values of the entries of  the strict lower
    triangular part of \f$A\f$ input in precisely the same  order as those
    for the row indices.
*/


//  *-*-*-*-*-*-*-*-*-  I C F S _ s o l v e _ s y s t e m  -*-*-*-*-*-*-*-*-

void icfs_solve_system( void **data,
                        ipc_ *status,
                        ipc_ n,
                        rpc_ sol[],
                        bool trans );

/*!<
 Solve the linear system \f$Lx=b\f$ or \f$L^T x=b\f$.

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
    its contains the solution \f$x\f$.

 @param[in] trans is a scalar variable of type bool, that specifies
    whether to solve the equation \f$L^Tx=b\f$ (trans=true) or
    \f$Lx=b\f$ (trans=false).
*/

// *-*-*-*-*-*-*-*-*-*-    I C F S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void icfs_information( void **data,
                      struct icfs_inform_type *inform,
                      ipc_ *status );

/*!<
  Provide output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see icfs_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    I C F S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void icfs_terminate( void **data,
                    struct icfs_control_type *control,
                    struct icfs_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information
              (see icfs_control_type)

  @param[out] inform   is a struct containing output information
              (see icfs_inform_type)
 */

/** \example icfst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
