//* \file galahad_nodend.h */

/*
 * THIS VERSION: GALAHAD 5.2 - 2025-03-23 AT 13:50 GMT
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NODEND C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.2. March 14th 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package nodend

  \nodendtion nodend_intro Introduction

  \subnodendtion nodend_purpose Purpose

  Find a symmetric row and column permutation P A P'
  of a symmetric, sparse matrix A with the aim of limiting 
  the fill-in during subsequent Cholesky-like factorization. 
  The package is actually a wrapper to the METIS_NodeND
  procedure from versions 4.0, 5.1 and 5.2 of the
  METIS package from the Karypis Lab.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subnodendtion nodend_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subnodendtion nodend_date Originally released

  May 2008, C interface March 2025

 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_NODEND_H
#define GALAHAD_NODEND_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

/**
 * control derived type as a C struct
 */
struct nodend_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// specify the version of METIS allowed. Possible values are
    /// '4.0', '5.1' or '5.2', where the latter is the defalt.
    char version[31];

    /// \brief
    /// error and warning diagnostics occur on stream error.
    ipc_ error;

    /// \brief
    /// general output occurs on stream out.
    ipc_ out;

    /// \brief
    /// the level of output required. <= 0 gives no output, >= 1 warning message
    ipc_ print_level;

    /// \brief
    /// should the method revert to METIS 5.2 if METIS 4.0 is requested but
    /// unavailable?
    bool no_metis_4_use_5_instead;

    /// \brief
    /// all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1)
    /// where .prefix contains the required string enclosed in
    /// quotes, e.g. "string" or 'string'
    char prefix[31];

    ipc_ metis4_ptype;
    ipc_ metis4_ctype;
    ipc_ metis4_itype;
    ipc_ metis4_rtype;
    ipc_ metis4_dbglvl;
    ipc_ metis4_oflags;
    ipc_ metis4_pfactor;
    ipc_ metis4_nseps;
    ipc_ metis5_ptype;
    ipc_ metis5_objtype;
    ipc_ metis5_ctype;
    ipc_ metis5_iptype;
    ipc_ metis5_rtype;
    ipc_ metis5_dbglvl;
    ipc_ metis5_niter;
    ipc_ metis5_ncuts;
    ipc_ metis5_seed;
    ipc_ metis5_no2hop;
    ipc_ metis5_minconn;
    ipc_ metis5_contig;
    ipc_ metis5_compress;
    ipc_ metis5_ccorder;
    ipc_ metis5_pfactor;
    ipc_ metis5_nseps;
    ipc_ metis5_ufactor;
    ipc_ metis5_niparts;
    ipc_ metis5_ondisk;
    ipc_ metis5_dropedges;
    ipc_ metis5_twohop;
    ipc_ metis5_fast;
};

/**
 * inform derived type as a C struct
 */
struct nodend_inform_type {

    /// \brief
    /// return status. Possible valuesa are:
    /// \li 0 successful return
    /// \li -1 An allocation error occurred.
    /// \li -2 A deallocation error occurred.
    /// \li -3 An input value has an illegal value.
    /// \li -26  The requested version of METIS is not available.
    ipc_ status;

    /// \brief
    /// the status of the last attempted allocation/deallocation.
    ipc_ alloc_status;

    /// \brief
    /// the name of the array for which an allocation/deallocation error
    /// occurred.
    char bad_alloc[81];

    /// \brief
    /// the actual version of METIS used.
    char version[4];

};

// *-*-*-*-*-*-*-*-*-    N O D E N D  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-

void nodend_initialize( void **data,
                        struct nodend_control_type *control,
                        ipc_ *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information
              (see nodend_control_type)

  @param[out] status is a scalar variable of type ipc_, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-    N O D E N D  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-

void nodend_read_specfile( struct nodend_control_type *control,
                           const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated
  with given keywords to the corresponding control parameters.
  By default, the spcification file will be named RUNNODEND.SPC and
  lie in the current directory.
  Refer to Table 2.1 in the fortran documentation provided in
  $GALAHAD/doc/nodend.pdf for a list of keywords that may be set.

  @param[in,out]  control is a struct containing control information
              (see nodend_control_type)

  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-    N O D E N D  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-

void nodend_information( void **data,
                         struct nodend_inform_type *inform,
                         ipc_ *status );

/*!<
  Provides output information

  @param[in,out] data  holds private internal data

  @param[out] inform   is a struct containing output information
              (see nodend_inform_type)

  @param[out] status is a scalar variable of type ipc_, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
