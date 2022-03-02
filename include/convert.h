//* \file convert.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-02-25 AT 07:13 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_CONVERT C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.0. February 25th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package convert
 
  \section convert_intro Introduction

  \subsection convert_purpose Purpose
  Given a real matrix \f$A\f$ stored in one format, convert it to another 

  \subsection convert_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection convert_date Originally released

  \subsection convert_terminology Terminology

  \subsection convert_method Method

  \subsection convert_references Reference

  \subsection convert_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_CONVERT_H 
#define GALAHAD_CONVERT_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct convert_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// unit for error messages
    int error;

    /// \brief
    /// unit for monitor output
    int out;

    /// \brief
    /// controls level of diagnostic output
    int print_level;

    /// \brief
    /// obtain the transpose of the input matrix?
    bool transpose;

    /// \brief
    /// add the values of entries in duplicate positions?
    bool sum_duplicates;

    /// \brief
    /// order row or column data by increasing index?
    bool order;

    /// \brief
    /// if space is critical, ensure allocated arrays are no bigger than needed
    bool space_critical;

    /// \brief
    /// exit if any deallocation fails
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
struct convert_time_type {

    /// \brief
    /// total cpu time spent in the package
    real_wp_ total;

    /// \brief
    /// total clock time spent in the package
    real_wp_ clock_total;
};

/**
 * inform derived type as a C struct
 */
struct convert_inform_type {

    /// \brief
    /// return status. See CONVERT_between_matrix_formats (etc) for details
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the number of duplicates found (-ve = not checked)
    int duplicates;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];

    /// \brief
    /// timings (see above)
    struct convert_time_type time;
};

// *-*-*-*-*-*-*-*-*-*-    C O N V E R T  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void convert_initialize( void **data, 
                     struct convert_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data holds private internal data

  @param[out] control is a struct containing control information 
              (see convert_control_type)

  @param[out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are (currently):
  \li  0. The import was succesful.
*/

// *-*-*-*-*-*-*-*-    C O N V E R T  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*

void convert_read_specfile( struct convert_control_type *control, 
                        const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control is a struct containing control information 
              (see convert_control_type)
  @param[in]  specfile is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    C O N V E R T  _ I M P O R T   -*-*-*-*-*-*-*-*-*

void convert_import( struct convert_control_type *control,
                 void **data,
                 int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see convert_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
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

// *-*-*-*-*-*-*-    C O N V E R T  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void convert_reset_control( struct convert_control_type *control,
                 void **data,
                 int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see convert_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-    C O N V E R T  _ I N F O R M A T I O N   -*-*-*-*-*-*-*

void convert_information( void **data,
                      struct convert_inform_type *inform,
                      int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] inform is a struct containing output information
              (see convert_inform_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-    C O N V E R T  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*

void convert_terminate( void **data, 
                    struct convert_control_type *control, 
                    struct convert_inform_type *inform );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see convert_control_type)

  @param[out] inform   is a struct containing output information
              (see convert_inform_type)
 */

/** \example convertt.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
