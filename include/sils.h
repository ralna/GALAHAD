//* \file sils.h */

/*
 * THIS VERSION: GALAHAD 3.3 - 29/11/2021 AT 13:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SILS C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 29th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

/*! \mainpage GALAHAD C package sils
 
  \section sils_intro Introduction

  \subsection sils_purpose Purpose

  \subsection sils_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection sils_date Originally released

  \subsection sils_terminology Terminology

  \subsection sils_method Method

  \subsection sils_references Reference

  \subsection sils_call_order Call order
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_SILS_H 
#define GALAHAD_SILS_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct sils_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;

    /// \brief
    /// MA27 internal integer controls
    int ICNTL[30];

    /// \brief
    /// Unit for error messages
    int lp;

    /// \brief
    /// Unit for warning messages
    int wp;

    /// \brief
    /// Unit for monitor output                                              NEW
    int mp;

    /// \brief
    /// Unit for statistical output                                          NEW
    int sp;

    /// \brief
    /// Controls level of diagnostic output
    int ldiag;

    /// \brief
    /// Initial size for real array for the factors. If less than nrlnec,
    /// default size used.
    int la;

    /// \brief
    /// Initial size for integer array for the factors. If less than nirnec,
    /// default size used.
    int liw;

    /// \brief
    /// Max. size for real array for the factors.
    int maxla;

    /// \brief
    /// Max. size for integer array for the factors.
    int maxliw;

    /// \brief
    /// Controls pivoting:
    /// 1  Numerical pivoting will be performed.
    /// 2  No pivoting will be performed and an error exit will occur
    /// immediately a pivot sign change is detected.
    /// 3  No pivoting will be performed and an error exit will occur if a
    /// zero pivot is detected.
    /// 4  No pivoting is performed but pivots are changed to all be positive.
    int pivoting;

    /// \brief
    /// Minimum number of eliminations in a step                          UNUSED
    int nemin;

    /// \brief
    /// Level 3 blocking in factorize                                     UNUSED
    int factorblocking;

    /// \brief
    /// Level 2 and 3 blocking in solve
    int solveblocking;

    /// \brief
    /// Controls threshold for detecting full rows in  analyse, registered as
    /// percentage of N, 100 Only fully dense rows detected (default)        NEW
    int thresh;

    /// \brief
    /// Controls ordering:                                                   NEW
    /// 0  AMD using MC47
    /// 1  User defined
    /// 2  AMD using MC50
    /// 3  Min deg as in MA57
    /// 4  Metis_nodend ordering
    /// 5  Ordering chosen depending on matrix characteristics.
    /// At the moment choices are MC50 or Metis_nodend
    /// >5  Presently equivalent to 5 but may chnage
    int ordering;

    /// \brief
    /// Controls scaling:                                                    NEW
    /// 0  No scaling
    /// >0  Scaling using MC64 but may change for > 1
    int scaling;

    /// \brief
    /// MA27 internal real controls
    real_wp_ CNTL[5];

    /// \brief
    /// Factor by which arrays sizes are to be increased if they are too small
    real_wp_ multiplier;

    /// \brief
    /// If previously allocated internal workspace arrays are greater than 
    /// reduce times the currently required sizes, they are reset to current 
    /// requirment
    real_wp_ reduce;

    /// \brief
    /// Pivot threshold
    real_wp_ u;

    /// \brief
    /// used for setting static pivot level                                  NEW
    real_wp_ static_tolerance;

    /// \brief
    /// used for switch to static                                            NEW
    real_wp_ static_level;

    /// \brief
    /// Anything less than this is considered zero
    real_wp_ tolerance;

    /// \brief
    /// used to monitor convergence in iterative refinement
    real_wp_ convergence;
};

/**
 * ainfo derived type as a C struct
 */
struct sils_ainfo_type {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// More information on failure                                          NEW
    int more;

    /// \brief
    /// Number of elimination steps
    int nsteps;

    /// \brief
    /// Size for a without compression
    int nrltot;

    /// \brief
    /// Size for iw without compression
    int nirtot;

    /// \brief
    /// Size for a with compression
    int nrlnec;

    /// \brief
    /// Size for iw with compression
    int nirnec;

    /// \brief
    /// Number of reals to hold factors
    int nrladu;

    /// \brief
    /// Number of integers to hold factors
    int niradu;

    /// \brief
    /// Number of compresses
    int ncmpa;

    /// \brief
    /// Number of indices out-of-range                                       NEW
    int oor;

    /// \brief
    /// Number of duplicates                                                 NEW
    int dup;

    /// \brief
    /// Forecast maximum front size                                          NEW
    int maxfrt;

    /// \brief
    /// STAT value after allocate failure                                    NEW
    int stat;

    /// \brief
    /// OLD
    int faulty;

    /// \brief
    /// Anticipated # ops. in assembly                                       NEW
    real_wp_ opsa;

    /// \brief
    /// Anticipated # ops. in elimin.                                        NEW
    real_wp_ opse;
};

/**
 * finfo derived type as a C struct
 */
struct sils_finfo_type {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// More information on failure                                          NEW
    int more;

    /// \brief
    /// Largest front size
    int maxfrt;

    /// \brief
    /// Number of entries in factors                                         NEW
    int nebdu;

    /// \brief
    /// Number of reals that hold factors
    int nrlbdu;

    /// \brief
    /// Number of integers that hold factors
    int nirbdu;

    /// \brief
    /// Size for a without compression
    int nrltot;

    /// \brief
    /// Size for iw without compression
    int nirtot;

    /// \brief
    /// Size for a with compression
    int nrlnec;

    /// \brief
    /// Size for iw with compression
    int nirnec;

    /// \brief
    /// Number of compresses of real data
    int ncmpbr;

    /// \brief
    /// Number of compresses of integer data
    int ncmpbi;

    /// \brief
    /// Number of 2x2 pivots
    int ntwo;

    /// \brief
    /// Number of negative eigenvalues
    int neig;

    /// \brief
    /// Number of delayed pivots (total)                                     NEW
    int delay;

    /// \brief
    /// Number of pivot sign changes (pivoting=3 )                           NEW
    int signc;

    /// \brief
    /// Number of static pivots chosen
    int nstatic;

    /// \brief
    /// First pivot modification when pivoting=4                             NEW
    int modstep;

    /// \brief
    /// Rank of original factorization
    int rank;

    /// \brief
    /// STAT value after allocate failure
    int stat;

    /// \brief
    /// OLD
    int faulty;

    /// \brief
    /// OLD
    int step;

    /// \brief
    /// # operations in assembly                                             NEW
    real_wp_ opsa;

    /// \brief
    /// # operations in elimination                                          NEW
    real_wp_ opse;

    /// \brief
    /// Additional # ops. for BLAS                                           NEW
    real_wp_ opsb;

    /// \brief
    /// Largest pivoting=4 mod.                                              NEW
    real_wp_ maxchange;

    /// \brief
    /// Minimum scaling factor
    real_wp_ smin;

    /// \brief
    /// Maximum scaling factor
    real_wp_ smax;
};

/**
 * sinfo derived type as a C struct
 */
struct sils_sinfo_type {

    /// \brief
    /// Flags success or failure case
    int flag;

    /// \brief
    /// STAT value after allocate failure
    int stat;

    /// \brief
    /// Cond # of matrix (cat 1 eqs)
    real_wp_ cond;

    /// \brief
    /// Cond # of matrix (cat 2 eqs)
    real_wp_ cond2;

    /// \brief
    /// Cond # of matrix (cat 1 eqs)
    real_wp_ berr;

    /// \brief
    /// Cond # of matrix (cat 2 eqs)
    real_wp_ berr2;

    /// \brief
    /// Estimate of forward error
    real_wp_ error;
};

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*

void sils_initialize( void **data, 
                      struct sils_control_type *control,
                     int *status );

/*!<
 Set default control values and initialize private data

  @param[in,out] data  holds private internal data
  @param[out] control  is a struct containing control information 
              (see sils_control_type)
  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-    S I L S  _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*

void sils_read_specfile( struct sils_control_type *control, 
                         const char specfile[] );

/*!<
  Read the content of a specification file, and assign values associated 
  with given keywords to the corresponding control parameters

  @param[in,out]  control  is a struct containing control information 
              (see sils_control_type)
  @param[in]  specfile  is a character string containing the name of
              the specification file
*/

// *-*-*-*-*-*-*-*-*-*-*-*-    S I L S  _ I M P O R T   -*-*-*-*-*-*-*-*-*-*

void sils_import( struct sils_control_type *control,
                  void **data,
                  int *status );

/*!<
 Import problem data into internal storage prior to solution. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sils_control_type)

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

// *-*-*-*-*-*-*-    S I L S  _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*

void sils_reset_control( struct sils_control_type *control,
                         void **data,
                         int *status );

/*!<
 Reset control parameters after import if required. 

 @param[in] control is a struct whose members provide control
  paramters for the remaining prcedures (see sils_control_type)

 @param[in,out] data holds private internal data

 @param[in,out] status is a scalar variable of type int, that gives
    the exit status from the package. Possible values are:
  \li  1. The import was succesful, and the package is ready for the solve phase
*/

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*

void sils_information( void **data,
                       struct sils_ainfo_type *ainfo,
                       struct sils_finfo_type *finfo,
                       struct sils_sinfo_type *sinfo,
                       int *status );

/*!<
  Provides output information

  @param[in,out] data holds private internal data

  @param[out] ainfo is a struct containing output information
              (see sils_ainfo_type) 

  @param[out] finfo is a struct containing output information
              (see sils_finfo_type) 

  @param[out] sinfo is a struct containing output information
              (see sils_sinfo_type) 

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
*/

// *-*-*-*-*-*-*-*-*-*-    S I L S  _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*

void sils_terminate( void **data, 
                     struct sils_control_type *control, 
                     int *status );

/*!<
  Deallocate all internal private storage

  @param[in,out] data  holds private internal data

  @param[out] control  is a struct containing control information 
              (see sils_control_type)

  @param[out] status is a scalar variable of type int, that gives
              the exit status from the package.
              Possible values are (currently):
  \li  0. The values were recorded succesfully
  \li \f$\neq\f$ 0. The Fortran STAT value of an allocate or 
          deallocate statement that has failed.
 */

/** \example silst.c
   This is an example of how to use the package.\n
 */

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
