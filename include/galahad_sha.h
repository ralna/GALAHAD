//* \file galahad_sha.h */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-01-12 AT 15:20 GMT.
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

  Find an approximation to a sparse Hessian using componentwise secant
  approximation.

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.

  \subsection sha_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  Julia interface, additionally A. Montoison and D. Orban, Polytechnique Montréal.

  \subsection sha_date Originally released

  April 2013, C interface January 2022.

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
    int error;

    /// \brief
    /// general output occurs on stream out
    int out;

    /// \brief
    /// the level of output required. <= 0 gives no output, = 1 gives a one-line
    /// summary for every iteration, = 2 gives a summary of the inner iteration
    /// for each iteration, >= 3 gives increasingly verbose (debugging) output
    int print_level;

    /// \brief
    /// which approximation algorithm should be used?
    /// \li 0 : unsymmetric (alg 2.1 in paper)
    /// \li 1 : symmetric (alg 2.2 in paper)
    /// \li 2 : composite (alg 2.3 in paper)
    /// \li 3 : composite 2 (alg 2.2/3 in paper)
    int approximation_algorithm;

    /// \brief
    /// which dense linear equation solver should be used?
    /// \li 1 : Gaussian elimination
    /// \li 2 : QR factorization
    /// \li 3 : singular-value decomposition
    /// \li 4 : singular-value decomposition with divide-and-conquer
    int dense_linear_solver;

    /// \brief
    /// the maximum sparse degree if the combined version is used
    int max_sparse_degree;

    /// \brief
    /// if available use an addition extra_differences differences
    int extra_differences;

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
    int status;

    /// \brief
    /// the status of the last attempted allocation/deallocation
    int alloc_status;

    /// \brief
    /// the maximum degree in the adgacency graph
    int max_degree;

    /// \brief
    /// the number of differences that will be needed
    int differences_needed;

    /// \brief
    /// the maximum reduced degree in the adgacency graph
    int max_reduced_degree;

    /// \brief
    /// the name of the array for which an allocation/deallocation error ocurred
    char bad_alloc[81];
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
