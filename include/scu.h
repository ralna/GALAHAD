//* \file scu.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 17:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SCU C INTERFACE  *-*-*-*-*-*-*-*-*-*-
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

/*! \mainpage GALAHAD C package scu
 
  \section scu_intro Introduction

  \subsection scu_purpose Purpose

  Compute the the <b>solution to an extended system of \f$n + m\f$ 
  sparse real linear equations in \f$n + m\f$ unknowns,</b>
  \f[\mbox{(1)}\;\; \mat{cc}{ A & B \\ C & D } \vect{x_1 \\ x_2} =  \vect{b_1 \\ b_2}\f]
  \manonly
   \n
     (1)  ( A  B ) ( x_1 ) = ( b_1 )
          ( C  D ) ( x_2 )   ( b_2 )
   \n
  \endmanonly
  in the case where the \f$n\f$ by \f$n\f$ matrix \f$A\f$ is nonsingular 
  and solutions to the systems 
  \f[A x  =  b \;\mbox{and}\; A^T y  =  c\f]
  \manonly
   \n
     A x  =  b  and  A^T y  =  c
   \n
  \endmanonly
  may be obtained from an external source, such as an existing 
  factorization.  The subroutine uses reverse communication to obtain 
  the solution to such smaller systems.  The method makes use of 
  the Schur complement matrix 
  \f[S = D - C A^{-1} B.\f]
  \manonly
   \n
     S = D - C A^{-1} B.\f]
   \n
  \endmanonly
  The Schur complement is stored and factorized as a dense matrix 
  and the subroutine is thus appropriate only if there is 
  sufficient storage for this matrix. Special advantage is taken 
  of symmetry and definiteness in the coefficient matrices. 
  Provision is made for introducing additional rows and columns 
  to, and removing existing rows and columns from, the extended 
  matrix. 

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.
  
  \subsection scu_authors Authors
  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection scu_date Originally released

  March 2005, C interface January 2022.

  \subsection scu_method Method

  The subroutine galahad_factorize forms the Schur complement 
  \f$S = D - C A^{-1} B\f$ of \f$ A\f$ 
  in the extended matrix by repeated reverse communication to 
  obtain the columns of 
  \f$A^{-1} B\f$. The Schur complement or its negative is then factorized 
  into its QR or, if possible, Cholesky factors. 

  The subroutine galahad_solve solves the extended system using 
  the following well-known scheme: 
   -# Compute the solution to \f$ A u  =  b_1\f$; 
   -# Compute \f$x_2\f$ from \f$ S x_2  =  b_2  -  C u\f$; 
   -# Compute the solution to \f$ A v  =  B x_2\f$; and 
   -# Compute \f$x_1 = u - v\f$. 

  The subroutines galahad_append and galahad_delete compute the factorization 
  of the Schur complement after a row and column have been appended 
  to, and removed from, the extended matrix, respectively. 
  The existing factorization is updated 
  to obtain the new one; this is normally more efficient than 
  forming the factorization from scratch. 

 */

/* \subsection scu_call_order Call order */
/* To solve a given problem, functions from the scu package must be called  */
/* in the following order: */

/* - \link scu_initialize \endlink - provide default control parameters and */
/*     set up initial data structures */
/* - \link scu_read_specfile \endlink (optional) - override control values  */
/*     by reading replacement values from a file */
/* - \link scu_form_and_factorize \endlink - form and factorize the  */
/*    Schur-complement matrix \f$S\f$ */
/* - \link scu_solve_system \endlink - solve the block system (1) */
/* - \link scu_add_rows_and_cols \endlink (optional) - update the factors of */
/*      the Schur-complement matrix when rows and columns are added to (1). */
/* - \link scu_delete_rows_and_cols \endlink (optional) - update the factors of */
/*      the Schur-complement matrix when rows and columns are removed from (1). */
/* - \link scu_information \endlink (optional) - recover information about */
/*   the solution and solution process */
/* - \link scu_terminate \endlink - deallocate data structures */

/* \latexonly */
/* See Section~\ref{examples} for examples of use. */
/* \endlatexonly */
/* \htmlonly */
/* See the <a href="examples.html">examples tab</a> for illustrations of use. */
/* \endhtmlonly */
/* \manonly */
/* See the examples section for illustrations of use. */
/* \endmanonly */



#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_SCU_H 
#define GALAHAD_SCU_H

// precision
#include "galahad_precision.h"

/**
 * control derived type as a C struct
 */
struct sha_control_type {

    /// \brief
    /// use C or Fortran sparse matrix indexing
    bool f_indexing;
};

/**
 * inform derived type as a C struct
 */
struct scu_inform_type {

    /// \brief
    /// the return status from the last attempted internal workspace array 
    /// allocation or deallocation. A non-zero value indicates that the 
    /// allocation or deallocation was unsuccessful, and corresponds to the 
    /// fortran STAT= value on the user’s system.
    int alloc_status;

    /// \brief
    /// the inertia of \f$S\f$ when the extended matrix is symmetric. 
    /// Specifically, inertia(i), i=0,1,2 give the number of positive, 
    /// negative and zero eigenvalues of \f$S\f$ respectively.
    int inertia[3];
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
