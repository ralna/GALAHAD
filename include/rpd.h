//* \file rpd.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-28 AT 17:00 GMT.
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
  \f[\mbox{minimize}\;\; \frac{1}{2} x^T H x g^T x + f 
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

  Currently, only the control and inform parameters are exposed;
  these are provided and used by other GALAHAD packages with C interfaces.
  
  \subsection rpd_authors Authors

  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.

  C interface, additionally J. Fowkes, STFC-Rutherford Appleton Laboratory.

  \subsection rpd_date Originally released

  January 2006, C interface January 2022.

  \subsection rpd_references Reference

  The QPBLIB format is defined in

  F. Furini, E. Traversi, P. Belotti, A. Frangioni, A. Gleixner, N. Gould, 
  L. Liberti, A. Lodi, R. Misener, H. Mittelmann, N. V. Sahinidis, 
  S. Vigerske and A. Wiegele  (2019).
  QPLIB: a library of quadratic programming instances,
  Mathematical Programming Computation <b>11</b> 237–265.
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef GALAHAD_RPD_H 
#define GALAHAD_RPD_H

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
struct rpd_inform_type {
    /// \brief
    ///  return status. Possible values are:
    /// \li  0  successful return
    /// \li -2  allocation failure 
    /// \li -3  end of file reached prematurely
    /// \li -4 other read error
    /// \li -5 unrecognised type
    int status;

    /// \brief
    /// the status of the last attempted allocation or deallocation
    int alloc_status;

    /// \brief
    /// the name of the array for which an allocation or deallocation 
    /// error ocurred
    char bad_alloc[11];

    /// \brief
    /// status from last read attempt
    int io_status;

    /// \brief
    /// number of last line read from i/o file
    int line;

    /// \brief
    /// problem type
    char p_type[4];
};

// end include guard
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
