/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:30 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 18 May 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA77 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA77 for full terms
 * and conditions. STFC may be contacted via hsl(at)stfc.ac.uk.
 *
 * Modified BSD licence (this header file only):
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of STFC nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL STFC BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef HSL_MA77D_H
#define HSL_MA77D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma77_default_control
#ifdef REAL_32
#ifdef INTEGER_64
#define ma77_control ma77_control_s_64
#define ma77_info ma77_info_s_64
#define ma77_default_control ma77_default_control_s_64
#define ma77_open_nelt ma77_open_nelt_s_64
#define ma77_open ma77_open_s_64
#define ma77_input_vars ma77_input_vars_s_64
#define ma77_input_reals ma77_input_reals_s_64
#define ma77_analyse ma77_analyse_s_64
#define ma77_factor ma77_factor_s_64
#define ma77_factor_solve ma77_factor_solve_s_64
#define ma77_solve ma77_solve_s_64
#define ma77_resid ma77_resid_s_64
#define ma77_scale ma77_scale_s_64
#define ma77_enquire_posdef ma77_enquire_posdef_s_64
#define ma77_enquire_indef ma77_enquire_indef_s_64
#define ma77_alter ma77_alter_s_64
#define ma77_restart ma77_restart_s_64
#define ma77_finalise ma77_finalise_s_64
#define ma77_solve_fredholm ma77_solve_fredholm_s_64
#define ma77_lmultiply ma77_lmultiply_s_64
#else
#define ma77_control ma77_control_s
#define ma77_info ma77_info_s
#define ma77_default_control ma77_default_control_s
#define ma77_open_nelt ma77_open_nelt_s
#define ma77_open ma77_open_s
#define ma77_input_vars ma77_input_vars_s
#define ma77_input_reals ma77_input_reals_s
#define ma77_analyse ma77_analyse_s
#define ma77_factor ma77_factor_s
#define ma77_factor_solve ma77_factor_solve_s
#define ma77_solve ma77_solve_s
#define ma77_resid ma77_resid_s
#define ma77_scale ma77_scale_s
#define ma77_enquire_posdef ma77_enquire_posdef_s
#define ma77_enquire_indef ma77_enquire_indef_s
#define ma77_alter ma77_alter_s
#define ma77_restart ma77_restart_s
#define ma77_finalise ma77_finalise_s
#define ma77_solve_fredholm ma77_solve_fredholm_s
#define ma77_lmultiply ma77_lmultiply_s
#endif
#else
#ifdef INTEGER_64
#define ma77_control ma77_control_d_64
#define ma77_info ma77_info_d_64
#define ma77_default_control ma77_default_control_d_64
#define ma77_open_nelt ma77_open_nelt_d_64
#define ma77_open ma77_open_d_64
#define ma77_input_vars ma77_input_vars_d_64
#define ma77_input_reals ma77_input_reals_d_64
#define ma77_analyse ma77_analyse_d_64
#define ma77_factor ma77_factor_d_64
#define ma77_factor_solve ma77_factor_solve_d_64
#define ma77_solve ma77_solve_d_64
#define ma77_resid ma77_resid_d_64
#define ma77_scale ma77_scale_d_64
#define ma77_enquire_posdef ma77_enquire_posdef_d_64
#define ma77_enquire_indef ma77_enquire_indef_d_64
#define ma77_alter ma77_alter_d_64
#define ma77_restart ma77_restart_d_64
#define ma77_finalise ma77_finalise_d_64
#define ma77_solve_fredholm ma77_solve_fredholm_d_64
#define ma77_lmultiply ma77_lmultiply_d_64
#else
#define ma77_control ma77_control_d
#define ma77_info ma77_info_d
#define ma77_default_control ma77_default_control_d
#define ma77_open_nelt ma77_open_nelt_d
#define ma77_open ma77_open_d
#define ma77_input_vars ma77_input_vars_d
#define ma77_input_reals ma77_input_reals_d
#define ma77_analyse ma77_analyse_d
#define ma77_factor ma77_factor_d
#define ma77_factor_solve ma77_factor_solve_d
#define ma77_solve ma77_solve_d
#define ma77_resid ma77_resid_d
#define ma77_scale ma77_scale_d
#define ma77_enquire_posdef ma77_enquire_posdef_d
#define ma77_enquire_indef ma77_enquire_indef_d
#define ma77_alter ma77_alter_d
#define ma77_restart ma77_restart_d
#define ma77_finalise ma77_finalise_d
#define ma77_solve_fredholm ma77_solve_fredholm_d
#define ma77_lmultiply ma77_lmultiply_d
#endif
#endif
#endif

/* Data type for user controls */
struct ma77_control {
   /* Note: 0 is false, non-zero is true */

   /* C/Fortran interface related controls */
   ipc_ f_arrays; /* Treat arrays as 1-based (Fortran) if true or 0-based (C) if
                    false. */

   /* Printing controls */
   ipc_ print_level;
   ipc_ unit_diagnostics;   /* unit for diagnostic messages
                              Printing is suppressed if unit_diagnostics < 0. */
   ipc_ unit_error;         /* unit for error messages
                              Printing is suppressed if unit_error  <  0.     */
   ipc_ unit_warning;       /* unit for warning messages
                              Printing is suppressed if unit_warning  <  0.   */

   /* Controls used by MA77_open */
   ipc_ bits;
   ipc_ buffer_lpage[2];
   ipc_ buffer_npage[2];
   hsl_longc_ file_size;
   hsl_longc_ maxstore;
   hsl_longc_ storage[3];

   /* Controls used by MA77_analyse */
   ipc_ nemin;  /* Node amalgamation parameter. A child node is merged with its
                  parent if they both involve fewer than nemin eliminations.*/

   /* Controls used by MA77_scale */
   ipc_ maxit;
   ipc_ infnorm;
   rpc_ thresh;

   /* Controls used by MA77_factor with posdef true */
   ipc_ nb54;

   /* Controls used by MA77_factor with posdef false */
   ipc_ action;    /* Keep going even if matrix is singular if true, or abort
                     if false */
   rpc_ multiplier;
   ipc_ nb64;
   ipc_ nbi;
   rpc_ small;
   rpc_ static_;
   hsl_longc_ storage_indef;
   rpc_ u;       /* Pivot tolerance*/
   rpc_ umin;    /* Minimum pivot tolerance*/

   /* Controls used by ma77_solve_fredholm */
   rpc_ consist_tol;   /* Tolerance for consistent singular system */

   /* Pad data structure to allow for future growth */
   ipc_ ispare[5]; hsl_longc_ lspare[5]; rpc_ rspare[5];
};

/***************************************************/

/* data type for returning information to user.*/
struct ma77_info {
   rpc_ detlog;
   ipc_ detsign;
   ipc_ flag;
   ipc_ iostat;
   ipc_ matrix_dup;
   ipc_ matrix_rank;
   ipc_ matrix_outrange;
   ipc_ maxdepth;
   ipc_ maxfront;
   hsl_longc_ minstore;
   ipc_ ndelay;
   hsl_longc_ nfactor;
   hsl_longc_ nflops;
   ipc_ niter;
   ipc_ nsup;
   ipc_ num_neg;
   ipc_ num_nothresh;
   ipc_ num_perturbed;
   ipc_ ntwo;
   ipc_ stat;
   ipc_ index[4];
   hsl_longc_ nio_read[2];
   hsl_longc_ nio_write[2];
   hsl_longc_ nwd_read[2];
   hsl_longc_ nwd_write[2];
   ipc_ num_file[4];
   hsl_longc_ storage[4];
   ipc_ tree_nodes;
   ipc_ unit_restart;
   ipc_ unused;
   rpc_ usmall;

   /* Pad data structure to allow for future growth */
   ipc_ ispare[5]; hsl_longc_ lspare[5]; rpc_ rspare[5];
};

/* Initialise control with default values */
void ma77_default_control(struct ma77_control *control);
void ma77_open_nelt(const ipc_ n, const char* fname1, const char* fname2,
   const char *fname3, const char *fname4, void **keep,
   const struct ma77_control *control, struct ma77_info *info,
   const ipc_ nelt);
void ma77_open(const ipc_ n, const char* fname1, const char* fname2,
   const char *fname3, const char *fname4, void **keep,
   const struct ma77_control *control, struct ma77_info *info);
void ma77_input_vars(const ipc_ idx, const ipc_ nvar, const ipc_ list[],
   void **keep, const struct ma77_control *control, struct ma77_info *info);
void ma77_input_reals(const ipc_ idx, const ipc_ length,
   const rpc_ reals[], void **keep, const struct ma77_control *control,
   struct ma77_info *info);
/* Analyse the sparsity pattern and prepare for factorization */
void ma77_analyse(const ipc_ order[], void **keep,
   const struct ma77_control *control, struct ma77_info *info);
/* To factorize the matrix */
void ma77_factor(const ipc_ posdef, void **keep, 
   const struct ma77_control *control, struct ma77_info *info,
   const rpc_ *scale);
/* To factorize the matrix AND solve AX = B */
void ma77_factor_solve(const ipc_ posdef, void **keep, 
   const struct ma77_control *control, struct ma77_info *info,
   const rpc_ *scale, const ipc_ nrhs, const ipc_ lx,
   rpc_ rhs[]);
/* To solve AX = B using the computed factors */
void ma77_solve(const ipc_ job, const ipc_ nrhs, const ipc_ lx, rpc_ x[],
   void **keep, const struct ma77_control *control, struct ma77_info *info,
   const rpc_ *scale);
void ma77_resid(const ipc_ nrhs, const ipc_ lx, const rpc_ x[],
   const ipc_ lresid, rpc_ resid[], void **keep, 
   const struct ma77_control *control, struct ma77_info *info,
   rpc_ *anorm_bnd);
void ma77_scale(rpc_ scale[], void **keep, 
   const struct ma77_control *control, struct ma77_info *info,
   rpc_ *anorm);
void ma77_enquire_posdef(rpc_ d[], void **keep, 
   const struct ma77_control *control, struct ma77_info *info);
void ma77_enquire_indef(int piv_order[], rpc_ d[], void **keep, 
   const struct ma77_control *control, struct ma77_info *info);
void ma77_alter(const rpc_ d[], void **keep, 
   const struct ma77_control *control, struct ma77_info *info);
void ma77_restart(const char *restart_file, const char *fname1, 
   const char *fname2, const char *fname3, const char *fname4, void **keep, 
   const struct ma77_control *control, struct ma77_info *info);
void ma77_solve_fredholm(int nrhs, ipc_ flag_out[], ipc_ lx, rpc_ x[],
   void **keep, const struct ma77_control *control,
   struct ma77_info *info, const rpc_ *scale);
void ma77_lmultiply(int trans, ipc_ k, ipc_ lx, rpc_ x[], ipc_ ly,
   rpc_ y[], void **keep, const struct ma77_control *control,
   struct ma77_info *info, const rpc_ *scale);
/* To clean up memory in keep */
void ma77_finalise(void **keep, const struct ma77_control *control,
   struct ma77_info *info);

#endif
