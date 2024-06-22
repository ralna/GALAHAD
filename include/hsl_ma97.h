/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:45 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 20 September 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * Version 2.8.1
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA97 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA97 for full terms
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

#ifndef HSL_MA97D_H
#define HSL_MA97D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma97_default_control
#ifdef REAL_32
#ifdef INTEGER_64
#define ma97_control ma97_control_s_64
#define ma97_info ma97_info_s_64
#define ma97_default_control ma97_default_control_s_64
#define ma97_analyse ma97_analyse_s_64
#define ma97_analyse_coord ma97_analyse_coord_s_64
#define ma97_factor ma97_factor_s_64
#define ma97_factor_solve ma97_factor_solve_s_64
#define ma97_solve ma97_solve_s_64
#define ma97_free_akeep ma97_free_akeep_s_64
#define ma97_free_fkeep ma97_free_fkeep_s_64
#define ma97_finalise ma97_finalise_s_64
#define ma97_enquire_posdef ma97_enquire_posdef_s_64
#define ma97_enquire_indef ma97_enquire_indef_s_64
#define ma97_alter ma97_alter_s_64
#define ma97_solve_fredholm ma97_solve_fredholm_s_64
#define ma97_lmultiply ma97_lmultiply_s_64
#define ma97_sparse_fwd_solve ma97_sparse_fwd_solve_s_64
#else
#define ma97_control ma97_control_s
#define ma97_info ma97_info_s
#define ma97_default_control ma97_default_control_s
#define ma97_analyse ma97_analyse_s
#define ma97_analyse_coord ma97_analyse_coord_s
#define ma97_factor ma97_factor_s
#define ma97_factor_solve ma97_factor_solve_s
#define ma97_solve ma97_solve_s
#define ma97_free_akeep ma97_free_akeep_s
#define ma97_free_fkeep ma97_free_fkeep_s
#define ma97_finalise ma97_finalise_s
#define ma97_enquire_posdef ma97_enquire_posdef_s
#define ma97_enquire_indef ma97_enquire_indef_s
#define ma97_alter ma97_alter_s
#define ma97_solve_fredholm ma97_solve_fredholm_s
#define ma97_lmultiply ma97_lmultiply_s
#define ma97_sparse_fwd_solve ma97_sparse_fwd_solve_s
#endif
#else
#ifdef INTEGER_64
#define ma97_control ma97_control_d_64
#define ma97_info ma97_info_d_64
#define ma97_default_control ma97_default_control_d_64
#define ma97_analyse ma97_analyse_d_64
#define ma97_analyse_coord ma97_analyse_coord_d_64
#define ma97_factor ma97_factor_d_64
#define ma97_factor_solve ma97_factor_solve_d_64
#define ma97_solve ma97_solve_d_64
#define ma97_free_akeep ma97_free_akeep_d_64
#define ma97_free_fkeep ma97_free_fkeep_d_64
#define ma97_finalise ma97_finalise_d_64
#define ma97_enquire_posdef ma97_enquire_posdef_d_64
#define ma97_enquire_indef ma97_enquire_indef_d_64
#define ma97_alter ma97_alter_d_64
#define ma97_solve_fredholm ma97_solve_fredholm_d_64
#define ma97_lmultiply ma97_lmultiply_d_64
#define ma97_sparse_fwd_solve ma97_sparse_fwd_solve_d_64
#else
#define ma97_control ma97_control_d
#define ma97_info ma97_info_d
#define ma97_default_control ma97_default_control_d
#define ma97_analyse ma97_analyse_d
#define ma97_analyse_coord ma97_analyse_coord_d
#define ma97_factor ma97_factor_d
#define ma97_factor_solve ma97_factor_solve_d
#define ma97_solve ma97_solve_d
#define ma97_free_akeep ma97_free_akeep_d
#define ma97_free_fkeep ma97_free_fkeep_d
#define ma97_finalise ma97_finalise_d
#define ma97_enquire_posdef ma97_enquire_posdef_d
#define ma97_enquire_indef ma97_enquire_indef_d
#define ma97_alter ma97_alter_d
#define ma97_solve_fredholm ma97_solve_fredholm_d
#define ma97_lmultiply ma97_lmultiply_d
#define ma97_sparse_fwd_solve ma97_sparse_fwd_solve_d
#endif
#endif
#endif

struct ma97_control {
    ipc_ f_arrays;            /* Use C or Fortran numbering */
    ipc_ action;              /* Continue on singularity if !=0 (true),
                                 otherwise abort */
    ipc_ nemin;               /* Supernode amalgamation if parent and child
                                 have fewer than nemin eliminations */
    rpc_ multiplier;          /* Amount of extra memory to allow for delays */
    ipc_ ordering;            /* Control scaling algorithm used:
                                 0 - user supplied order (order absent=identity)
                                 1 - AMD
                                 2 - MD (as in MA27)
                                 3 - METIS nested dissection
                                 4 - MA47
                                 5 - Automatic choice between 1 and 3 */
    ipc_ print_level;         /* <0 for no printing, 0 for basic, >1 for most */
    ipc_ scaling;             /* 0 user/none, 1 mc64, 2 mc77 */
    rpc_ small;               /* Minimum value to count as non-zero */
    rpc_ u;                   /* Pivoting parameter */
    ipc_ unit_diagnostics;    /* Fortran unit for diagnostics (<0 disables) */
    ipc_ unit_error;          /* Fortran unit for error msgs (<0 disables) */
    ipc_ unit_warning;        /* Fortran unit for warning msgs (<0 disables) */
    hsl_longc_ factor_min;    /* Min number of flops for parallel execution */
    ipc_ solve_blas3;         /* Use BLAS3 in solve in true, else BLAS2 */
    hsl_longc_ solve_min;     /* Min number of entries for parallel exection */
    ipc_ solve_mf;            /* If true use m/f solve, else use s/n */
    rpc_ consist_tol;         /* Consistent equation tolerance */

    /* Reserve space for future interface changes */
    ipc_ ispare[5]; rpc_ rspare[10];
};

struct ma97_info {
    ipc_ flag;                /* <0 on error */
    ipc_ flag68;
    ipc_ flag77;
    ipc_ matrix_dup;          /* number duplicate entries in A */
    ipc_ matrix_rank;         /* matrix rank */
    ipc_ matrix_outrange;     /* number of out of range entries in A */
    ipc_ matrix_missing_diag; /* number of zero diagonal entries in A */
    ipc_ maxdepth;            /* height of assembly tree */
    ipc_ maxfront;            /* maximum no. rows in a supernode */
    ipc_ num_delay;           /* number of times a pivot was delayed */
    hsl_longc_ num_factor;    /* number of entries in L */
    hsl_longc_ num_flops;     /* number of floating point operations */
    ipc_ num_neg;             /* number of negative pivots */
    ipc_ num_sup;             /* number of supernodes in assembly tree */
    ipc_ num_two;             /* number of 2x2 pivots */
    ipc_ ordering;            /* ordering used (as per control.ordering) */
    ipc_ stat;                /* error code from failed memory allocation */
    ipc_ maxsupernode;        /* maximum no. columns in a supernode */

    /* Reserve space for future interface changes */
    ipc_ ispare[4]; rpc_ rspare[10];
};

/* Set default values of control */
void ma97_default_control(struct ma97_control *control);
/* Perform symbolic analysis of matrix (sparse column entry) */
void ma97_analyse(int check, ipc_ n, const ipc_ ptr[], const ipc_ row[],
      rpc_ val[], void **akeep, const struct ma97_control *control,
      struct ma97_info *info, ipc_ order[]);
/* Perform symbolic analysis of matrix (coordinate entry) */
void ma97_analyse_coord(int n, ipc_ ne, const ipc_ row[], const ipc_ col[],
      rpc_ val[], void **akeep, const struct ma97_control *control,
      struct ma97_info *info, ipc_ order[]);
/* Perform numerical factorization, following call to ma97_analyse */
void ma97_factor(int matrix_type, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info,
      rpc_ scale[]);
/* Perform numerical factorization and solve, following call to ma97_analyse */
void ma97_factor_solve(int matrix_type, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], ipc_ nrhs, rpc_ x[], ipc_ ldx,
      void **akeep, void **fkeep, const struct ma97_control *control,
      struct ma97_info *info, rpc_ scale[]);
/* Perform forward and back substitutions, following call to ma97_factor */
void ma97_solve(int job, ipc_ nrhs, rpc_ x[], ipc_ ldx,
      void **akeep, void **fkeep, const struct ma97_control *control,
      struct ma97_info *info);
/* Free memory in akeep */
void ma97_free_akeep(void **akeep);
/* Free memory in fkeep */
void ma97_free_fkeep(void **fkeep);
/* Free memory in akeep and fkeep */
void ma97_finalise(void **akeep, void **fkeep);
/* Return diagonal entries of L */
void ma97_enquire_posdef(void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info,
      rpc_ d[]);
/* Return diagonal, subdiagonal and/or pivot order of D */
void ma97_enquire_indef(void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info,
      ipc_ *piv_order, rpc_ *d);
/* Alter diagonal and subdiagonal of D */
void ma97_alter(const rpc_ d[], void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info);
/* Fredholm alternative for singular systems */
void ma97_solve_fredholm(int nrhs,  ipc_ flag_out[], rpc_ x[],
      ipc_ ldx, void **akeep, void **fkeep, 
      const struct ma97_control *control,
      struct ma97_info *info);
/* Form (S^{-1}PL) X or (S^{-1}PL)^T X */
void ma97_lmultiply(int trans, ipc_ k, const rpc_ x[], ipc_ ldx,
      rpc_ y[], ipc_ ldy, void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info);
/* Perform a sparse forward solve */
void ma97_sparse_fwd_solve(int nbi, const ipc_ bindex[],
      const rpc_ b[], const ipc_ order[], ipc_ *nxi, ipc_ xindex[],
      rpc_ x[], void **akeep, void **fkeep,
      const struct ma97_control *control, struct ma97_info *info);

#endif
