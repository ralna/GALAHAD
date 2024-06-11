/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-11 AT 10:00 GMT
 * COPYRIGHT (c) 2021 Science and Technology Facilities Council (STFC)
 * Original date 30 March 2021
 * All rights reserved
 *
 * Written by: Gabriele Santi and Jaroslav Fowkes
 *
 * Version 1.0
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA57 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA57 for full terms
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

#ifndef HSL_MA57D_H
#define HSL_MA57D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma57_default_control
#ifdef REAL_32
#ifdef INTEGER_64
#define ma57_default_control      ma57_default_control_s_64
#define ma57_init_factors         ma57_init_factors_s_64
#define ma57_control              ma57_control_s_64
#define ma57_ainfo                ma57_ainfo_s_64
#define ma57_finfo                ma57_finfo_s_64
#define ma57_sinfo                ma57_sinfo_s_64
#define ma57_analyse              ma57_analyse_s_64
#define ma57_factorize            ma57_factorize_s_64
#define ma57_solve                ma57_solve_s_64
#define ma57_finalize             ma57_finalize_s_64
#define ma57_enquire_perm    	  ma57_enquire_perm_s_64
#define ma57_enquire_pivots  	  ma57_enquire_pivots_s_64
#define ma57_enquire_d       	  ma57_enquire_d_s_64
#define ma57_enquire_perturbation ma57_enquire_perturbation_s_64
#define ma57_enquire_scaling      ma57_enquire_scaling_s_64
#define ma57_alter_d              ma57_alter_d_s_64
#define ma57_part_solve           ma57_part_solve_s_64
#define ma57_sparse_lsolve        ma57_sparse_lsolve_s_64
#define ma57_fredholm_alternative ma57_fredholm_alternative_s_64
#define ma57_lmultiply            ma57_lmultiply_s_64
#define ma57_get_factors          ma57_get_factors_s_64
#else
#define ma57_default_control      ma57_default_control_s
#define ma57_init_factors         ma57_init_factors_s
#define ma57_control              ma57_control_s
#define ma57_ainfo                ma57_ainfo_s
#define ma57_finfo                ma57_finfo_s
#define ma57_sinfo                ma57_sinfo_s
#define ma57_analyse              ma57_analyse_s
#define ma57_factorize            ma57_factorize_s
#define ma57_solve                ma57_solve_s
#define ma57_finalize             ma57_finalize_s
#define ma57_enquire_perm    	  ma57_enquire_perm_s
#define ma57_enquire_pivots  	  ma57_enquire_pivots_s
#define ma57_enquire_d       	  ma57_enquire_d_s
#define ma57_enquire_perturbation ma57_enquire_perturbation_s
#define ma57_enquire_scaling      ma57_enquire_scaling_s
#define ma57_alter_d              ma57_alter_d_s
#define ma57_part_solve           ma57_part_solve_s
#define ma57_sparse_lsolve        ma57_sparse_lsolve_s
#define ma57_fredholm_alternative ma57_fredholm_alternative_s
#define ma57_lmultiply            ma57_lmultiply_s
#define ma57_get_factors          ma57_get_factors_s
#endif
#else
#ifdef INTEGER_64
#define ma57_default_control      ma57_default_control_d_64
#define ma57_init_factors         ma57_init_factors_d_64
#define ma57_control              ma57_control_d_64
#define ma57_ainfo                ma57_ainfo_d_64
#define ma57_finfo                ma57_finfo_d_64
#define ma57_sinfo                ma57_sinfo_d_64
#define ma57_analyse              ma57_analyse_d_64
#define ma57_factorize            ma57_factorize_d_64
#define ma57_solve                ma57_solve_d_64
#define ma57_finalize             ma57_finalize_d_64
#define ma57_enquire_perm    	  ma57_enquire_perm_d_64
#define ma57_enquire_pivots  	  ma57_enquire_pivots_d_64
#define ma57_enquire_d       	  ma57_enquire_d_d_64
#define ma57_enquire_perturbation ma57_enquire_perturbation_d_64
#define ma57_enquire_scaling      ma57_enquire_scaling_d_64
#define ma57_alter_d              ma57_alter_d_d_64
#define ma57_part_solve           ma57_part_solve_d_64
#define ma57_sparse_lsolve        ma57_sparse_lsolve_d_64
#define ma57_fredholm_alternative ma57_fredholm_alternative_d_64
#define ma57_lmultiply            ma57_lmultiply_d_64
#define ma57_get_factors          ma57_get_factors_d_64
#else
#define ma57_default_control      ma57_default_control_d
#define ma57_init_factors         ma57_init_factors_d
#define ma57_control              ma57_control_d
#define ma57_ainfo                ma57_ainfo_d
#define ma57_finfo                ma57_finfo_d
#define ma57_sinfo                ma57_sinfo_d
#define ma57_analyse              ma57_analyse_d
#define ma57_factorize            ma57_factorize_d
#define ma57_solve                ma57_solve_d
#define ma57_finalize             ma57_finalize_d
#define ma57_enquire_perm    	  ma57_enquire_perm_d
#define ma57_enquire_pivots  	  ma57_enquire_pivots_d
#define ma57_enquire_d       	  ma57_enquire_d_d
#define ma57_enquire_perturbation ma57_enquire_perturbation_d
#define ma57_enquire_scaling      ma57_enquire_scaling_d
#define ma57_alter_d              ma57_alter_d_d
#define ma57_part_solve           ma57_part_solve_d
#define ma57_sparse_lsolve        ma57_sparse_lsolve_d
#define ma57_fredholm_alternative ma57_fredholm_alternative_d
#define ma57_lmultiply            ma57_lmultiply_d
#define ma57_get_factors          ma57_get_factors_d
#endif
#endif
#endif

struct ma57_control {
  ipc_ f_arrays;         /* Use C or Fortran based indexing for sparse matrices */
  rpc_ multiplier;       /* Factor by which arrays sizes are to be */
                                    /* increased if they are too small */
  rpc_ reduce;           /* if previously allocated internal workspace arrays */
                         /*  are greater than reduce times the currently */
                         /*  required sizes, they are reset to current requirments */
  rpc_ u;                /* Pivot threshold */
  rpc_ static_tolerance; /* used for setting static pivot level */
  rpc_ static_level;     /* used for switch to static */
  rpc_ tolerance;        /* anything less than this is considered zero */
  rpc_ convergence;      /* used to monitor convergence in iterative */
                                    /* refinement */
  rpc_ consist;          /* used in test for consistency when using */
                                    /* fredholm alternative */
  ipc_ lp;               /* Unit for error messages */
  ipc_ wp;               /* Unit for warning messages */
  ipc_ mp;               /* Unit for monitor output */
  ipc_ sp;               /* Unit for statistical output */
  ipc_ ldiag;            /* Controls level of diagnostic output */
  ipc_ nemin;            /* Minimum number of eliminations in a step */
  ipc_ factorblocking;   /* Level 3 blocking in factorize */
  ipc_ solveblocking;    /* Level 2 and 3 blocking in solve */
  ipc_ la;     		 /* Initial size for real array for the factors. */
              		 /* If less than nrlnec, default size used. */
  ipc_ liw;    		 /* Initial size for integer array for the factors. */
              		 /* If less than nirnec, default size used. */
  ipc_ maxla;  		 /* Max. size for real array for the factors. */
  ipc_ maxliw; 		 /* Max. size for integer array for the factors. */
  ipc_ pivoting;  	 /* Controls pivoting: */
                 	 /*  1  Numerical pivoting will be performed. */
                 	 /*  2  No pivoting will be performed and an error exit will */
                 	 /*     occur immediately a pivot sign change is detected. */
                 	 /*  3  No pivoting will be performed and an error exit will */
                 	 /*     occur if a zero pivot is detected. */
                 	 /*  4  No pivoting is performed but pivots are changed to */
                 	 /*     all be positive. */
  ipc_ thresh;            /* Controls threshold for detecting full rows in analyse */
                         /*     Registered as percentage of N */
                         /* 100 Only fully dense rows detected (default) */
  ipc_ ordering;  	 /* Controls ordering: */
                 	 /*  Note that this is overridden by using optional parameter */
                 	 /*  perm in analyse call with equivalent action to 1. */
                 	 /*  0  AMD using MC47 */
                 	 /*  1  User defined */
                 	 /*  2  AMD using MC50 */
                 	 /*  3  Min deg as in MA57 */
                 	 /*  4  Metis_nodend ordering */
                 	 /* >4  Presently equivalent to 0 but may chnage */
  ipc_ scaling;           /* Controls scaling: */
  			 /*  0  No scaling */
  			 /* >0  Scaling using MC64 but may change for > 1 */
  ipc_ rank_deficient;    /* Controls handling rank deficiency: */
  			 /*  0  No control */
  			 /* >0  Small entries removed during factorization */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; rpc_ rspare[10];
};

struct ma57_ainfo {
  rpc_ opsa;  /* Anticipated # operations in assembly */
  rpc_ opse;  /* Anticipated # operations in elimination */
  ipc_ flag;        	 /* Flags success or failure case */
  ipc_ more;        	 /* More information on failure */
  ipc_ nsteps;      	 /* Number of elimination steps */
  ipc_ nrltot;      	 /* Size for a without compression */
  ipc_ nirtot;      	 /* Size for iw without compression */
  ipc_ nrlnec;      	 /* Size for a with compression */
  ipc_ nirnec;      	 /* Size for iw with compression */
  ipc_ nrladu;      	 /* Number of reals to hold factors */
  ipc_ niradu;      	 /* Number of integers to hold factors */
  ipc_ ncmpa;       	 /* Number of compresses */
  ipc_ ordering;    	 /* Indicates the ordering actually used */
  ipc_ oor;         	 /* Number of indices out-of-range */
  ipc_ dup;         	 /* Number of duplicates */
  ipc_ maxfrt;      	 /* Forecast maximum front size */
  ipc_ stat;        	 /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; rpc_ rspare[10];
};

struct ma57_finfo {
  rpc_ opsa;       /* Number of operations in assembly */
  rpc_ opse;       /* Number of operations in elimination */
  rpc_ opsb;       /* Additional number of operations for BLAS */
  rpc_ maxchange;  /* Largest pivot modification when pivoting=4 */
  rpc_ smin;       /* Minimum scaling factor */
  rpc_ smax;       /* Maximum scaling factor */
  ipc_ flag;         	      /* Flags success or failure case */
  ipc_ more;         	      /* More information on failure */
  ipc_ maxfrt;       	      /* Largest front size */
  ipc_ nebdu;        	      /* Number of entries in factors */
  ipc_ nrlbdu;       	      /* Number of reals that hold factors */
  ipc_ nirbdu;       	      /* Number of integers that hold factors */
  ipc_ nrltot;       	      /* Size for a without compression */
  ipc_ nirtot;       	      /* Size for iw without compression */
  ipc_ nrlnec;       	      /* Size for a with compression */
  ipc_ nirnec;       	      /* Size for iw with compression */
  ipc_ ncmpbr;       	      /* Number of compresses of real data */
  ipc_ ncmpbi;       	      /* Number of compresses of integer data */
  ipc_ ntwo;         	      /* Number of 2x2 pivots */
  ipc_ neig;         	      /* Number of negative eigenvalues */
  ipc_ delay;        	      /* Number of delayed pivots (total) */
  ipc_ signc;        	      /* Number of pivot sign changes (pivoting=3) */
  ipc_ static_;       	      /* Number of static pivots chosen */
  ipc_ modstep;      	      /* First pivot modification when pivoting=4 */
  ipc_ rank;         	      /* Rank of original factorization */
  ipc_ stat;                   /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; rpc_ rspare[10];
};

struct ma57_sinfo {
  rpc_ cond;   /* Condition number of matrix (category 1 equations) */
  rpc_ cond2;  /* Condition number of matrix (category 2 equations) */
  rpc_ berr;   /* Condition number of matrix (category 1 equations) */
  rpc_ berr2;  /* Condition number of matrix (category 2 equations) */
  rpc_ error;  /* Estimate of forward error using above data */
  ipc_ flag;       	  /* Flags success or failure case */
  ipc_ stat;       	  /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; rpc_ rspare[10];
};

/*** BASE SUBROUTINES OF ORIGINAL MA57 F77 CODE ***/
/* Set default values of control */
void ma57_default_control(struct ma57_control *control);
/* Fortran-side initialization of factors */
void ma57_init_factors(void **factors);
/* Perform symbolic analysis of matrix */
void ma57_analyse(ipc_ n, ipc_ ne, const ipc_ row[], const ipc_ col[],
		    void **factors, const struct ma57_control *control,
		    struct ma57_ainfo *ainfo, const ipc_ perm[]);
/* Perform numerical factorization, following call to ma57_analyse */
void ma57_factorize(ipc_ n, ipc_ ne, const ipc_ row[], const ipc_ col[],
		      const rpc_ val[], void **factors,
		      const struct ma57_control *control, 
                      struct ma57_finfo *finfo);
/* Perform forward and back substitutions, following call to ma57_factor */
void ma57_solve(ipc_ n, ipc_ ne, const ipc_ row[], const ipc_ col[],
		  const rpc_ val[], void **factors, ipc_ nrhs, rpc_ x[],
		  const struct ma57_control *control, 
                  struct ma57_sinfo *sinfo,
		  const rpc_ rhs[], ipc_ iter, ipc_ cond);
/* Free memory in factors and control */
void ma57_finalize(void **factors, struct ma57_control *control, 
                     ipc_ *info);

/*** ADDITIONAL FEATURES OF HSL_MA57 F90 CODE ***/
/****** ENQUIRE functions */
void ma57_enquire_perm(const struct ma57_control *control, 
                         void **factors, ipc_ perm[]);
void ma57_enquire_pivots(const struct ma57_control *control, void **factors,
			   ipc_ pivots[]);
void ma57_enquire_d(void **factors, rpc_ d[]);
void ma57_enquire_perturbation(void **factors, rpc_ perturbation[]);
void ma57_enquire_scaling(void **factors, rpc_ scaling[]);

void ma57_alter_d(void **factors, const rpc_ d[], ipc_ *info);
void ma57_part_solve(void **factors, const struct ma57_control *control,
		     char part, ipc_ nrhs, rpc_ x[], ipc_ *info);
void ma57_sparse_lsolve(void **factors, const struct ma57_control *control,
			ipc_ nzrhs, const ipc_ irhs[], ipc_ *nzsoln, 
                        ipc_ isoln[], rpc_ x[], struct ma57_sinfo *sinfo);
void ma57_fredholm_alternative(void **factors, 
                                 const struct ma57_control *control,
				 rpc_ x[], rpc_ fredx[],
				 struct ma57_sinfo *sinfo);
void ma57_lmultiply(void **factors, const struct ma57_control *control,
		      char trans, rpc_ x[], rpc_ y[],
		      struct ma57_sinfo *sinfo);
void ma57_get_factors(void **factors, const struct ma57_control *control,
			ipc_ *nzl, ipc_ iptrl[], ipc_ lrows[], rpc_ lvals[], 
                        ipc_ *nzd, ipc_ iptrd[], ipc_ drows[], rpc_ dvals[], 
                        ipc_ perm[], ipc_ invperm[], rpc_ scale[], 
                        struct ma57_sinfo *sinfo);

#endif
