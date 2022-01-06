/*
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

#ifndef ma57_default_control
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

typedef double ma57pkgtype_d_;
typedef double ma57realtype_d_;

struct ma57_control_d {
  int f_arrays;                     /* Use C or Fortran based indexing for sparse matrices */
  ma57realtype_d_ multiplier;       /* Factor by which arrays sizes are to be */
                                    /* increased if they are too small */
  ma57realtype_d_ reduce;           /* if previously allocated internal workspace arrays */
                                    /*  are greater than reduce times the currently */
                                    /*  required sizes, they are reset to current requirments */
  ma57realtype_d_ u;                /* Pivot threshold */
  ma57realtype_d_ static_tolerance; /* used for setting static pivot level */
  ma57realtype_d_ static_level;     /* used for switch to static */
  ma57realtype_d_ tolerance;        /* anything less than this is considered zero */
  ma57realtype_d_ convergence;      /* used to monitor convergence in iterative */
                                    /* refinement */
  ma57realtype_d_ consist;          /* used in test for consistency when using */
                                    /* fredholm alternative */
  int lp;                           /* Unit for error messages */
  int wp;                           /* Unit for warning messages */
  int mp;                           /* Unit for monitor output */
  int sp;                           /* Unit for statistical output */
  int ldiag;                        /* Controls level of diagnostic output */
  int nemin;                        /* Minimum number of eliminations in a step */
  int factorblocking;               /* Level 3 blocking in factorize */
  int solveblocking;                /* Level 2 and 3 blocking in solve */
  int la;     			    /* Initial size for real array for the factors. */
              			    /* If less than nrlnec, default size used. */
  int liw;    			    /* Initial size for integer array for the factors. */
              			    /* If less than nirnec, default size used. */
  int maxla;  			    /* Max. size for real array for the factors. */
  int maxliw; 			    /* Max. size for integer array for the factors. */
  int pivoting;  		    /* Controls pivoting: */
                 		    /*  1  Numerical pivoting will be performed. */
                 		    /*  2  No pivoting will be performed and an error exit will */
                 		    /*     occur immediately a pivot sign change is detected. */
                 		    /*  3  No pivoting will be performed and an error exit will */
                 		    /*     occur if a zero pivot is detected. */
                 		    /*  4  No pivoting is performed but pivots are changed to */
                 		    /*     all be positive. */
  int thresh;                       /* Controls threshold for detecting full rows in analyse */
                                    /*     Registered as percentage of N */
                                    /* 100 Only fully dense rows detected (default) */
  int ordering;  		    /* Controls ordering: */
                 		    /*  Note that this is overridden by using optional parameter */
                 		    /*  perm in analyse call with equivalent action to 1. */
                 		    /*  0  AMD using MC47 */
                 		    /*  1  User defined */
                 		    /*  2  AMD using MC50 */
                 		    /*  3  Min deg as in MA57 */
                 		    /*  4  Metis_nodend ordering */
                 		    /* >4  Presently equivalent to 0 but may chnage */
  int scaling;                      /* Controls scaling: */
  				    /*  0  No scaling */
  				    /* >0  Scaling using MC64 but may change for > 1 */
  int rank_deficient;               /* Controls handling rank deficiency: */
  				    /*  0  No control */
  				    /* >0  Small entries removed during factorization */

  /* Reserve space for future interface changes */
  int ispare[5]; ma57realtype_d_ rspare[10];
};

struct ma57_ainfo_d {
  ma57realtype_d_ opsa;  /* Anticipated # operations in assembly */
  ma57realtype_d_ opse;  /* Anticipated # operations in elimination */
  int flag;        	 /* Flags success or failure case */
  int more;        	 /* More information on failure */
  int nsteps;      	 /* Number of elimination steps */
  int nrltot;      	 /* Size for a without compression */
  int nirtot;      	 /* Size for iw without compression */
  int nrlnec;      	 /* Size for a with compression */
  int nirnec;      	 /* Size for iw with compression */
  int nrladu;      	 /* Number of reals to hold factors */
  int niradu;      	 /* Number of integers to hold factors */
  int ncmpa;       	 /* Number of compresses */
  int ordering;    	 /* Indicates the ordering actually used */
  int oor;         	 /* Number of indices out-of-range */
  int dup;         	 /* Number of duplicates */
  int maxfrt;      	 /* Forecast maximum front size */
  int stat;        	 /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  int ispare[5]; ma57realtype_d_ rspare[10];
};

struct ma57_finfo_d {
  ma57realtype_d_ opsa;       /* Number of operations in assembly */
  ma57realtype_d_ opse;       /* Number of operations in elimination */
  ma57realtype_d_ opsb;       /* Additional number of operations for BLAS */
  ma57realtype_d_ maxchange;  /* Largest pivot modification when pivoting=4 */
  ma57realtype_d_ smin;       /* Minimum scaling factor */
  ma57realtype_d_ smax;       /* Maximum scaling factor */
  int flag;         	      /* Flags success or failure case */
  int more;         	      /* More information on failure */
  int maxfrt;       	      /* Largest front size */
  int nebdu;        	      /* Number of entries in factors */
  int nrlbdu;       	      /* Number of reals that hold factors */
  int nirbdu;       	      /* Number of integers that hold factors */
  int nrltot;       	      /* Size for a without compression */
  int nirtot;       	      /* Size for iw without compression */
  int nrlnec;       	      /* Size for a with compression */
  int nirnec;       	      /* Size for iw with compression */
  int ncmpbr;       	      /* Number of compresses of real data */
  int ncmpbi;       	      /* Number of compresses of integer data */
  int ntwo;         	      /* Number of 2x2 pivots */
  int neig;         	      /* Number of negative eigenvalues */
  int delay;        	      /* Number of delayed pivots (total) */
  int signc;        	      /* Number of pivot sign changes (pivoting=3) */
  int static_;       	      /* Number of static pivots chosen */
  int modstep;      	      /* First pivot modification when pivoting=4 */
  int rank;         	      /* Rank of original factorization */
  int stat;                   /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  int ispare[5]; ma57realtype_d_ rspare[10];
};

struct ma57_sinfo_d {
  ma57realtype_d_ cond;   /* Condition number of matrix (category 1 equations) */
  ma57realtype_d_ cond2;  /* Condition number of matrix (category 2 equations) */
  ma57realtype_d_ berr;   /* Condition number of matrix (category 1 equations) */
  ma57realtype_d_ berr2;  /* Condition number of matrix (category 2 equations) */
  ma57realtype_d_ error;  /* Estimate of forward error using above data */
  int flag;       	  /* Flags success or failure case */
  int stat;       	  /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  int ispare[5]; ma57realtype_d_ rspare[10];
};

/*** BASE SUBROUTINES OF ORIGINAL MA57 F77 CODE ***/
/* Set default values of control */
void ma57_default_control_d(struct ma57_control_d *control);
/* Fortran-side initialization of factors */
void ma57_init_factors_d(void **factors);
/* Perform symbolic analysis of matrix */
void ma57_analyse_d(int n, int ne, const int row[], const int col[],
		    void **factors, const struct ma57_control_d *control,
		    struct ma57_ainfo_d *ainfo, const int perm[]);
/* Perform numerical factorization, following call to ma57_analyse */
void ma57_factorize_d(int n, int ne, const int row[], const int col[],
		      const ma57pkgtype_d_ val[], void **factors,
		      const struct ma57_control_d *control, struct ma57_finfo_d *finfo);
/* Perform forward and back substitutions, following call to ma57_factor */
void ma57_solve_d(int n, int ne, const int row[], const int col[],
		  const ma57pkgtype_d_ val[], void **factors, int nrhs, ma57pkgtype_d_ x[],
		  const struct ma57_control_d *control, struct ma57_sinfo_d *sinfo,
		  const ma57pkgtype_d_ rhs[], int iter, int cond);
/* Free memory in factors and control */
void ma57_finalize_d(void **factors, struct ma57_control_d *control, int *info);

/*** ADDITIONAL FEATURES OF HSL_MA57 F90 CODE ***/
/****** ENQUIRE functions */
void ma57_enquire_perm_d(const struct ma57_control_d *control, void **factors, int perm[]);
void ma57_enquire_pivots_d(const struct ma57_control_d *control, void **factors,
			   int pivots[]);
void ma57_enquire_d_d(void **factors, ma57pkgtype_d_ d[]);
void ma57_enquire_perturbation_d(void **factors, ma57pkgtype_d_ perturbation[]);
void ma57_enquire_scaling_d(void **factors, ma57pkgtype_d_ scaling[]);

void ma57_alter_d_d(void **factors, const ma57pkgtype_d_ d[], int *info);
void ma57_part_solve_d(void **factors, const struct ma57_control_d *control,
		     char part, int nrhs, ma57pkgtype_d_ x[], int *info);
void ma57_sparse_lsolve_d(void **factors, const struct ma57_control_d *control,
			int nzrhs, const int irhs[], int *nzsoln, int isoln[],
			ma57pkgtype_d_ x[], struct ma57_sinfo_d *sinfo);
void ma57_fredholm_alternative_d(void **factors, const struct ma57_control_d *control,
				 ma57pkgtype_d_ x[], ma57pkgtype_d_ fredx[],
				 struct ma57_sinfo_d *sinfo);
void ma57_lmultiply_d(void **factors, const struct ma57_control_d *control,
		      char trans, ma57pkgtype_d_ x[], ma57pkgtype_d_ y[],
		      struct ma57_sinfo_d *sinfo);
void ma57_get_factors_d(void **factors, const struct ma57_control_d *control,
			int *nzl, int iptrl[], int lrows[], ma57pkgtype_d_ lvals[], int *nzd,
			int iptrd[], int drows[], ma57pkgtype_d_ dvals[], int perm[],
			int invperm[], ma57pkgtype_d_ scale[], struct ma57_sinfo_d *sinfo);

#endif
