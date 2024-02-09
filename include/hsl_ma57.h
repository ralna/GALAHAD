//* \file hsl_ma57.h */

/*
 * COPYRIGHT (c) 2021 Science and Technology Facilities Council (STFC)
 * Original date 30 March 2021
 * All rights reserved
 *
 * Written by: Gabriele Santi and Jaroslav Fowkes
 * Version 1.0
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
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

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef HSL_MA57_H
#define HSL_MA57_H

// precision
#include "galahad_precision.h"

struct ma57_control {
  ipc_ f_arrays;        /* Use C or Fortran based indexing for sparse matrices */
  real_wp_ multiplier; /* Factor by which arrays sizes are to be */
                       /* increased if they are too small */
  real_wp_ reduce;     /* if previously allocated internal workspace arrays */
                       /* are greater than reduce times the currently */
                     /* required sizes, they are reset to current requirments */
  real_wp_ u;          /* Pivot threshold */
  real_wp_ static_tolerance; /* used for setting static pivot level */
  real_wp_ static_level; /* used for switch to static */
  real_wp_ tolerance;    /* anything less than this is considered zero */
  real_wp_ convergence;  /* used to monitor convergence in iterative */
                                /* refinement */
  real_wp_ consist;      /* used in test for consistency when using */
                         /* fredholm alternative */
  ipc_ lp;                /* Unit for error messages */
  ipc_ wp;                /* Unit for warning messages */
  ipc_ mp;                /* Unit for monitor output */
  ipc_ sp;                /* Unit for statistical output */
  ipc_ ldiag;             /* Controls level of diagnostic output */
  ipc_ nemin;             /* Minimum number of eliminations in a step */
  ipc_ factorblocking;    /* Level 3 blocking in factorize */
  ipc_ solveblocking;     /* Level 2 and 3 blocking in solve */
  ipc_ la;     		 /* Initial size for real array for the factors. */
              		 /* If less than nrlnec, default size used. */
  ipc_ liw;    		 /* Initial size for integer array for the factors. */
              		 /* If less than nirnec, default size used. */
  ipc_ maxla;  		 /* Max. size for real array for the factors. */
  ipc_ maxliw; 		 /* Max. size for integer array for the factors. */
  ipc_ pivoting;  	 /* Controls pivoting: */
                 	 /*  1  Numerical pivoting will be performed. */
                 	 /*  2 No pivoting will be performed and an error exit will */
                 	 /*    occur immediately a pivot sign change is detected. */
                 	 /*  3 No pivoting will be performed and an error exit will */
                 	 /*     occur if a zero pivot is detected. */
                 	 /*  4 No pivoting is performed but pivots are changed to */
                 	 /*     all be positive. */
  ipc_ thresh;            /* Controls threshold for detecting full rows in analyse */
                         /*     Registered as percentage of N */
                         /* 100 Only fully dense rows detected (default) */
  ipc_ ordering;  	 /* Controls ordering: */
                 	 /*  Note that this is overridden by using optional parameter */
                 	 /* perm in analyse call with equivalent action to 1. */
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
  ipc_ ispare[5]; real_wp_ rspare[10];
};

struct ma57_ainfo {
  real_wp_ opsa;         /* Anticipated # operations in assembly */
  real_wp_ opse;         /* Anticipated # operations in elimination */
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
  ipc_ ispare[5]; real_wp_ rspare[10];
};

struct ma57_finfo {
  real_wp_ opsa;       /* Number of operations in assembly */
  real_wp_ opse;       /* Number of operations in elimination */
  real_wp_ opsb;       /* Additional number of operations for BLAS */
  real_wp_ maxchange;  /* Largest pivot modification when pivoting=4 */
  real_wp_ smin;       /* Minimum scaling factor */
  real_wp_ smax;       /* Maximum scaling factor */
  ipc_ flag;            /* Flags success or failure case */
  ipc_ more;            /* More information on failure */
  ipc_ maxfrt;          /* Largest front size */
  ipc_ nebdu;           /* Number of entries in factors */
  ipc_ nrlbdu;          /* Number of reals that hold factors */
  ipc_ nirbdu;          /* Number of integers that hold factors */
  ipc_ nrltot;          /* Size for a without compression */
  ipc_ nirtot;          /* Size for iw without compression */
  ipc_ nrlnec;          /* Size for a with compression */
  ipc_ nirnec;          /* Size for iw with compression */
  ipc_ ncmpbr;          /* Number of compresses of real data */
  ipc_ ncmpbi;          /* Number of compresses of integer data */
  ipc_ ntwo;            /* Number of 2x2 pivots */
  ipc_ neig;            /* Number of negative eigenvalues */
  ipc_ delay;           /* Number of delayed pivots (total) */
  ipc_ signc;           /* Number of pivot sign changes (pivoting=3) */
  ipc_ static_;         /* Number of static pivots chosen */
  ipc_ modstep;         /* First pivot modification when pivoting=4 */
  ipc_ rank;            /* Rank of original factorization */
  ipc_ stat;            /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; real_wp_ rspare[10];
};

struct ma57_sinfo {
  real_wp_ cond;   /* Condition number of matrix (category 1 equations) */
  real_wp_ cond2;  /* Condition number of matrix (category 2 equations) */
  real_wp_ berr;   /* Condition number of matrix (category 1 equations) */
  real_wp_ berr2;  /* Condition number of matrix (category 2 equations) */
  real_wp_ error;  /* Estimate of forward error using above data */
  ipc_ flag;       	  /* Flags success or failure case */
  ipc_ stat;       	  /* STAT value after allocate failure */

  /* Reserve space for future interface changes */
  ipc_ ispare[5]; real_wp_ rspare[10];
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
