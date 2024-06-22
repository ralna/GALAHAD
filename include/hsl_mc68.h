/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:40 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 2 March 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MC68 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MC68 for full terms
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

#ifndef HSL_MC68I
#define HSL_MC68I

/* precision */
#include "hsl_precision.h"

#ifndef mc68_default_control
#ifdef INTEGER_64
#define mc68_control mc68_control_i_64
#define mc68_info mc68_info_i_64
#define mc68_default_control mc68_default_control_i_64
#define mc68_order mc68_order_i_64
#else
#define mc68_control mc68_control_i
#define mc68_info mc68_info_i
#define mc68_default_control mc68_default_control_i
#define mc68_order mc68_order_i
#endif
#endif

struct mc68_control {
   /* Extra options for C version */
   ipc_ f_array_in;      /* 0 for C array indexing, 1 for Fortran indexing */
   ipc_ f_array_out;     /* 0 for C array indexing, 1 for Fortran indexing
                         * NOTE: 2x2 pivot information discarded if C indexing
                         * is used for output! */
   ipc_ min_l_workspace; /* Initial size of workspace, as argument in Fortran */
   /* Options from Fortran version */
   ipc_ lp;              /* stream number for error messages */
   ipc_ wp;              /* stream number for warning messages */
   ipc_ mp;              /* stream number for diagnostic messages */
   ipc_ nemin;           /* stream number for diagnostic messages */
   ipc_ print_level;     /* amount of informational output required */
   ipc_ row_full_thresh; /* percentage threshold for full row */
   ipc_ row_search;      /* Number of rows searched for pivot with ord=6 */
};

struct mc68_info {
   ipc_ flag;              /* error/warning flag */
   ipc_ iostat;            /* holds Fortran iostat parameter */
   ipc_ stat;              /* holds Fortran stat parameter */
   ipc_ out_range;         /* holds number of out of range entries ignored */
   ipc_ duplicate;         /* holds number of duplicate entries */
   ipc_ n_compressions;    /* holds number of compressions in order */
   ipc_ n_zero_eigs;       /* holds the number of zero eigs from ma47 */
   hsl_longc_ l_workspace; /* holds length of workspace iw used in order */
   ipc_ zb01_info;         /* holds flag from zb01_expand1 call */
   ipc_ n_dense_rows;      /* holds number of dense rows from amdd */
};

/* Set default values for control struct */
void mc68_default_control(struct mc68_control *control);
/* Perform ordering */
void mc68_order(ipc_ ord, ipc_ n, const ipc_ ptr[], const ipc_ row[],
   ipc_ perm[], const struct mc68_control *control, struct mc68_info *info);

#endif
