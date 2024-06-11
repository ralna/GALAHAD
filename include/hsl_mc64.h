/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-11 AT 09:25 GMT
 * COPYRIGHT (c) 2012 Science and Technology Facilities Council
 * Original date 12 June 2012
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * Version 2.4.3
 *
 * History: See ChangeLog
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

#ifndef HSL_MC64D_H
#define HSL_MC64D_H

/* precision */
#include "hsl_precision.h"

#ifndef mc64_default_control
#ifdef REAL_32
#ifdef INTEGER_64
#define mc64_control mc64_control_s_64
#define mc64_info mc64_info_s_64
#define mc64_default_control mc64_default_control_s_64
#define mc64_matching mc64_matching_s_64
#else
#define mc64_control mc64_control_s
#define mc64_info mc64_info_s
#define mc64_default_control mc64_default_control_s
#define mc64_matching mc64_matching_s
#endif
#else
#ifdef INTEGER_64
#define mc64_control mc64_control_d_64
#define mc64_info mc64_info_d_64
#define mc64_default_control mc64_default_control_d_64
#define mc64_matching mc64_matching_d_64
#else
#define mc64_control mc64_control_d
#define mc64_info mc64_info_d
#define mc64_default_control mc64_default_control_d
#define mc64_matching mc64_matching_d
#endif
#endif
#endif

struct mc64_control {
   ipc_ f_arrays;
   ipc_ lp;
   ipc_ wp;
   ipc_ sp;
   ipc_ ldiag;
   ipc_ checking;
};

struct mc64_info {
   ipc_ flag;
   ipc_ more;
   ipc_ strucrank;
   ipc_ stat;
};

/* Set default values of control */
void mc64_default_control(struct mc64_control *control);
/* Find a matching, and (optionally) scaling */
void mc64_matching(ipc_ job, ipc_ matrix_type, ipc_ m, ipc_ n, const ipc_ *ptr,
   const ipc_ *row, const rpc_ *cval,
   const struct mc64_control *control,
   struct mc64_info *info, ipc_ *perm, rpc_ *scale);

#endif
