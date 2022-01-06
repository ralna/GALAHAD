/* 
 * COPYRIGHT (c) 2012 Science and Technology Facilities Council
 * Original date 12 June 2012
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * Version 2.2.0
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

#ifndef HSL_MC64S_H
#define HSL_MC64S_H

#ifndef mc64_default_control
#define mc64_control mc64_control_s
#define mc64_info mc64_info_s
#define mc64_default_control mc64_default_control_s
#define mc64_matching mc64_matching_s
#endif

typedef float mc64pkgtype_s_;

struct mc64_control_s {
   int f_arrays;
   int lp;
   int wp;
   int sp;
   int ldiag;
   int checking;
};

struct mc64_info_s {
   int flag;
   int more;
   int strucrank;
   int stat;
};

/* Set default values of control */
void mc64_default_control_s(struct mc64_control *control);
/* Find a matching, and (optionally) scaling */
void mc64_matching_s(int job, int matrix_type, int m, int n, const int *ptr,
   const int *row, const mc64pkgtype_s_ *cval,
   const struct mc64_control *control,
   struct mc64_info *info, int *perm, mc64pkgtype_s_ *scale);

#endif
