/* 
 * COPYRIGHT (c) 2012 Science and Technology Facilities Council
 * Original date 12 June 2012
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 * Version 2.2.0
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
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
#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef HSL_MC64_H
#define HSL_MC64_H

// precision
#include "galahad_precision.h"

struct mc64_control {
   int f_arrays;
   int lp;
   int wp;
   int sp;
   int ldiag;
   int checking;
};

struct mc64_info {
   int flag;
   int more;
   int strucrank;
   int stat;
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
