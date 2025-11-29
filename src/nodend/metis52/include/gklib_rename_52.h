/*!
\file

 * Copyright 1997, Regents of the University of Minnesota
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * $Id: gklib_rename.h 10395 2011-06-23 23:28:06Z karypis $
 *
 */


#ifndef _LIBMETIS_GKLIB_RENAME_H_
#define _LIBMETIS_GKLIB_RENAME_H_

/* 64-bit integer procedures */

#ifdef INTEGER_64

#define iAllocMatrix iAllocMatrix_52_64
#define iargmax iargmax_52_64
#define iargmax_n iargmax_n_52_64
#define iargmin iargmin_52_64
#define iarray2csr iarray2csr_52_64
#define iaxpy iaxpy_52_64
#define icopy icopy_52_64
#define idot idot_52_64
#define iFreeMatrix iFreeMatrix_52_64
#define iincset iincset_52_64
#define ikvAllocMatrix ikvAllocMatrix_52_64
#define ikvcopy ikvcopy_52_64
#define ikvFreeMatrix ikvFreeMatrix_52_64
#define ikvmalloc ikvmalloc_52_64
#define ikvrealloc ikvrealloc_52_64
#define ikvset ikvset_52_64
#define ikvSetMatrix ikvSetMatrix_52_64
#define ikvsmalloc ikvsmalloc_52_64
#define ikvsortd ikvsortd_52_64
#define ikvsorti ikvsorti_52_64
#define ikvsortii ikvsortii_52_64
#define imalloc imalloc_52_64
#define imax imax_52_64
#define imin imin_52_64
#define inorm2 inorm2_52_64
#define ipqCheckHeap ipqCheckHeap_52_64
#define ipqCreate ipqCreate_52_64
#define ipqDelete ipqDelete_52_64
#define ipqDestroy ipqDestroy_52_64
#define ipqFree ipqFree_52_64
#define ipqGetTop ipqGetTop_52_64
#define ipqInit ipqInit_52_64
#define ipqInsert ipqInsert_52_64
#define ipqLength ipqLength_52_64
#define ipqReset ipqReset_52_64
#define ipqSeeKey ipqSeeKey_52_64
#define ipqSeeTopKey ipqSeeTopKey_52_64
#define ipqSeeTopVal ipqSeeTopVal_52_64
#define ipqUpdate ipqUpdate_52_64
#define irand irand_52_64
#define irandArrayPermute irandArrayPermute_52_64
#define irandArrayPermuteFine irandArrayPermuteFine_52_64
#define irandInRange irandInRange_52_64
#define irealloc irealloc_52_64
#define iscale iscale_52_64
#define iset iset_52_64
#define iSetMatrix iSetMatrix_52_64
#define ismalloc ismalloc_52_64
#define isortd isortd_52_64
#define isorti isorti_52_64
#define isrand isrand_52_64
#define isum isum_52_64
#define rAllocMatrix rAllocMatrix_52_64
#define rargmax rargmax_52_64
#define rargmax_n rargmax_n_52_64
#define rargmin rargmin_52_64
#define raxpy raxpy_52_64
#define rcopy rcopy_52_64
#define rdot rdot_52_64
#define rFreeMatrix rFreeMatrix_52_64
#define rincset rincset_52_64
#define rkvAllocMatrix rkvAllocMatrix_52_64
#define rkvcopy rkvcopy_52_64
#define rkvFreeMatrix rkvFreeMatrix_52_64
#define rkvmalloc rkvmalloc_52_64
#define rkvrealloc rkvrealloc_52_64
#define rkvset rkvset_52_64
#define rkvSetMatrix rkvSetMatrix_52_64
#define rkvsmalloc rkvsmalloc_52_64
#define rkvsortd rkvsortd_52_64
#define rkvsorti rkvsorti_52_64
#define rmalloc rmalloc_52_64
#define rmax rmax_52_64
#define rmin rmin_52_64
#define rnorm2 rnorm2_52_64
#define rpqCheckHeap rpqCheckHeap_52_64
#define rpqCreate rpqCreate_52_64
#define rpqDelete rpqDelete_52_64
#define rpqDestroy rpqDestroy_52_64
#define rpqFree rpqFree_52_64
#define rpqGetTop rpqGetTop_52_64
#define rpqInit rpqInit_52_64
#define rpqInsert rpqInsert_52_64
#define rpqLength rpqLength_52_64
#define rpqReset rpqReset_52_64
#define rpqSeeKey rpqSeeKey_52_64
#define rpqSeeTopKey rpqSeeTopKey_52_64
#define rpqSeeTopVal rpqSeeTopVal_52_64
#define rpqUpdate rpqUpdate_52_64
#define rrealloc rrealloc_52_64
#define rscale rscale_52_64
#define rset rset_52_64
#define rSetMatrix rSetMatrix_52_64
#define rsmalloc rsmalloc_52_64
#define rsortd rsortd_52_64
#define rsorti rsorti_52_64
#define rsum rsum_52_64
#define uvwsorti uvwsorti_52_64
#define re_compile_pattern re_compile_pattern_52_64
#define re_set_syntax re_set_syntax_52_64
#define re_compile_fastmap re_compile_fastmap_52_64
#define regcomp regcomp_52_64
#define regerror regerror_52_64
#define regfree regfree_52_64
#define regexec regexec_52_64
#define re_match re_match_52_64
#define re_search re_search_52_64
#define re_match_2 re_match_2_52_64
#define re_search_2 re_search_2_52_64
#define re_set_registers re_set_registers_52_64
#define re_syntax_options re_syntax_options_52_64

/* 32-bit integer procedures */

#else

#define iAllocMatrix iAllocMatrix_52
#define iargmax iargmax_52
#define iargmax_n iargmax_n_52
#define iargmin iargmin_52
#define iarray2csr iarray2csr_52
#define iaxpy iaxpy_52
#define icopy icopy_52
#define idot idot_52
#define iFreeMatrix iFreeMatrix_52
#define iincset iincset_52
#define ikvAllocMatrix ikvAllocMatrix_52
#define ikvcopy ikvcopy_52
#define ikvFreeMatrix ikvFreeMatrix_52
#define ikvmalloc ikvmalloc_52
#define ikvrealloc ikvrealloc_52
#define ikvset ikvset_52
#define ikvSetMatrix ikvSetMatrix_52
#define ikvsmalloc ikvsmalloc_52
#define ikvsortd ikvsortd_52
#define ikvsorti ikvsorti_52
#define ikvsortii ikvsortii_52
#define imalloc imalloc_52
#define imax imax_52
#define imin imin_52
#define inorm2 inorm2_52
#define ipqCheckHeap ipqCheckHeap_52
#define ipqCreate ipqCreate_52
#define ipqDelete ipqDelete_52
#define ipqDestroy ipqDestroy_52
#define ipqFree ipqFree_52
#define ipqGetTop ipqGetTop_52
#define ipqInit ipqInit_52
#define ipqInsert ipqInsert_52
#define ipqLength ipqLength_52
#define ipqReset ipqReset_52
#define ipqSeeKey ipqSeeKey_52
#define ipqSeeTopKey ipqSeeTopKey_52
#define ipqSeeTopVal ipqSeeTopVal_52
#define ipqUpdate ipqUpdate_52
#define irand irand_52
#define irandArrayPermute irandArrayPermute_52
#define irandArrayPermuteFine irandArrayPermuteFine_52
#define irandInRange irandInRange_52
#define irealloc irealloc_52
#define iscale iscale_52
#define iset iset_52
#define iSetMatrix iSetMatrix_52
#define ismalloc ismalloc_52
#define isortd isortd_52
#define isorti isorti_52
#define isrand isrand_52
#define isum isum_52
#define rAllocMatrix rAllocMatrix_52
#define rargmax rargmax_52
#define rargmax_n rargmax_n_52
#define rargmin rargmin_52
#define raxpy raxpy_52
#define rcopy rcopy_52
#define rdot rdot_52
#define rFreeMatrix rFreeMatrix_52
#define rincset rincset_52
#define rkvAllocMatrix rkvAllocMatrix_52
#define rkvcopy rkvcopy_52
#define rkvFreeMatrix rkvFreeMatrix_52
#define rkvmalloc rkvmalloc_52
#define rkvrealloc rkvrealloc_52
#define rkvset rkvset_52
#define rkvSetMatrix rkvSetMatrix_52
#define rkvsmalloc rkvsmalloc_52
#define rkvsortd rkvsortd_52
#define rkvsorti rkvsorti_52
#define rmalloc rmalloc_52
#define rmax rmax_52
#define rmin rmin_52
#define rnorm2 rnorm2_52
#define rpqCheckHeap rpqCheckHeap_52
#define rpqCreate rpqCreate_52
#define rpqDelete rpqDelete_52
#define rpqDestroy rpqDestroy_52
#define rpqFree rpqFree_52
#define rpqGetTop rpqGetTop_52
#define rpqInit rpqInit_52
#define rpqInsert rpqInsert_52
#define rpqLength rpqLength_52
#define rpqReset rpqReset_52
#define rpqSeeKey rpqSeeKey_52
#define rpqSeeTopKey rpqSeeTopKey_52
#define rpqSeeTopVal rpqSeeTopVal_52
#define rpqUpdate rpqUpdate_52
#define rrealloc rrealloc_52
#define rscale rscale_52
#define rset rset_52
#define rSetMatrix rSetMatrix_52
#define rsmalloc rsmalloc_52
#define rsortd rsortd_52
#define rsorti rsorti_52
#define rsum rsum_52
#define uvwsorti uvwsorti_52
#define re_compile_pattern re_compile_pattern_52
#define re_set_syntax re_set_syntax_52
#define re_compile_fastmap re_compile_fastmap_52
#define regcomp regcomp_52
#define regerror regerror_52
#define regfree regfree_52
#define regexec regexec_52
#define re_match re_match_52
#define re_search re_search_52
#define re_match_2 re_match_2_52
#define re_search_2 re_search_2_52
#define re_set_registers re_set_registers_52
#define re_syntax_options re_syntax_options_52

#endif

#endif


