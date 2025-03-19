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

#define iAllocMatrix iAllocMatrix_51_64
#define iargmax iargmax_51_64
#define iargmax_n iargmax_n_51_64
#define iargmin iargmin_51_64
#define iarray2csr iarray2csr_51_64
#define iaxpy iaxpy_51_64
#define icopy icopy_51_64
#define idot idot_51_64
#define iFreeMatrix iFreeMatrix_51_64
#define iincset iincset_51_64
#define ikvAllocMatrix ikvAllocMatrix_51_64
#define ikvcopy ikvcopy_51_64
#define ikvFreeMatrix ikvFreeMatrix_51_64
#define ikvmalloc ikvmalloc_51_64
#define ikvrealloc ikvrealloc_51_64
#define ikvset ikvset_51_64
#define ikvSetMatrix ikvSetMatrix_51_64
#define ikvsmalloc ikvsmalloc_51_64
#define ikvsortd ikvsortd_51_64
#define ikvsorti ikvsorti_51_64
#define ikvsortii ikvsortii_51_64
#define imalloc imalloc_51_64
#define imax imax_51_64
#define imin imin_51_64
#define inorm2 inorm2_51_64
#define ipqCheckHeap ipqCheckHeap_51_64
#define ipqCreate ipqCreate_51_64
#define ipqDelete ipqDelete_51_64
#define ipqDestroy ipqDestroy_51_64
#define ipqFree ipqFree_51_64
#define ipqGetTop ipqGetTop_51_64
#define ipqInit ipqInit_51_64
#define ipqInsert ipqInsert_51_64
#define ipqLength ipqLength_51_64
#define ipqReset ipqReset_51_64
#define ipqSeeKey ipqSeeKey_51_64
#define ipqSeeTopKey ipqSeeTopKey_51_64
#define ipqSeeTopVal ipqSeeTopVal_51_64
#define ipqUpdate ipqUpdate_51_64
#define irand irand_51_64
#define irandArrayPermute irandArrayPermute_51_64
#define irandArrayPermuteFine irandArrayPermuteFine_51_64
#define irandInRange irandInRange_51_64
#define irealloc irealloc_51_64
#define iscale iscale_51_64
#define iset iset_51_64
#define iSetMatrix iSetMatrix_51_64
#define ismalloc ismalloc_51_64
#define isortd isortd_51_64
#define isorti isorti_51_64
#define isrand isrand_51_64
#define isum isum_51_64
#define rAllocMatrix rAllocMatrix_51_64
#define rargmax rargmax_51_64
#define rargmax_n rargmax_n_51_64
#define rargmin rargmin_51_64
#define raxpy raxpy_51_64
#define rcopy rcopy_51_64
#define rdot rdot_51_64
#define rFreeMatrix rFreeMatrix_51_64
#define rincset rincset_51_64
#define rkvAllocMatrix rkvAllocMatrix_51_64
#define rkvcopy rkvcopy_51_64
#define rkvFreeMatrix rkvFreeMatrix_51_64
#define rkvmalloc rkvmalloc_51_64
#define rkvrealloc rkvrealloc_51_64
#define rkvset rkvset_51_64
#define rkvSetMatrix rkvSetMatrix_51_64
#define rkvsmalloc rkvsmalloc_51_64
#define rkvsortd rkvsortd_51_64
#define rkvsorti rkvsorti_51_64
#define rmalloc rmalloc_51_64
#define rmax rmax_51_64
#define rmin rmin_51_64
#define rnorm2 rnorm2_51_64
#define rpqCheckHeap rpqCheckHeap_51_64
#define rpqCreate rpqCreate_51_64
#define rpqDelete rpqDelete_51_64
#define rpqDestroy rpqDestroy_51_64
#define rpqFree rpqFree_51_64
#define rpqGetTop rpqGetTop_51_64
#define rpqInit rpqInit_51_64
#define rpqInsert rpqInsert_51_64
#define rpqLength rpqLength_51_64
#define rpqReset rpqReset_51_64
#define rpqSeeKey rpqSeeKey_51_64
#define rpqSeeTopKey rpqSeeTopKey_51_64
#define rpqSeeTopVal rpqSeeTopVal_51_64
#define rpqUpdate rpqUpdate_51_64
#define rrealloc rrealloc_51_64
#define rscale rscale_51_64
#define rset rset_51_64
#define rSetMatrix rSetMatrix_51_64
#define rsmalloc rsmalloc_51_64
#define rsortd rsortd_51_64
#define rsorti rsorti_51_64
#define rsum rsum_51_64
#define uvwsorti uvwsorti_51_64

/* 32-bit integer procedures */

#else

#define iAllocMatrix iAllocMatrix_51
#define iargmax iargmax_51
#define iargmax_n iargmax_n_51
#define iargmin iargmin_51
#define iarray2csr iarray2csr_51
#define iaxpy iaxpy_51
#define icopy icopy_51
#define idot idot_51
#define iFreeMatrix iFreeMatrix_51
#define iincset iincset_51
#define ikvAllocMatrix ikvAllocMatrix_51
#define ikvcopy ikvcopy_51
#define ikvFreeMatrix ikvFreeMatrix_51
#define ikvmalloc ikvmalloc_51
#define ikvrealloc ikvrealloc_51
#define ikvset ikvset_51
#define ikvSetMatrix ikvSetMatrix_51
#define ikvsmalloc ikvsmalloc_51
#define ikvsortd ikvsortd_51
#define ikvsorti ikvsorti_51
#define ikvsortii ikvsortii_51
#define imalloc imalloc_51
#define imax imax_51
#define imin imin_51
#define inorm2 inorm2_51
#define ipqCheckHeap ipqCheckHeap_51
#define ipqCreate ipqCreate_51
#define ipqDelete ipqDelete_51
#define ipqDestroy ipqDestroy_51
#define ipqFree ipqFree_51
#define ipqGetTop ipqGetTop_51
#define ipqInit ipqInit_51
#define ipqInsert ipqInsert_51
#define ipqLength ipqLength_51
#define ipqReset ipqReset_51
#define ipqSeeKey ipqSeeKey_51
#define ipqSeeTopKey ipqSeeTopKey_51
#define ipqSeeTopVal ipqSeeTopVal_51
#define ipqUpdate ipqUpdate_51
#define irand irand_51
#define irandArrayPermute irandArrayPermute_51
#define irandArrayPermuteFine irandArrayPermuteFine_51
#define irandInRange irandInRange_51
#define irealloc irealloc_51
#define iscale iscale_51
#define iset iset_51
#define iSetMatrix iSetMatrix_51
#define ismalloc ismalloc_51
#define isortd isortd_51
#define isorti isorti_51
#define isrand isrand_51
#define isum isum_51
#define rAllocMatrix rAllocMatrix_51
#define rargmax rargmax_51
#define rargmax_n rargmax_n_51
#define rargmin rargmin_51
#define raxpy raxpy_51
#define rcopy rcopy_51
#define rdot rdot_51
#define rFreeMatrix rFreeMatrix_51
#define rincset rincset_51
#define rkvAllocMatrix rkvAllocMatrix_51
#define rkvcopy rkvcopy_51
#define rkvFreeMatrix rkvFreeMatrix_51
#define rkvmalloc rkvmalloc_51
#define rkvrealloc rkvrealloc_51
#define rkvset rkvset_51
#define rkvSetMatrix rkvSetMatrix_51
#define rkvsmalloc rkvsmalloc_51
#define rkvsortd rkvsortd_51
#define rkvsorti rkvsorti_51
#define rmalloc rmalloc_51
#define rmax rmax_51
#define rmin rmin_51
#define rnorm2 rnorm2_51
#define rpqCheckHeap rpqCheckHeap_51
#define rpqCreate rpqCreate_51
#define rpqDelete rpqDelete_51
#define rpqDestroy rpqDestroy_51
#define rpqFree rpqFree_51
#define rpqGetTop rpqGetTop_51
#define rpqInit rpqInit_51
#define rpqInsert rpqInsert_51
#define rpqLength rpqLength_51
#define rpqReset rpqReset_51
#define rpqSeeKey rpqSeeKey_51
#define rpqSeeTopKey rpqSeeTopKey_51
#define rpqSeeTopVal rpqSeeTopVal_51
#define rpqUpdate rpqUpdate_51
#define rrealloc rrealloc_51
#define rscale rscale_51
#define rset rset_51
#define rSetMatrix rSetMatrix_51
#define rsmalloc rsmalloc_51
#define rsortd rsortd_51
#define rsorti rsorti_51
#define rsum rsum_51
#define uvwsorti uvwsorti_51

#endif

#endif


