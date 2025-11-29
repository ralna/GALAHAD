/*!
\file

 *
 * This file contains aliases for GKlib 5.2 procedures
 *
 * Nick Gould, STFC-RAL, 2025-03-19, revised 2025-11-28
 *
 */

#ifndef _LIBMETIS_GK_RENAME_H_
#define _LIBMETIS_GK_RENAME_H_

/* 64-bit integer procedures */

#ifdef INTEGER_64

#define errexit errexit_52_64
#define getpathname getpathname_52_64
#define gk_AllocMatrix gk_AllocMatrix_52_64
#define gk_cmalloc gk_cmalloc_52_64
#define gk_CPUSeconds gk_CPUSeconds_52_64
#define gk_creadfilebin gk_creadfilebin_52_64
#define gk_cur_jbufs gk_cur_jbufs_52_64
#define gk_cwritefilebin gk_cwritefilebin_52_64
#define gk_dexists gk_dexists_52_64
#define gk_dkvmalloc gk_dkvmalloc_52_64
#define gk_dmalloc gk_dmalloc_52_64
#define gk_dreadfilebin gk_dreadfilebin_52_64
#define gk_dwritefilebin gk_dwritefilebin_52_64
#define gk_errexit gk_errexit_52_64
#define gk_fclose gk_fclose_52_64
#define gk_fexists gk_fexists_52_64
#define gk_fmalloc gk_fmalloc_52_64
#define gk_fopen gk_fopen_52_64
#define gk_freadfilebin gk_freadfilebin_52_64
#define gk_free gk_free_52_64
#define gk_FreeMatrix gk_FreeMatrix_52_64
#define gk_fwritefilebin gk_fwritefilebin_52_64
#define gk_getbasename gk_getbasename_52_64
#define gk_GetCurMemoryUsed gk_GetCurMemoryUsed_52_64
#define gk_getextname gk_getextname_52_64
#define gk_getfilename gk_getfilename_52_64
#define gk_getfilestats gk_getfilestats_52_64
#define gk_getfsize gk_getfsize_52_64
#define gk_getline gk_getline_52_64
#define gk_GetMaxMemoryUsed gk_GetMaxMemoryUsed_52_64
#define gk_GetProcVmPeak gk_GetProcVmPeak_52_64
#define gk_GetStringID gk_GetStringID_52_64
#define gk_GetVMInfo gk_GetVMInfo_52_64
#define gk_gkmcoreAdd gk_gkmcoreAdd_52_64
#define gk_gkmcoreCreate gk_gkmcoreCreate_52_64
#define gk_gkmcoreDel gk_gkmcoreDel_52_64
#define gk_gkmcoreDestroy gk_gkmcoreDestroy_52_64
#define gk_gkmcorePop gk_gkmcorePop_52_64
#define gk_gkmcorePush gk_gkmcorePush_52_64
#define gk_i32malloc gk_i32malloc_52_64
#define gk_i32readfile gk_i32readfile_52_64
#define gk_i32readfilebin gk_i32readfilebin_52_64
#define gk_i32writefilebin gk_i32writefilebin_52_64
#define gk_i64malloc gk_i64malloc_52_64
#define gk_i64readfile gk_i64readfile_52_64
#define gk_i64readfilebin gk_i64readfilebin_52_64
#define gk_i64writefilebin gk_i64writefilebin_52_64
#define gk_idxset gk_idxset_52_64
#define gk_idxsmalloc gk_idxsmalloc_52_64
#define gk_jbuf gk_jbuf_52_64
#define gk_jbufs gk_jbufs_52_64
#define gk_malloc gk_malloc_52_64
#define gk_malloc_cleanup gk_malloc_cleanup_52_64
#define gk_malloc_init gk_malloc_init_52_64
#define gk_mcoreAdd gk_mcoreAdd_52_64
#define gk_mcoreCreate gk_mcoreCreate_52_64
#define gk_mcoreDel gk_mcoreDel_52_64
#define gk_mcoreDestroy gk_mcoreDestroy_52_64
#define gk_mcoreMalloc gk_mcoreMalloc_52_64
#define gk_mcorePop gk_mcorePop_52_64
#define gk_mcorePush gk_mcorePush_52_64
#define gk_mkpath gk_mkpath_52_64
#define gk_NonLocalExit_Handler gk_NonLocalExit_Handler_52_64
#define gk_randinit gk_randinit_52_64
#define gk_randint32 gk_randint32_52_64
#define gk_randint64 gk_randint64_52_64
#define gk_readfile gk_readfile_52_64
#define gk_realloc gk_realloc_52_64
#define gk_rmpath gk_rmpath_52_64
#define gk_set_exit_on_error gk_set_exit_on_error_52_64
#define gk_SetSignalHandlers gk_SetSignalHandlers_52_64
#define gk_sigthrow gk_sigthrow_52_64
#define gk_sigtrap gk_sigtrap_52_64
#define gk_siguntrap gk_siguntrap_52_64
#define gk_str2time gk_str2time_52_64
#define gk_strcasecmp gk_strcasecmp_52_64
#define gk_strchr_replace gk_strchr_replace_52_64
#define gk_strdup gk_strdup_52_64
#define gk_strerror gk_strerror_52_64
#define gk_strhprune gk_strhprune_52_64
#define gk_strrcmp gk_strrcmp_52_64
#define gk_strstr_replace gk_strstr_replace_52_64
#define gk_strtolower gk_strtolower_52_64
#define gk_strtoupper gk_strtoupper_52_64
#define gk_strtprune gk_strtprune_52_64
#define gk_time2str gk_time2str_52_64
#define gk_UnsetSignalHandlers gk_UnsetSignalHandlers_52_64
#define gk_WClockSeconds gk_WClockSeconds_52_64
#define gk_zmalloc gk_zmalloc_52_64
#define gk_zreadfile gk_zreadfile_52_64
#define gk_zreadfilebin gk_zreadfilebin_52_64
#define gk_zwritefilebin gk_zwritefilebin_52_64
#define PrintBackTrace PrintBackTrace_52_64

/* 32-bit integer procedures */

#else

#define errexit errexit_52
#define getpathname getpathname_52
#define gk_AllocMatrix gk_AllocMatrix_52
#define gk_cmalloc gk_cmalloc_52
#define gk_CPUSeconds gk_CPUSeconds_52
#define gk_creadfilebin gk_creadfilebin_52
#define gk_cur_jbufs gk_cur_jbufs_52
#define gk_cwritefilebin gk_cwritefilebin_52
#define gk_dexists gk_dexists_52
#define gk_dkvmalloc gk_dkvmalloc_52
#define gk_dmalloc gk_dmalloc_52
#define gk_dreadfilebin gk_dreadfilebin_52
#define gk_dwritefilebin gk_dwritefilebin_52
#define gk_errexit gk_errexit_52
#define gk_fclose gk_fclose_52
#define gk_fexists gk_fexists_52
#define gk_fmalloc gk_fmalloc_52
#define gk_fopen gk_fopen_52
#define gk_freadfilebin gk_freadfilebin_52
#define gk_free gk_free_52
#define gk_FreeMatrix gk_FreeMatrix_52
#define gk_fwritefilebin gk_fwritefilebin_52
#define gk_getbasename gk_getbasename_52
#define gk_GetCurMemoryUsed gk_GetCurMemoryUsed_52
#define gk_getextname gk_getextname_52
#define gk_getfilename gk_getfilename_52
#define gk_getfilestats gk_getfilestats_52
#define gk_getfsize gk_getfsize_52
#define gk_getline gk_getline_52
#define gk_GetMaxMemoryUsed gk_GetMaxMemoryUsed_52
#define gk_GetProcVmPeak gk_GetProcVmPeak_52
#define gk_GetStringID gk_GetStringID_52
#define gk_GetVMInfo gk_GetVMInfo_52
#define gk_gkmcoreAdd gk_gkmcoreAdd_52
#define gk_gkmcoreCreate gk_gkmcoreCreate_52
#define gk_gkmcoreDel gk_gkmcoreDel_52
#define gk_gkmcoreDestroy gk_gkmcoreDestroy_52
#define gk_gkmcorePop gk_gkmcorePop_52
#define gk_gkmcorePush gk_gkmcorePush_52
#define gk_i32malloc gk_i32malloc_52
#define gk_i32readfile gk_i32readfile_52
#define gk_i32readfilebin gk_i32readfilebin_52
#define gk_i32writefilebin gk_i32writefilebin_52
#define gk_i64malloc gk_i64malloc_52
#define gk_i64readfile gk_i64readfile_52
#define gk_i64readfilebin gk_i64readfilebin_52
#define gk_i64writefilebin gk_i64writefilebin_52
#define gk_idxset gk_idxset_52
#define gk_idxsmalloc gk_idxsmalloc_52
#define gk_jbuf gk_jbuf_52
#define gk_jbufs gk_jbufs_52
#define gk_malloc gk_malloc_52
#define gk_malloc_cleanup gk_malloc_cleanup_52
#define gk_malloc_init gk_malloc_init_52
#define gk_mcoreAdd gk_mcoreAdd_52
#define gk_mcoreCreate gk_mcoreCreate_52
#define gk_mcoreDel gk_mcoreDel_52
#define gk_mcoreDestroy gk_mcoreDestroy_52
#define gk_mcoreMalloc gk_mcoreMalloc_52
#define gk_mcorePop gk_mcorePop_52
#define gk_mcorePush gk_mcorePush_52
#define gk_mkpath gk_mkpath_52
#define gk_NonLocalExit_Handler gk_NonLocalExit_Handler_52
#define gk_randinit gk_randinit_52
#define gk_randint32 gk_randint32_52
#define gk_randint64 gk_randint64_52
#define gk_readfile gk_readfile_52
#define gk_realloc gk_realloc_52
#define gk_rmpath gk_rmpath_52
#define gk_set_exit_on_error gk_set_exit_on_error_52
#define gk_SetSignalHandlers gk_SetSignalHandlers_52
#define gk_sigthrow gk_sigthrow_52
#define gk_sigtrap gk_sigtrap_52
#define gk_siguntrap gk_siguntrap_52
#define gk_str2time gk_str2time_52
#define gk_strcasecmp gk_strcasecmp_52
#define gk_strchr_replace gk_strchr_replace_52
#define gk_strdup gk_strdup_52
#define gk_strerror gk_strerror_52
#define gk_strhprune gk_strhprune_52
#define gk_strrcmp gk_strrcmp_52
#define gk_strstr_replace gk_strstr_replace_52
#define gk_strtolower gk_strtolower_52
#define gk_strtoupper gk_strtoupper_52
#define gk_strtprune gk_strtprune_52
#define gk_time2str gk_time2str_52
#define gk_UnsetSignalHandlers gk_UnsetSignalHandlers_52
#define gk_WClockSeconds gk_WClockSeconds_52
#define gk_zmalloc gk_zmalloc_52
#define gk_zreadfile gk_zreadfile_52
#define gk_zreadfilebin gk_zreadfilebin_52
#define gk_zwritefilebin gk_zwritefilebin_52
#define PrintBackTrace PrintBackTrace_52

#endif

#endif


