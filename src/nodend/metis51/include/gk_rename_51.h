/*!
\file
 *
 * This file contains aliases for GKlib 5.1 procedures
 *
 * Nick Gould, STFC-RAL, 2025-03-19, revised 2025-11-28
 *
 */


#ifndef _LIBMETIS_GK_RENAME_H_
#define _LIBMETIS_GK_RENAME_H_

/* 64-bit integer procedures */

#ifdef INTEGER_64

#define errexit errexit_51_64
#define getpathname getpathname_51_64
#define gk_AllocMatrix gk_AllocMatrix_51_64
#define gk_cmalloc gk_cmalloc_51_64
#define gk_CPUSeconds gk_CPUSeconds_51_64
#define gk_cur_jbufs gk_cur_jbufs_51_64
#define gk_dexists gk_dexists_51_64
#define gk_dkvmalloc gk_dkvmalloc_51_64
#define gk_dmalloc gk_dmalloc_51_64
#define gk_dreadfilebin gk_dreadfilebin_51_64
#define gk_errexit gk_errexit_51_64
#define gk_fclose gk_fclose_51_64
#define gk_fexists gk_fexists_51_64
#define gk_fmalloc gk_fmalloc_51_64
#define gk_fopen gk_fopen_51_64
#define gk_freadfilebin gk_freadfilebin_51_64
#define gk_free gk_free_51_64
#define gk_FreeMatrix gk_FreeMatrix_51_64
#define gk_fwritefilebin gk_fwritefilebin_51_64
#define gk_getbasename gk_getbasename_51_64
#define gk_GetCurMemoryUsed gk_GetCurMemoryUsed_51_64
#define gk_getextname gk_getextname_51_64
#define gk_getfilename gk_getfilename_51_64
#define gk_getfilestats gk_getfilestats_51_64
#define gk_getfsize gk_getfsize_51_64
#define gk_getline gk_getline_51_64
#define gk_GetMaxMemoryUsed gk_GetMaxMemoryUsed_51_64
#define gk_GetStringID gk_GetStringID_51_64
#define gk_gkmcoreAdd gk_gkmcoreAdd_51_64
#define gk_gkmcoreCreate gk_gkmcoreCreate_51_64
#define gk_gkmcoreDel gk_gkmcoreDel_51_64
#define gk_gkmcoreDestroy gk_gkmcoreDestroy_51_64
#define gk_gkmcorePop gk_gkmcorePop_51_64
#define gk_gkmcorePush gk_gkmcorePush_51_64
#define gk_i32malloc gk_i32malloc_51_64
#define gk_i32readfile gk_i32readfile_51_64
#define gk_i32readfilebin gk_i32readfilebin_51_64
#define gk_i64malloc gk_i64malloc_51_64
#define gk_i64readfile gk_i64readfile_51_64
#define gk_i64readfilebin gk_i64readfilebin_51_64
#define gk_idxsmalloc gk_idxsmalloc_51_64
#define gk_jbuf gk_jbuf_51_64
#define gk_jbufs gk_jbufs_51_64
#define gk_malloc gk_malloc_51_64
#define gk_malloc_cleanup gk_malloc_cleanup_51_64
#define gk_malloc_init gk_malloc_init_51_64
#define gk_mcoreAdd gk_mcoreAdd_51_64
#define gk_mcoreCreate gk_mcoreCreate_51_64
#define gk_mcoreDel gk_mcoreDel_51_64
#define gk_mcoreDestroy gk_mcoreDestroy_51_64
#define gk_mcoreMalloc gk_mcoreMalloc_51_64
#define gk_mcorePop gk_mcorePop_51_64
#define gk_mcorePush gk_mcorePush_51_64
#define gk_mkpath gk_mkpath_51_64
#define gk_NonLocalExit_Handler gk_NonLocalExit_Handler_51_64
#define gk_randinit gk_randinit_51_64
#define gk_randint32 gk_randint32_51_64
#define gk_randint64 gk_randint64_51_64
#define gk_readfile gk_readfile_51_64
#define gk_realloc gk_realloc_51_64
#define gk_rmpath gk_rmpath_51_64
#define gk_set_exit_on_error gk_set_exit_on_error_51_64
#define gk_SetSignalHandlers gk_SetSignalHandlers_51_64
#define gk_sigthrow gk_sigthrow_51_64
#define gk_sigtrap gk_sigtrap_51_64
#define gk_siguntrap gk_siguntrap_51_64
#define gk_str2time gk_str2time_51_64
#define gk_strcasecmp gk_strcasecmp_51_64
#define gk_strchr_replace gk_strchr_replace_51_64
#define gk_strdup gk_strdup_51_64
#define gk_strerror gk_strerror_51_64
#define gk_strhprune gk_strhprune_51_64
#define gk_strrcmp gk_strrcmp_51_64
#define gk_strstr_replace gk_strstr_replace_51_64
#define gk_strtolower gk_strtolower_51_64
#define gk_strtoupper gk_strtoupper_51_64
#define gk_strtprune gk_strtprune_51_64
#define gk_time2str gk_time2str_51_64
#define gk_UnsetSignalHandlers gk_UnsetSignalHandlers_51_64
#define gk_WClockSeconds gk_WClockSeconds_51_64
#define PrintBackTrace PrintBackTrace_51_64

/* 32-bit integer procedures */

#else

#define errexit errexit_51
#define getpathname getpathname_51
#define gk_AllocMatrix gk_AllocMatrix_51
#define gk_cmalloc gk_cmalloc_51
#define gk_CPUSeconds gk_CPUSeconds_51
#define gk_cur_jbufs gk_cur_jbufs_51
#define gk_dexists gk_dexists_51
#define gk_dkvmalloc gk_dkvmalloc_51
#define gk_dmalloc gk_dmalloc_51
#define gk_dreadfilebin gk_dreadfilebin_51
#define gk_errexit gk_errexit_51
#define gk_fclose gk_fclose_51
#define gk_fexists gk_fexists_51
#define gk_fmalloc gk_fmalloc_51
#define gk_fopen gk_fopen_51
#define gk_freadfilebin gk_freadfilebin_51
#define gk_free gk_free_51
#define gk_FreeMatrix gk_FreeMatrix_51
#define gk_fwritefilebin gk_fwritefilebin_51
#define gk_getbasename gk_getbasename_51
#define gk_GetCurMemoryUsed gk_GetCurMemoryUsed_51
#define gk_getextname gk_getextname_51
#define gk_getfilename gk_getfilename_51
#define gk_getfilestats gk_getfilestats_51
#define gk_getfsize gk_getfsize_51
#define gk_getline gk_getline_51
#define gk_GetMaxMemoryUsed gk_GetMaxMemoryUsed_51
#define gk_GetStringID gk_GetStringID_51
#define gk_gkmcoreAdd gk_gkmcoreAdd_51
#define gk_gkmcoreCreate gk_gkmcoreCreate_51
#define gk_gkmcoreDel gk_gkmcoreDel_51
#define gk_gkmcoreDestroy gk_gkmcoreDestroy_51
#define gk_gkmcorePop gk_gkmcorePop_51
#define gk_gkmcorePush gk_gkmcorePush_51
#define gk_i32malloc gk_i32malloc_51
#define gk_i32readfile gk_i32readfile_51
#define gk_i32readfilebin gk_i32readfilebin_51
#define gk_i64malloc gk_i64malloc_51
#define gk_i64readfile gk_i64readfile_51
#define gk_i64readfilebin gk_i64readfilebin_51
#define gk_idxsmalloc gk_idxsmalloc_51
#define gk_jbuf gk_jbuf_51
#define gk_jbufs gk_jbufs_51
#define gk_malloc gk_malloc_51
#define gk_malloc_cleanup gk_malloc_cleanup_51
#define gk_malloc_init gk_malloc_init_51
#define gk_mcoreAdd gk_mcoreAdd_51
#define gk_mcoreCreate gk_mcoreCreate_51
#define gk_mcoreDel gk_mcoreDel_51
#define gk_mcoreDestroy gk_mcoreDestroy_51
#define gk_mcoreMalloc gk_mcoreMalloc_51
#define gk_mcorePop gk_mcorePop_51
#define gk_mcorePush gk_mcorePush_51
#define gk_mkpath gk_mkpath_51
#define gk_NonLocalExit_Handler gk_NonLocalExit_Handler_51
#define gk_randinit gk_randinit_51
#define gk_randint32 gk_randint32_51
#define gk_randint64 gk_randint64_51
#define gk_readfile gk_readfile_51
#define gk_realloc gk_realloc_51
#define gk_rmpath gk_rmpath_51
#define gk_set_exit_on_error gk_set_exit_on_error_51
#define gk_SetSignalHandlers gk_SetSignalHandlers_51
#define gk_sigthrow gk_sigthrow_51
#define gk_sigtrap gk_sigtrap_51
#define gk_siguntrap gk_siguntrap_51
#define gk_str2time gk_str2time_51
#define gk_strcasecmp gk_strcasecmp_51
#define gk_strchr_replace gk_strchr_replace_51
#define gk_strdup gk_strdup_51
#define gk_strerror gk_strerror_51
#define gk_strhprune gk_strhprune_51
#define gk_strrcmp gk_strrcmp_51
#define gk_strstr_replace gk_strstr_replace_51
#define gk_strtolower gk_strtolower_51
#define gk_strtoupper gk_strtoupper_51
#define gk_strtprune gk_strtprune_51
#define gk_time2str gk_time2str_51
#define gk_UnsetSignalHandlers gk_UnsetSignalHandlers_51
#define gk_WClockSeconds gk_WClockSeconds_51
#define PrintBackTrace PrintBackTrace_51

#endif

#endif


