/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * modified by Nick Gould, STFC-RAL, 2025-03-0 to remove non metis_nd 
 * components, _4 suffix and 64-bit integer-support added to all procedures
 *
 * $Id: rename.h,v 1.1 1998/11/27 17:59:29 karypis Exp $
 *
 */

#ifndef _LIBMETIS_RENAME_H_
#define _LIBMETIS_RENAME_H_

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4_64

/* balance.c */
#define Balance2Way			Balance2Way_4_64
#define Bnd2WayBalance			Bnd2WayBalance_4_64
#define General2WayBalance		General2WayBalance_4_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4_64

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4_64
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4_64
#define SetUpCoarseGraph		SetUpCoarseGraph_4_64
#define ReAdjustMemory			ReAdjustMemory_4_64

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4_64

/* compress.c */
#define CompressGraph			CompressGraph_4_64
#define PruneGraph			PruneGraph_4_64

/* debug.c */
#define ComputeCut			ComputeCut_4_64
#define CheckBnd			CheckBnd_4_64
#define CheckNodeBnd			CheckNodeBnd_4_64
#define CheckNodePartitionParams	CheckNodePartitionParams_4_64
#define IsSeparable			IsSeparable_4_64

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4_64
#define Change2FNumbering		Change2FNumbering_4_64
#define Change2FNumbering2		Change2FNumbering2_4_64
#define Change2FNumberingOrder		Change2FNumberingOrder_4_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4_64

/* graph.c */
#define SetUpGraph			SetUpGraph_4_64
#define SetUpGraphKway 			SetUpGraphKway_4_64
#define SetUpGraph2			SetUpGraph2_4_64
#define VolSetUpGraph			VolSetUpGraph_4_64
#define RandomizeGraph			RandomizeGraph_4_64
#define IsConnectedSubdomain		IsConnectedSubdomain_4_64
#define IsConnected			IsConnected_4_64
#define IsConnected2			IsConnected2_4_64
#define FindComponents			FindComponents_4_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4_64
#define InitSeparator			InitSeparator_4_64
#define GrowBisection			GrowBisection_4_64
#define GrowBisectionNode		GrowBisectionNode_4_64
#define RandomBisection			RandomBisection_4_64

/* match.c */
#define Match_RM			Match_RM_4_64
#define Match_RM_NVW			Match_RM_NVW_4_64
#define Match_HEM			Match_HEM_4_64
#define Match_SHEM			Match_SHEM_4_64

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4_64
#define FreeWorkSpace			FreeWorkSpace_4_64
#define WspaceAvail			WspaceAvail_4_64
#define idxwspacemalloc			idxwspacemalloc_4_64
#define idxwspacefree			idxwspacefree_4_64
#define fwspacemalloc			fwspacemalloc_4_64
#define CreateGraph			CreateGraph_4_64
#define InitGraph			InitGraph_4_64
#define FreeGraph			FreeGraph_4_64

/* mincover.c */
#define MinCover			MinCover_4_64
#define MinCover_Augment		MinCover_Augment_4_64
#define MinCover_Decompose		MinCover_Decompose_4_64
#define MinCover_ColDFS			MinCover_ColDFS_4_64
#define MinCover_RowDFS			MinCover_RowDFS_4_64

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4_64
#define MCMatch_HEM			MCMatch_HEM_4_64
#define MCMatch_SHEM			MCMatch_SHEM_4_64
#define MCMatch_SHEBM			MCMatch_SHEBM_4_64
#define MCMatch_SBHEM			MCMatch_SBHEM_4_64
#define BetterVBalance			BetterVBalance_4_64
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4_64

/* mmd.c */
#define genmmd				genmmd_4_64
#define mmdelm				mmdelm_4_64
#define mmdint				mmdint_4_64
#define mmdnum				mmdnum_4_64
#define mmdupd				mmdupd_4_64

/* myqsort.c */
#define iidxsort			iidxsort_4_64
#define iintsort			iintsort_4_64
#define ikeysort			ikeysort_4_64
#define ikeyvalsort			ikeyvalsort_4_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4_64
#define MlevelNodeBisection		MlevelNodeBisection_4_64
#define SplitGraphOrder			SplitGraphOrder_4_64
#define MMDOrder			MMDOrder_4_64
#define SplitGraphOrderCC		SplitGraphOrderCC_4_64

/* pqueue.c */
#define PQueueInit			PQueueInit_4_64
#define PQueueReset			PQueueReset_4_64
#define PQueueFree			PQueueFree_4_64
#define PQueueInsert			PQueueInsert_4_64
#define PQueueDelete			PQueueDelete_4_64
#define PQueueUpdate			PQueueUpdate_4_64
#define PQueueUpdateUp			PQueueUpdateUp_4_64
#define PQueueGetMax			PQueueGetMax_4_64
#define PQueueSeeMax			PQueueSeeMax_4_64
#define CheckHeap			CheckHeap_4_64


/* refine.c */
#define Refine2Way			Refine2Way_4_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4_64
#define Project2WayPartition		Project2WayPartition_4_64


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4_64
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4_64


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4_64
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4_64
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4_64
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4_64
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4_64


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4_64
#define Project2WayNodePartition	Project2WayNodePartition_4_64

/* timing.c */
#define InitTimers			InitTimers_4_64
#define PrintTimers			PrintTimers_4_64
#define seconds				seconds_4_64

/* util.c */
#define errexit				errexit_4_64
#define GK_free				GK_free_4_64
#ifndef DMALLOC
#define imalloc				imalloc_4_64
#define idxmalloc			idxmalloc_4_64
#define fmalloc				fmalloc_4_64
#define ismalloc			ismalloc_4_64
#define idxsmalloc			idxsmalloc_4_64
#define GKmalloc			GKmalloc_4_64
#endif
#define iset				iset_4_64
#define idxset				idxset_4_64
#define sset				sset_4_64
#define iamax				iamax_4_64
#define idxamax				idxamax_4_64
#define idxamax_strd			idxamax_strd_4_64
#define samax				samax_4_64
#define samax2				samax2_4_64
#define idxamin				idxamin_4_64
#define samin				samin_4_64
#define idxsum				idxsum_4_64
#define idxsum_strd			idxsum_strd_4_64
#define idxadd				idxadd_4_64
#define charsum				charsum_4_64
#define isum				isum_4_64
#define ssum				ssum_4_64
#define ssum_strd			ssum_strd_4_64
#define sscale				sscale_4_64
#define snorm2				snorm2_4_64
#define sdot				sdot_4_64
#define saxpy				saxpy_4_64
#define RandomPermute			RandomPermute_4_64
#define ispow2				ispow2_4_64
#define InitRandom			InitRandom_4_64
#define ilog2				ilog2_4_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4

/* balance.c */
#define Balance2Way			Balance2Way_4
#define Bnd2WayBalance			Bnd2WayBalance_4
#define General2WayBalance		General2WayBalance_4

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4
#define SetUpCoarseGraph		SetUpCoarseGraph_4
#define ReAdjustMemory			ReAdjustMemory_4

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4

/* compress.c */
#define CompressGraph			CompressGraph_4
#define PruneGraph			PruneGraph_4

/* debug.c */
#define ComputeCut			ComputeCut_4
#define CheckBnd			CheckBnd_4
#define CheckNodeBnd			CheckNodeBnd_4
#define CheckNodePartitionParams	CheckNodePartitionParams_4
#define IsSeparable			IsSeparable_4

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4
#define Change2FNumbering		Change2FNumbering_4
#define Change2FNumbering2		Change2FNumbering2_4
#define Change2FNumberingOrder		Change2FNumberingOrder_4
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4

/* graph.c */
#define SetUpGraph			SetUpGraph_4
#define SetUpGraphKway 			SetUpGraphKway_4
#define SetUpGraph2			SetUpGraph2_4
#define VolSetUpGraph			VolSetUpGraph_4
#define RandomizeGraph			RandomizeGraph_4
#define IsConnectedSubdomain		IsConnectedSubdomain_4
#define IsConnected			IsConnected_4
#define IsConnected2			IsConnected2_4
#define FindComponents			FindComponents_4

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4
#define InitSeparator			InitSeparator_4
#define GrowBisection			GrowBisection_4
#define GrowBisectionNode		GrowBisectionNode_4
#define RandomBisection			RandomBisection_4

/* match.c */
#define Match_RM			Match_RM_4
#define Match_RM_NVW			Match_RM_NVW_4
#define Match_HEM			Match_HEM_4
#define Match_SHEM			Match_SHEM_4

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4
#define FreeWorkSpace			FreeWorkSpace_4
#define WspaceAvail			WspaceAvail_4
#define idxwspacemalloc			idxwspacemalloc_4
#define idxwspacefree			idxwspacefree_4
#define fwspacemalloc			fwspacemalloc_4
#define CreateGraph			CreateGraph_4
#define InitGraph			InitGraph_4
#define FreeGraph			FreeGraph_4

/* mincover.c */
#define MinCover			MinCover_4
#define MinCover_Augment		MinCover_Augment_4
#define MinCover_Decompose		MinCover_Decompose_4
#define MinCover_ColDFS			MinCover_ColDFS_4
#define MinCover_RowDFS			MinCover_RowDFS_4

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4
#define MCMatch_HEM			MCMatch_HEM_4
#define MCMatch_SHEM			MCMatch_SHEM_4
#define MCMatch_SHEBM			MCMatch_SHEBM_4
#define MCMatch_SBHEM			MCMatch_SBHEM_4
#define BetterVBalance			BetterVBalance_4
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4

/* mmd.c */
#define genmmd				genmmd_4
#define mmdelm				mmdelm_4
#define mmdint				mmdint_4
#define mmdnum				mmdnum_4
#define mmdupd				mmdupd_4

/* myqsort.c */
#define iidxsort			iidxsort_4
#define iintsort			iintsort_4
#define ikeysort			ikeysort_4
#define ikeyvalsort			ikeyvalsort_4

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4
#define MlevelNodeBisection		MlevelNodeBisection_4
#define SplitGraphOrder			SplitGraphOrder_4
#define MMDOrder			MMDOrder_4
#define SplitGraphOrderCC		SplitGraphOrderCC_4

/* pqueue.c */
#define PQueueInit			PQueueInit_4
#define PQueueReset			PQueueReset_4
#define PQueueFree			PQueueFree_4
#define PQueueInsert			PQueueInsert_4
#define PQueueDelete			PQueueDelete_4
#define PQueueUpdate			PQueueUpdate_4
#define PQueueUpdateUp			PQueueUpdateUp_4
#define PQueueGetMax			PQueueGetMax_4
#define PQueueSeeMax			PQueueSeeMax_4
#define CheckHeap			CheckHeap_4


/* refine.c */
#define Refine2Way			Refine2Way_4
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4
#define Project2WayPartition		Project2WayPartition_4


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4
#define Project2WayNodePartition	Project2WayNodePartition_4

/* timing.c */
#define InitTimers			InitTimers_4
#define PrintTimers			PrintTimers_4
#define seconds				seconds_4

/* util.c */
#define errexit				errexit_4
#define GK_free				GK_free_4
#ifndef DMALLOC
#define imalloc				imalloc_4
#define idxmalloc			idxmalloc_4
#define fmalloc				fmalloc_4
#define ismalloc			ismalloc_4
#define idxsmalloc			idxsmalloc_4
#define GKmalloc			GKmalloc_4
#endif
#define iset				iset_4
#define idxset				idxset_4
#define sset				sset_4
#define iamax				iamax_4
#define idxamax				idxamax_4
#define idxamax_strd			idxamax_strd_4
#define samax				samax_4
#define samax2				samax2_4
#define idxamin				idxamin_4
#define samin				samin_4
#define idxsum				idxsum_4
#define idxsum_strd			idxsum_strd_4
#define idxadd				idxadd_4
#define charsum				charsum_4
#define isum				isum_4
#define ssum				ssum_4
#define ssum_strd			ssum_strd_4
#define sscale				sscale_4
#define snorm2				snorm2_4
#define sdot				sdot_4
#define saxpy				saxpy_4
#define RandomPermute			RandomPermute_4
#define ispow2				ispow2_4
#define InitRandom			InitRandom_4
#define ilog2				ilog2_4


#endif
#endif


