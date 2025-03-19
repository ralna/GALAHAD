/* double precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4d_64

/* balance.c */
#define Balance2Way			Balance2Way_4d_64
#define Bnd2WayBalance			Bnd2WayBalance_4d_64
#define General2WayBalance		General2WayBalance_4d_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4d_64

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4d_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4d_64
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4d_64
#define SetUpCoarseGraph		SetUpCoarseGraph_4d_64
#define ReAdjustMemory			ReAdjustMemory_4d_64

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4d_64

/* compress.c */
#define CompressGraph			CompressGraph_4d_64
#define PruneGraph			PruneGraph_4d_64

/* debug.c */
#define ComputeCut			ComputeCut_4d_64
#define CheckBnd			CheckBnd_4d_64
#define CheckNodeBnd			CheckNodeBnd_4d_64
#define CheckNodePartitionParams	CheckNodePartitionParams_4d_64
#define IsSeparable			IsSeparable_4d_64

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4d_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4d_64
#define Change2FNumbering		Change2FNumbering_4d_64
#define Change2FNumbering2		Change2FNumbering2_4d_64
#define Change2FNumberingOrder		Change2FNumberingOrder_4d_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4d_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4d_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4d_64

/* graph.c */
#define SetUpGraph			SetUpGraph_4d_64
#define SetUpGraphKway 			SetUpGraphKway_4d_64
#define SetUpGraph2			SetUpGraph2_4d_64
#define VolSetUpGraph			VolSetUpGraph_4d_64
#define RandomizeGraph			RandomizeGraph_4d_64
#define IsConnectedSubdomain		IsConnectedSubdomain_4d_64
#define IsConnected			IsConnected_4d_64
#define IsConnected2			IsConnected2_4d_64
#define FindComponents			FindComponents_4d_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4d_64
#define InitSeparator			InitSeparator_4d_64
#define GrowBisection			GrowBisection_4d_64
#define GrowBisectionNode		GrowBisectionNode_4d_64
#define RandomBisection			RandomBisection_4d_64

/* match.c */
#define Match_RM			Match_RM_4d_64
#define Match_RM_NVW			Match_RM_NVW_4d_64
#define Match_HEM			Match_HEM_4d_64
#define Match_SHEM			Match_SHEM_4d_64

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4d_64
#define FreeWorkSpace			FreeWorkSpace_4d_64
#define WspaceAvail			WspaceAvail_4d_64
#define idxwspacemalloc			idxwspacemalloc_4d_64
#define idxwspacefree			idxwspacefree_4d_64
#define fwspacemalloc			fwspacemalloc_4d_64
#define CreateGraph			CreateGraph_4d_64
#define InitGraph			InitGraph_4d_64
#define FreeGraph			FreeGraph_4d_64

/* mincover.c */
#define MinCover			MinCover_4d_64
#define MinCover_Augment		MinCover_Augment_4d_64
#define MinCover_Decompose		MinCover_Decompose_4d_64
#define MinCover_ColDFS			MinCover_ColDFS_4d_64
#define MinCover_RowDFS			MinCover_RowDFS_4d_64

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4d_64
#define MCMatch_HEM			MCMatch_HEM_4d_64
#define MCMatch_SHEM			MCMatch_SHEM_4d_64
#define MCMatch_SHEBM			MCMatch_SHEBM_4d_64
#define MCMatch_SBHEM			MCMatch_SBHEM_4d_64
#define BetterVBalance			BetterVBalance_4d_64
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4d_64

/* mmd.c */
#define genmmd				genmmd_4d_64
#define mmdelm				mmdelm_4d_64
#define mmdint				mmdint_4d_64
#define mmdnum				mmdnum_4d_64
#define mmdupd				mmdupd_4d_64

/* myqsort.c */
#define iidxsort			iidxsort_4d_64
#define iintsort			iintsort_4d_64
#define ikeysort			ikeysort_4d_64
#define ikeyvalsort			ikeyvalsort_4d_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4d_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4d_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4d_64
#define MlevelNodeBisection		MlevelNodeBisection_4d_64
#define SplitGraphOrder			SplitGraphOrder_4d_64
#define MMDOrder			MMDOrder_4d_64
#define SplitGraphOrderCC		SplitGraphOrderCC_4d_64

/* pqueue.c */
#define PQueueInit			PQueueInit_4d_64
#define PQueueReset			PQueueReset_4d_64
#define PQueueFree			PQueueFree_4d_64
#define PQueueInsert			PQueueInsert_4d_64
#define PQueueDelete			PQueueDelete_4d_64
#define PQueueUpdate			PQueueUpdate_4d_64
#define PQueueUpdateUp			PQueueUpdateUp_4d_64
#define PQueueGetMax			PQueueGetMax_4d_64
#define PQueueSeeMax			PQueueSeeMax_4d_64
#define CheckHeap			CheckHeap_4d_64


/* refine.c */
#define Refine2Way			Refine2Way_4d_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4d_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4d_64
#define Project2WayPartition		Project2WayPartition_4d_64


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4d_64
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4d_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4d_64


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4d_64
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4d_64
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4d_64
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4d_64
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4d_64


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4d_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4d_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4d_64
#define Project2WayNodePartition	Project2WayNodePartition_4d_64

/* timing.c */
#define InitTimers			InitTimers_4d_64
#define PrintTimers			PrintTimers_4d_64
#define seconds				seconds_4d_64

/* util.c */
#define errexit				errexit_4d_64
#define GK_free				GK_free_4d_64
#ifndef DMALLOC
#define imalloc				imalloc_4d_64
#define idxmalloc			idxmalloc_4d_64
#define fmalloc				fmalloc_4d_64
#define ismalloc			ismalloc_4d_64
#define idxsmalloc			idxsmalloc_4d_64
#define GKmalloc			GKmalloc_4d_64
#endif
#define iset				iset_4d_64
#define idxset				idxset_4d_64
#define sset				sset_4d_64
#define iamax				iamax_4d_64
#define idxamax				idxamax_4d_64
#define idxamax_strd			idxamax_strd_4d_64
#define samax				samax_4d_64
#define samax2				samax2_4d_64
#define idxamin				idxamin_4d_64
#define samin				samin_4d_64
#define idxsum				idxsum_4d_64
#define idxsum_strd			idxsum_strd_4d_64
#define idxadd				idxadd_4d_64
#define charsum				charsum_4d_64
#define isum				isum_4d_64
#define ssum				ssum_4d_64
#define ssum_strd			ssum_strd_4d_64
#define sscale				sscale_4d_64
#define snorm2				snorm2_4d_64
#define sdot				sdot_4d_64
#define saxpy				saxpy_4d_64
#define RandomPermute			RandomPermute_4d_64
#define ispow2				ispow2_4d_64
#define InitRandom			InitRandom_4d_64
#define ilog2				ilog2_4d_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4d

/* balance.c */
#define Balance2Way			Balance2Way_4d
#define Bnd2WayBalance			Bnd2WayBalance_4d
#define General2WayBalance		General2WayBalance_4d

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4d

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4d
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4d
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4d
#define SetUpCoarseGraph		SetUpCoarseGraph_4d
#define ReAdjustMemory			ReAdjustMemory_4d

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4d

/* compress.c */
#define CompressGraph			CompressGraph_4d
#define PruneGraph			PruneGraph_4d

/* debug.c */
#define ComputeCut			ComputeCut_4d
#define CheckBnd			CheckBnd_4d
#define CheckNodeBnd			CheckNodeBnd_4d
#define CheckNodePartitionParams	CheckNodePartitionParams_4d
#define IsSeparable			IsSeparable_4d

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4d

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4d
#define Change2FNumbering		Change2FNumbering_4d
#define Change2FNumbering2		Change2FNumbering2_4d
#define Change2FNumberingOrder		Change2FNumberingOrder_4d
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4d
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4d
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4d

/* graph.c */
#define SetUpGraph			SetUpGraph_4d
#define SetUpGraphKway 			SetUpGraphKway_4d
#define SetUpGraph2			SetUpGraph2_4d
#define VolSetUpGraph			VolSetUpGraph_4d
#define RandomizeGraph			RandomizeGraph_4d
#define IsConnectedSubdomain		IsConnectedSubdomain_4d
#define IsConnected			IsConnected_4d
#define IsConnected2			IsConnected2_4d
#define FindComponents			FindComponents_4d

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4d
#define InitSeparator			InitSeparator_4d
#define GrowBisection			GrowBisection_4d
#define GrowBisectionNode		GrowBisectionNode_4d
#define RandomBisection			RandomBisection_4d

/* match.c */
#define Match_RM			Match_RM_4d
#define Match_RM_NVW			Match_RM_NVW_4d
#define Match_HEM			Match_HEM_4d
#define Match_SHEM			Match_SHEM_4d

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4d
#define FreeWorkSpace			FreeWorkSpace_4d
#define WspaceAvail			WspaceAvail_4d
#define idxwspacemalloc			idxwspacemalloc_4d
#define idxwspacefree			idxwspacefree_4d
#define fwspacemalloc			fwspacemalloc_4d
#define CreateGraph			CreateGraph_4d
#define InitGraph			InitGraph_4d
#define FreeGraph			FreeGraph_4d

/* mincover.c */
#define MinCover			MinCover_4d
#define MinCover_Augment		MinCover_Augment_4d
#define MinCover_Decompose		MinCover_Decompose_4d
#define MinCover_ColDFS			MinCover_ColDFS_4d
#define MinCover_RowDFS			MinCover_RowDFS_4d

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4d
#define MCMatch_HEM			MCMatch_HEM_4d
#define MCMatch_SHEM			MCMatch_SHEM_4d
#define MCMatch_SHEBM			MCMatch_SHEBM_4d
#define MCMatch_SBHEM			MCMatch_SBHEM_4d
#define BetterVBalance			BetterVBalance_4d
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4d

/* mmd.c */
#define genmmd				genmmd_4d
#define mmdelm				mmdelm_4d
#define mmdint				mmdint_4d
#define mmdnum				mmdnum_4d
#define mmdupd				mmdupd_4d

/* myqsort.c */
#define iidxsort			iidxsort_4d
#define iintsort			iintsort_4d
#define ikeysort			ikeysort_4d
#define ikeyvalsort			ikeyvalsort_4d

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4d
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4d
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4d
#define MlevelNodeBisection		MlevelNodeBisection_4d
#define SplitGraphOrder			SplitGraphOrder_4d
#define MMDOrder			MMDOrder_4d
#define SplitGraphOrderCC		SplitGraphOrderCC_4d

/* pqueue.c */
#define PQueueInit			PQueueInit_4d
#define PQueueReset			PQueueReset_4d
#define PQueueFree			PQueueFree_4d
#define PQueueInsert			PQueueInsert_4d
#define PQueueDelete			PQueueDelete_4d
#define PQueueUpdate			PQueueUpdate_4d
#define PQueueUpdateUp			PQueueUpdateUp_4d
#define PQueueGetMax			PQueueGetMax_4d
#define PQueueSeeMax			PQueueSeeMax_4d
#define CheckHeap			CheckHeap_4d


/* refine.c */
#define Refine2Way			Refine2Way_4d
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4d
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4d
#define Project2WayPartition		Project2WayPartition_4d


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4d
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4d
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4d


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4d
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4d
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4d
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4d
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4d


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4d
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4d
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4d
#define Project2WayNodePartition	Project2WayNodePartition_4d

/* timing.c */
#define InitTimers			InitTimers_4d
#define PrintTimers			PrintTimers_4d
#define seconds				seconds_4d

/* util.c */
#define errexit				errexit_4d
#define GK_free				GK_free_4d
#ifndef DMALLOC
#define imalloc				imalloc_4d
#define idxmalloc			idxmalloc_4d
#define fmalloc				fmalloc_4d
#define ismalloc			ismalloc_4d
#define idxsmalloc			idxsmalloc_4d
#define GKmalloc			GKmalloc_4d
#endif
#define iset				iset_4d
#define idxset				idxset_4d
#define sset				sset_4d
#define iamax				iamax_4d
#define idxamax				idxamax_4d
#define idxamax_strd			idxamax_strd_4d
#define samax				samax_4d
#define samax2				samax2_4d
#define idxamin				idxamin_4d
#define samin				samin_4d
#define idxsum				idxsum_4d
#define idxsum_strd			idxsum_strd_4d
#define idxadd				idxadd_4d
#define charsum				charsum_4d
#define isum				isum_4d
#define ssum				ssum_4d
#define ssum_strd			ssum_strd_4d
#define sscale				sscale_4d
#define snorm2				snorm2_4d
#define sdot				sdot_4d
#define saxpy				saxpy_4d
#define RandomPermute			RandomPermute_4d
#define ispow2				ispow2_4d
#define InitRandom			InitRandom_4d
#define ilog2				ilog2_4d

#endif
