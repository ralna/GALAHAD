/* single precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4s_64

/* balance.c */
#define Balance2Way			Balance2Way_4s_64
#define Bnd2WayBalance			Bnd2WayBalance_4s_64
#define General2WayBalance		General2WayBalance_4s_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4s_64

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4s_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4s_64
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4s_64
#define SetUpCoarseGraph		SetUpCoarseGraph_4s_64
#define ReAdjustMemory			ReAdjustMemory_4s_64

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4s_64

/* compress.c */
#define CompressGraph			CompressGraph_4s_64
#define PruneGraph			PruneGraph_4s_64

/* debug.c */
#define ComputeCut			ComputeCut_4s_64
#define CheckBnd			CheckBnd_4s_64
#define CheckNodeBnd			CheckNodeBnd_4s_64
#define CheckNodePartitionParams	CheckNodePartitionParams_4s_64
#define IsSeparable			IsSeparable_4s_64

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4s_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4s_64
#define Change2FNumbering		Change2FNumbering_4s_64
#define Change2FNumbering2		Change2FNumbering2_4s_64
#define Change2FNumberingOrder		Change2FNumberingOrder_4s_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4s_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4s_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4s_64

/* graph.c */
#define SetUpGraph			SetUpGraph_4s_64
#define SetUpGraphKway 			SetUpGraphKway_4s_64
#define SetUpGraph2			SetUpGraph2_4s_64
#define VolSetUpGraph			VolSetUpGraph_4s_64
#define RandomizeGraph			RandomizeGraph_4s_64
#define IsConnectedSubdomain		IsConnectedSubdomain_4s_64
#define IsConnected			IsConnected_4s_64
#define IsConnected2			IsConnected2_4s_64
#define FindComponents			FindComponents_4s_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4s_64
#define InitSeparator			InitSeparator_4s_64
#define GrowBisection			GrowBisection_4s_64
#define GrowBisectionNode		GrowBisectionNode_4s_64
#define RandomBisection			RandomBisection_4s_64

/* match.c */
#define Match_RM			Match_RM_4s_64
#define Match_RM_NVW			Match_RM_NVW_4s_64
#define Match_HEM			Match_HEM_4s_64
#define Match_SHEM			Match_SHEM_4s_64

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4s_64
#define FreeWorkSpace			FreeWorkSpace_4s_64
#define WspaceAvail			WspaceAvail_4s_64
#define idxwspacemalloc			idxwspacemalloc_4s_64
#define idxwspacefree			idxwspacefree_4s_64
#define fwspacemalloc			fwspacemalloc_4s_64
#define CreateGraph			CreateGraph_4s_64
#define InitGraph			InitGraph_4s_64
#define FreeGraph			FreeGraph_4s_64

/* mincover.c */
#define MinCover			MinCover_4s_64
#define MinCover_Augment		MinCover_Augment_4s_64
#define MinCover_Decompose		MinCover_Decompose_4s_64
#define MinCover_ColDFS			MinCover_ColDFS_4s_64
#define MinCover_RowDFS			MinCover_RowDFS_4s_64

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4s_64
#define MCMatch_HEM			MCMatch_HEM_4s_64
#define MCMatch_SHEM			MCMatch_SHEM_4s_64
#define MCMatch_SHEBM			MCMatch_SHEBM_4s_64
#define MCMatch_SBHEM			MCMatch_SBHEM_4s_64
#define BetterVBalance			BetterVBalance_4s_64
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4s_64

/* mmd.c */
#define genmmd				genmmd_4s_64
#define mmdelm				mmdelm_4s_64
#define mmdint				mmdint_4s_64
#define mmdnum				mmdnum_4s_64
#define mmdupd				mmdupd_4s_64

/* myqsort.c */
#define iidxsort			iidxsort_4s_64
#define iintsort			iintsort_4s_64
#define ikeysort			ikeysort_4s_64
#define ikeyvalsort			ikeyvalsort_4s_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4s_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4s_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4s_64
#define MlevelNodeBisection		MlevelNodeBisection_4s_64
#define SplitGraphOrder			SplitGraphOrder_4s_64
#define MMDOrder			MMDOrder_4s_64
#define SplitGraphOrderCC		SplitGraphOrderCC_4s_64

/* pqueue.c */
#define PQueueInit			PQueueInit_4s_64
#define PQueueReset			PQueueReset_4s_64
#define PQueueFree			PQueueFree_4s_64
#define PQueueInsert			PQueueInsert_4s_64
#define PQueueDelete			PQueueDelete_4s_64
#define PQueueUpdate			PQueueUpdate_4s_64
#define PQueueUpdateUp			PQueueUpdateUp_4s_64
#define PQueueGetMax			PQueueGetMax_4s_64
#define PQueueSeeMax			PQueueSeeMax_4s_64
#define CheckHeap			CheckHeap_4s_64


/* refine.c */
#define Refine2Way			Refine2Way_4s_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4s_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4s_64
#define Project2WayPartition		Project2WayPartition_4s_64


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4s_64
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4s_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4s_64


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4s_64
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4s_64
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4s_64
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4s_64
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4s_64


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4s_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4s_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4s_64
#define Project2WayNodePartition	Project2WayNodePartition_4s_64

/* timing.c */
#define InitTimers			InitTimers_4s_64
#define PrintTimers			PrintTimers_4s_64
#define seconds				seconds_4s_64

/* util.c */
#define errexit				errexit_4s_64
#define GK_free				GK_free_4s_64
#ifndef DMALLOC
#define imalloc				imalloc_4s_64
#define idxmalloc			idxmalloc_4s_64
#define fmalloc				fmalloc_4s_64
#define ismalloc			ismalloc_4s_64
#define idxsmalloc			idxsmalloc_4s_64
#define GKmalloc			GKmalloc_4s_64
#endif
#define iset				iset_4s_64
#define idxset				idxset_4s_64
#define sset				sset_4s_64
#define iamax				iamax_4s_64
#define idxamax				idxamax_4s_64
#define idxamax_strd			idxamax_strd_4s_64
#define samax				samax_4s_64
#define samax2				samax2_4s_64
#define idxamin				idxamin_4s_64
#define samin				samin_4s_64
#define idxsum				idxsum_4s_64
#define idxsum_strd			idxsum_strd_4s_64
#define idxadd				idxadd_4s_64
#define charsum				charsum_4s_64
#define isum				isum_4s_64
#define ssum				ssum_4s_64
#define ssum_strd			ssum_strd_4s_64
#define sscale				sscale_4s_64
#define snorm2				snorm2_4s_64
#define sdot				sdot_4s_64
#define saxpy				saxpy_4s_64
#define RandomPermute			RandomPermute_4s_64
#define ispow2				ispow2_4s_64
#define InitRandom			InitRandom_4s_64
#define ilog2				ilog2_4s_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4s

/* balance.c */
#define Balance2Way			Balance2Way_4s
#define Bnd2WayBalance			Bnd2WayBalance_4s
#define General2WayBalance		General2WayBalance_4s

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4s

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4s
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4s
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4s
#define SetUpCoarseGraph		SetUpCoarseGraph_4s
#define ReAdjustMemory			ReAdjustMemory_4s

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4s

/* compress.c */
#define CompressGraph			CompressGraph_4s
#define PruneGraph			PruneGraph_4s

/* debug.c */
#define ComputeCut			ComputeCut_4s
#define CheckBnd			CheckBnd_4s
#define CheckNodeBnd			CheckNodeBnd_4s
#define CheckNodePartitionParams	CheckNodePartitionParams_4s
#define IsSeparable			IsSeparable_4s

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4s

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4s
#define Change2FNumbering		Change2FNumbering_4s
#define Change2FNumbering2		Change2FNumbering2_4s
#define Change2FNumberingOrder		Change2FNumberingOrder_4s
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4s
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4s
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4s

/* graph.c */
#define SetUpGraph			SetUpGraph_4s
#define SetUpGraphKway 			SetUpGraphKway_4s
#define SetUpGraph2			SetUpGraph2_4s
#define VolSetUpGraph			VolSetUpGraph_4s
#define RandomizeGraph			RandomizeGraph_4s
#define IsConnectedSubdomain		IsConnectedSubdomain_4s
#define IsConnected			IsConnected_4s
#define IsConnected2			IsConnected2_4s
#define FindComponents			FindComponents_4s

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4s
#define InitSeparator			InitSeparator_4s
#define GrowBisection			GrowBisection_4s
#define GrowBisectionNode		GrowBisectionNode_4s
#define RandomBisection			RandomBisection_4s

/* match.c */
#define Match_RM			Match_RM_4s
#define Match_RM_NVW			Match_RM_NVW_4s
#define Match_HEM			Match_HEM_4s
#define Match_SHEM			Match_SHEM_4s

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4s
#define FreeWorkSpace			FreeWorkSpace_4s
#define WspaceAvail			WspaceAvail_4s
#define idxwspacemalloc			idxwspacemalloc_4s
#define idxwspacefree			idxwspacefree_4s
#define fwspacemalloc			fwspacemalloc_4s
#define CreateGraph			CreateGraph_4s
#define InitGraph			InitGraph_4s
#define FreeGraph			FreeGraph_4s

/* mincover.c */
#define MinCover			MinCover_4s
#define MinCover_Augment		MinCover_Augment_4s
#define MinCover_Decompose		MinCover_Decompose_4s
#define MinCover_ColDFS			MinCover_ColDFS_4s
#define MinCover_RowDFS			MinCover_RowDFS_4s

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4s
#define MCMatch_HEM			MCMatch_HEM_4s
#define MCMatch_SHEM			MCMatch_SHEM_4s
#define MCMatch_SHEBM			MCMatch_SHEBM_4s
#define MCMatch_SBHEM			MCMatch_SBHEM_4s
#define BetterVBalance			BetterVBalance_4s
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4s

/* mmd.c */
#define genmmd				genmmd_4s
#define mmdelm				mmdelm_4s
#define mmdint				mmdint_4s
#define mmdnum				mmdnum_4s
#define mmdupd				mmdupd_4s

/* myqsort.c */
#define iidxsort			iidxsort_4s
#define iintsort			iintsort_4s
#define ikeysort			ikeysort_4s
#define ikeyvalsort			ikeyvalsort_4s

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4s
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4s
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4s
#define MlevelNodeBisection		MlevelNodeBisection_4s
#define SplitGraphOrder			SplitGraphOrder_4s
#define MMDOrder			MMDOrder_4s
#define SplitGraphOrderCC		SplitGraphOrderCC_4s

/* pqueue.c */
#define PQueueInit			PQueueInit_4s
#define PQueueReset			PQueueReset_4s
#define PQueueFree			PQueueFree_4s
#define PQueueInsert			PQueueInsert_4s
#define PQueueDelete			PQueueDelete_4s
#define PQueueUpdate			PQueueUpdate_4s
#define PQueueUpdateUp			PQueueUpdateUp_4s
#define PQueueGetMax			PQueueGetMax_4s
#define PQueueSeeMax			PQueueSeeMax_4s
#define CheckHeap			CheckHeap_4s


/* refine.c */
#define Refine2Way			Refine2Way_4s
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4s
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4s
#define Project2WayPartition		Project2WayPartition_4s


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4s
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4s
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4s


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4s
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4s
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4s
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4s
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4s


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4s
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4s
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4s
#define Project2WayNodePartition	Project2WayNodePartition_4s

/* timing.c */
#define InitTimers			InitTimers_4s
#define PrintTimers			PrintTimers_4s
#define seconds				seconds_4s

/* util.c */
#define errexit				errexit_4s
#define GK_free				GK_free_4s
#ifndef DMALLOC
#define imalloc				imalloc_4s
#define idxmalloc			idxmalloc_4s
#define fmalloc				fmalloc_4s
#define ismalloc			ismalloc_4s
#define idxsmalloc			idxsmalloc_4s
#define GKmalloc			GKmalloc_4s
#endif
#define iset				iset_4s
#define idxset				idxset_4s
#define sset				sset_4s
#define iamax				iamax_4s
#define idxamax				idxamax_4s
#define idxamax_strd			idxamax_strd_4s
#define samax				samax_4s
#define samax2				samax2_4s
#define idxamin				idxamin_4s
#define samin				samin_4s
#define idxsum				idxsum_4s
#define idxsum_strd			idxsum_strd_4s
#define idxadd				idxadd_4s
#define charsum				charsum_4s
#define isum				isum_4s
#define ssum				ssum_4s
#define ssum_strd			ssum_strd_4s
#define sscale				sscale_4s
#define snorm2				snorm2_4s
#define sdot				sdot_4s
#define saxpy				saxpy_4s
#define RandomPermute			RandomPermute_4s
#define ispow2				ispow2_4s
#define InitRandom			InitRandom_4s
#define ilog2				ilog2_4s

#endif
