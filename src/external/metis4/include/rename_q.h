/* quadruple precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4q_64

/* balance.c */
#define Balance2Way			Balance2Way_4q_64
#define Bnd2WayBalance			Bnd2WayBalance_4q_64
#define General2WayBalance		General2WayBalance_4q_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4q_64

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4q_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4q_64
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4q_64
#define SetUpCoarseGraph		SetUpCoarseGraph_4q_64
#define ReAdjustMemory			ReAdjustMemory_4q_64

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4q_64

/* compress.c */
#define CompressGraph			CompressGraph_4q_64
#define PruneGraph			PruneGraph_4q_64

/* debug.c */
#define ComputeCut			ComputeCut_4q_64
#define CheckBnd			CheckBnd_4q_64
#define CheckNodeBnd			CheckNodeBnd_4q_64
#define CheckNodePartitionParams	CheckNodePartitionParams_4q_64
#define IsSeparable			IsSeparable_4q_64

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4q_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4q_64
#define Change2FNumbering		Change2FNumbering_4q_64
#define Change2FNumbering2		Change2FNumbering2_4q_64
#define Change2FNumberingOrder		Change2FNumberingOrder_4q_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4q_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4q_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4q_64

/* graph.c */
#define SetUpGraph			SetUpGraph_4q_64
#define SetUpGraphKway 			SetUpGraphKway_4q_64
#define SetUpGraph2			SetUpGraph2_4q_64
#define VolSetUpGraph			VolSetUpGraph_4q_64
#define RandomizeGraph			RandomizeGraph_4q_64
#define IsConnectedSubdomain		IsConnectedSubdomain_4q_64
#define IsConnected			IsConnected_4q_64
#define IsConnected2			IsConnected2_4q_64
#define FindComponents			FindComponents_4q_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4q_64
#define InitSeparator			InitSeparator_4q_64
#define GrowBisection			GrowBisection_4q_64
#define GrowBisectionNode		GrowBisectionNode_4q_64
#define RandomBisection			RandomBisection_4q_64

/* match.c */
#define Match_RM			Match_RM_4q_64
#define Match_RM_NVW			Match_RM_NVW_4q_64
#define Match_HEM			Match_HEM_4q_64
#define Match_SHEM			Match_SHEM_4q_64

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4q_64
#define FreeWorkSpace			FreeWorkSpace_4q_64
#define WspaceAvail			WspaceAvail_4q_64
#define idxwspacemalloc			idxwspacemalloc_4q_64
#define idxwspacefree			idxwspacefree_4q_64
#define fwspacemalloc			fwspacemalloc_4q_64
#define CreateGraph			CreateGraph_4q_64
#define InitGraph			InitGraph_4q_64
#define FreeGraph			FreeGraph_4q_64

/* mincover.c */
#define MinCover			MinCover_4q_64
#define MinCover_Augment		MinCover_Augment_4q_64
#define MinCover_Decompose		MinCover_Decompose_4q_64
#define MinCover_ColDFS			MinCover_ColDFS_4q_64
#define MinCover_RowDFS			MinCover_RowDFS_4q_64

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4q_64
#define MCMatch_HEM			MCMatch_HEM_4q_64
#define MCMatch_SHEM			MCMatch_SHEM_4q_64
#define MCMatch_SHEBM			MCMatch_SHEBM_4q_64
#define MCMatch_SBHEM			MCMatch_SBHEM_4q_64
#define BetterVBalance			BetterVBalance_4q_64
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4q_64

/* mmd.c */
#define genmmd				genmmd_4q_64
#define mmdelm				mmdelm_4q_64
#define mmdint				mmdint_4q_64
#define mmdnum				mmdnum_4q_64
#define mmdupd				mmdupd_4q_64

/* myqsort.c */
#define iidxsort			iidxsort_4q_64
#define iintsort			iintsort_4q_64
#define ikeysort			ikeysort_4q_64
#define ikeyvalsort			ikeyvalsort_4q_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4q_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4q_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4q_64
#define MlevelNodeBisection		MlevelNodeBisection_4q_64
#define SplitGraphOrder			SplitGraphOrder_4q_64
#define MMDOrder			MMDOrder_4q_64
#define SplitGraphOrderCC		SplitGraphOrderCC_4q_64

/* pqueue.c */
#define PQueueInit			PQueueInit_4q_64
#define PQueueReset			PQueueReset_4q_64
#define PQueueFree			PQueueFree_4q_64
#define PQueueInsert			PQueueInsert_4q_64
#define PQueueDelete			PQueueDelete_4q_64
#define PQueueUpdate			PQueueUpdate_4q_64
#define PQueueUpdateUp			PQueueUpdateUp_4q_64
#define PQueueGetMax			PQueueGetMax_4q_64
#define PQueueSeeMax			PQueueSeeMax_4q_64
#define CheckHeap			CheckHeap_4q_64


/* refine.c */
#define Refine2Way			Refine2Way_4q_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4q_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4q_64
#define Project2WayPartition		Project2WayPartition_4q_64


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4q_64
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4q_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4q_64


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4q_64
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4q_64
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4q_64
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4q_64
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4q_64


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4q_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4q_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4q_64
#define Project2WayNodePartition	Project2WayNodePartition_4q_64

/* timing.c */
#define InitTimers			InitTimers_4q_64
#define PrintTimers			PrintTimers_4q_64
#define seconds				seconds_4q_64

/* util.c */
#define errexit				errexit_4q_64
#define GK_free				GK_free_4q_64
#ifndef DMALLOC
#define imalloc				imalloc_4q_64
#define idxmalloc			idxmalloc_4q_64
#define fmalloc				fmalloc_4q_64
#define ismalloc			ismalloc_4q_64
#define idxsmalloc			idxsmalloc_4q_64
#define GKmalloc			GKmalloc_4q_64
#endif
#define iset				iset_4q_64
#define idxset				idxset_4q_64
#define sset				sset_4q_64
#define iamax				iamax_4q_64
#define idxamax				idxamax_4q_64
#define idxamax_strd			idxamax_strd_4q_64
#define samax				samax_4q_64
#define samax2				samax2_4q_64
#define idxamin				idxamin_4q_64
#define samin				samin_4q_64
#define idxsum				idxsum_4q_64
#define idxsum_strd			idxsum_strd_4q_64
#define idxadd				idxadd_4q_64
#define charsum				charsum_4q_64
#define isum				isum_4q_64
#define ssum				ssum_4q_64
#define ssum_strd			ssum_strd_4q_64
#define sscale				sscale_4q_64
#define snorm2				snorm2_4q_64
#define sdot				sdot_4q_64
#define saxpy				saxpy_4q_64
#define RandomPermute			RandomPermute_4q_64
#define ispow2				ispow2_4q_64
#define InitRandom			InitRandom_4q_64
#define ilog2				ilog2_4q_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_4q

/* balance.c */
#define Balance2Way			Balance2Way_4q
#define Bnd2WayBalance			Bnd2WayBalance_4q
#define General2WayBalance		General2WayBalance_4q

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_4q

/* ccgraph.c */
#define CreateCoarseGraph		CreateCoarseGraph_4q
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_4q
#define CreateCoarseGraph_NVW 		CreateCoarseGraph_NVW_4q
#define SetUpCoarseGraph		SetUpCoarseGraph_4q
#define ReAdjustMemory			ReAdjustMemory_4q

/* coarsen.c */
#define Coarsen2Way			Coarsen2Way_4q

/* compress.c */
#define CompressGraph			CompressGraph_4q
#define PruneGraph			PruneGraph_4q

/* debug.c */
#define ComputeCut			ComputeCut_4q
#define CheckBnd			CheckBnd_4q
#define CheckNodeBnd			CheckNodeBnd_4q
#define CheckNodePartitionParams	CheckNodePartitionParams_4q
#define IsSeparable			IsSeparable_4q

/* fm.c */
#define FM_2WayEdgeRefine		FM_2WayEdgeRefine_4q

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_4q
#define Change2FNumbering		Change2FNumbering_4q
#define Change2FNumbering2		Change2FNumbering2_4q
#define Change2FNumberingOrder		Change2FNumberingOrder_4q
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_4q
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_4q
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_4q

/* graph.c */
#define SetUpGraph			SetUpGraph_4q
#define SetUpGraphKway 			SetUpGraphKway_4q
#define SetUpGraph2			SetUpGraph2_4q
#define VolSetUpGraph			VolSetUpGraph_4q
#define RandomizeGraph			RandomizeGraph_4q
#define IsConnectedSubdomain		IsConnectedSubdomain_4q
#define IsConnected			IsConnected_4q
#define IsConnected2			IsConnected2_4q
#define FindComponents			FindComponents_4q

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_4q
#define InitSeparator			InitSeparator_4q
#define GrowBisection			GrowBisection_4q
#define GrowBisectionNode		GrowBisectionNode_4q
#define RandomBisection			RandomBisection_4q

/* match.c */
#define Match_RM			Match_RM_4q
#define Match_RM_NVW			Match_RM_NVW_4q
#define Match_HEM			Match_HEM_4q
#define Match_SHEM			Match_SHEM_4q

/* memory.c */
#define AllocateWorkSpace		AllocateWorkSpace_4q
#define FreeWorkSpace			FreeWorkSpace_4q
#define WspaceAvail			WspaceAvail_4q
#define idxwspacemalloc			idxwspacemalloc_4q
#define idxwspacefree			idxwspacefree_4q
#define fwspacemalloc			fwspacemalloc_4q
#define CreateGraph			CreateGraph_4q
#define InitGraph			InitGraph_4q
#define FreeGraph			FreeGraph_4q

/* mincover.c */
#define MinCover			MinCover_4q
#define MinCover_Augment		MinCover_Augment_4q
#define MinCover_Decompose		MinCover_Decompose_4q
#define MinCover_ColDFS			MinCover_ColDFS_4q
#define MinCover_RowDFS			MinCover_RowDFS_4q

/* mmatch.c */
#define MCMatch_RM			MCMatch_RM_4q
#define MCMatch_HEM			MCMatch_HEM_4q
#define MCMatch_SHEM			MCMatch_SHEM_4q
#define MCMatch_SHEBM			MCMatch_SHEBM_4q
#define MCMatch_SBHEM			MCMatch_SBHEM_4q
#define BetterVBalance			BetterVBalance_4q
#define AreAllVwgtsBelowFast		AreAllVwgtsBelowFast_4q

/* mmd.c */
#define genmmd				genmmd_4q
#define mmdelm				mmdelm_4q
#define mmdint				mmdint_4q
#define mmdnum				mmdnum_4q
#define mmdupd				mmdupd_4q

/* myqsort.c */
#define iidxsort			iidxsort_4q
#define iintsort			iintsort_4q
#define ikeysort			ikeysort_4q
#define ikeyvalsort			ikeyvalsort_4q

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_4q
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_4q
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_4q
#define MlevelNodeBisection		MlevelNodeBisection_4q
#define SplitGraphOrder			SplitGraphOrder_4q
#define MMDOrder			MMDOrder_4q
#define SplitGraphOrderCC		SplitGraphOrderCC_4q

/* pqueue.c */
#define PQueueInit			PQueueInit_4q
#define PQueueReset			PQueueReset_4q
#define PQueueFree			PQueueFree_4q
#define PQueueInsert			PQueueInsert_4q
#define PQueueDelete			PQueueDelete_4q
#define PQueueUpdate			PQueueUpdate_4q
#define PQueueUpdateUp			PQueueUpdateUp_4q
#define PQueueGetMax			PQueueGetMax_4q
#define PQueueSeeMax			PQueueSeeMax_4q
#define CheckHeap			CheckHeap_4q


/* refine.c */
#define Refine2Way			Refine2Way_4q
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_4q
#define Compute2WayPartitionParams	Compute2WayPartitionParams_4q
#define Project2WayPartition		Project2WayPartition_4q


/* separator.c */
#define ConstructSeparator		ConstructSeparator_4q
#define ConstructMinCoverSeparator0	ConstructMinCoverSeparator0_4q
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_4q


/* sfm.c */
#define FM_2WayNodeRefine		FM_2WayNodeRefine_4q
#define FM_2WayNodeRefineEqWgt		FM_2WayNodeRefineEqWgt_4q
#define FM_2WayNodeRefine_OneSided	FM_2WayNodeRefine_OneSided_4q
#define FM_2WayNodeBalance		FM_2WayNodeBalance_4q
#define ComputeMaxNodeGain		ComputeMaxNodeGain_4q


/* srefine.c */
#define Refine2WayNode			Refine2WayNode_4q
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_4q
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_4q
#define Project2WayNodePartition	Project2WayNodePartition_4q

/* timing.c */
#define InitTimers			InitTimers_4q
#define PrintTimers			PrintTimers_4q
#define seconds				seconds_4q

/* util.c */
#define errexit				errexit_4q
#define GK_free				GK_free_4q
#ifndef DMALLOC
#define imalloc				imalloc_4q
#define idxmalloc			idxmalloc_4q
#define fmalloc				fmalloc_4q
#define ismalloc			ismalloc_4q
#define idxsmalloc			idxsmalloc_4q
#define GKmalloc			GKmalloc_4q
#endif
#define iset				iset_4q
#define idxset				idxset_4q
#define sset				sset_4q
#define iamax				iamax_4q
#define idxamax				idxamax_4q
#define idxamax_strd			idxamax_strd_4q
#define samax				samax_4q
#define samax2				samax2_4q
#define idxamin				idxamin_4q
#define samin				samin_4q
#define idxsum				idxsum_4q
#define idxsum_strd			idxsum_strd_4q
#define idxadd				idxadd_4q
#define charsum				charsum_4q
#define isum				isum_4q
#define ssum				ssum_4q
#define ssum_strd			ssum_strd_4q
#define sscale				sscale_4q
#define snorm2				snorm2_4q
#define sdot				sdot_4q
#define saxpy				saxpy_4q
#define RandomPermute			RandomPermute_4q
#define ispow2				ispow2_4q
#define InitRandom			InitRandom_4q
#define ilog2				ilog2_4q

#endif
