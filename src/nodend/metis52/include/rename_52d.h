/* double precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52d_64
#define METIS_Free                      METIS_Free_52d_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52d_64

/* balance.c */
#define Balance2Way			Balance2Way_52d_64
#define Bnd2WayBalance			Bnd2WayBalance_52d_64
#define General2WayBalance		General2WayBalance_52d_64
#define McGeneral2WayBalance            McGeneral2WayBalance_52d_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52d_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52d_64
#define CheckInputGraphWeights          CheckInputGraphWeights_52d_64
#define FixGraph                        FixGraph_52d_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52d_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52d_64
#define Match_RM                        Match_RM_52d_64
#define Match_SHEM                      Match_SHEM_52d_64
#define Match_2Hop                      Match_2Hop_52d_64
#define Match_2HopAny                   Match_2HopAny_52d_64
#define Match_2HopAll                   Match_2HopAll_52d_64
#define Match_JC                        Match_JC_52d_64
#define PrintCGraphStats                PrintCGraphStats_52d_64
#define CreateCoarseGraph		CreateCoarseGraph_52d_64
#define SetupCoarseGraph		SetupCoarseGraph_52d_64
#define ReAdjustMemory			ReAdjustMemory_52d_64

/* compress.c */
#define CompressGraph			CompressGraph_52d_64
#define PruneGraph			PruneGraph_52d_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52d_64
#define IsConnected                     IsConnected_52d_64
#define IsConnectedSubdomain            IsConnectedSubdomain_52d_64
#define FindSepInducedComponents        FindSepInducedComponents_52d_64
#define EliminateComponents             EliminateComponents_52d_64
#define MoveGroupContigForCut           MoveGroupContigForCut_52d_64
#define MoveGroupContigForVol           MoveGroupContigForVol_52d_64
#define ComputeBFSOrdering              ComputeBFSOrdering_52d_64

/* debug.c */
#define ComputeCut			ComputeCut_52d_64
#define ComputeVolume			ComputeVolume_52d_64
#define ComputeMaxCut			ComputeMaxCut_52d_64
#define CheckBnd			CheckBnd_52d_64
#define CheckBnd2			CheckBnd2_52d_64
#define CheckNodeBnd			CheckNodeBnd_52d_64
#define CheckRInfo			CheckRInfo_52d_64
#define CheckNodePartitionParams	CheckNodePartitionParams_52d_64
#define IsSeparable			IsSeparable_52d_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52d_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52d_64
#define FM_2WayCutRefine                FM_2WayCutRefine_52d_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52d_64
#define SelectQueue                     SelectQueue_52d_64
#define Print2WayRefineStats            Print2WayRefineStats_52d_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52d_64
#define Change2FNumbering		Change2FNumbering_52d_64
#define Change2FNumbering2		Change2FNumbering2_52d_64
#define Change2FNumberingOrder		Change2FNumberingOrder_52d_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52d_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52d_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52d_64

/* graph.c */
#define SetupGraph			SetupGraph_52d_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52d_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52d_64
#define SetupGraph_label                SetupGraph_label_52d_64
#define SetupSplitGraph                 SetupSplitGraph_52d_64
#define CreateGraph                     CreateGraph_52d_64
#define InitGraph                       InitGraph_52d_64
#define FreeSData                       FreeSData_52d_64
#define FreeRData                       FreeRData_52d_64
#define FreeGraph                       FreeGraph_52d_64
#define graph_WriteToDisk               graph_WriteToDisk_52d_64
#define graph_ReadFromDisk              graph_ReadFromDisk_52d_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52d_64
#define InitSeparator			InitSeparator_52d_64
#define RandomBisection			RandomBisection_52d_64
#define GrowBisection			GrowBisection_52d_64
#define McRandomBisection               McRandomBisection_52d_64
#define McGrowBisection                 McGrowBisection_52d_64
#define GrowBisectionNode		GrowBisectionNode_52d_64
#define GrowBisectionNode2		GrowBisectionNode2_52d_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52d_64
#define InitKWayPartitioning            InitKWayPartitioning_52d_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52d_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52d_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52d_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52d_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52d_64
#define IsArticulationNode              IsArticulationNode_52d_64
#define KWayVolUpdate                   KWayVolUpdate_52d_64
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52d_64
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52d_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52d_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52d_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52d_64
#define ProjectKWayPartition		ProjectKWayPartition_52d_64
#define ComputeKWayBoundary		ComputeKWayBoundary_52d_64
#define ComputeKWayVolGains             ComputeKWayVolGains_52d_64
#define IsBalanced			IsBalanced_52d_64

/* mcutil */
#define rvecle                          rvecle_52d_64
#define rvecge                          rvecge_52d_64
#define rvecsumle                       rvecsumle_52d_64
#define rvecmaxdiff                     rvecmaxdiff_52d_64
#define ivecle                          ivecle_52d_64
#define ivecge                          ivecge_52d_64
#define ivecaxpylez                     ivecaxpylez_52d_64
#define ivecaxpygez                     ivecaxpygez_52d_64
#define BetterVBalance                  BetterVBalance_52d_64
#define BetterBalance2Way               BetterBalance2Way_52d_64
#define BetterBalanceKWay               BetterBalanceKWay_52d_64
#define ComputeLoadImbalance            ComputeLoadImbalance_52d_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52d_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52d_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52d_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52d_64
#define FindCommonElements              FindCommonElements_52d_64
#define CreateGraphNodal                CreateGraphNodal_52d_64
#define FindCommonNodes                 FindCommonNodes_52d_64
#define CreateMesh                      CreateMesh_52d_64
#define InitMesh                        InitMesh_52d_64
#define FreeMesh                        FreeMesh_52d_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52d_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52d_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52d_64
#define PrintSubDomainGraph             PrintSubDomainGraph_52d_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52d_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52d_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52d_64

/* mincover.c */
#define MinCover			MinCover_52d_64
#define MinCover_Augment		MinCover_Augment_52d_64
#define MinCover_Decompose		MinCover_Decompose_52d_64
#define MinCover_ColDFS			MinCover_ColDFS_52d_64
#define MinCover_RowDFS			MinCover_RowDFS_52d_64

/* mmd.c */
#define genmmd				genmmd_52d_64
#define mmdelm				mmdelm_52d_64
#define mmdint				mmdint_52d_64
#define mmdnum				mmdnum_52d_64
#define mmdupd				mmdupd_52d_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52d_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52d_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52d_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52d_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52d_64
#define SplitGraphOrder			SplitGraphOrder_52d_64
#define SplitGraphOrderCC		SplitGraphOrderCC_52d_64
#define MMDOrder			MMDOrder_52d_64

/* options.c */
#define SetupCtrl                       SetupCtrl_52d_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52d_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52d_64
#define PrintCtrl                       PrintCtrl_52d_64
#define FreeCtrl                        FreeCtrl_52d_64
#define CheckParams                     CheckParams_52d_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52d_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52d_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52d_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52d_64
#define MultilevelBisect		MultilevelBisect_52d_64
#define SplitGraphPart			SplitGraphPart_52d_64

/* refine.c */
#define Refine2Way			Refine2Way_52d_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52d_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52d_64
#define Project2WayPartition		Project2WayPartition_52d_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52d_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52d_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52d_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52d_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52d_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52d_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52d_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52d_64
#define Project2WayNodePartition	Project2WayNodePartition_52d_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52d_64
#define ComputePartitionBalance		ComputePartitionBalance_52d_64
#define ComputeElementBalance		ComputeElementBalance_52d_64

/* timing.c */
#define InitTimers			InitTimers_52d_64
#define PrintTimers			PrintTimers_52d_64

/* util.c */
#define iargmax_strd                    iargmax_strd_52d_64
#define iargmax_nrm                     iargmax_nrm_52d_64
#define iargmax2_nrm                    iargmax2_nrm_52d_64
#define rargmax2                        rargmax2_52d_64
#define InitRandom                      InitRandom_52d_64
#define metis_rcode                     metis_rcode_52d_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52d_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52d_64
#define FreeWorkSpace                   FreeWorkSpace_52d_64
#define wspacemalloc                    wspacemalloc_52d_64
#define wspacepush                      wspacepush_52d_64
#define wspacepop                       wspacepop_52d_64
#define iwspacemalloc                   iwspacemalloc_52d_64
#define rwspacemalloc                   rwspacemalloc_52d_64
#define ikvwspacemalloc                 ikvwspacemalloc_52d_64
#define cnbrpoolReset                   cnbrpoolReset_52d_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_52d_64
#define vnbrpoolReset                   vnbrpoolReset_52d_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_52d_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52d
#define METIS_Free                      METIS_Free_52d
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52d

/* balance.c */
#define Balance2Way			Balance2Way_52d
#define Bnd2WayBalance			Bnd2WayBalance_52d
#define General2WayBalance		General2WayBalance_52d
#define McGeneral2WayBalance            McGeneral2WayBalance_52d

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52d

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52d
#define CheckInputGraphWeights          CheckInputGraphWeights_52d
#define FixGraph                        FixGraph_52d

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52d
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52d
#define Match_RM                        Match_RM_52d
#define Match_SHEM                      Match_SHEM_52d
#define Match_2Hop                      Match_2Hop_52d
#define Match_2HopAny                   Match_2HopAny_52d
#define Match_2HopAll                   Match_2HopAll_52d
#define Match_JC                        Match_JC_52d
#define PrintCGraphStats                PrintCGraphStats_52d
#define CreateCoarseGraph		CreateCoarseGraph_52d
#define SetupCoarseGraph		SetupCoarseGraph_52d
#define ReAdjustMemory			ReAdjustMemory_52d

/* compress.c */
#define CompressGraph			CompressGraph_52d
#define PruneGraph			PruneGraph_52d

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52d
#define IsConnected                     IsConnected_52d
#define IsConnectedSubdomain            IsConnectedSubdomain_52d
#define FindSepInducedComponents        FindSepInducedComponents_52d
#define EliminateComponents             EliminateComponents_52d
#define MoveGroupContigForCut           MoveGroupContigForCut_52d
#define MoveGroupContigForVol           MoveGroupContigForVol_52d
#define ComputeBFSOrdering              ComputeBFSOrdering_52d

/* debug.c */
#define ComputeCut			ComputeCut_52d
#define ComputeVolume			ComputeVolume_52d
#define ComputeMaxCut			ComputeMaxCut_52d
#define CheckBnd			CheckBnd_52d
#define CheckBnd2			CheckBnd2_52d
#define CheckNodeBnd			CheckNodeBnd_52d
#define CheckRInfo			CheckRInfo_52d
#define CheckNodePartitionParams	CheckNodePartitionParams_52d
#define IsSeparable			IsSeparable_52d
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52d

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52d
#define FM_2WayCutRefine                FM_2WayCutRefine_52d
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52d
#define SelectQueue                     SelectQueue_52d
#define Print2WayRefineStats            Print2WayRefineStats_52d

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52d
#define Change2FNumbering		Change2FNumbering_52d
#define Change2FNumbering2		Change2FNumbering2_52d
#define Change2FNumberingOrder		Change2FNumberingOrder_52d
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52d
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52d
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52d

/* graph.c */
#define SetupGraph			SetupGraph_52d
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52d
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52d
#define SetupGraph_label                SetupGraph_label_52d
#define SetupSplitGraph                 SetupSplitGraph_52d
#define CreateGraph                     CreateGraph_52d
#define InitGraph                       InitGraph_52d
#define FreeSData                       FreeSData_52d
#define FreeRData                       FreeRData_52d
#define FreeGraph                       FreeGraph_52d
#define graph_WriteToDisk               graph_WriteToDisk_52d
#define graph_ReadFromDisk              graph_ReadFromDisk_52d

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52d
#define InitSeparator			InitSeparator_52d
#define RandomBisection			RandomBisection_52d
#define GrowBisection			GrowBisection_52d
#define McRandomBisection               McRandomBisection_52d
#define McGrowBisection                 McGrowBisection_52d
#define GrowBisectionNode		GrowBisectionNode_52d
#define GrowBisectionNode2		GrowBisectionNode2_52d

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52d
#define InitKWayPartitioning            InitKWayPartitioning_52d

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52d
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52d
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52d
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52d
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52d
#define IsArticulationNode              IsArticulationNode_52d
#define KWayVolUpdate                   KWayVolUpdate_52d
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52d
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52d

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52d
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52d
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52d
#define ProjectKWayPartition		ProjectKWayPartition_52d
#define ComputeKWayBoundary		ComputeKWayBoundary_52d
#define ComputeKWayVolGains             ComputeKWayVolGains_52d
#define IsBalanced			IsBalanced_52d

/* mcutil */
#define rvecle                          rvecle_52d
#define rvecge                          rvecge_52d
#define rvecsumle                       rvecsumle_52d
#define rvecmaxdiff                     rvecmaxdiff_52d
#define ivecle                          ivecle_52d
#define ivecge                          ivecge_52d
#define ivecaxpylez                     ivecaxpylez_52d
#define ivecaxpygez                     ivecaxpygez_52d
#define BetterVBalance                  BetterVBalance_52d
#define BetterBalance2Way               BetterBalance2Way_52d
#define BetterBalanceKWay               BetterBalanceKWay_52d
#define ComputeLoadImbalance            ComputeLoadImbalance_52d
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52d
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52d
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52d

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52d
#define FindCommonElements              FindCommonElements_52d
#define CreateGraphNodal                CreateGraphNodal_52d
#define FindCommonNodes                 FindCommonNodes_52d
#define CreateMesh                      CreateMesh_52d
#define InitMesh                        InitMesh_52d
#define FreeMesh                        FreeMesh_52d

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52d

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52d
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52d
#define PrintSubDomainGraph             PrintSubDomainGraph_52d
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52d
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52d
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52d

/* mincover.c */
#define MinCover			MinCover_52d
#define MinCover_Augment		MinCover_Augment_52d
#define MinCover_Decompose		MinCover_Decompose_52d
#define MinCover_ColDFS			MinCover_ColDFS_52d
#define MinCover_RowDFS			MinCover_RowDFS_52d

/* mmd.c */
#define genmmd				genmmd_52d
#define mmdelm				mmdelm_52d
#define mmdint				mmdint_52d
#define mmdnum				mmdnum_52d
#define mmdupd				mmdupd_52d


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52d
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52d
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52d
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52d
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52d
#define SplitGraphOrder			SplitGraphOrder_52d
#define SplitGraphOrderCC		SplitGraphOrderCC_52d
#define MMDOrder			MMDOrder_52d

/* options.c */
#define SetupCtrl                       SetupCtrl_52d
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52d
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52d
#define PrintCtrl                       PrintCtrl_52d
#define FreeCtrl                        FreeCtrl_52d
#define CheckParams                     CheckParams_52d

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52d
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52d
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52d

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52d
#define MultilevelBisect		MultilevelBisect_52d
#define SplitGraphPart			SplitGraphPart_52d

/* refine.c */
#define Refine2Way			Refine2Way_52d
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52d
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52d
#define Project2WayPartition		Project2WayPartition_52d

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52d
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52d

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52d
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52d
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52d

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52d
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52d
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52d
#define Project2WayNodePartition	Project2WayNodePartition_52d

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52d
#define ComputePartitionBalance		ComputePartitionBalance_52d
#define ComputeElementBalance		ComputeElementBalance_52d

/* timing.c */
#define InitTimers			InitTimers_52d
#define PrintTimers			PrintTimers_52d

/* util.c */
#define iargmax_strd                    iargmax_strd_52d
#define iargmax_nrm                     iargmax_nrm_52d
#define iargmax2_nrm                    iargmax2_nrm_52d
#define rargmax2                        rargmax2_52d
#define InitRandom                      InitRandom_52d
#define metis_rcode                     metis_rcode_52d

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52d
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52d
#define FreeWorkSpace                   FreeWorkSpace_52d
#define wspacemalloc                    wspacemalloc_52d
#define wspacepush                      wspacepush_52d
#define wspacepop                       wspacepop_52d
#define iwspacemalloc                   iwspacemalloc_52d
#define rwspacemalloc                   rwspacemalloc_52d
#define ikvwspacemalloc                 ikvwspacemalloc_52d
#define cnbrpoolReset                   cnbrpoolReset_52d
#define cnbrpoolGetNext                 cnbrpoolGetNext_52d
#define vnbrpoolReset                   vnbrpoolReset_52d
#define vnbrpoolGetNext                 vnbrpoolGetNext_52d

#endif
