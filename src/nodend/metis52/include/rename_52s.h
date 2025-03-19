/* single precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52s_64
#define METIS_Free                      METIS_Free_52s_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52s_64

/* balance.c */
#define Balance2Way			Balance2Way_52s_64
#define Bnd2WayBalance			Bnd2WayBalance_52s_64
#define General2WayBalance		General2WayBalance_52s_64
#define McGeneral2WayBalance            McGeneral2WayBalance_52s_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52s_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52s_64
#define CheckInputGraphWeights          CheckInputGraphWeights_52s_64
#define FixGraph                        FixGraph_52s_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52s_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52s_64
#define Match_RM                        Match_RM_52s_64
#define Match_SHEM                      Match_SHEM_52s_64
#define Match_2Hop                      Match_2Hop_52s_64
#define Match_2HopAny                   Match_2HopAny_52s_64
#define Match_2HopAll                   Match_2HopAll_52s_64
#define Match_JC                        Match_JC_52s_64
#define PrintCGraphStats                PrintCGraphStats_52s_64
#define CreateCoarseGraph		CreateCoarseGraph_52s_64
#define SetupCoarseGraph		SetupCoarseGraph_52s_64
#define ReAdjustMemory			ReAdjustMemory_52s_64

/* compress.c */
#define CompressGraph			CompressGraph_52s_64
#define PruneGraph			PruneGraph_52s_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52s_64
#define IsConnected                     IsConnected_52s_64
#define IsConnectedSubdomain            IsConnectedSubdomain_52s_64
#define FindSepInducedComponents        FindSepInducedComponents_52s_64
#define EliminateComponents             EliminateComponents_52s_64
#define MoveGroupContigForCut           MoveGroupContigForCut_52s_64
#define MoveGroupContigForVol           MoveGroupContigForVol_52s_64
#define ComputeBFSOrdering              ComputeBFSOrdering_52s_64

/* debug.c */
#define ComputeCut			ComputeCut_52s_64
#define ComputeVolume			ComputeVolume_52s_64
#define ComputeMaxCut			ComputeMaxCut_52s_64
#define CheckBnd			CheckBnd_52s_64
#define CheckBnd2			CheckBnd2_52s_64
#define CheckNodeBnd			CheckNodeBnd_52s_64
#define CheckRInfo			CheckRInfo_52s_64
#define CheckNodePartitionParams	CheckNodePartitionParams_52s_64
#define IsSeparable			IsSeparable_52s_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52s_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52s_64
#define FM_2WayCutRefine                FM_2WayCutRefine_52s_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52s_64
#define SelectQueue                     SelectQueue_52s_64
#define Print2WayRefineStats            Print2WayRefineStats_52s_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52s_64
#define Change2FNumbering		Change2FNumbering_52s_64
#define Change2FNumbering2		Change2FNumbering2_52s_64
#define Change2FNumberingOrder		Change2FNumberingOrder_52s_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52s_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52s_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52s_64

/* graph.c */
#define SetupGraph			SetupGraph_52s_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52s_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52s_64
#define SetupGraph_label                SetupGraph_label_52s_64
#define SetupSplitGraph                 SetupSplitGraph_52s_64
#define CreateGraph                     CreateGraph_52s_64
#define InitGraph                       InitGraph_52s_64
#define FreeSData                       FreeSData_52s_64
#define FreeRData                       FreeRData_52s_64
#define FreeGraph                       FreeGraph_52s_64
#define graph_WriteToDisk               graph_WriteToDisk_52s_64
#define graph_ReadFromDisk              graph_ReadFromDisk_52s_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52s_64
#define InitSeparator			InitSeparator_52s_64
#define RandomBisection			RandomBisection_52s_64
#define GrowBisection			GrowBisection_52s_64
#define McRandomBisection               McRandomBisection_52s_64
#define McGrowBisection                 McGrowBisection_52s_64
#define GrowBisectionNode		GrowBisectionNode_52s_64
#define GrowBisectionNode2		GrowBisectionNode2_52s_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52s_64
#define InitKWayPartitioning            InitKWayPartitioning_52s_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52s_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52s_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52s_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52s_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52s_64
#define IsArticulationNode              IsArticulationNode_52s_64
#define KWayVolUpdate                   KWayVolUpdate_52s_64
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52s_64
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52s_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52s_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52s_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52s_64
#define ProjectKWayPartition		ProjectKWayPartition_52s_64
#define ComputeKWayBoundary		ComputeKWayBoundary_52s_64
#define ComputeKWayVolGains             ComputeKWayVolGains_52s_64
#define IsBalanced			IsBalanced_52s_64

/* mcutil */
#define rvecle                          rvecle_52s_64
#define rvecge                          rvecge_52s_64
#define rvecsumle                       rvecsumle_52s_64
#define rvecmaxdiff                     rvecmaxdiff_52s_64
#define ivecle                          ivecle_52s_64
#define ivecge                          ivecge_52s_64
#define ivecaxpylez                     ivecaxpylez_52s_64
#define ivecaxpygez                     ivecaxpygez_52s_64
#define BetterVBalance                  BetterVBalance_52s_64
#define BetterBalance2Way               BetterBalance2Way_52s_64
#define BetterBalanceKWay               BetterBalanceKWay_52s_64
#define ComputeLoadImbalance            ComputeLoadImbalance_52s_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52s_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52s_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52s_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52s_64
#define FindCommonElements              FindCommonElements_52s_64
#define CreateGraphNodal                CreateGraphNodal_52s_64
#define FindCommonNodes                 FindCommonNodes_52s_64
#define CreateMesh                      CreateMesh_52s_64
#define InitMesh                        InitMesh_52s_64
#define FreeMesh                        FreeMesh_52s_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52s_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52s_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52s_64
#define PrintSubDomainGraph             PrintSubDomainGraph_52s_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52s_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52s_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52s_64

/* mincover.c */
#define MinCover			MinCover_52s_64
#define MinCover_Augment		MinCover_Augment_52s_64
#define MinCover_Decompose		MinCover_Decompose_52s_64
#define MinCover_ColDFS			MinCover_ColDFS_52s_64
#define MinCover_RowDFS			MinCover_RowDFS_52s_64

/* mmd.c */
#define genmmd				genmmd_52s_64
#define mmdelm				mmdelm_52s_64
#define mmdint				mmdint_52s_64
#define mmdnum				mmdnum_52s_64
#define mmdupd				mmdupd_52s_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52s_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52s_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52s_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52s_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52s_64
#define SplitGraphOrder			SplitGraphOrder_52s_64
#define SplitGraphOrderCC		SplitGraphOrderCC_52s_64
#define MMDOrder			MMDOrder_52s_64

/* options.c */
#define SetupCtrl                       SetupCtrl_52s_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52s_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52s_64
#define PrintCtrl                       PrintCtrl_52s_64
#define FreeCtrl                        FreeCtrl_52s_64
#define CheckParams                     CheckParams_52s_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52s_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52s_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52s_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52s_64
#define MultilevelBisect		MultilevelBisect_52s_64
#define SplitGraphPart			SplitGraphPart_52s_64

/* refine.c */
#define Refine2Way			Refine2Way_52s_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52s_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52s_64
#define Project2WayPartition		Project2WayPartition_52s_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52s_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52s_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52s_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52s_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52s_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52s_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52s_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52s_64
#define Project2WayNodePartition	Project2WayNodePartition_52s_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52s_64
#define ComputePartitionBalance		ComputePartitionBalance_52s_64
#define ComputeElementBalance		ComputeElementBalance_52s_64

/* timing.c */
#define InitTimers			InitTimers_52s_64
#define PrintTimers			PrintTimers_52s_64

/* util.c */
#define iargmax_strd                    iargmax_strd_52s_64
#define iargmax_nrm                     iargmax_nrm_52s_64
#define iargmax2_nrm                    iargmax2_nrm_52s_64
#define rargmax2                        rargmax2_52s_64
#define InitRandom                      InitRandom_52s_64
#define metis_rcode                     metis_rcode_52s_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52s_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52s_64
#define FreeWorkSpace                   FreeWorkSpace_52s_64
#define wspacemalloc                    wspacemalloc_52s_64
#define wspacepush                      wspacepush_52s_64
#define wspacepop                       wspacepop_52s_64
#define iwspacemalloc                   iwspacemalloc_52s_64
#define rwspacemalloc                   rwspacemalloc_52s_64
#define ikvwspacemalloc                 ikvwspacemalloc_52s_64
#define cnbrpoolReset                   cnbrpoolReset_52s_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_52s_64
#define vnbrpoolReset                   vnbrpoolReset_52s_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_52s_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52s
#define METIS_Free                      METIS_Free_52s
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52s

/* balance.c */
#define Balance2Way			Balance2Way_52s
#define Bnd2WayBalance			Bnd2WayBalance_52s
#define General2WayBalance		General2WayBalance_52s
#define McGeneral2WayBalance            McGeneral2WayBalance_52s

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52s

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52s
#define CheckInputGraphWeights          CheckInputGraphWeights_52s
#define FixGraph                        FixGraph_52s

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52s
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52s
#define Match_RM                        Match_RM_52s
#define Match_SHEM                      Match_SHEM_52s
#define Match_2Hop                      Match_2Hop_52s
#define Match_2HopAny                   Match_2HopAny_52s
#define Match_2HopAll                   Match_2HopAll_52s
#define Match_JC                        Match_JC_52s
#define PrintCGraphStats                PrintCGraphStats_52s
#define CreateCoarseGraph		CreateCoarseGraph_52s
#define SetupCoarseGraph		SetupCoarseGraph_52s
#define ReAdjustMemory			ReAdjustMemory_52s

/* compress.c */
#define CompressGraph			CompressGraph_52s
#define PruneGraph			PruneGraph_52s

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52s
#define IsConnected                     IsConnected_52s
#define IsConnectedSubdomain            IsConnectedSubdomain_52s
#define FindSepInducedComponents        FindSepInducedComponents_52s
#define EliminateComponents             EliminateComponents_52s
#define MoveGroupContigForCut           MoveGroupContigForCut_52s
#define MoveGroupContigForVol           MoveGroupContigForVol_52s
#define ComputeBFSOrdering              ComputeBFSOrdering_52s

/* debug.c */
#define ComputeCut			ComputeCut_52s
#define ComputeVolume			ComputeVolume_52s
#define ComputeMaxCut			ComputeMaxCut_52s
#define CheckBnd			CheckBnd_52s
#define CheckBnd2			CheckBnd2_52s
#define CheckNodeBnd			CheckNodeBnd_52s
#define CheckRInfo			CheckRInfo_52s
#define CheckNodePartitionParams	CheckNodePartitionParams_52s
#define IsSeparable			IsSeparable_52s
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52s

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52s
#define FM_2WayCutRefine                FM_2WayCutRefine_52s
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52s
#define SelectQueue                     SelectQueue_52s
#define Print2WayRefineStats            Print2WayRefineStats_52s

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52s
#define Change2FNumbering		Change2FNumbering_52s
#define Change2FNumbering2		Change2FNumbering2_52s
#define Change2FNumberingOrder		Change2FNumberingOrder_52s
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52s
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52s
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52s

/* graph.c */
#define SetupGraph			SetupGraph_52s
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52s
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52s
#define SetupGraph_label                SetupGraph_label_52s
#define SetupSplitGraph                 SetupSplitGraph_52s
#define CreateGraph                     CreateGraph_52s
#define InitGraph                       InitGraph_52s
#define FreeSData                       FreeSData_52s
#define FreeRData                       FreeRData_52s
#define FreeGraph                       FreeGraph_52s
#define graph_WriteToDisk               graph_WriteToDisk_52s
#define graph_ReadFromDisk              graph_ReadFromDisk_52s

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52s
#define InitSeparator			InitSeparator_52s
#define RandomBisection			RandomBisection_52s
#define GrowBisection			GrowBisection_52s
#define McRandomBisection               McRandomBisection_52s
#define McGrowBisection                 McGrowBisection_52s
#define GrowBisectionNode		GrowBisectionNode_52s
#define GrowBisectionNode2		GrowBisectionNode2_52s

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52s
#define InitKWayPartitioning            InitKWayPartitioning_52s

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52s
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52s
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52s
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52s
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52s
#define IsArticulationNode              IsArticulationNode_52s
#define KWayVolUpdate                   KWayVolUpdate_52s
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52s
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52s

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52s
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52s
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52s
#define ProjectKWayPartition		ProjectKWayPartition_52s
#define ComputeKWayBoundary		ComputeKWayBoundary_52s
#define ComputeKWayVolGains             ComputeKWayVolGains_52s
#define IsBalanced			IsBalanced_52s

/* mcutil */
#define rvecle                          rvecle_52s
#define rvecge                          rvecge_52s
#define rvecsumle                       rvecsumle_52s
#define rvecmaxdiff                     rvecmaxdiff_52s
#define ivecle                          ivecle_52s
#define ivecge                          ivecge_52s
#define ivecaxpylez                     ivecaxpylez_52s
#define ivecaxpygez                     ivecaxpygez_52s
#define BetterVBalance                  BetterVBalance_52s
#define BetterBalance2Way               BetterBalance2Way_52s
#define BetterBalanceKWay               BetterBalanceKWay_52s
#define ComputeLoadImbalance            ComputeLoadImbalance_52s
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52s
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52s
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52s

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52s
#define FindCommonElements              FindCommonElements_52s
#define CreateGraphNodal                CreateGraphNodal_52s
#define FindCommonNodes                 FindCommonNodes_52s
#define CreateMesh                      CreateMesh_52s
#define InitMesh                        InitMesh_52s
#define FreeMesh                        FreeMesh_52s

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52s

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52s
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52s
#define PrintSubDomainGraph             PrintSubDomainGraph_52s
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52s
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52s
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52s

/* mincover.c */
#define MinCover			MinCover_52s
#define MinCover_Augment		MinCover_Augment_52s
#define MinCover_Decompose		MinCover_Decompose_52s
#define MinCover_ColDFS			MinCover_ColDFS_52s
#define MinCover_RowDFS			MinCover_RowDFS_52s

/* mmd.c */
#define genmmd				genmmd_52s
#define mmdelm				mmdelm_52s
#define mmdint				mmdint_52s
#define mmdnum				mmdnum_52s
#define mmdupd				mmdupd_52s


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52s
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52s
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52s
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52s
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52s
#define SplitGraphOrder			SplitGraphOrder_52s
#define SplitGraphOrderCC		SplitGraphOrderCC_52s
#define MMDOrder			MMDOrder_52s

/* options.c */
#define SetupCtrl                       SetupCtrl_52s
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52s
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52s
#define PrintCtrl                       PrintCtrl_52s
#define FreeCtrl                        FreeCtrl_52s
#define CheckParams                     CheckParams_52s

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52s
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52s
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52s

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52s
#define MultilevelBisect		MultilevelBisect_52s
#define SplitGraphPart			SplitGraphPart_52s

/* refine.c */
#define Refine2Way			Refine2Way_52s
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52s
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52s
#define Project2WayPartition		Project2WayPartition_52s

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52s
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52s

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52s
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52s
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52s

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52s
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52s
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52s
#define Project2WayNodePartition	Project2WayNodePartition_52s

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52s
#define ComputePartitionBalance		ComputePartitionBalance_52s
#define ComputeElementBalance		ComputeElementBalance_52s

/* timing.c */
#define InitTimers			InitTimers_52s
#define PrintTimers			PrintTimers_52s

/* util.c */
#define iargmax_strd                    iargmax_strd_52s
#define iargmax_nrm                     iargmax_nrm_52s
#define iargmax2_nrm                    iargmax2_nrm_52s
#define rargmax2                        rargmax2_52s
#define InitRandom                      InitRandom_52s
#define metis_rcode                     metis_rcode_52s

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52s
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52s
#define FreeWorkSpace                   FreeWorkSpace_52s
#define wspacemalloc                    wspacemalloc_52s
#define wspacepush                      wspacepush_52s
#define wspacepop                       wspacepop_52s
#define iwspacemalloc                   iwspacemalloc_52s
#define rwspacemalloc                   rwspacemalloc_52s
#define ikvwspacemalloc                 ikvwspacemalloc_52s
#define cnbrpoolReset                   cnbrpoolReset_52s
#define cnbrpoolGetNext                 cnbrpoolGetNext_52s
#define vnbrpoolReset                   vnbrpoolReset_52s
#define vnbrpoolGetNext                 vnbrpoolGetNext_52s

#endif
