/* quadruple precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52q_64
#define METIS_Free                      METIS_Free_52q_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52q_64

/* balance.c */
#define Balance2Way			Balance2Way_52q_64
#define Bnd2WayBalance			Bnd2WayBalance_52q_64
#define General2WayBalance		General2WayBalance_52q_64
#define McGeneral2WayBalance            McGeneral2WayBalance_52q_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52q_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52q_64
#define CheckInputGraphWeights          CheckInputGraphWeights_52q_64
#define FixGraph                        FixGraph_52q_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52q_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52q_64
#define Match_RM                        Match_RM_52q_64
#define Match_SHEM                      Match_SHEM_52q_64
#define Match_2Hop                      Match_2Hop_52q_64
#define Match_2HopAny                   Match_2HopAny_52q_64
#define Match_2HopAll                   Match_2HopAll_52q_64
#define Match_JC                        Match_JC_52q_64
#define PrintCGraphStats                PrintCGraphStats_52q_64
#define CreateCoarseGraph		CreateCoarseGraph_52q_64
#define SetupCoarseGraph		SetupCoarseGraph_52q_64
#define ReAdjustMemory			ReAdjustMemory_52q_64

/* compress.c */
#define CompressGraph			CompressGraph_52q_64
#define PruneGraph			PruneGraph_52q_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52q_64
#define IsConnected                     IsConnected_52q_64
#define IsConnectedSubdomain            IsConnectedSubdomain_52q_64
#define FindSepInducedComponents        FindSepInducedComponents_52q_64
#define EliminateComponents             EliminateComponents_52q_64
#define MoveGroupContigForCut           MoveGroupContigForCut_52q_64
#define MoveGroupContigForVol           MoveGroupContigForVol_52q_64
#define ComputeBFSOrdering              ComputeBFSOrdering_52q_64

/* debug.c */
#define ComputeCut			ComputeCut_52q_64
#define ComputeVolume			ComputeVolume_52q_64
#define ComputeMaxCut			ComputeMaxCut_52q_64
#define CheckBnd			CheckBnd_52q_64
#define CheckBnd2			CheckBnd2_52q_64
#define CheckNodeBnd			CheckNodeBnd_52q_64
#define CheckRInfo			CheckRInfo_52q_64
#define CheckNodePartitionParams	CheckNodePartitionParams_52q_64
#define IsSeparable			IsSeparable_52q_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52q_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52q_64
#define FM_2WayCutRefine                FM_2WayCutRefine_52q_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52q_64
#define SelectQueue                     SelectQueue_52q_64
#define Print2WayRefineStats            Print2WayRefineStats_52q_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52q_64
#define Change2FNumbering		Change2FNumbering_52q_64
#define Change2FNumbering2		Change2FNumbering2_52q_64
#define Change2FNumberingOrder		Change2FNumberingOrder_52q_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52q_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52q_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52q_64

/* graph.c */
#define SetupGraph			SetupGraph_52q_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52q_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52q_64
#define SetupGraph_label                SetupGraph_label_52q_64
#define SetupSplitGraph                 SetupSplitGraph_52q_64
#define CreateGraph                     CreateGraph_52q_64
#define InitGraph                       InitGraph_52q_64
#define FreeSData                       FreeSData_52q_64
#define FreeRData                       FreeRData_52q_64
#define FreeGraph                       FreeGraph_52q_64
#define graph_WriteToDisk               graph_WriteToDisk_52q_64
#define graph_ReadFromDisk              graph_ReadFromDisk_52q_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52q_64
#define InitSeparator			InitSeparator_52q_64
#define RandomBisection			RandomBisection_52q_64
#define GrowBisection			GrowBisection_52q_64
#define McRandomBisection               McRandomBisection_52q_64
#define McGrowBisection                 McGrowBisection_52q_64
#define GrowBisectionNode		GrowBisectionNode_52q_64
#define GrowBisectionNode2		GrowBisectionNode2_52q_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52q_64
#define InitKWayPartitioning            InitKWayPartitioning_52q_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52q_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52q_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52q_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52q_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52q_64
#define IsArticulationNode              IsArticulationNode_52q_64
#define KWayVolUpdate                   KWayVolUpdate_52q_64
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52q_64
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52q_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52q_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52q_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52q_64
#define ProjectKWayPartition		ProjectKWayPartition_52q_64
#define ComputeKWayBoundary		ComputeKWayBoundary_52q_64
#define ComputeKWayVolGains             ComputeKWayVolGains_52q_64
#define IsBalanced			IsBalanced_52q_64

/* mcutil */
#define rvecle                          rvecle_52q_64
#define rvecge                          rvecge_52q_64
#define rvecsumle                       rvecsumle_52q_64
#define rvecmaxdiff                     rvecmaxdiff_52q_64
#define ivecle                          ivecle_52q_64
#define ivecge                          ivecge_52q_64
#define ivecaxpylez                     ivecaxpylez_52q_64
#define ivecaxpygez                     ivecaxpygez_52q_64
#define BetterVBalance                  BetterVBalance_52q_64
#define BetterBalance2Way               BetterBalance2Way_52q_64
#define BetterBalanceKWay               BetterBalanceKWay_52q_64
#define ComputeLoadImbalance            ComputeLoadImbalance_52q_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52q_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52q_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52q_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52q_64
#define FindCommonElements              FindCommonElements_52q_64
#define CreateGraphNodal                CreateGraphNodal_52q_64
#define FindCommonNodes                 FindCommonNodes_52q_64
#define CreateMesh                      CreateMesh_52q_64
#define InitMesh                        InitMesh_52q_64
#define FreeMesh                        FreeMesh_52q_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52q_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52q_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52q_64
#define PrintSubDomainGraph             PrintSubDomainGraph_52q_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52q_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52q_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52q_64

/* mincover.c */
#define MinCover			MinCover_52q_64
#define MinCover_Augment		MinCover_Augment_52q_64
#define MinCover_Decompose		MinCover_Decompose_52q_64
#define MinCover_ColDFS			MinCover_ColDFS_52q_64
#define MinCover_RowDFS			MinCover_RowDFS_52q_64

/* mmd.c */
#define genmmd				genmmd_52q_64
#define mmdelm				mmdelm_52q_64
#define mmdint				mmdint_52q_64
#define mmdnum				mmdnum_52q_64
#define mmdupd				mmdupd_52q_64

/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52q_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52q_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52q_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52q_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52q_64
#define SplitGraphOrder			SplitGraphOrder_52q_64
#define SplitGraphOrderCC		SplitGraphOrderCC_52q_64
#define MMDOrder			MMDOrder_52q_64

/* options.c */
#define SetupCtrl                       SetupCtrl_52q_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52q_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52q_64
#define PrintCtrl                       PrintCtrl_52q_64
#define FreeCtrl                        FreeCtrl_52q_64
#define CheckParams                     CheckParams_52q_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52q_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52q_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52q_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52q_64
#define MultilevelBisect		MultilevelBisect_52q_64
#define SplitGraphPart			SplitGraphPart_52q_64

/* refine.c */
#define Refine2Way			Refine2Way_52q_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52q_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52q_64
#define Project2WayPartition		Project2WayPartition_52q_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52q_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52q_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52q_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52q_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52q_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52q_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52q_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52q_64
#define Project2WayNodePartition	Project2WayNodePartition_52q_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52q_64
#define ComputePartitionBalance		ComputePartitionBalance_52q_64
#define ComputeElementBalance		ComputeElementBalance_52q_64

/* timing.c */
#define InitTimers			InitTimers_52q_64
#define PrintTimers			PrintTimers_52q_64

/* util.c */
#define iargmax_strd                    iargmax_strd_52q_64
#define iargmax_nrm                     iargmax_nrm_52q_64
#define iargmax2_nrm                    iargmax2_nrm_52q_64
#define rargmax2                        rargmax2_52q_64
#define InitRandom                      InitRandom_52q_64
#define metis_rcode                     metis_rcode_52q_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52q_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52q_64
#define FreeWorkSpace                   FreeWorkSpace_52q_64
#define wspacemalloc                    wspacemalloc_52q_64
#define wspacepush                      wspacepush_52q_64
#define wspacepop                       wspacepop_52q_64
#define iwspacemalloc                   iwspacemalloc_52q_64
#define rwspacemalloc                   rwspacemalloc_52q_64
#define ikvwspacemalloc                 ikvwspacemalloc_52q_64
#define cnbrpoolReset                   cnbrpoolReset_52q_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_52q_64
#define vnbrpoolReset                   vnbrpoolReset_52q_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_52q_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52q
#define METIS_Free                      METIS_Free_52q
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52q

/* balance.c */
#define Balance2Way			Balance2Way_52q
#define Bnd2WayBalance			Bnd2WayBalance_52q
#define General2WayBalance		General2WayBalance_52q
#define McGeneral2WayBalance            McGeneral2WayBalance_52q

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52q

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52q
#define CheckInputGraphWeights          CheckInputGraphWeights_52q
#define FixGraph                        FixGraph_52q

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52q
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52q
#define Match_RM                        Match_RM_52q
#define Match_SHEM                      Match_SHEM_52q
#define Match_2Hop                      Match_2Hop_52q
#define Match_2HopAny                   Match_2HopAny_52q
#define Match_2HopAll                   Match_2HopAll_52q
#define Match_JC                        Match_JC_52q
#define PrintCGraphStats                PrintCGraphStats_52q
#define CreateCoarseGraph		CreateCoarseGraph_52q
#define SetupCoarseGraph		SetupCoarseGraph_52q
#define ReAdjustMemory			ReAdjustMemory_52q

/* compress.c */
#define CompressGraph			CompressGraph_52q
#define PruneGraph			PruneGraph_52q

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52q
#define IsConnected                     IsConnected_52q
#define IsConnectedSubdomain            IsConnectedSubdomain_52q
#define FindSepInducedComponents        FindSepInducedComponents_52q
#define EliminateComponents             EliminateComponents_52q
#define MoveGroupContigForCut           MoveGroupContigForCut_52q
#define MoveGroupContigForVol           MoveGroupContigForVol_52q
#define ComputeBFSOrdering              ComputeBFSOrdering_52q

/* debug.c */
#define ComputeCut			ComputeCut_52q
#define ComputeVolume			ComputeVolume_52q
#define ComputeMaxCut			ComputeMaxCut_52q
#define CheckBnd			CheckBnd_52q
#define CheckBnd2			CheckBnd2_52q
#define CheckNodeBnd			CheckNodeBnd_52q
#define CheckRInfo			CheckRInfo_52q
#define CheckNodePartitionParams	CheckNodePartitionParams_52q
#define IsSeparable			IsSeparable_52q
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52q

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52q
#define FM_2WayCutRefine                FM_2WayCutRefine_52q
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52q
#define SelectQueue                     SelectQueue_52q
#define Print2WayRefineStats            Print2WayRefineStats_52q

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52q
#define Change2FNumbering		Change2FNumbering_52q
#define Change2FNumbering2		Change2FNumbering2_52q
#define Change2FNumberingOrder		Change2FNumberingOrder_52q
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52q
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52q
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52q

/* graph.c */
#define SetupGraph			SetupGraph_52q
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52q
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52q
#define SetupGraph_label                SetupGraph_label_52q
#define SetupSplitGraph                 SetupSplitGraph_52q
#define CreateGraph                     CreateGraph_52q
#define InitGraph                       InitGraph_52q
#define FreeSData                       FreeSData_52q
#define FreeRData                       FreeRData_52q
#define FreeGraph                       FreeGraph_52q
#define graph_WriteToDisk               graph_WriteToDisk_52q
#define graph_ReadFromDisk              graph_ReadFromDisk_52q

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52q
#define InitSeparator			InitSeparator_52q
#define RandomBisection			RandomBisection_52q
#define GrowBisection			GrowBisection_52q
#define McRandomBisection               McRandomBisection_52q
#define McGrowBisection                 McGrowBisection_52q
#define GrowBisectionNode		GrowBisectionNode_52q
#define GrowBisectionNode2		GrowBisectionNode2_52q

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52q
#define InitKWayPartitioning            InitKWayPartitioning_52q

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52q
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52q
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52q
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52q
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52q
#define IsArticulationNode              IsArticulationNode_52q
#define KWayVolUpdate                   KWayVolUpdate_52q
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52q
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52q

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52q
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52q
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52q
#define ProjectKWayPartition		ProjectKWayPartition_52q
#define ComputeKWayBoundary		ComputeKWayBoundary_52q
#define ComputeKWayVolGains             ComputeKWayVolGains_52q
#define IsBalanced			IsBalanced_52q

/* mcutil */
#define rvecle                          rvecle_52q
#define rvecge                          rvecge_52q
#define rvecsumle                       rvecsumle_52q
#define rvecmaxdiff                     rvecmaxdiff_52q
#define ivecle                          ivecle_52q
#define ivecge                          ivecge_52q
#define ivecaxpylez                     ivecaxpylez_52q
#define ivecaxpygez                     ivecaxpygez_52q
#define BetterVBalance                  BetterVBalance_52q
#define BetterBalance2Way               BetterBalance2Way_52q
#define BetterBalanceKWay               BetterBalanceKWay_52q
#define ComputeLoadImbalance            ComputeLoadImbalance_52q
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52q
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52q
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52q

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52q
#define FindCommonElements              FindCommonElements_52q
#define CreateGraphNodal                CreateGraphNodal_52q
#define FindCommonNodes                 FindCommonNodes_52q
#define CreateMesh                      CreateMesh_52q
#define InitMesh                        InitMesh_52q
#define FreeMesh                        FreeMesh_52q

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52q

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52q
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52q
#define PrintSubDomainGraph             PrintSubDomainGraph_52q
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52q
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52q
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52q

/* mincover.c */
#define MinCover			MinCover_52q
#define MinCover_Augment		MinCover_Augment_52q
#define MinCover_Decompose		MinCover_Decompose_52q
#define MinCover_ColDFS			MinCover_ColDFS_52q
#define MinCover_RowDFS			MinCover_RowDFS_52q

/* mmd.c */
#define genmmd				genmmd_52q
#define mmdelm				mmdelm_52q
#define mmdint				mmdint_52q
#define mmdnum				mmdnum_52q
#define mmdupd				mmdupd_52q


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52q
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52q
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52q
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52q
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52q
#define SplitGraphOrder			SplitGraphOrder_52q
#define SplitGraphOrderCC		SplitGraphOrderCC_52q
#define MMDOrder			MMDOrder_52q

/* options.c */
#define SetupCtrl                       SetupCtrl_52q
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52q
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52q
#define PrintCtrl                       PrintCtrl_52q
#define FreeCtrl                        FreeCtrl_52q
#define CheckParams                     CheckParams_52q

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52q
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52q
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52q

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52q
#define MultilevelBisect		MultilevelBisect_52q
#define SplitGraphPart			SplitGraphPart_52q

/* refine.c */
#define Refine2Way			Refine2Way_52q
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52q
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52q
#define Project2WayPartition		Project2WayPartition_52q

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52q
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52q

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52q
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52q
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52q

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52q
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52q
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52q
#define Project2WayNodePartition	Project2WayNodePartition_52q

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52q
#define ComputePartitionBalance		ComputePartitionBalance_52q
#define ComputeElementBalance		ComputeElementBalance_52q

/* timing.c */
#define InitTimers			InitTimers_52q
#define PrintTimers			PrintTimers_52q

/* util.c */
#define iargmax_strd                    iargmax_strd_52q
#define iargmax_nrm                     iargmax_nrm_52q
#define iargmax2_nrm                    iargmax2_nrm_52q
#define rargmax2                        rargmax2_52q
#define InitRandom                      InitRandom_52q
#define metis_rcode                     metis_rcode_52q

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52q
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52q
#define FreeWorkSpace                   FreeWorkSpace_52q
#define wspacemalloc                    wspacemalloc_52q
#define wspacepush                      wspacepush_52q
#define wspacepop                       wspacepop_52q
#define iwspacemalloc                   iwspacemalloc_52q
#define rwspacemalloc                   rwspacemalloc_52q
#define ikvwspacemalloc                 ikvwspacemalloc_52q
#define cnbrpoolReset                   cnbrpoolReset_52q
#define cnbrpoolGetNext                 cnbrpoolGetNext_52q
#define vnbrpoolReset                   vnbrpoolReset_52q
#define vnbrpoolGetNext                 vnbrpoolGetNext_52q

#endif
