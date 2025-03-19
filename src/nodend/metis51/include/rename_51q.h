/* quadruple precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51q_64
#define METIS_Free                      METIS_Free_51q_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51q_64

/* balance.c */
#define Balance2Way			Balance2Way_51q_64
#define Bnd2WayBalance			Bnd2WayBalance_51q_64
#define General2WayBalance		General2WayBalance_51q_64
#define McGeneral2WayBalance            McGeneral2WayBalance_51q_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51q_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51q_64
#define CheckInputGraphWeights          CheckInputGraphWeights_51q_64
#define FixGraph                        FixGraph_51q_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51q_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51q_64
#define Match_RM                        Match_RM_51q_64
#define Match_SHEM                      Match_SHEM_51q_64
#define Match_2Hop                      Match_2Hop_51q_64
#define Match_2HopAny                   Match_2HopAny_51q_64
#define Match_2HopAll                   Match_2HopAll_51q_64
#define PrintCGraphStats                PrintCGraphStats_51q_64
#define CreateCoarseGraph		CreateCoarseGraph_51q_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51q_64
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51q_64
#define SetupCoarseGraph		SetupCoarseGraph_51q_64
#define ReAdjustMemory			ReAdjustMemory_51q_64

/* compress.c */
#define CompressGraph			CompressGraph_51q_64
#define PruneGraph			PruneGraph_51q_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51q_64
#define IsConnected                     IsConnected_51q_64
#define IsConnectedSubdomain            IsConnectedSubdomain_51q_64
#define FindSepInducedComponents        FindSepInducedComponents_51q_64
#define EliminateComponents             EliminateComponents_51q_64
#define MoveGroupContigForCut           MoveGroupContigForCut_51q_64
#define MoveGroupContigForVol           MoveGroupContigForVol_51q_64
#define ComputeBFSOrdering              ComputeBFSOrdering_51q_64

/* debug.c */
#define ComputeCut			ComputeCut_51q_64
#define ComputeVolume			ComputeVolume_51q_64
#define ComputeMaxCut			ComputeMaxCut_51q_64
#define CheckBnd			CheckBnd_51q_64
#define CheckBnd2			CheckBnd2_51q_64
#define CheckNodeBnd			CheckNodeBnd_51q_64
#define CheckRInfo			CheckRInfo_51q_64
#define CheckNodePartitionParams	CheckNodePartitionParams_51q_64
#define IsSeparable			IsSeparable_51q_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51q_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51q_64
#define FM_2WayCutRefine                FM_2WayCutRefine_51q_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51q_64
#define SelectQueue                     SelectQueue_51q_64
#define Print2WayRefineStats            Print2WayRefineStats_51q_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51q_64
#define Change2FNumbering		Change2FNumbering_51q_64
#define Change2FNumbering2		Change2FNumbering2_51q_64
#define Change2FNumberingOrder		Change2FNumberingOrder_51q_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51q_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51q_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51q_64

/* graph.c */
#define SetupGraph			SetupGraph_51q_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51q_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51q_64
#define SetupGraph_label                SetupGraph_label_51q_64
#define SetupSplitGraph                 SetupSplitGraph_51q_64
#define CreateGraph                     CreateGraph_51q_64
#define InitGraph                       InitGraph_51q_64
#define FreeRData                       FreeRData_51q_64
#define FreeGraph                       FreeGraph_51q_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51q_64
#define InitSeparator			InitSeparator_51q_64
#define RandomBisection			RandomBisection_51q_64
#define GrowBisection			GrowBisection_51q_64
#define McRandomBisection               McRandomBisection_51q_64
#define McGrowBisection                 McGrowBisection_51q_64
#define GrowBisectionNode		GrowBisectionNode_51q_64
#define GrowBisectionNode2		GrowBisectionNode2_51q_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51q_64
#define InitKWayPartitioning            InitKWayPartitioning_51q_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51q_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51q_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51q_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51q_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51q_64
#define IsArticulationNode              IsArticulationNode_51q_64
#define KWayVolUpdate                   KWayVolUpdate_51q_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51q_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51q_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51q_64
#define ProjectKWayPartition		ProjectKWayPartition_51q_64
#define ComputeKWayBoundary		ComputeKWayBoundary_51q_64
#define ComputeKWayVolGains             ComputeKWayVolGains_51q_64
#define IsBalanced			IsBalanced_51q_64

/* mcutil */
#define rvecle                          rvecle_51q_64
#define rvecge                          rvecge_51q_64
#define rvecsumle                       rvecsumle_51q_64
#define rvecmaxdiff                     rvecmaxdiff_51q_64
#define ivecle                          ivecle_51q_64
#define ivecge                          ivecge_51q_64
#define ivecaxpylez                     ivecaxpylez_51q_64
#define ivecaxpygez                     ivecaxpygez_51q_64
#define BetterVBalance                  BetterVBalance_51q_64
#define BetterBalance2Way               BetterBalance2Way_51q_64
#define BetterBalanceKWay               BetterBalanceKWay_51q_64
#define ComputeLoadImbalance            ComputeLoadImbalance_51q_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51q_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51q_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51q_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51q_64
#define FindCommonElements              FindCommonElements_51q_64
#define CreateGraphNodal                CreateGraphNodal_51q_64
#define FindCommonNodes                 FindCommonNodes_51q_64
#define CreateMesh                      CreateMesh_51q_64
#define InitMesh                        InitMesh_51q_64
#define FreeMesh                        FreeMesh_51q_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51q_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51q_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51q_64
#define PrintSubDomainGraph             PrintSubDomainGraph_51q_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51q_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51q_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51q_64

/* mincover.c */
#define MinCover			MinCover_51q_64
#define MinCover_Augment		MinCover_Augment_51q_64
#define MinCover_Decompose		MinCover_Decompose_51q_64
#define MinCover_ColDFS			MinCover_ColDFS_51q_64
#define MinCover_RowDFS			MinCover_RowDFS_51q_64

/* mmd.c */
#define genmmd				genmmd_51q_64
#define mmdelm				mmdelm_51q_64
#define mmdint				mmdint_51q_64
#define mmdnum				mmdnum_51q_64
#define mmdupd				mmdupd_51q_64


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51q_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51q_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51q_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51q_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51q_64
#define SplitGraphOrder			SplitGraphOrder_51q_64
#define SplitGraphOrderCC		SplitGraphOrderCC_51q_64
#define MMDOrder			MMDOrder_51q_64

/* options.c */
#define SetupCtrl                       SetupCtrl_51q_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51q_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51q_64
#define PrintCtrl                       PrintCtrl_51q_64
#define FreeCtrl                        FreeCtrl_51q_64
#define CheckParams                     CheckParams_51q_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51q_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51q_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51q_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51q_64
#define MultilevelBisect		MultilevelBisect_51q_64
#define SplitGraphPart			SplitGraphPart_51q_64

/* refine.c */
#define Refine2Way			Refine2Way_51q_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51q_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51q_64
#define Project2WayPartition		Project2WayPartition_51q_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51q_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51q_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51q_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51q_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51q_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51q_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51q_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51q_64
#define Project2WayNodePartition	Project2WayNodePartition_51q_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51q_64
#define ComputePartitionBalance		ComputePartitionBalance_51q_64
#define ComputeElementBalance		ComputeElementBalance_51q_64

/* timing.c */
#define InitTimers			InitTimers_51q_64
#define PrintTimers			PrintTimers_51q_64

/* util.c */
#define iargmax_strd                    iargmax_strd_51q_64
#define iargmax_nrm                     iargmax_nrm_51q_64
#define iargmax2_nrm                    iargmax2_nrm_51q_64
#define rargmax2                        rargmax2_51q_64
#define InitRandom                      InitRandom_51q_64
#define metis_rcode                     metis_rcode_51q_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51q_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51q_64
#define FreeWorkSpace                   FreeWorkSpace_51q_64
#define wspacemalloc                    wspacemalloc_51q_64
#define wspacepush                      wspacepush_51q_64
#define wspacepop                       wspacepop_51q_64
#define iwspacemalloc                   iwspacemalloc_51q_64
#define rwspacemalloc                   rwspacemalloc_51q_64
#define ikvwspacemalloc                 ikvwspacemalloc_51q_64
#define cnbrpoolReset                   cnbrpoolReset_51q_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_51q_64
#define vnbrpoolReset                   vnbrpoolReset_51q_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_51q_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51q
#define METIS_Free                      METIS_Free_51q
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51q

/* balance.c */
#define Balance2Way			Balance2Way_51q
#define Bnd2WayBalance			Bnd2WayBalance_51q
#define General2WayBalance		General2WayBalance_51q
#define McGeneral2WayBalance            McGeneral2WayBalance_51q

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51q

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51q
#define CheckInputGraphWeights          CheckInputGraphWeights_51q
#define FixGraph                        FixGraph_51q

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51q
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51q
#define Match_RM                        Match_RM_51q
#define Match_SHEM                      Match_SHEM_51q
#define Match_2Hop                      Match_2Hop_51q
#define Match_2HopAny                   Match_2HopAny_51q
#define Match_2HopAll                   Match_2HopAll_51q
#define PrintCGraphStats                PrintCGraphStats_51q
#define CreateCoarseGraph		CreateCoarseGraph_51q
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51q
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51q
#define SetupCoarseGraph		SetupCoarseGraph_51q
#define ReAdjustMemory			ReAdjustMemory_51q

/* compress.c */
#define CompressGraph			CompressGraph_51q
#define PruneGraph			PruneGraph_51q

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51q
#define IsConnected                     IsConnected_51q
#define IsConnectedSubdomain            IsConnectedSubdomain_51q
#define FindSepInducedComponents        FindSepInducedComponents_51q
#define EliminateComponents             EliminateComponents_51q
#define MoveGroupContigForCut           MoveGroupContigForCut_51q
#define MoveGroupContigForVol           MoveGroupContigForVol_51q
#define ComputeBFSOrdering              ComputeBFSOrdering_51q

/* debug.c */
#define ComputeCut			ComputeCut_51q
#define ComputeVolume			ComputeVolume_51q
#define ComputeMaxCut			ComputeMaxCut_51q
#define CheckBnd			CheckBnd_51q
#define CheckBnd2			CheckBnd2_51q
#define CheckNodeBnd			CheckNodeBnd_51q
#define CheckRInfo			CheckRInfo_51q
#define CheckNodePartitionParams	CheckNodePartitionParams_51q
#define IsSeparable			IsSeparable_51q
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51q

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51q
#define FM_2WayCutRefine                FM_2WayCutRefine_51q
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51q
#define SelectQueue                     SelectQueue_51q
#define Print2WayRefineStats            Print2WayRefineStats_51q

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51q
#define Change2FNumbering		Change2FNumbering_51q
#define Change2FNumbering2		Change2FNumbering2_51q
#define Change2FNumberingOrder		Change2FNumberingOrder_51q
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51q
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51q
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51q

/* graph.c */
#define SetupGraph			SetupGraph_51q
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51q
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51q
#define SetupGraph_label                SetupGraph_label_51q
#define SetupSplitGraph                 SetupSplitGraph_51q
#define CreateGraph                     CreateGraph_51q
#define InitGraph                       InitGraph_51q
#define FreeRData                       FreeRData_51q
#define FreeGraph                       FreeGraph_51q

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51q
#define InitSeparator			InitSeparator_51q
#define RandomBisection			RandomBisection_51q
#define GrowBisection			GrowBisection_51q
#define McRandomBisection               McRandomBisection_51q
#define McGrowBisection                 McGrowBisection_51q
#define GrowBisectionNode		GrowBisectionNode_51q
#define GrowBisectionNode2		GrowBisectionNode2_51q

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51q
#define InitKWayPartitioning            InitKWayPartitioning_51q

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51q
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51q
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51q
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51q
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51q
#define IsArticulationNode              IsArticulationNode_51q
#define KWayVolUpdate                   KWayVolUpdate_51q

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51q
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51q
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51q
#define ProjectKWayPartition		ProjectKWayPartition_51q
#define ComputeKWayBoundary		ComputeKWayBoundary_51q
#define ComputeKWayVolGains             ComputeKWayVolGains_51q
#define IsBalanced			IsBalanced_51q

/* mcutil */
#define rvecle                          rvecle_51q
#define rvecge                          rvecge_51q
#define rvecsumle                       rvecsumle_51q
#define rvecmaxdiff                     rvecmaxdiff_51q
#define ivecle                          ivecle_51q
#define ivecge                          ivecge_51q
#define ivecaxpylez                     ivecaxpylez_51q
#define ivecaxpygez                     ivecaxpygez_51q
#define BetterVBalance                  BetterVBalance_51q
#define BetterBalance2Way               BetterBalance2Way_51q
#define BetterBalanceKWay               BetterBalanceKWay_51q
#define ComputeLoadImbalance            ComputeLoadImbalance_51q
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51q
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51q
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51q

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51q
#define FindCommonElements              FindCommonElements_51q
#define CreateGraphNodal                CreateGraphNodal_51q
#define FindCommonNodes                 FindCommonNodes_51q
#define CreateMesh                      CreateMesh_51q
#define InitMesh                        InitMesh_51q
#define FreeMesh                        FreeMesh_51q

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51q

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51q
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51q
#define PrintSubDomainGraph             PrintSubDomainGraph_51q
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51q
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51q
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51q

/* mincover.c */
#define MinCover			MinCover_51q
#define MinCover_Augment		MinCover_Augment_51q
#define MinCover_Decompose		MinCover_Decompose_51q
#define MinCover_ColDFS			MinCover_ColDFS_51q
#define MinCover_RowDFS			MinCover_RowDFS_51q

/* mmd.c */
#define genmmd				genmmd_51q
#define mmdelm				mmdelm_51q
#define mmdint				mmdint_51q
#define mmdnum				mmdnum_51q
#define mmdupd				mmdupd_51q


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51q
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51q
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51q
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51q
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51q
#define SplitGraphOrder			SplitGraphOrder_51q
#define SplitGraphOrderCC		SplitGraphOrderCC_51q
#define MMDOrder			MMDOrder_51q

/* options.c */
#define SetupCtrl                       SetupCtrl_51q
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51q
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51q
#define PrintCtrl                       PrintCtrl_51q
#define FreeCtrl                        FreeCtrl_51q
#define CheckParams                     CheckParams_51q

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51q
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51q
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51q

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51q
#define MultilevelBisect		MultilevelBisect_51q
#define SplitGraphPart			SplitGraphPart_51q

/* refine.c */
#define Refine2Way			Refine2Way_51q
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51q
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51q
#define Project2WayPartition		Project2WayPartition_51q

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51q
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51q

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51q
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51q
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51q

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51q
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51q
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51q
#define Project2WayNodePartition	Project2WayNodePartition_51q

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51q
#define ComputePartitionBalance		ComputePartitionBalance_51q
#define ComputeElementBalance		ComputeElementBalance_51q

/* timing.c */
#define InitTimers			InitTimers_51q
#define PrintTimers			PrintTimers_51q

/* util.c */
#define iargmax_strd                    iargmax_strd_51q
#define iargmax_nrm                     iargmax_nrm_51q
#define iargmax2_nrm                    iargmax2_nrm_51q
#define rargmax2                        rargmax2_51q
#define InitRandom                      InitRandom_51q
#define metis_rcode                     metis_rcode_51q

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51q
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51q
#define FreeWorkSpace                   FreeWorkSpace_51q
#define wspacemalloc                    wspacemalloc_51q
#define wspacepush                      wspacepush_51q
#define wspacepop                       wspacepop_51q
#define iwspacemalloc                   iwspacemalloc_51q
#define rwspacemalloc                   rwspacemalloc_51q
#define ikvwspacemalloc                 ikvwspacemalloc_51q
#define cnbrpoolReset                   cnbrpoolReset_51q
#define cnbrpoolGetNext                 cnbrpoolGetNext_51q
#define vnbrpoolReset                   vnbrpoolReset_51q
#define vnbrpoolGetNext                 vnbrpoolGetNext_51q

#endif
