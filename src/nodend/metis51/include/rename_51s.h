/* single precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51s_64
#define METIS_Free                      METIS_Free_51s_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51s_64

/* balance.c */
#define Balance2Way			Balance2Way_51s_64
#define Bnd2WayBalance			Bnd2WayBalance_51s_64
#define General2WayBalance		General2WayBalance_51s_64
#define McGeneral2WayBalance            McGeneral2WayBalance_51s_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51s_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51s_64
#define CheckInputGraphWeights          CheckInputGraphWeights_51s_64
#define FixGraph                        FixGraph_51s_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51s_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51s_64
#define Match_RM                        Match_RM_51s_64
#define Match_SHEM                      Match_SHEM_51s_64
#define Match_2Hop                      Match_2Hop_51s_64
#define Match_2HopAny                   Match_2HopAny_51s_64
#define Match_2HopAll                   Match_2HopAll_51s_64
#define PrintCGraphStats                PrintCGraphStats_51s_64
#define CreateCoarseGraph		CreateCoarseGraph_51s_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51s_64
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51s_64
#define SetupCoarseGraph		SetupCoarseGraph_51s_64
#define ReAdjustMemory			ReAdjustMemory_51s_64

/* compress.c */
#define CompressGraph			CompressGraph_51s_64
#define PruneGraph			PruneGraph_51s_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51s_64
#define IsConnected                     IsConnected_51s_64
#define IsConnectedSubdomain            IsConnectedSubdomain_51s_64
#define FindSepInducedComponents        FindSepInducedComponents_51s_64
#define EliminateComponents             EliminateComponents_51s_64
#define MoveGroupContigForCut           MoveGroupContigForCut_51s_64
#define MoveGroupContigForVol           MoveGroupContigForVol_51s_64
#define ComputeBFSOrdering              ComputeBFSOrdering_51s_64

/* debug.c */
#define ComputeCut			ComputeCut_51s_64
#define ComputeVolume			ComputeVolume_51s_64
#define ComputeMaxCut			ComputeMaxCut_51s_64
#define CheckBnd			CheckBnd_51s_64
#define CheckBnd2			CheckBnd2_51s_64
#define CheckNodeBnd			CheckNodeBnd_51s_64
#define CheckRInfo			CheckRInfo_51s_64
#define CheckNodePartitionParams	CheckNodePartitionParams_51s_64
#define IsSeparable			IsSeparable_51s_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51s_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51s_64
#define FM_2WayCutRefine                FM_2WayCutRefine_51s_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51s_64
#define SelectQueue                     SelectQueue_51s_64
#define Print2WayRefineStats            Print2WayRefineStats_51s_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51s_64
#define Change2FNumbering		Change2FNumbering_51s_64
#define Change2FNumbering2		Change2FNumbering2_51s_64
#define Change2FNumberingOrder		Change2FNumberingOrder_51s_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51s_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51s_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51s_64

/* graph.c */
#define SetupGraph			SetupGraph_51s_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51s_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51s_64
#define SetupGraph_label                SetupGraph_label_51s_64
#define SetupSplitGraph                 SetupSplitGraph_51s_64
#define CreateGraph                     CreateGraph_51s_64
#define InitGraph                       InitGraph_51s_64
#define FreeRData                       FreeRData_51s_64
#define FreeGraph                       FreeGraph_51s_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51s_64
#define InitSeparator			InitSeparator_51s_64
#define RandomBisection			RandomBisection_51s_64
#define GrowBisection			GrowBisection_51s_64
#define McRandomBisection               McRandomBisection_51s_64
#define McGrowBisection                 McGrowBisection_51s_64
#define GrowBisectionNode		GrowBisectionNode_51s_64
#define GrowBisectionNode2		GrowBisectionNode2_51s_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51s_64
#define InitKWayPartitioning            InitKWayPartitioning_51s_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51s_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51s_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51s_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51s_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51s_64
#define IsArticulationNode              IsArticulationNode_51s_64
#define KWayVolUpdate                   KWayVolUpdate_51s_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51s_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51s_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51s_64
#define ProjectKWayPartition		ProjectKWayPartition_51s_64
#define ComputeKWayBoundary		ComputeKWayBoundary_51s_64
#define ComputeKWayVolGains             ComputeKWayVolGains_51s_64
#define IsBalanced			IsBalanced_51s_64

/* mcutil */
#define rvecle                          rvecle_51s_64
#define rvecge                          rvecge_51s_64
#define rvecsumle                       rvecsumle_51s_64
#define rvecmaxdiff                     rvecmaxdiff_51s_64
#define ivecle                          ivecle_51s_64
#define ivecge                          ivecge_51s_64
#define ivecaxpylez                     ivecaxpylez_51s_64
#define ivecaxpygez                     ivecaxpygez_51s_64
#define BetterVBalance                  BetterVBalance_51s_64
#define BetterBalance2Way               BetterBalance2Way_51s_64
#define BetterBalanceKWay               BetterBalanceKWay_51s_64
#define ComputeLoadImbalance            ComputeLoadImbalance_51s_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51s_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51s_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51s_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51s_64
#define FindCommonElements              FindCommonElements_51s_64
#define CreateGraphNodal                CreateGraphNodal_51s_64
#define FindCommonNodes                 FindCommonNodes_51s_64
#define CreateMesh                      CreateMesh_51s_64
#define InitMesh                        InitMesh_51s_64
#define FreeMesh                        FreeMesh_51s_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51s_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51s_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51s_64
#define PrintSubDomainGraph             PrintSubDomainGraph_51s_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51s_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51s_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51s_64

/* mincover.c */
#define MinCover			MinCover_51s_64
#define MinCover_Augment		MinCover_Augment_51s_64
#define MinCover_Decompose		MinCover_Decompose_51s_64
#define MinCover_ColDFS			MinCover_ColDFS_51s_64
#define MinCover_RowDFS			MinCover_RowDFS_51s_64

/* mmd.c */
#define genmmd				genmmd_51s_64
#define mmdelm				mmdelm_51s_64
#define mmdint				mmdint_51s_64
#define mmdnum				mmdnum_51s_64
#define mmdupd				mmdupd_51s_64


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51s_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51s_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51s_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51s_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51s_64
#define SplitGraphOrder			SplitGraphOrder_51s_64
#define SplitGraphOrderCC		SplitGraphOrderCC_51s_64
#define MMDOrder			MMDOrder_51s_64

/* options.c */
#define SetupCtrl                       SetupCtrl_51s_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51s_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51s_64
#define PrintCtrl                       PrintCtrl_51s_64
#define FreeCtrl                        FreeCtrl_51s_64
#define CheckParams                     CheckParams_51s_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51s_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51s_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51s_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51s_64
#define MultilevelBisect		MultilevelBisect_51s_64
#define SplitGraphPart			SplitGraphPart_51s_64

/* refine.c */
#define Refine2Way			Refine2Way_51s_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51s_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51s_64
#define Project2WayPartition		Project2WayPartition_51s_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51s_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51s_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51s_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51s_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51s_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51s_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51s_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51s_64
#define Project2WayNodePartition	Project2WayNodePartition_51s_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51s_64
#define ComputePartitionBalance		ComputePartitionBalance_51s_64
#define ComputeElementBalance		ComputeElementBalance_51s_64

/* timing.c */
#define InitTimers			InitTimers_51s_64
#define PrintTimers			PrintTimers_51s_64

/* util.c */
#define iargmax_strd                    iargmax_strd_51s_64
#define iargmax_nrm                     iargmax_nrm_51s_64
#define iargmax2_nrm                    iargmax2_nrm_51s_64
#define rargmax2                        rargmax2_51s_64
#define InitRandom                      InitRandom_51s_64
#define metis_rcode                     metis_rcode_51s_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51s_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51s_64
#define FreeWorkSpace                   FreeWorkSpace_51s_64
#define wspacemalloc                    wspacemalloc_51s_64
#define wspacepush                      wspacepush_51s_64
#define wspacepop                       wspacepop_51s_64
#define iwspacemalloc                   iwspacemalloc_51s_64
#define rwspacemalloc                   rwspacemalloc_51s_64
#define ikvwspacemalloc                 ikvwspacemalloc_51s_64
#define cnbrpoolReset                   cnbrpoolReset_51s_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_51s_64
#define vnbrpoolReset                   vnbrpoolReset_51s_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_51s_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51s
#define METIS_Free                      METIS_Free_51s
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51s

/* balance.c */
#define Balance2Way			Balance2Way_51s
#define Bnd2WayBalance			Bnd2WayBalance_51s
#define General2WayBalance		General2WayBalance_51s
#define McGeneral2WayBalance            McGeneral2WayBalance_51s

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51s

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51s
#define CheckInputGraphWeights          CheckInputGraphWeights_51s
#define FixGraph                        FixGraph_51s

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51s
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51s
#define Match_RM                        Match_RM_51s
#define Match_SHEM                      Match_SHEM_51s
#define Match_2Hop                      Match_2Hop_51s
#define Match_2HopAny                   Match_2HopAny_51s
#define Match_2HopAll                   Match_2HopAll_51s
#define PrintCGraphStats                PrintCGraphStats_51s
#define CreateCoarseGraph		CreateCoarseGraph_51s
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51s
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51s
#define SetupCoarseGraph		SetupCoarseGraph_51s
#define ReAdjustMemory			ReAdjustMemory_51s

/* compress.c */
#define CompressGraph			CompressGraph_51s
#define PruneGraph			PruneGraph_51s

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51s
#define IsConnected                     IsConnected_51s
#define IsConnectedSubdomain            IsConnectedSubdomain_51s
#define FindSepInducedComponents        FindSepInducedComponents_51s
#define EliminateComponents             EliminateComponents_51s
#define MoveGroupContigForCut           MoveGroupContigForCut_51s
#define MoveGroupContigForVol           MoveGroupContigForVol_51s
#define ComputeBFSOrdering              ComputeBFSOrdering_51s

/* debug.c */
#define ComputeCut			ComputeCut_51s
#define ComputeVolume			ComputeVolume_51s
#define ComputeMaxCut			ComputeMaxCut_51s
#define CheckBnd			CheckBnd_51s
#define CheckBnd2			CheckBnd2_51s
#define CheckNodeBnd			CheckNodeBnd_51s
#define CheckRInfo			CheckRInfo_51s
#define CheckNodePartitionParams	CheckNodePartitionParams_51s
#define IsSeparable			IsSeparable_51s
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51s

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51s
#define FM_2WayCutRefine                FM_2WayCutRefine_51s
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51s
#define SelectQueue                     SelectQueue_51s
#define Print2WayRefineStats            Print2WayRefineStats_51s

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51s
#define Change2FNumbering		Change2FNumbering_51s
#define Change2FNumbering2		Change2FNumbering2_51s
#define Change2FNumberingOrder		Change2FNumberingOrder_51s
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51s
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51s
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51s

/* graph.c */
#define SetupGraph			SetupGraph_51s
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51s
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51s
#define SetupGraph_label                SetupGraph_label_51s
#define SetupSplitGraph                 SetupSplitGraph_51s
#define CreateGraph                     CreateGraph_51s
#define InitGraph                       InitGraph_51s
#define FreeRData                       FreeRData_51s
#define FreeGraph                       FreeGraph_51s

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51s
#define InitSeparator			InitSeparator_51s
#define RandomBisection			RandomBisection_51s
#define GrowBisection			GrowBisection_51s
#define McRandomBisection               McRandomBisection_51s
#define McGrowBisection                 McGrowBisection_51s
#define GrowBisectionNode		GrowBisectionNode_51s
#define GrowBisectionNode2		GrowBisectionNode2_51s

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51s
#define InitKWayPartitioning            InitKWayPartitioning_51s

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51s
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51s
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51s
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51s
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51s
#define IsArticulationNode              IsArticulationNode_51s
#define KWayVolUpdate                   KWayVolUpdate_51s

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51s
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51s
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51s
#define ProjectKWayPartition		ProjectKWayPartition_51s
#define ComputeKWayBoundary		ComputeKWayBoundary_51s
#define ComputeKWayVolGains             ComputeKWayVolGains_51s
#define IsBalanced			IsBalanced_51s

/* mcutil */
#define rvecle                          rvecle_51s
#define rvecge                          rvecge_51s
#define rvecsumle                       rvecsumle_51s
#define rvecmaxdiff                     rvecmaxdiff_51s
#define ivecle                          ivecle_51s
#define ivecge                          ivecge_51s
#define ivecaxpylez                     ivecaxpylez_51s
#define ivecaxpygez                     ivecaxpygez_51s
#define BetterVBalance                  BetterVBalance_51s
#define BetterBalance2Way               BetterBalance2Way_51s
#define BetterBalanceKWay               BetterBalanceKWay_51s
#define ComputeLoadImbalance            ComputeLoadImbalance_51s
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51s
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51s
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51s

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51s
#define FindCommonElements              FindCommonElements_51s
#define CreateGraphNodal                CreateGraphNodal_51s
#define FindCommonNodes                 FindCommonNodes_51s
#define CreateMesh                      CreateMesh_51s
#define InitMesh                        InitMesh_51s
#define FreeMesh                        FreeMesh_51s

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51s

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51s
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51s
#define PrintSubDomainGraph             PrintSubDomainGraph_51s
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51s
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51s
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51s

/* mincover.c */
#define MinCover			MinCover_51s
#define MinCover_Augment		MinCover_Augment_51s
#define MinCover_Decompose		MinCover_Decompose_51s
#define MinCover_ColDFS			MinCover_ColDFS_51s
#define MinCover_RowDFS			MinCover_RowDFS_51s

/* mmd.c */
#define genmmd				genmmd_51s
#define mmdelm				mmdelm_51s
#define mmdint				mmdint_51s
#define mmdnum				mmdnum_51s
#define mmdupd				mmdupd_51s


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51s
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51s
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51s
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51s
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51s
#define SplitGraphOrder			SplitGraphOrder_51s
#define SplitGraphOrderCC		SplitGraphOrderCC_51s
#define MMDOrder			MMDOrder_51s

/* options.c */
#define SetupCtrl                       SetupCtrl_51s
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51s
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51s
#define PrintCtrl                       PrintCtrl_51s
#define FreeCtrl                        FreeCtrl_51s
#define CheckParams                     CheckParams_51s

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51s
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51s
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51s

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51s
#define MultilevelBisect		MultilevelBisect_51s
#define SplitGraphPart			SplitGraphPart_51s

/* refine.c */
#define Refine2Way			Refine2Way_51s
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51s
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51s
#define Project2WayPartition		Project2WayPartition_51s

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51s
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51s

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51s
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51s
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51s

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51s
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51s
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51s
#define Project2WayNodePartition	Project2WayNodePartition_51s

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51s
#define ComputePartitionBalance		ComputePartitionBalance_51s
#define ComputeElementBalance		ComputeElementBalance_51s

/* timing.c */
#define InitTimers			InitTimers_51s
#define PrintTimers			PrintTimers_51s

/* util.c */
#define iargmax_strd                    iargmax_strd_51s
#define iargmax_nrm                     iargmax_nrm_51s
#define iargmax2_nrm                    iargmax2_nrm_51s
#define rargmax2                        rargmax2_51s
#define InitRandom                      InitRandom_51s
#define metis_rcode                     metis_rcode_51s

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51s
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51s
#define FreeWorkSpace                   FreeWorkSpace_51s
#define wspacemalloc                    wspacemalloc_51s
#define wspacepush                      wspacepush_51s
#define wspacepop                       wspacepop_51s
#define iwspacemalloc                   iwspacemalloc_51s
#define rwspacemalloc                   rwspacemalloc_51s
#define ikvwspacemalloc                 ikvwspacemalloc_51s
#define cnbrpoolReset                   cnbrpoolReset_51s
#define cnbrpoolGetNext                 cnbrpoolGetNext_51s
#define vnbrpoolReset                   vnbrpoolReset_51s
#define vnbrpoolGetNext                 vnbrpoolGetNext_51s

#endif
