/* double precision procedures */

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51d_64
#define METIS_Free                      METIS_Free_51d_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51d_64

/* balance.c */
#define Balance2Way			Balance2Way_51d_64
#define Bnd2WayBalance			Bnd2WayBalance_51d_64
#define General2WayBalance		General2WayBalance_51d_64
#define McGeneral2WayBalance            McGeneral2WayBalance_51d_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51d_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51d_64
#define CheckInputGraphWeights          CheckInputGraphWeights_51d_64
#define FixGraph                        FixGraph_51d_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51d_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51d_64
#define Match_RM                        Match_RM_51d_64
#define Match_SHEM                      Match_SHEM_51d_64
#define Match_2Hop                      Match_2Hop_51d_64
#define Match_2HopAny                   Match_2HopAny_51d_64
#define Match_2HopAll                   Match_2HopAll_51d_64
#define PrintCGraphStats                PrintCGraphStats_51d_64
#define CreateCoarseGraph		CreateCoarseGraph_51d_64
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51d_64
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51d_64
#define SetupCoarseGraph		SetupCoarseGraph_51d_64
#define ReAdjustMemory			ReAdjustMemory_51d_64

/* compress.c */
#define CompressGraph			CompressGraph_51d_64
#define PruneGraph			PruneGraph_51d_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51d_64
#define IsConnected                     IsConnected_51d_64
#define IsConnectedSubdomain            IsConnectedSubdomain_51d_64
#define FindSepInducedComponents        FindSepInducedComponents_51d_64
#define EliminateComponents             EliminateComponents_51d_64
#define MoveGroupContigForCut           MoveGroupContigForCut_51d_64
#define MoveGroupContigForVol           MoveGroupContigForVol_51d_64
#define ComputeBFSOrdering              ComputeBFSOrdering_51d_64

/* debug.c */
#define ComputeCut			ComputeCut_51d_64
#define ComputeVolume			ComputeVolume_51d_64
#define ComputeMaxCut			ComputeMaxCut_51d_64
#define CheckBnd			CheckBnd_51d_64
#define CheckBnd2			CheckBnd2_51d_64
#define CheckNodeBnd			CheckNodeBnd_51d_64
#define CheckRInfo			CheckRInfo_51d_64
#define CheckNodePartitionParams	CheckNodePartitionParams_51d_64
#define IsSeparable			IsSeparable_51d_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51d_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51d_64
#define FM_2WayCutRefine                FM_2WayCutRefine_51d_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51d_64
#define SelectQueue                     SelectQueue_51d_64
#define Print2WayRefineStats            Print2WayRefineStats_51d_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51d_64
#define Change2FNumbering		Change2FNumbering_51d_64
#define Change2FNumbering2		Change2FNumbering2_51d_64
#define Change2FNumberingOrder		Change2FNumberingOrder_51d_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51d_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51d_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51d_64

/* graph.c */
#define SetupGraph			SetupGraph_51d_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51d_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51d_64
#define SetupGraph_label                SetupGraph_label_51d_64
#define SetupSplitGraph                 SetupSplitGraph_51d_64
#define CreateGraph                     CreateGraph_51d_64
#define InitGraph                       InitGraph_51d_64
#define FreeRData                       FreeRData_51d_64
#define FreeGraph                       FreeGraph_51d_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51d_64
#define InitSeparator			InitSeparator_51d_64
#define RandomBisection			RandomBisection_51d_64
#define GrowBisection			GrowBisection_51d_64
#define McRandomBisection               McRandomBisection_51d_64
#define McGrowBisection                 McGrowBisection_51d_64
#define GrowBisectionNode		GrowBisectionNode_51d_64
#define GrowBisectionNode2		GrowBisectionNode2_51d_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51d_64
#define InitKWayPartitioning            InitKWayPartitioning_51d_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51d_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51d_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51d_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51d_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51d_64
#define IsArticulationNode              IsArticulationNode_51d_64
#define KWayVolUpdate                   KWayVolUpdate_51d_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51d_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51d_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51d_64
#define ProjectKWayPartition		ProjectKWayPartition_51d_64
#define ComputeKWayBoundary		ComputeKWayBoundary_51d_64
#define ComputeKWayVolGains             ComputeKWayVolGains_51d_64
#define IsBalanced			IsBalanced_51d_64

/* mcutil */
#define rvecle                          rvecle_51d_64
#define rvecge                          rvecge_51d_64
#define rvecsumle                       rvecsumle_51d_64
#define rvecmaxdiff                     rvecmaxdiff_51d_64
#define ivecle                          ivecle_51d_64
#define ivecge                          ivecge_51d_64
#define ivecaxpylez                     ivecaxpylez_51d_64
#define ivecaxpygez                     ivecaxpygez_51d_64
#define BetterVBalance                  BetterVBalance_51d_64
#define BetterBalance2Way               BetterBalance2Way_51d_64
#define BetterBalanceKWay               BetterBalanceKWay_51d_64
#define ComputeLoadImbalance            ComputeLoadImbalance_51d_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51d_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51d_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51d_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51d_64
#define FindCommonElements              FindCommonElements_51d_64
#define CreateGraphNodal                CreateGraphNodal_51d_64
#define FindCommonNodes                 FindCommonNodes_51d_64
#define CreateMesh                      CreateMesh_51d_64
#define InitMesh                        InitMesh_51d_64
#define FreeMesh                        FreeMesh_51d_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51d_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51d_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51d_64
#define PrintSubDomainGraph             PrintSubDomainGraph_51d_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51d_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51d_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51d_64

/* mincover.c */
#define MinCover			MinCover_51d_64
#define MinCover_Augment		MinCover_Augment_51d_64
#define MinCover_Decompose		MinCover_Decompose_51d_64
#define MinCover_ColDFS			MinCover_ColDFS_51d_64
#define MinCover_RowDFS			MinCover_RowDFS_51d_64

/* mmd.c */
#define genmmd				genmmd_51d_64
#define mmdelm				mmdelm_51d_64
#define mmdint				mmdint_51d_64
#define mmdnum				mmdnum_51d_64
#define mmdupd				mmdupd_51d_64


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51d_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51d_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51d_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51d_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51d_64
#define SplitGraphOrder			SplitGraphOrder_51d_64
#define SplitGraphOrderCC		SplitGraphOrderCC_51d_64
#define MMDOrder			MMDOrder_51d_64

/* options.c */
#define SetupCtrl                       SetupCtrl_51d_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51d_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51d_64
#define PrintCtrl                       PrintCtrl_51d_64
#define FreeCtrl                        FreeCtrl_51d_64
#define CheckParams                     CheckParams_51d_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51d_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51d_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51d_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51d_64
#define MultilevelBisect		MultilevelBisect_51d_64
#define SplitGraphPart			SplitGraphPart_51d_64

/* refine.c */
#define Refine2Way			Refine2Way_51d_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51d_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51d_64
#define Project2WayPartition		Project2WayPartition_51d_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51d_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51d_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51d_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51d_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51d_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51d_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51d_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51d_64
#define Project2WayNodePartition	Project2WayNodePartition_51d_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51d_64
#define ComputePartitionBalance		ComputePartitionBalance_51d_64
#define ComputeElementBalance		ComputeElementBalance_51d_64

/* timing.c */
#define InitTimers			InitTimers_51d_64
#define PrintTimers			PrintTimers_51d_64

/* util.c */
#define iargmax_strd                    iargmax_strd_51d_64
#define iargmax_nrm                     iargmax_nrm_51d_64
#define iargmax2_nrm                    iargmax2_nrm_51d_64
#define rargmax2                        rargmax2_51d_64
#define InitRandom                      InitRandom_51d_64
#define metis_rcode                     metis_rcode_51d_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51d_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51d_64
#define FreeWorkSpace                   FreeWorkSpace_51d_64
#define wspacemalloc                    wspacemalloc_51d_64
#define wspacepush                      wspacepush_51d_64
#define wspacepop                       wspacepop_51d_64
#define iwspacemalloc                   iwspacemalloc_51d_64
#define rwspacemalloc                   rwspacemalloc_51d_64
#define ikvwspacemalloc                 ikvwspacemalloc_51d_64
#define cnbrpoolReset                   cnbrpoolReset_51d_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_51d_64
#define vnbrpoolReset                   vnbrpoolReset_51d_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_51d_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_51d
#define METIS_Free                      METIS_Free_51d
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_51d

/* balance.c */
#define Balance2Way			Balance2Way_51d
#define Bnd2WayBalance			Bnd2WayBalance_51d
#define General2WayBalance		General2WayBalance_51d
#define McGeneral2WayBalance            McGeneral2WayBalance_51d

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51d

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51d
#define CheckInputGraphWeights          CheckInputGraphWeights_51d
#define FixGraph                        FixGraph_51d

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51d
#define CoarsenGraphNlevels		CoarsenGraphNlevels_51d
#define Match_RM                        Match_RM_51d
#define Match_SHEM                      Match_SHEM_51d
#define Match_2Hop                      Match_2Hop_51d
#define Match_2HopAny                   Match_2HopAny_51d
#define Match_2HopAll                   Match_2HopAll_51d
#define PrintCGraphStats                PrintCGraphStats_51d
#define CreateCoarseGraph		CreateCoarseGraph_51d
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51d
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51d
#define SetupCoarseGraph		SetupCoarseGraph_51d
#define ReAdjustMemory			ReAdjustMemory_51d

/* compress.c */
#define CompressGraph			CompressGraph_51d
#define PruneGraph			PruneGraph_51d

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51d
#define IsConnected                     IsConnected_51d
#define IsConnectedSubdomain            IsConnectedSubdomain_51d
#define FindSepInducedComponents        FindSepInducedComponents_51d
#define EliminateComponents             EliminateComponents_51d
#define MoveGroupContigForCut           MoveGroupContigForCut_51d
#define MoveGroupContigForVol           MoveGroupContigForVol_51d
#define ComputeBFSOrdering              ComputeBFSOrdering_51d

/* debug.c */
#define ComputeCut			ComputeCut_51d
#define ComputeVolume			ComputeVolume_51d
#define ComputeMaxCut			ComputeMaxCut_51d
#define CheckBnd			CheckBnd_51d
#define CheckBnd2			CheckBnd2_51d
#define CheckNodeBnd			CheckNodeBnd_51d
#define CheckRInfo			CheckRInfo_51d
#define CheckNodePartitionParams	CheckNodePartitionParams_51d
#define IsSeparable			IsSeparable_51d
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51d

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51d
#define FM_2WayCutRefine                FM_2WayCutRefine_51d
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51d
#define SelectQueue                     SelectQueue_51d
#define Print2WayRefineStats            Print2WayRefineStats_51d

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51d
#define Change2FNumbering		Change2FNumbering_51d
#define Change2FNumbering2		Change2FNumbering2_51d
#define Change2FNumberingOrder		Change2FNumberingOrder_51d
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51d
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51d
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51d

/* graph.c */
#define SetupGraph			SetupGraph_51d
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51d
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51d
#define SetupGraph_label                SetupGraph_label_51d
#define SetupSplitGraph                 SetupSplitGraph_51d
#define CreateGraph                     CreateGraph_51d
#define InitGraph                       InitGraph_51d
#define FreeRData                       FreeRData_51d
#define FreeGraph                       FreeGraph_51d

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51d
#define InitSeparator			InitSeparator_51d
#define RandomBisection			RandomBisection_51d
#define GrowBisection			GrowBisection_51d
#define McRandomBisection               McRandomBisection_51d
#define McGrowBisection                 McGrowBisection_51d
#define GrowBisectionNode		GrowBisectionNode_51d
#define GrowBisectionNode2		GrowBisectionNode2_51d

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51d
#define InitKWayPartitioning            InitKWayPartitioning_51d

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51d
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51d
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51d
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51d
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51d
#define IsArticulationNode              IsArticulationNode_51d
#define KWayVolUpdate                   KWayVolUpdate_51d

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51d
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51d
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51d
#define ProjectKWayPartition		ProjectKWayPartition_51d
#define ComputeKWayBoundary		ComputeKWayBoundary_51d
#define ComputeKWayVolGains             ComputeKWayVolGains_51d
#define IsBalanced			IsBalanced_51d

/* mcutil */
#define rvecle                          rvecle_51d
#define rvecge                          rvecge_51d
#define rvecsumle                       rvecsumle_51d
#define rvecmaxdiff                     rvecmaxdiff_51d
#define ivecle                          ivecle_51d
#define ivecge                          ivecge_51d
#define ivecaxpylez                     ivecaxpylez_51d
#define ivecaxpygez                     ivecaxpygez_51d
#define BetterVBalance                  BetterVBalance_51d
#define BetterBalance2Way               BetterBalance2Way_51d
#define BetterBalanceKWay               BetterBalanceKWay_51d
#define ComputeLoadImbalance            ComputeLoadImbalance_51d
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51d
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51d
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51d

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51d
#define FindCommonElements              FindCommonElements_51d
#define CreateGraphNodal                CreateGraphNodal_51d
#define FindCommonNodes                 FindCommonNodes_51d
#define CreateMesh                      CreateMesh_51d
#define InitMesh                        InitMesh_51d
#define FreeMesh                        FreeMesh_51d

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51d

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51d
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51d
#define PrintSubDomainGraph             PrintSubDomainGraph_51d
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51d
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51d
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51d

/* mincover.c */
#define MinCover			MinCover_51d
#define MinCover_Augment		MinCover_Augment_51d
#define MinCover_Decompose		MinCover_Decompose_51d
#define MinCover_ColDFS			MinCover_ColDFS_51d
#define MinCover_RowDFS			MinCover_RowDFS_51d

/* mmd.c */
#define genmmd				genmmd_51d
#define mmdelm				mmdelm_51d
#define mmdint				mmdint_51d
#define mmdnum				mmdnum_51d
#define mmdupd				mmdupd_51d


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51d
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51d
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51d
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51d
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51d
#define SplitGraphOrder			SplitGraphOrder_51d
#define SplitGraphOrderCC		SplitGraphOrderCC_51d
#define MMDOrder			MMDOrder_51d

/* options.c */
#define SetupCtrl                       SetupCtrl_51d
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51d
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51d
#define PrintCtrl                       PrintCtrl_51d
#define FreeCtrl                        FreeCtrl_51d
#define CheckParams                     CheckParams_51d

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51d
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51d
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51d

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51d
#define MultilevelBisect		MultilevelBisect_51d
#define SplitGraphPart			SplitGraphPart_51d

/* refine.c */
#define Refine2Way			Refine2Way_51d
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51d
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51d
#define Project2WayPartition		Project2WayPartition_51d

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51d
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51d

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51d
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51d
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51d

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51d
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51d
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51d
#define Project2WayNodePartition	Project2WayNodePartition_51d

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51d
#define ComputePartitionBalance		ComputePartitionBalance_51d
#define ComputeElementBalance		ComputeElementBalance_51d

/* timing.c */
#define InitTimers			InitTimers_51d
#define PrintTimers			PrintTimers_51d

/* util.c */
#define iargmax_strd                    iargmax_strd_51d
#define iargmax_nrm                     iargmax_nrm_51d
#define iargmax2_nrm                    iargmax2_nrm_51d
#define rargmax2                        rargmax2_51d
#define InitRandom                      InitRandom_51d
#define metis_rcode                     metis_rcode_51d

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51d
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51d
#define FreeWorkSpace                   FreeWorkSpace_51d
#define wspacemalloc                    wspacemalloc_51d
#define wspacepush                      wspacepush_51d
#define wspacepop                       wspacepop_51d
#define iwspacemalloc                   iwspacemalloc_51d
#define rwspacemalloc                   rwspacemalloc_51d
#define ikvwspacemalloc                 ikvwspacemalloc_51d
#define cnbrpoolReset                   cnbrpoolReset_51d
#define cnbrpoolGetNext                 cnbrpoolGetNext_51d
#define vnbrpoolReset                   vnbrpoolReset_51d
#define vnbrpoolGetNext                 vnbrpoolGetNext_51d

#endif
