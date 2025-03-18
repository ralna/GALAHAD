/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h (modified by Nick Gould, STFC-RAL, 2025-03-01, to provide 
 *   64-bit integer support, additional prototypes for METIS_nodeND, 
 *   METIS_free and METIS_SetDefaultOptions, and to add _52 suffixes to 
 *   avoid possible conflicts)
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * $Id: rename.h 20398 2016-11-22 17:17:12Z karypis $
 *
 */

#ifndef _LIBMETIS_RENAME_H_
#define _LIBMETIS_RENAME_H_

/* 64-bit integer procedures */

#ifdef INTEGER_64

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52_64
#define METIS_Free                      METIS_Free_52_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52_64

/* balance.c */
#define Balance2Way			Balance2Way_52_64
#define Bnd2WayBalance			Bnd2WayBalance_52_64
#define General2WayBalance		General2WayBalance_52_64
#define McGeneral2WayBalance            McGeneral2WayBalance_52_64

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52_64

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52_64
#define CheckInputGraphWeights          CheckInputGraphWeights_52_64
#define FixGraph                        FixGraph_52_64

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52_64
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52_64
#define Match_RM                        Match_RM_52_64
#define Match_SHEM                      Match_SHEM_52_64
#define Match_2Hop                      Match_2Hop_52_64
#define Match_2HopAny                   Match_2HopAny_52_64
#define Match_2HopAll                   Match_2HopAll_52_64
#define Match_JC                        Match_JC_52_64
#define PrintCGraphStats                PrintCGraphStats_52_64
#define CreateCoarseGraph		CreateCoarseGraph_52_64
#define SetupCoarseGraph		SetupCoarseGraph_52_64
#define ReAdjustMemory			ReAdjustMemory_52_64

/* compress.c */
#define CompressGraph			CompressGraph_52_64
#define PruneGraph			PruneGraph_52_64

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52_64
#define IsConnected                     IsConnected_52_64
#define IsConnectedSubdomain            IsConnectedSubdomain_52_64
#define FindSepInducedComponents        FindSepInducedComponents_52_64
#define EliminateComponents             EliminateComponents_52_64
#define MoveGroupContigForCut           MoveGroupContigForCut_52_64
#define MoveGroupContigForVol           MoveGroupContigForVol_52_64
#define ComputeBFSOrdering              ComputeBFSOrdering_52_64

/* debug.c */
#define ComputeCut			ComputeCut_52_64
#define ComputeVolume			ComputeVolume_52_64
#define ComputeMaxCut			ComputeMaxCut_52_64
#define CheckBnd			CheckBnd_52_64
#define CheckBnd2			CheckBnd2_52_64
#define CheckNodeBnd			CheckNodeBnd_52_64
#define CheckRInfo			CheckRInfo_52_64
#define CheckNodePartitionParams	CheckNodePartitionParams_52_64
#define IsSeparable			IsSeparable_52_64
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52_64

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52_64
#define FM_2WayCutRefine                FM_2WayCutRefine_52_64
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52_64
#define SelectQueue                     SelectQueue_52_64
#define Print2WayRefineStats            Print2WayRefineStats_52_64

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52_64
#define Change2FNumbering		Change2FNumbering_52_64
#define Change2FNumbering2		Change2FNumbering2_52_64
#define Change2FNumberingOrder		Change2FNumberingOrder_52_64
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52_64
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52_64
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52_64

/* graph.c */
#define SetupGraph			SetupGraph_52_64
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52_64
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52_64
#define SetupGraph_label                SetupGraph_label_52_64
#define SetupSplitGraph                 SetupSplitGraph_52_64
#define CreateGraph                     CreateGraph_52_64
#define InitGraph                       InitGraph_52_64
#define FreeSData                       FreeSData_52_64
#define FreeRData                       FreeRData_52_64
#define FreeGraph                       FreeGraph_52_64
#define graph_WriteToDisk               graph_WriteToDisk_52_64
#define graph_ReadFromDisk              graph_ReadFromDisk_52_64

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52_64
#define InitSeparator			InitSeparator_52_64
#define RandomBisection			RandomBisection_52_64
#define GrowBisection			GrowBisection_52_64
#define McRandomBisection               McRandomBisection_52_64
#define McGrowBisection                 McGrowBisection_52_64
#define GrowBisectionNode		GrowBisectionNode_52_64
#define GrowBisectionNode2		GrowBisectionNode2_52_64

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52_64
#define InitKWayPartitioning            InitKWayPartitioning_52_64

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52_64
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52_64
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52_64
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52_64
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52_64
#define IsArticulationNode              IsArticulationNode_52_64
#define KWayVolUpdate                   KWayVolUpdate_52_64
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52_64
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52_64

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52_64
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52_64
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52_64
#define ProjectKWayPartition		ProjectKWayPartition_52_64
#define ComputeKWayBoundary		ComputeKWayBoundary_52_64
#define ComputeKWayVolGains             ComputeKWayVolGains_52_64
#define IsBalanced			IsBalanced_52_64

/* mcutil */
#define rvecle                          rvecle_52_64
#define rvecge                          rvecge_52_64
#define rvecsumle                       rvecsumle_52_64
#define rvecmaxdiff                     rvecmaxdiff_52_64
#define ivecle                          ivecle_52_64
#define ivecge                          ivecge_52_64
#define ivecaxpylez                     ivecaxpylez_52_64
#define ivecaxpygez                     ivecaxpygez_52_64
#define BetterVBalance                  BetterVBalance_52_64
#define BetterBalance2Way               BetterBalance2Way_52_64
#define BetterBalanceKWay               BetterBalanceKWay_52_64
#define ComputeLoadImbalance            ComputeLoadImbalance_52_64
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52_64
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52_64
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52_64

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52_64
#define FindCommonElements              FindCommonElements_52_64
#define CreateGraphNodal                CreateGraphNodal_52_64
#define FindCommonNodes                 FindCommonNodes_52_64
#define CreateMesh                      CreateMesh_52_64
#define InitMesh                        InitMesh_52_64
#define FreeMesh                        FreeMesh_52_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52_64

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52_64
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52_64
#define PrintSubDomainGraph             PrintSubDomainGraph_52_64
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52_64
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52_64
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52_64

/* mincover.c */
#define MinCover			MinCover_52_64
#define MinCover_Augment		MinCover_Augment_52_64
#define MinCover_Decompose		MinCover_Decompose_52_64
#define MinCover_ColDFS			MinCover_ColDFS_52_64
#define MinCover_RowDFS			MinCover_RowDFS_52_64

/* mmd.c */
#define genmmd				genmmd_52_64
#define mmdelm				mmdelm_52_64
#define mmdint				mmdint_52_64
#define mmdnum				mmdnum_52_64
#define mmdupd				mmdupd_52_64


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52_64
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52_64
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52_64
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52_64
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52_64
#define SplitGraphOrder			SplitGraphOrder_52_64
#define SplitGraphOrderCC		SplitGraphOrderCC_52_64
#define MMDOrder			MMDOrder_52_64

/* options.c */
#define SetupCtrl                       SetupCtrl_52_64
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52_64
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52_64
#define PrintCtrl                       PrintCtrl_52_64
#define FreeCtrl                        FreeCtrl_52_64
#define CheckParams                     CheckParams_52_64

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52_64
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52_64
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52_64

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52_64
#define MultilevelBisect		MultilevelBisect_52_64
#define SplitGraphPart			SplitGraphPart_52_64

/* refine.c */
#define Refine2Way			Refine2Way_52_64
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52_64
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52_64
#define Project2WayPartition		Project2WayPartition_52_64

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52_64
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52_64
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52_64
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52_64

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52_64
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52_64
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52_64
#define Project2WayNodePartition	Project2WayNodePartition_52_64

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52_64
#define ComputePartitionBalance		ComputePartitionBalance_52_64
#define ComputeElementBalance		ComputeElementBalance_52_64

/* timing.c */
#define InitTimers			InitTimers_52_64
#define PrintTimers			PrintTimers_52_64

/* util.c */
#define iargmax_strd                    iargmax_strd_52_64
#define iargmax_nrm                     iargmax_nrm_52_64
#define iargmax2_nrm                    iargmax2_nrm_52_64
#define rargmax2                        rargmax2_52_64
#define InitRandom                      InitRandom_52_64
#define metis_rcode                     metis_rcode_52_64

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52_64
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52_64
#define FreeWorkSpace                   FreeWorkSpace_52_64
#define wspacemalloc                    wspacemalloc_52_64
#define wspacepush                      wspacepush_52_64
#define wspacepop                       wspacepop_52_64
#define iwspacemalloc                   iwspacemalloc_52_64
#define rwspacemalloc                   rwspacemalloc_52_64
#define ikvwspacemalloc                 ikvwspacemalloc_52_64
#define cnbrpoolReset                   cnbrpoolReset_52_64
#define cnbrpoolGetNext                 cnbrpoolGetNext_52_64
#define vnbrpoolReset                   vnbrpoolReset_52_64
#define vnbrpoolGetNext                 vnbrpoolGetNext_52_64

/* 32-bit integer procedures */

#else

/* user-facing procedures */

#define METIS_NodeND                    METIS_NodeND_52
#define METIS_Free                      METIS_Free_52
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_52

/* balance.c */
#define Balance2Way			Balance2Way_52
#define Bnd2WayBalance			Bnd2WayBalance_52
#define General2WayBalance		General2WayBalance_52
#define McGeneral2WayBalance            McGeneral2WayBalance_52

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_52

/* checkgraph.c */
#define CheckGraph                      CheckGraph_52
#define CheckInputGraphWeights          CheckInputGraphWeights_52
#define FixGraph                        FixGraph_52

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_52
#define CoarsenGraphNlevels		CoarsenGraphNlevels_52
#define Match_RM                        Match_RM_52
#define Match_SHEM                      Match_SHEM_52
#define Match_2Hop                      Match_2Hop_52
#define Match_2HopAny                   Match_2HopAny_52
#define Match_2HopAll                   Match_2HopAll_52
#define Match_JC                        Match_JC_52
#define PrintCGraphStats                PrintCGraphStats_52
#define CreateCoarseGraph		CreateCoarseGraph_52
#define SetupCoarseGraph		SetupCoarseGraph_52
#define ReAdjustMemory			ReAdjustMemory_52

/* compress.c */
#define CompressGraph			CompressGraph_52
#define PruneGraph			PruneGraph_52

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_52
#define IsConnected                     IsConnected_52
#define IsConnectedSubdomain            IsConnectedSubdomain_52
#define FindSepInducedComponents        FindSepInducedComponents_52
#define EliminateComponents             EliminateComponents_52
#define MoveGroupContigForCut           MoveGroupContigForCut_52
#define MoveGroupContigForVol           MoveGroupContigForVol_52
#define ComputeBFSOrdering              ComputeBFSOrdering_52

/* debug.c */
#define ComputeCut			ComputeCut_52
#define ComputeVolume			ComputeVolume_52
#define ComputeMaxCut			ComputeMaxCut_52
#define CheckBnd			CheckBnd_52
#define CheckBnd2			CheckBnd2_52
#define CheckNodeBnd			CheckNodeBnd_52
#define CheckRInfo			CheckRInfo_52
#define CheckNodePartitionParams	CheckNodePartitionParams_52
#define IsSeparable			IsSeparable_52
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_52

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_52
#define FM_2WayCutRefine                FM_2WayCutRefine_52
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_52
#define SelectQueue                     SelectQueue_52
#define Print2WayRefineStats            Print2WayRefineStats_52

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_52
#define Change2FNumbering		Change2FNumbering_52
#define Change2FNumbering2		Change2FNumbering2_52
#define Change2FNumberingOrder		Change2FNumberingOrder_52
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_52
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_52
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_52

/* graph.c */
#define SetupGraph			SetupGraph_52
#define SetupGraph_adjrsum              SetupGraph_adjrsum_52
#define SetupGraph_tvwgt                SetupGraph_tvwgt_52
#define SetupGraph_label                SetupGraph_label_52
#define SetupSplitGraph                 SetupSplitGraph_52
#define CreateGraph                     CreateGraph_52
#define InitGraph                       InitGraph_52
#define FreeSData                       FreeSData_52
#define FreeRData                       FreeRData_52
#define FreeGraph                       FreeGraph_52
#define graph_WriteToDisk               graph_WriteToDisk_52
#define graph_ReadFromDisk              graph_ReadFromDisk_52

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_52
#define InitSeparator			InitSeparator_52
#define RandomBisection			RandomBisection_52
#define GrowBisection			GrowBisection_52
#define McRandomBisection               McRandomBisection_52
#define McGrowBisection                 McGrowBisection_52
#define GrowBisectionNode		GrowBisectionNode_52
#define GrowBisectionNode2		GrowBisectionNode2_52

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_52
#define InitKWayPartitioning            InitKWayPartitioning_52

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_52
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_52
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_52
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_52
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_52
#define IsArticulationNode              IsArticulationNode_52
#define KWayVolUpdate                   KWayVolUpdate_52
#define Greedy_KWayEdgeStats            Greedy_KWayEdgeStats_52
#define Greedy_KWayEdgeCutOptimize      Greedy_KWayEdgeCutOptimize_52

/* kwayrefine.c */
#define RefineKWay			RefineKWay_52
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_52
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_52
#define ProjectKWayPartition		ProjectKWayPartition_52
#define ComputeKWayBoundary		ComputeKWayBoundary_52
#define ComputeKWayVolGains             ComputeKWayVolGains_52
#define IsBalanced			IsBalanced_52

/* mcutil */
#define rvecle                          rvecle_52
#define rvecge                          rvecge_52
#define rvecsumle                       rvecsumle_52
#define rvecmaxdiff                     rvecmaxdiff_52
#define ivecle                          ivecle_52
#define ivecge                          ivecge_52
#define ivecaxpylez                     ivecaxpylez_52
#define ivecaxpygez                     ivecaxpygez_52
#define BetterVBalance                  BetterVBalance_52
#define BetterBalance2Way               BetterBalance2Way_52
#define BetterBalanceKWay               BetterBalanceKWay_52
#define ComputeLoadImbalance            ComputeLoadImbalance_52
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_52
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_52
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_52

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_52
#define FindCommonElements              FindCommonElements_52
#define CreateGraphNodal                CreateGraphNodal_52
#define FindCommonNodes                 FindCommonNodes_52
#define CreateMesh                      CreateMesh_52
#define InitMesh                        InitMesh_52
#define FreeMesh                        FreeMesh_52

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_52

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_52
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_52
#define PrintSubDomainGraph             PrintSubDomainGraph_52
#define EliminateSubDomainEdges         EliminateSubDomainEdges_52
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_52
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_52

/* mincover.c */
#define MinCover			MinCover_52
#define MinCover_Augment		MinCover_Augment_52
#define MinCover_Decompose		MinCover_Decompose_52
#define MinCover_ColDFS			MinCover_ColDFS_52
#define MinCover_RowDFS			MinCover_RowDFS_52

/* mmd.c */
#define genmmd				genmmd_52
#define mmdelm				mmdelm_52
#define mmdint				mmdint_52
#define mmdnum				mmdnum_52
#define mmdupd				mmdupd_52


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_52
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_52
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_52
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_52
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_52
#define SplitGraphOrder			SplitGraphOrder_52
#define SplitGraphOrderCC		SplitGraphOrderCC_52
#define MMDOrder			MMDOrder_52

/* options.c */
#define SetupCtrl                       SetupCtrl_52
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_52
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_52
#define PrintCtrl                       PrintCtrl_52
#define FreeCtrl                        FreeCtrl_52
#define CheckParams                     CheckParams_52

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_52
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_52
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_52

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_52
#define MultilevelBisect		MultilevelBisect_52
#define SplitGraphPart			SplitGraphPart_52

/* refine.c */
#define Refine2Way			Refine2Way_52
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_52
#define Compute2WayPartitionParams	Compute2WayPartitionParams_52
#define Project2WayPartition		Project2WayPartition_52

/* separator.c */
#define ConstructSeparator		ConstructSeparator_52
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_52

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_52
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_52
#define FM_2WayNodeBalance              FM_2WayNodeBalance_52

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_52
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_52
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_52
#define Project2WayNodePartition	Project2WayNodePartition_52

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_52
#define ComputePartitionBalance		ComputePartitionBalance_52
#define ComputeElementBalance		ComputeElementBalance_52

/* timing.c */
#define InitTimers			InitTimers_52
#define PrintTimers			PrintTimers_52

/* util.c */
#define iargmax_strd                    iargmax_strd_52
#define iargmax_nrm                     iargmax_nrm_52
#define iargmax2_nrm                    iargmax2_nrm_52
#define rargmax2                        rargmax2_52
#define InitRandom                      InitRandom_52
#define metis_rcode                     metis_rcode_52

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_52
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_52
#define FreeWorkSpace                   FreeWorkSpace_52
#define wspacemalloc                    wspacemalloc_52
#define wspacepush                      wspacepush_52
#define wspacepop                       wspacepop_52
#define iwspacemalloc                   iwspacemalloc_52
#define rwspacemalloc                   rwspacemalloc_52
#define ikvwspacemalloc                 ikvwspacemalloc_52
#define cnbrpoolReset                   cnbrpoolReset_52
#define cnbrpoolGetNext                 cnbrpoolGetNext_52
#define vnbrpoolReset                   vnbrpoolReset_52
#define vnbrpoolGetNext                 vnbrpoolGetNext_52

#endif
#endif


