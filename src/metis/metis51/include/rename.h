/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * non metis_nd components removed, _51 suffix added to all procedures
 *
 * $Id: rename.h 13933 2013-03-29 22:20:46Z karypis $
 *
 */


#ifndef _LIBMETIS_RENAME_H_
#define _LIBMETIS_RENAME_H_


/* balance.c */
#define Balance2Way			Balance2Way_51
#define Bnd2WayBalance			Bnd2WayBalance_51
#define General2WayBalance		General2WayBalance_51
#define McGeneral2WayBalance            McGeneral2WayBalance_51

/* bucketsort.c */
#define BucketSortKeysInc		BucketSortKeysInc_51

/* checkgraph.c */
#define CheckGraph                      CheckGraph_51
#define CheckInputGraphWeights          CheckInputGraphWeights_51
#define FixGraph                        FixGraph_51

/* coarsen.c */
#define CoarsenGraph			CoarsenGraph_51
#define Match_RM                        Match_RM_51
#define Match_SHEM                      Match_SHEM_51
#define Match_2Hop                      Match_2Hop_51
#define Match_2HopAny                   Match_2HopAny_51
#define Match_2HopAll                   Match_2HopAll_51
#define PrintCGraphStats                PrintCGraphStats_51
#define CreateCoarseGraph		CreateCoarseGraph_51
#define CreateCoarseGraphNoMask		CreateCoarseGraphNoMask_51
#define CreateCoarseGraphPerm		CreateCoarseGraphPerm_51
#define SetupCoarseGraph		SetupCoarseGraph_51
#define ReAdjustMemory			ReAdjustMemory_51

/* compress.c */
#define CompressGraph			CompressGraph_51
#define PruneGraph			PruneGraph_51

/* contig.c */
#define FindPartitionInducedComponents  FindPartitionInducedComponents_51
#define IsConnected                     IsConnected_51
#define IsConnectedSubdomain            IsConnectedSubdomain_51
#define FindSepInducedComponents        FindSepInducedComponents_51
#define EliminateComponents             EliminateComponents_51
#define MoveGroupContigForCut           MoveGroupContigForCut_51
#define MoveGroupContigForVol           MoveGroupContigForVol_51

/* debug.c */
#define ComputeCut			ComputeCut_51
#define ComputeVolume			ComputeVolume_51
#define ComputeMaxCut			ComputeMaxCut_51
#define CheckBnd			CheckBnd_51
#define CheckBnd2			CheckBnd2_51
#define CheckNodeBnd			CheckNodeBnd_51
#define CheckRInfo			CheckRInfo_51
#define CheckNodePartitionParams	CheckNodePartitionParams_51
#define IsSeparable			IsSeparable_51
#define CheckKWayVolPartitionParams     CheckKWayVolPartitionParams_51

/* fm.c */
#define FM_2WayRefine                   FM_2WayRefine_51
#define FM_2WayCutRefine                FM_2WayCutRefine_51
#define FM_Mc2WayCutRefine              FM_Mc2WayCutRefine_51
#define SelectQueue                     SelectQueue_51
#define Print2WayRefineStats            Print2WayRefineStats_51

/* fortran.c */
#define Change2CNumbering		Change2CNumbering_51
#define Change2FNumbering		Change2FNumbering_51
#define Change2FNumbering2		Change2FNumbering2_51
#define Change2FNumberingOrder		Change2FNumberingOrder_51
#define ChangeMesh2CNumbering		ChangeMesh2CNumbering_51
#define ChangeMesh2FNumbering		ChangeMesh2FNumbering_51
#define ChangeMesh2FNumbering2		ChangeMesh2FNumbering2_51

/* graph.c */
#define SetupGraph			SetupGraph_51
#define SetupGraph_adjrsum              SetupGraph_adjrsum_51
#define SetupGraph_tvwgt                SetupGraph_tvwgt_51
#define SetupGraph_label                SetupGraph_label_51
#define SetupSplitGraph                 SetupSplitGraph_51
#define CreateGraph                     CreateGraph_51
#define InitGraph                       InitGraph_51
#define FreeRData                       FreeRData_51
#define FreeGraph                       FreeGraph_51

/* initpart.c */
#define Init2WayPartition		Init2WayPartition_51
#define InitSeparator			InitSeparator_51
#define RandomBisection			RandomBisection_51
#define GrowBisection			GrowBisection_51
#define McRandomBisection               McRandomBisection_51
#define McGrowBisection                 McGrowBisection_51
#define GrowBisectionNode		GrowBisectionNode_51

/* kmetis.c */
#define MlevelKWayPartitioning		MlevelKWayPartitioning_51
#define InitKWayPartitioning            InitKWayPartitioning_51

/* kwayfm.c */
#define Greedy_KWayOptimize		Greedy_KWayOptimize_51
#define Greedy_KWayCutOptimize		Greedy_KWayCutOptimize_51
#define Greedy_KWayVolOptimize          Greedy_KWayVolOptimize_51
#define Greedy_McKWayCutOptimize        Greedy_McKWayCutOptimize_51
#define Greedy_McKWayVolOptimize        Greedy_McKWayVolOptimize_51
#define IsArticulationNode              IsArticulationNode_51
#define KWayVolUpdate                   KWayVolUpdate_51

/* kwayrefine.c */
#define RefineKWay			RefineKWay_51
#define AllocateKWayPartitionMemory	AllocateKWayPartitionMemory_51
#define ComputeKWayPartitionParams	ComputeKWayPartitionParams_51
#define ProjectKWayPartition		ProjectKWayPartition_51
#define ComputeKWayBoundary		ComputeKWayBoundary_51
#define ComputeKWayVolGains             ComputeKWayVolGains_51
#define IsBalanced			IsBalanced_51

/* mcutil */
#define rvecle                          rvecle_51
#define rvecge                          rvecge_51
#define rvecsumle                       rvecsumle_51
#define rvecmaxdiff                     rvecmaxdiff_51
#define ivecle                          ivecle_51
#define ivecge                          ivecge_51
#define ivecaxpylez                     ivecaxpylez_51
#define ivecaxpygez                     ivecaxpygez_51
#define BetterVBalance                  BetterVBalance_51
#define BetterBalance2Way               BetterBalance2Way_51
#define BetterBalanceKWay               BetterBalanceKWay_51
#define ComputeLoadImbalance            ComputeLoadImbalance_51
#define ComputeLoadImbalanceDiff        ComputeLoadImbalanceDiff_51
#define ComputeLoadImbalanceDiffVec     ComputeLoadImbalanceDiffVec_51
#define ComputeLoadImbalanceVec         ComputeLoadImbalanceVec_51

/* mesh.c */
#define CreateGraphDual                 CreateGraphDual_51
#define FindCommonElements              FindCommonElements_51
#define CreateGraphNodal                CreateGraphNodal_51
#define FindCommonNodes                 FindCommonNodes_51
#define CreateMesh                      CreateMesh_51
#define InitMesh                        InitMesh_51
#define FreeMesh                        FreeMesh_51

/* meshpart.c */
#define InduceRowPartFromColumnPart     InduceRowPartFromColumnPart_51

/* minconn.c */
#define ComputeSubDomainGraph           ComputeSubDomainGraph_51
#define UpdateEdgeSubDomainGraph        UpdateEdgeSubDomainGraph_51
#define PrintSubDomainGraph             PrintSubDomainGraph_51
#define EliminateSubDomainEdges         EliminateSubDomainEdges_51
#define MoveGroupMinConnForCut          MoveGroupMinConnForCut_51
#define MoveGroupMinConnForVol          MoveGroupMinConnForVol_51

/* mincover.c */
#define MinCover			MinCover_51
#define MinCover_Augment		MinCover_Augment_51
#define MinCover_Decompose		MinCover_Decompose_51
#define MinCover_ColDFS			MinCover_ColDFS_51
#define MinCover_RowDFS			MinCover_RowDFS_51

/* mmd.c */
#define genmmd				genmmd_51
#define mmdelm				mmdelm_51
#define mmdint				mmdint_51
#define mmdnum				mmdnum_51
#define mmdupd				mmdupd_51


/* ometis.c */
#define MlevelNestedDissection		MlevelNestedDissection_51
#define MlevelNestedDissectionCC	MlevelNestedDissectionCC_51
#define MlevelNodeBisectionMultiple	MlevelNodeBisectionMultiple_51
#define MlevelNodeBisectionL2		MlevelNodeBisectionL2_51
#define MlevelNodeBisectionL1		MlevelNodeBisectionL1_51
#define SplitGraphOrder			SplitGraphOrder_51
#define SplitGraphOrderCC		SplitGraphOrderCC_51
#define MMDOrder			MMDOrder_51

/* options.c */
#define SetupCtrl                       SetupCtrl_51
#define SetupKWayBalMultipliers         SetupKWayBalMultipliers_51
#define Setup2WayBalMultipliers         Setup2WayBalMultipliers_51
#define PrintCtrl                       PrintCtrl_51
#define FreeCtrl                        FreeCtrl_51
#define CheckParams                     CheckParams_51

/* parmetis.c */
#define MlevelNestedDissectionP		MlevelNestedDissectionP_51
#define FM_2WayNodeRefine1SidedP        FM_2WayNodeRefine1SidedP_51
#define FM_2WayNodeRefine2SidedP        FM_2WayNodeRefine2SidedP_51

/* pmetis.c */
#define MlevelRecursiveBisection	MlevelRecursiveBisection_51
#define MultilevelBisect		MultilevelBisect_51
#define SplitGraphPart			SplitGraphPart_51

/* refine.c */
#define Refine2Way			Refine2Way_51
#define Allocate2WayPartitionMemory	Allocate2WayPartitionMemory_51
#define Compute2WayPartitionParams	Compute2WayPartitionParams_51
#define Project2WayPartition		Project2WayPartition_51

/* separator.c */
#define ConstructSeparator		ConstructSeparator_51
#define ConstructMinCoverSeparator	ConstructMinCoverSeparator_51

/* sfm.c */
#define FM_2WayNodeRefine2Sided         FM_2WayNodeRefine2Sided_51
#define FM_2WayNodeRefine1Sided         FM_2WayNodeRefine1Sided_51
#define FM_2WayNodeBalance              FM_2WayNodeBalance_51

/* srefine.c */
#define Refine2WayNode			Refine2WayNode_51
#define Allocate2WayNodePartitionMemory	Allocate2WayNodePartitionMemory_51
#define Compute2WayNodePartitionParams	Compute2WayNodePartitionParams_51
#define Project2WayNodePartition	Project2WayNodePartition_51

/* stat.c */
#define ComputePartitionInfoBipartite   ComputePartitionInfoBipartite_51
#define ComputePartitionBalance		ComputePartitionBalance_51
#define ComputeElementBalance		ComputeElementBalance_51

/* timing.c */
#define InitTimers			InitTimers_51
#define PrintTimers			PrintTimers_51

/* util.c */
#define iargmax_strd                    iargmax_strd_51
#define iargmax_nrm                     iargmax_nrm_51
#define iargmax2_nrm                    iargmax2_nrm_51
#define rargmax2                        rargmax2_51
#define InitRandom                      InitRandom_51
#define metis_rcode                     metis_rcode_51

/* wspace.c */
#define AllocateWorkSpace               AllocateWorkSpace_51
#define AllocateRefinementWorkSpace     AllocateRefinementWorkSpace_51
#define FreeWorkSpace                   FreeWorkSpace_51
#define wspacemalloc                    wspacemalloc_51
#define wspacepush                      wspacepush_51
#define wspacepop                       wspacepop_51
#define iwspacemalloc                   iwspacemalloc_51
#define rwspacemalloc                   rwspacemalloc_51
#define ikvwspacemalloc                 ikvwspacemalloc_51
#define cnbrpoolReset                   cnbrpoolReset_51
#define cnbrpoolGetNext                 cnbrpoolGetNext_51
#define vnbrpoolReset                   vnbrpoolReset_51
#define vnbrpoolGetNext                 vnbrpoolGetNext_51

#endif


