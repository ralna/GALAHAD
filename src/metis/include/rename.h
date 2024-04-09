/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h (modified by Nick Gould, STFC-RAL, 2024-03-25, to provide 
 *   additional prototypes for METIS_nodeND, METIS_free and 
 *   METIS_SetDefaultOptions, and to change the libmetis_ suffix to 
 *   galmetis_ to avoid possible conflicts)
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

#define METIS_NodeND                    METIS_NodeND_64
#define METIS_Free                      METIS_Free_64
#define METIS_SetDefaultOptions         METIS_SetDefaultOptions_64

/* balance.c */
#define Balance2Way			galmetis__Balance2Way_64
#define Bnd2WayBalance			galmetis__Bnd2WayBalance_64
#define General2WayBalance		galmetis__General2WayBalance_64
#define McGeneral2WayBalance            galmetis__McGeneral2WayBalance_64

/* bucketsort.c */
#define BucketSortKeysInc		galmetis__BucketSortKeysInc_64

/* checkgraph.c */
#define CheckGraph                      galmetis__CheckGraph_64
#define CheckInputGraphWeights          galmetis__CheckInputGraphWeights_64
#define FixGraph                        galmetis__FixGraph_64

/* coarsen.c */
#define CoarsenGraph			galmetis__CoarsenGraph_64
#define Match_RM                        galmetis__Match_RM_64
#define Match_SHEM                      galmetis__Match_SHEM_64
#define Match_2Hop                      galmetis__Match_2Hop_64
#define Match_2HopAny                   galmetis__Match_2HopAny_64
#define Match_2HopAll                   galmetis__Match_2HopAll_64
#define Match_JC                        galmetis__Match_JC_64
#define PrintCGraphStats                galmetis__PrintCGraphStats_64
#define CreateCoarseGraph		galmetis__CreateCoarseGraph_64
#define SetupCoarseGraph		galmetis__SetupCoarseGraph_64
#define ReAdjustMemory			galmetis__ReAdjustMemory_64

/* compress.c */
#define CompressGraph			galmetis__CompressGraph_64
#define PruneGraph			galmetis__PruneGraph_64

/* contig.c */
#define FindPartitionInducedComponents  galmetis__FindPartitionInducedComponents_64
#define IsConnected                     galmetis__IsConnected_64
#define IsConnectedSubdomain            galmetis__IsConnectedSubdomain_64
#define FindSepInducedComponents        galmetis__FindSepInducedComponents_64
#define EliminateComponents             galmetis__EliminateComponents_64
#define MoveGroupContigForCut           galmetis__MoveGroupContigForCut_64
#define MoveGroupContigForVol           galmetis__MoveGroupContigForVol_64

/* debug.c */
#define ComputeCut			galmetis__ComputeCut_64
#define ComputeVolume			galmetis__ComputeVolume_64
#define ComputeMaxCut			galmetis__ComputeMaxCut_64
#define CheckBnd			galmetis__CheckBnd_64
#define CheckBnd2			galmetis__CheckBnd2_64
#define CheckNodeBnd			galmetis__CheckNodeBnd_64
#define CheckRInfo			galmetis__CheckRInfo_64
#define CheckNodePartitionParams	galmetis__CheckNodePartitionParams_64
#define IsSeparable			galmetis__IsSeparable_64
#define CheckKWayVolPartitionParams     galmetis__CheckKWayVolPartitionParams_64

/* fm.c */
#define FM_2WayRefine                   galmetis__FM_2WayRefine_64
#define FM_2WayCutRefine                galmetis__FM_2WayCutRefine_64
#define FM_Mc2WayCutRefine              galmetis__FM_Mc2WayCutRefine_64
#define SelectQueue                     galmetis__SelectQueue_64
#define Print2WayRefineStats            galmetis__Print2WayRefineStats_64

/* fortran.c */
#define Change2CNumbering		galmetis__Change2CNumbering_64
#define Change2FNumbering		galmetis__Change2FNumbering_64
#define Change2FNumbering2		galmetis__Change2FNumbering2_64
#define Change2FNumberingOrder		galmetis__Change2FNumberingOrder_64
#define ChangeMesh2CNumbering		galmetis__ChangeMesh2CNumbering_64
#define ChangeMesh2FNumbering		galmetis__ChangeMesh2FNumbering_64
#define ChangeMesh2FNumbering2		galmetis__ChangeMesh2FNumbering2_64

/* graph.c */
#define SetupGraph			galmetis__SetupGraph_64
#define SetupGraph_adjrsum              galmetis__SetupGraph_adjrsum_64
#define SetupGraph_tvwgt                galmetis__SetupGraph_tvwgt_64
#define SetupGraph_label                galmetis__SetupGraph_label_64
#define SetupSplitGraph                 galmetis__SetupSplitGraph_64
#define CreateGraph                     galmetis__CreateGraph_64
#define InitGraph                       galmetis__InitGraph_64
#define FreeSData                       galmetis__FreeSData_64
#define FreeRData                       galmetis__FreeRData_64
#define FreeGraph                       galmetis__FreeGraph_64
#define graph_WriteToDisk               galmetis__graph_WriteToDisk_64
#define graph_ReadFromDisk              galmetis__graph_ReadFromDisk_64

/* initpart.c */
#define Init2WayPartition		galmetis__Init2WayPartition_64
#define InitSeparator			galmetis__InitSeparator_64
#define RandomBisection			galmetis__RandomBisection_64
#define GrowBisection			galmetis__GrowBisection_64
#define McRandomBisection               galmetis__McRandomBisection_64
#define McGrowBisection                 galmetis__McGrowBisection_64
#define GrowBisectionNode		galmetis__GrowBisectionNode_64

/* kmetis.c */
#define MlevelKWayPartitioning		galmetis__MlevelKWayPartitioning_64
#define InitKWayPartitioning            galmetis__InitKWayPartitioning_64

/* kwayfm.c */
#define Greedy_KWayOptimize		galmetis__Greedy_KWayOptimize_64
#define Greedy_KWayCutOptimize		galmetis__Greedy_KWayCutOptimize_64
#define Greedy_KWayVolOptimize          galmetis__Greedy_KWayVolOptimize_64
#define Greedy_McKWayCutOptimize        galmetis__Greedy_McKWayCutOptimize_64
#define Greedy_McKWayVolOptimize        galmetis__Greedy_McKWayVolOptimize_64
#define IsArticulationNode              galmetis__IsArticulationNode_64
#define KWayVolUpdate                   galmetis__KWayVolUpdate_64

/* kwayrefine.c */
#define RefineKWay			galmetis__RefineKWay_64
#define AllocateKWayPartitionMemory	galmetis__AllocateKWayPartitionMemory_64
#define ComputeKWayPartitionParams	galmetis__ComputeKWayPartitionParams_64
#define ProjectKWayPartition		galmetis__ProjectKWayPartition_64
#define ComputeKWayBoundary		galmetis__ComputeKWayBoundary_64
#define ComputeKWayVolGains             galmetis__ComputeKWayVolGains_64
#define IsBalanced			galmetis__IsBalanced_64

/* mcutil */
#define rvecle                          galmetis__rvecle_64
#define rvecge                          galmetis__rvecge_64
#define rvecsumle                       galmetis__rvecsumle_64
#define rvecmaxdiff                     galmetis__rvecmaxdiff_64
#define ivecle                          galmetis__ivecle_64
#define ivecge                          galmetis__ivecge_64
#define ivecaxpylez                     galmetis__ivecaxpylez_64
#define ivecaxpygez                     galmetis__ivecaxpygez_64
#define BetterVBalance                  galmetis__BetterVBalance_64
#define BetterBalance2Way               galmetis__BetterBalance2Way_64
#define BetterBalanceKWay               galmetis__BetterBalanceKWay_64
#define ComputeLoadImbalance            galmetis__ComputeLoadImbalance_64
#define ComputeLoadImbalanceDiff        galmetis__ComputeLoadImbalanceDiff_64
#define ComputeLoadImbalanceDiffVec     galmetis__ComputeLoadImbalanceDiffVec_64
#define ComputeLoadImbalanceVec         galmetis__ComputeLoadImbalanceVec_64

/* mesh.c */
#define CreateGraphDual                 galmetis__CreateGraphDual_64
#define FindCommonElements              galmetis__FindCommonElements_64
#define CreateGraphNodal                galmetis__CreateGraphNodal_64
#define FindCommonNodes                 galmetis__FindCommonNodes_64
#define CreateMesh                      galmetis__CreateMesh_64
#define InitMesh                        galmetis__InitMesh_64
#define FreeMesh                        galmetis__FreeMesh_64

/* meshpart.c */
#define InduceRowPartFromColumnPart     galmetis__InduceRowPartFromColumnPart_64

/* minconn.c */
#define ComputeSubDomainGraph           galmetis__ComputeSubDomainGraph_64
#define UpdateEdgeSubDomainGraph        galmetis__UpdateEdgeSubDomainGraph_64
#define PrintSubDomainGraph             galmetis__PrintSubDomainGraph_64
#define EliminateSubDomainEdges         galmetis__EliminateSubDomainEdges_64
#define MoveGroupMinConnForCut          galmetis__MoveGroupMinConnForCut_64
#define MoveGroupMinConnForVol          galmetis__MoveGroupMinConnForVol_64

/* mincover.c */
#define MinCover			galmetis__MinCover_64
#define MinCover_Augment		galmetis__MinCover_Augment_64
#define MinCover_Decompose		galmetis__MinCover_Decompose_64
#define MinCover_ColDFS			galmetis__MinCover_ColDFS_64
#define MinCover_RowDFS			galmetis__MinCover_RowDFS_64

/* mmd.c */
#define genmmd				galmetis__genmmd_64
#define mmdelm				galmetis__mmdelm_64
#define mmdint				galmetis__mmdint_64
#define mmdnum				galmetis__mmdnum_64
#define mmdupd				galmetis__mmdupd_64


/* ometis.c */
#define MlevelNestedDissection		galmetis__MlevelNestedDissection_64
#define MlevelNestedDissectionCC	galmetis__MlevelNestedDissectionCC_64
#define MlevelNodeBisectionMultiple	galmetis__MlevelNodeBisectionMultiple_64
#define MlevelNodeBisectionL2		galmetis__MlevelNodeBisectionL2_64
#define MlevelNodeBisectionL1		galmetis__MlevelNodeBisectionL1_64
#define SplitGraphOrder			galmetis__SplitGraphOrder_64
#define SplitGraphOrderCC		galmetis__SplitGraphOrderCC_64
#define MMDOrder			galmetis__MMDOrder_64

/* options.c */
#define SetupCtrl                       galmetis__SetupCtrl_64
#define SetupKWayBalMultipliers         galmetis__SetupKWayBalMultipliers_64
#define Setup2WayBalMultipliers         galmetis__Setup2WayBalMultipliers_64
#define PrintCtrl                       galmetis__PrintCtrl_64
#define FreeCtrl                        galmetis__FreeCtrl_64
#define CheckParams                     galmetis__CheckParams_64

/* parmetis.c */
#define MlevelNestedDissectionP		galmetis__MlevelNestedDissectionP_64
#define FM_2WayNodeRefine1SidedP        galmetis__FM_2WayNodeRefine1SidedP_64
#define FM_2WayNodeRefine2SidedP        galmetis__FM_2WayNodeRefine2SidedP_64

/* pmetis.c */
#define MlevelRecursiveBisection	galmetis__MlevelRecursiveBisection_64
#define MultilevelBisect		galmetis__MultilevelBisect_64
#define SplitGraphPart			galmetis__SplitGraphPart_64

/* refine.c */
#define Refine2Way			galmetis__Refine2Way_64
#define Allocate2WayPartitionMemory	galmetis__Allocate2WayPartitionMemory_64
#define Compute2WayPartitionParams	galmetis__Compute2WayPartitionParams_64
#define Project2WayPartition		galmetis__Project2WayPartition_64

/* separator.c */
#define ConstructSeparator		galmetis__ConstructSeparator_64
#define ConstructMinCoverSeparator	galmetis__ConstructMinCoverSeparator_64

/* sfm.c */
#define FM_2WayNodeRefine2Sided         galmetis__FM_2WayNodeRefine2Sided_64
#define FM_2WayNodeRefine1Sided         galmetis__FM_2WayNodeRefine1Sided_64
#define FM_2WayNodeBalance              galmetis__FM_2WayNodeBalance_64

/* srefine.c */
#define Refine2WayNode			galmetis__Refine2WayNode_64
#define Allocate2WayNodePartitionMemory	galmetis__Allocate2WayNodePartitionMemory_64
#define Compute2WayNodePartitionParams	galmetis__Compute2WayNodePartitionParams_64
#define Project2WayNodePartition	galmetis__Project2WayNodePartition_64

/* stat.c */
#define ComputePartitionInfoBipartite   galmetis__ComputePartitionInfoBipartite_64
#define ComputePartitionBalance		galmetis__ComputePartitionBalance_64
#define ComputeElementBalance		galmetis__ComputeElementBalance_64

/* timing.c */
#define InitTimers			galmetis__InitTimers_64
#define PrintTimers			galmetis__PrintTimers_64

/* util.c */
#define iargmax_strd                    galmetis__iargmax_strd_64
#define iargmax_nrm                     galmetis__iargmax_nrm_64
#define iargmax2_nrm                    galmetis__iargmax2_nrm_64
#define rargmax2                        galmetis__rargmax2_64
#define InitRandom                      galmetis__InitRandom_64
#define metis_rcode                     galmetis__metis_rcode_64

/* wspace.c */
#define AllocateWorkSpace               galmetis__AllocateWorkSpace_64
#define AllocateRefinementWorkSpace     galmetis__AllocateRefinementWorkSpace_64
#define FreeWorkSpace                   galmetis__FreeWorkSpace_64
#define wspacemalloc                    galmetis__wspacemalloc_64
#define wspacepush                      galmetis__wspacepush_64
#define wspacepop                       galmetis__wspacepop_64
#define iwspacemalloc                   galmetis__iwspacemalloc_64
#define rwspacemalloc                   galmetis__rwspacemalloc_64
#define ikvwspacemalloc                 galmetis__ikvwspacemalloc_64
#define cnbrpoolReset                   galmetis__cnbrpoolReset_64
#define cnbrpoolGetNext                 galmetis__cnbrpoolGetNext_64
#define vnbrpoolReset                   galmetis__vnbrpoolReset_64
#define vnbrpoolGetNext                 galmetis__vnbrpoolGetNext_64

/* 32-bit integer procedures */

#else

/* balance.c */
#define Balance2Way			galmetis__Balance2Way
#define Bnd2WayBalance			galmetis__Bnd2WayBalance
#define General2WayBalance		galmetis__General2WayBalance
#define McGeneral2WayBalance            galmetis__McGeneral2WayBalance

/* bucketsort.c */
#define BucketSortKeysInc		galmetis__BucketSortKeysInc

/* checkgraph.c */
#define CheckGraph                      galmetis__CheckGraph
#define CheckInputGraphWeights          galmetis__CheckInputGraphWeights
#define FixGraph                        galmetis__FixGraph

/* coarsen.c */
#define CoarsenGraph			galmetis__CoarsenGraph
#define Match_RM                        galmetis__Match_RM
#define Match_SHEM                      galmetis__Match_SHEM
#define Match_2Hop                      galmetis__Match_2Hop
#define Match_2HopAny                   galmetis__Match_2HopAny
#define Match_2HopAll                   galmetis__Match_2HopAll
#define Match_JC                        galmetis__Match_JC
#define PrintCGraphStats                galmetis__PrintCGraphStats
#define CreateCoarseGraph		galmetis__CreateCoarseGraph
#define SetupCoarseGraph		galmetis__SetupCoarseGraph
#define ReAdjustMemory			galmetis__ReAdjustMemory

/* compress.c */
#define CompressGraph			galmetis__CompressGraph
#define PruneGraph			galmetis__PruneGraph

/* contig.c */
#define FindPartitionInducedComponents  galmetis__FindPartitionInducedComponents
#define IsConnected                     galmetis__IsConnected
#define IsConnectedSubdomain            galmetis__IsConnectedSubdomain
#define FindSepInducedComponents        galmetis__FindSepInducedComponents
#define EliminateComponents             galmetis__EliminateComponents
#define MoveGroupContigForCut           galmetis__MoveGroupContigForCut
#define MoveGroupContigForVol           galmetis__MoveGroupContigForVol

/* debug.c */
#define ComputeCut			galmetis__ComputeCut
#define ComputeVolume			galmetis__ComputeVolume
#define ComputeMaxCut			galmetis__ComputeMaxCut
#define CheckBnd			galmetis__CheckBnd
#define CheckBnd2			galmetis__CheckBnd2
#define CheckNodeBnd			galmetis__CheckNodeBnd
#define CheckRInfo			galmetis__CheckRInfo
#define CheckNodePartitionParams	galmetis__CheckNodePartitionParams
#define IsSeparable			galmetis__IsSeparable
#define CheckKWayVolPartitionParams     galmetis__CheckKWayVolPartitionParams

/* fm.c */
#define FM_2WayRefine                   galmetis__FM_2WayRefine
#define FM_2WayCutRefine                galmetis__FM_2WayCutRefine
#define FM_Mc2WayCutRefine              galmetis__FM_Mc2WayCutRefine
#define SelectQueue                     galmetis__SelectQueue
#define Print2WayRefineStats            galmetis__Print2WayRefineStats

/* fortran.c */
#define Change2CNumbering		galmetis__Change2CNumbering
#define Change2FNumbering		galmetis__Change2FNumbering
#define Change2FNumbering2		galmetis__Change2FNumbering2
#define Change2FNumberingOrder		galmetis__Change2FNumberingOrder
#define ChangeMesh2CNumbering		galmetis__ChangeMesh2CNumbering
#define ChangeMesh2FNumbering		galmetis__ChangeMesh2FNumbering
#define ChangeMesh2FNumbering2		galmetis__ChangeMesh2FNumbering2

/* graph.c */
#define SetupGraph			galmetis__SetupGraph
#define SetupGraph_adjrsum              galmetis__SetupGraph_adjrsum
#define SetupGraph_tvwgt                galmetis__SetupGraph_tvwgt
#define SetupGraph_label                galmetis__SetupGraph_label
#define SetupSplitGraph                 galmetis__SetupSplitGraph
#define CreateGraph                     galmetis__CreateGraph
#define InitGraph                       galmetis__InitGraph
#define FreeSData                       galmetis__FreeSData
#define FreeRData                       galmetis__FreeRData
#define FreeGraph                       galmetis__FreeGraph
#define graph_WriteToDisk               galmetis__graph_WriteToDisk
#define graph_ReadFromDisk              galmetis__graph_ReadFromDisk

/* initpart.c */
#define Init2WayPartition		galmetis__Init2WayPartition
#define InitSeparator			galmetis__InitSeparator
#define RandomBisection			galmetis__RandomBisection
#define GrowBisection			galmetis__GrowBisection
#define McRandomBisection               galmetis__McRandomBisection
#define McGrowBisection                 galmetis__McGrowBisection
#define GrowBisectionNode		galmetis__GrowBisectionNode

/* kmetis.c */
#define MlevelKWayPartitioning		galmetis__MlevelKWayPartitioning
#define InitKWayPartitioning            galmetis__InitKWayPartitioning

/* kwayfm.c */
#define Greedy_KWayOptimize		galmetis__Greedy_KWayOptimize
#define Greedy_KWayCutOptimize		galmetis__Greedy_KWayCutOptimize
#define Greedy_KWayVolOptimize          galmetis__Greedy_KWayVolOptimize
#define Greedy_McKWayCutOptimize        galmetis__Greedy_McKWayCutOptimize
#define Greedy_McKWayVolOptimize        galmetis__Greedy_McKWayVolOptimize
#define IsArticulationNode              galmetis__IsArticulationNode
#define KWayVolUpdate                   galmetis__KWayVolUpdate

/* kwayrefine.c */
#define RefineKWay			galmetis__RefineKWay
#define AllocateKWayPartitionMemory	galmetis__AllocateKWayPartitionMemory
#define ComputeKWayPartitionParams	galmetis__ComputeKWayPartitionParams
#define ProjectKWayPartition		galmetis__ProjectKWayPartition
#define ComputeKWayBoundary		galmetis__ComputeKWayBoundary
#define ComputeKWayVolGains             galmetis__ComputeKWayVolGains
#define IsBalanced			galmetis__IsBalanced

/* mcutil */
#define rvecle                          galmetis__rvecle
#define rvecge                          galmetis__rvecge
#define rvecsumle                       galmetis__rvecsumle
#define rvecmaxdiff                     galmetis__rvecmaxdiff
#define ivecle                          galmetis__ivecle
#define ivecge                          galmetis__ivecge
#define ivecaxpylez                     galmetis__ivecaxpylez
#define ivecaxpygez                     galmetis__ivecaxpygez
#define BetterVBalance                  galmetis__BetterVBalance
#define BetterBalance2Way               galmetis__BetterBalance2Way
#define BetterBalanceKWay               galmetis__BetterBalanceKWay
#define ComputeLoadImbalance            galmetis__ComputeLoadImbalance
#define ComputeLoadImbalanceDiff        galmetis__ComputeLoadImbalanceDiff
#define ComputeLoadImbalanceDiffVec     galmetis__ComputeLoadImbalanceDiffVec
#define ComputeLoadImbalanceVec         galmetis__ComputeLoadImbalanceVec

/* mesh.c */
#define CreateGraphDual                 galmetis__CreateGraphDual
#define FindCommonElements              galmetis__FindCommonElements
#define CreateGraphNodal                galmetis__CreateGraphNodal
#define FindCommonNodes                 galmetis__FindCommonNodes
#define CreateMesh                      galmetis__CreateMesh
#define InitMesh                        galmetis__InitMesh
#define FreeMesh                        galmetis__FreeMesh

/* meshpart.c */
#define InduceRowPartFromColumnPart     galmetis__InduceRowPartFromColumnPart

/* minconn.c */
#define ComputeSubDomainGraph           galmetis__ComputeSubDomainGraph
#define UpdateEdgeSubDomainGraph        galmetis__UpdateEdgeSubDomainGraph
#define PrintSubDomainGraph             galmetis__PrintSubDomainGraph
#define EliminateSubDomainEdges         galmetis__EliminateSubDomainEdges
#define MoveGroupMinConnForCut          galmetis__MoveGroupMinConnForCut
#define MoveGroupMinConnForVol          galmetis__MoveGroupMinConnForVol

/* mincover.c */
#define MinCover			galmetis__MinCover
#define MinCover_Augment		galmetis__MinCover_Augment
#define MinCover_Decompose		galmetis__MinCover_Decompose
#define MinCover_ColDFS			galmetis__MinCover_ColDFS
#define MinCover_RowDFS			galmetis__MinCover_RowDFS

/* mmd.c */
#define genmmd				galmetis__genmmd
#define mmdelm				galmetis__mmdelm
#define mmdint				galmetis__mmdint
#define mmdnum				galmetis__mmdnum
#define mmdupd				galmetis__mmdupd


/* ometis.c */
#define MlevelNestedDissection		galmetis__MlevelNestedDissection
#define MlevelNestedDissectionCC	galmetis__MlevelNestedDissectionCC
#define MlevelNodeBisectionMultiple	galmetis__MlevelNodeBisectionMultiple
#define MlevelNodeBisectionL2		galmetis__MlevelNodeBisectionL2
#define MlevelNodeBisectionL1		galmetis__MlevelNodeBisectionL1
#define SplitGraphOrder			galmetis__SplitGraphOrder
#define SplitGraphOrderCC		galmetis__SplitGraphOrderCC
#define MMDOrder			galmetis__MMDOrder

/* options.c */
#define SetupCtrl                       galmetis__SetupCtrl
#define SetupKWayBalMultipliers         galmetis__SetupKWayBalMultipliers
#define Setup2WayBalMultipliers         galmetis__Setup2WayBalMultipliers
#define PrintCtrl                       galmetis__PrintCtrl
#define FreeCtrl                        galmetis__FreeCtrl
#define CheckParams                     galmetis__CheckParams

/* parmetis.c */
#define MlevelNestedDissectionP		galmetis__MlevelNestedDissectionP
#define FM_2WayNodeRefine1SidedP        galmetis__FM_2WayNodeRefine1SidedP
#define FM_2WayNodeRefine2SidedP        galmetis__FM_2WayNodeRefine2SidedP

/* pmetis.c */
#define MlevelRecursiveBisection	galmetis__MlevelRecursiveBisection
#define MultilevelBisect		galmetis__MultilevelBisect
#define SplitGraphPart			galmetis__SplitGraphPart

/* refine.c */
#define Refine2Way			galmetis__Refine2Way
#define Allocate2WayPartitionMemory	galmetis__Allocate2WayPartitionMemory
#define Compute2WayPartitionParams	galmetis__Compute2WayPartitionParams
#define Project2WayPartition		galmetis__Project2WayPartition

/* separator.c */
#define ConstructSeparator		galmetis__ConstructSeparator
#define ConstructMinCoverSeparator	galmetis__ConstructMinCoverSeparator

/* sfm.c */
#define FM_2WayNodeRefine2Sided         galmetis__FM_2WayNodeRefine2Sided 
#define FM_2WayNodeRefine1Sided         galmetis__FM_2WayNodeRefine1Sided
#define FM_2WayNodeBalance              galmetis__FM_2WayNodeBalance

/* srefine.c */
#define Refine2WayNode			galmetis__Refine2WayNode
#define Allocate2WayNodePartitionMemory	galmetis__Allocate2WayNodePartitionMemory
#define Compute2WayNodePartitionParams	galmetis__Compute2WayNodePartitionParams
#define Project2WayNodePartition	galmetis__Project2WayNodePartition

/* stat.c */
#define ComputePartitionInfoBipartite   galmetis__ComputePartitionInfoBipartite
#define ComputePartitionBalance		galmetis__ComputePartitionBalance
#define ComputeElementBalance		galmetis__ComputeElementBalance

/* timing.c */
#define InitTimers			galmetis__InitTimers
#define PrintTimers			galmetis__PrintTimers

/* util.c */
#define iargmax_strd                    galmetis__iargmax_strd 
#define iargmax_nrm                     galmetis__iargmax_nrm
#define iargmax2_nrm                    galmetis__iargmax2_nrm
#define rargmax2                        galmetis__rargmax2
#define InitRandom                      galmetis__InitRandom
#define metis_rcode                     galmetis__metis_rcode

/* wspace.c */
#define AllocateWorkSpace               galmetis__AllocateWorkSpace
#define AllocateRefinementWorkSpace     galmetis__AllocateRefinementWorkSpace
#define FreeWorkSpace                   galmetis__FreeWorkSpace
#define wspacemalloc                    galmetis__wspacemalloc
#define wspacepush                      galmetis__wspacepush
#define wspacepop                       galmetis__wspacepop
#define iwspacemalloc                   galmetis__iwspacemalloc
#define rwspacemalloc                   galmetis__rwspacemalloc
#define ikvwspacemalloc                 galmetis__ikvwspacemalloc
#define cnbrpoolReset                   galmetis__cnbrpoolReset
#define cnbrpoolGetNext                 galmetis__cnbrpoolGetNext
#define vnbrpoolReset                   galmetis__vnbrpoolReset
#define vnbrpoolGetNext                 galmetis__vnbrpoolGetNext

#endif
#endif


