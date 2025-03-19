/*
 * protobin.h 
 *
 * This file contains function prototypes
 *
 * Started 11/1/99
 * George
 *
 * $Id: proto.h 10513 2011-07-07 22:06:03Z karypis $
 *
 */

#ifndef _PROTOBIN_H_
#define _PROTOBIN_H_

/* smbfactor.c */
void ComputeFillIn(graph_t *graph, idx_t *perm, idx_t *iperm,
         size_t *r_maxlnz, size_t *r_opc);
idx_t smbfct(idx_t neqns, idx_t *xadj, idx_t *adjncy, idx_t *perm, 
          idx_t *invp, idx_t *xlnz, idx_t *maxlnz, idx_t *xnzsub, 
          idx_t *nzsub, idx_t *maxsub);


/* cmdline.c */
params_t *parse_cmdline(int argc, char *argv[]);

/* gpmetis.c */
void GPPrintInfo(params_t *params, graph_t *graph);
void GPReportResults(params_t *params, graph_t *graph, idx_t *part, idx_t edgecut);

/* ndmetis.c */
void NDPrintInfo(params_t *params, graph_t *graph);
void NDReportResults(params_t *params, graph_t *graph, idx_t *perm, idx_t *iperm);

/* mpmetis.c */
void MPPrintInfo(params_t *params, mesh_t *mesh);
void MPReportResults(params_t *params, mesh_t *mesh, idx_t *epart, idx_t *npart, 
         idx_t edgecut);

/* m2gmetis.c */
void M2GPrintInfo(params_t *params, mesh_t *mesh);
void M2GReportResults(params_t *params, mesh_t *mesh, graph_t *graph);

/* stat.c */
void ComputePartitionInfo(params_t *params, graph_t *graph, idx_t *where);


#endif 
