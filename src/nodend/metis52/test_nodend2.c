/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * test_nodend.c (simplified version of test/mtest.c by Nick Gould, STFC-RAL,
 * 2024-03, in which a specific test example is given rather than one read from
 * a data file, the test is only performed for metis_nodend, and where
 * unused variables and other post-Metis-4 changes have been removed/fixed)
 *
 * This file is a comprehensive tester for all the graph partitioning/ordering
 * routines of METIS
 *
 * Started 9/18/98
 * George
 *
 * $Id: mtest.c,v 1.1 2002/08/10 04:34:53 karypis Exp $
 *
 */

#include "metislib_52.h"
#include "test_proto_52.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main( )
{

/*  idx_t n = 10; */
  idx_t n = 2;
  idx_t *adj = (int *)  malloc((n+1) * sizeof(idx_t));
  idx_t *adjncy = (int *)  malloc((2) * sizeof(idx_t));

/* dense 2x2 example */

  adj[0]=0;
  adjncy[0]= 1;
  adj[1]=1;
  adjncy[1]= 0;
  adj[2]=2;

/*
  for (i=0; i<n+1; i++)
     printf("adj %d\n", adj[i]);

  for (i=0; i<2*(n-1); i++)
     printf("adjncy %d\n", adjncy[i]);
*/

/*  printf("%s", METISTITLE); */
  printf(" ** start metis nodend 5.2");
  fflush(stdout);
  clock_t begin = clock();
  Test_ND(n, adj, adjncy);
  clock_t end = clock();
  double time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf(" ** total execution time: %f seconds\n", time_elapsed);
  printf(" ** end test\n");

  gk_free((void **)&adj, &adjncy, LTERM);
}

/*********************************************************
* tests the the node nested-disection partitioning routine
**********************************************************/
void Test_ND(idx_t nvtxs, idx_t *xadj, idx_t *adjncy)
{
  idx_t i, rcode;
  idx_t options[METIS_NOPTIONS];
  idx_t *perm, *iperm, *vwgt;

  vwgt = imalloc(nvtxs, "vwgt");
  for (i=0; i<nvtxs; i++)
    vwgt[i] = 1+RandomInRange(10);


  perm = imalloc(nvtxs, "perm");
  iperm = imalloc(nvtxs, "iperm");



  /*==========================================================================*/
  /*printf("\nTesting METIS_NodeND\n");*/

  options[0] = 0;

  /*
  options[METIS_OPTION_OBJTYPE] =  METIS_OBJTYPE_NODE;
  options[METIS_OPTION_RTYPE] =    METIS_RTYPE_SEP1SIDED;
  options[METIS_OPTION_IPTYPE] =   METIS_IPTYPE_EDGE;
  options[METIS_OPTION_NSEPS] =    1;
  options[METIS_OPTION_NITER] =    10;
  options[METIS_OPTION_UFACTOR] =  OMETIS_DEFAULT_UFACTOR;
  options[METIS_OPTION_COMPRESS] = 1;
  options[METIS_OPTION_CCORDER] =  0;
  options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
  options[METIS_OPTION_NO2HOP] = 0;
  options[METIS_OPTION_ONDISK] = 0;
  options[METIS_OPTION_SEED] = -1;
  options[METIS_OPTION_DBGLVL] = 0;
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_DROPEDGES] = 0;
  */ 

  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_UFACTOR]  = 1;

  /* print option values if desired */
  int debug = 0; /* switch to 1 for debugging */
  if(debug==1){
    printf("options[METIS_OPTION_OBJTYPE] = %" d_ipc_ "\n",
            options[METIS_OPTION_OBJTYPE]);
    printf("options[METIS_OPTION_RTYPE] = %" d_ipc_ "\n",
            options[METIS_OPTION_RTYPE]);
    printf("options[METIS_OPTION_IPTYPE] = %" d_ipc_ "\n",
            options[METIS_OPTION_IPTYPE]);
    printf("options[METIS_OPTION_NSEPS] = %" d_ipc_ "\n",
            options[METIS_OPTION_NSEPS]);
    printf("options[METIS_OPTION_NITER] = %" d_ipc_ "\n",
            options[METIS_OPTION_NITER]);
    printf("options[METIS_OPTION_UFACTOR] = %" d_ipc_ "\n",
            options[METIS_OPTION_UFACTOR]);
    printf("options[METIS_OPTION_COMPRESS] = %" d_ipc_ "\n",
            options[METIS_OPTION_COMPRESS]);
    printf("options[METIS_OPTION_CCORDER] = %" d_ipc_ "\n",
            options[METIS_OPTION_CCORDER]);
    printf("options[METIS_OPTION_CTYPE] = %" d_ipc_ "\n",
            options[METIS_OPTION_CTYPE]);
    printf("options[METIS_OPTION_NO2HOP] = %" d_ipc_ "\n",
            options[METIS_OPTION_NO2HOP]);
    printf("options[METIS_OPTION_ONDISK] = %" d_ipc_ "\n",
            options[METIS_OPTION_ONDISK]);
    printf("options[METIS_OPTION_SEED] = %" d_ipc_ "\n",
            options[METIS_OPTION_SEED]);
    printf("options[METIS_OPTION_DBGLVL] = %" d_ipc_ "\n",
            options[METIS_OPTION_DBGLVL]);
    printf("options[METIS_OPTION_NUMBERING] = %" d_ipc_ "\n",
            options[METIS_OPTION_NUMBERING]);
    printf("options[METIS_OPTION_DROPEDGES] = %" d_ipc_ "\n",
            options[METIS_OPTION_DROPEDGES]);
    printf("options[METIS_OPTION_PTYPE] = %" d_ipc_ "\n",
            options[METIS_OPTION_PTYPE]);
    printf("options[METIS_OPTION_NCUTS] = %" d_ipc_ "\n",
            options[METIS_OPTION_NCUTS]);
    printf("options[METIS_OPTION_MINCONN] = %" d_ipc_ "\n",
            options[METIS_OPTION_MINCONN]);
    printf("options[METIS_OPTION_CONTIG] = %" d_ipc_ "\n",
            options[METIS_OPTION_CONTIG]);
    printf("options[METIS_OPTION_PFACTOR] = %" d_ipc_ "\n",
            options[METIS_OPTION_PFACTOR]);
  }

/*
   options[METIS_OPTION_UFACTOR] = 1;
   options[METIS_OPTION_NUMBERING] = -1;
   options[METIS_OPTION_OBJTYPE] = - 1;
   options[METIS_OPTION_PTYPE] = -1;
   options[METIS_OPTION_NCUTS] = -1;
   options[METIS_OPTION_MINCONN] = 1;
   options[METIS_OPTION_CONTIG] = -1;
   options[METIS_OPTION_PFACTOR] = 0;
   options[METIS_OPTION_PFACTOR] = 10;
   options[METIS_OPTION_DBGLVL] = 1;
*/
  if(debug==1){
    printf("options[METIS_OPTION_NIPARTS] = %" d_ipc_ "\n",
            options[METIS_OPTION_NIPARTS]);
    printf("options[METIS_OPTION_ONDISK] = %" d_ipc_ "\n",
            options[METIS_OPTION_ONDISK]);
    printf("options[METIS_OPTION_TWOHOP] = %" d_ipc_ "\n",
            options[METIS_OPTION_TWOHOP]);
    printf("options[METIS_OPTION_FAST] = %" d_ipc_ "\n",
            options[METIS_OPTION_FAST]);
  }
/*
  options[METIS_OPTION_PTYPE] = 1;
  options[METIS_OPTION_CTYPE] = 1;
  options[METIS_OPTION_IPTYPE] = 1;
  options[METIS_OPTION_RTYPE] = 1;
  options[METIS_OPTION_DBGLVL]  = 0;
  options[METIS_OPTION_OBJTYPE]  = 0;
  options[METIS_OPTION_PFACTOR] = 0;
  options[METIS_OPTION_NSEPS] = 1;
*/

  METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf(" test: ok");
  else
    printf(" test: err-code %"PRIDX"]", rcode);
  fflush(stdout);

  printf("\n");

  gk_free((void **)&vwgt, &perm, &iperm, LTERM);
}


/*************************************************************************
* This function verifies that the partitioning was computed correctly
**************************************************************************/
int VerifyND(idx_t nvtxs, idx_t *perm, idx_t *iperm)
{
  idx_t i, rcode=0;

  for (i=0; i<nvtxs; i++) {
    if (i != perm[iperm[i]])
      rcode = 1;
  }

  for (i=0; i<nvtxs; i++) {
    if (i != iperm[perm[i]])
      rcode = 2;
  }

  MALLOC_CHECK(NULL);

  return rcode;
}
