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

#include "metislib.h"
#include "proto.h"
#include "test_proto.h"
#include <stdio.h>
#include <stdlib.h>

int main( )
{

  idx_t n = 10;
  idx_t i, l;
  idx_t *adj = (idx_t *)  malloc((n+1) * sizeof(idx_t));
  idx_t *adjncy = (idx_t *)  malloc((2*(n-1)) * sizeof(idx_t));

/* tri-diagonal matrix example */

  l = 0;
  adj[0]=l;
  adjncy[l]= 1;

  for(i=1; i<n-1; i++) {
    l++;
    adj[i]=l;
    adjncy[l]= i-1;
    l++;
    adjncy[l]= i+1;
  }
  l++;
  adj[n-1]=l;
  adjncy[l]= n-2;
  l++;
  adj[n]=l;

/*
  for (i=0; i<n+1; i++)
     printf("adj %d\n", adj[i]);

  for (i=0; i<2*(n-1); i++)
     printf("adjncy %d\n", adjncy[i]);
*/

  printf("%s", METISTITLE);

  Test_ND(n, adj, adjncy);

  printf("\nTesting completed\n");

  gk_free((void **)&adj, &adjncy, LTERM);
}

/*************************************************************************
* This function tests the regular graph partitioning routines
**************************************************************************/
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
  printf("\nTesting METIS_NodeND\n");

  options[0] = 0;

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

  METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("\nTest:ok");
  else
    printf("\nTest:err-code %"PRIDX"]", rcode);
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
