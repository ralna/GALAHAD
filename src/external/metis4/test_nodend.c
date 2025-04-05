/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * test_nodend.c (simplified version of test/mtest.c by Nick Gould, STFC-RAL,
 * 2024-03, in which a specific test example is given rather than one read from
 * a data file, the test is only performed for metis_nodend, and where
 * unused variables have been removed/fixed)
 *
 * Started 9/18/98
 * George
 *
 * $Id: mtest.c,v 1.1 2002/08/10 04:34:53 karypis Exp $
 *
 */

#include "metis.h"
#include <time.h>

int main( )
{

/*  int n = 10; */
  int n = 1000000;
  int i, l;
  int *adj = (int *)  malloc((n+1) * sizeof(int));
  int *adjncy = (int *)  malloc((2*(n-1)) * sizeof(int));

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

  printf("%s", METISTITLE);
*/

  printf("start testing\n");
/*  exit(0); */
  Test_ND(n, adj, adjncy);

  printf("\nTesting completed\n");

  GK_free((void **)&adj, &adjncy, LTERM);
}

/*******************************************************************
* This function tests the node nested-disection partitioning routine
********************************************************************/
void Test_ND(int nvtxs, idxtype *xadj, idxtype *adjncy)
{
  int tstnum, rcode;
  int numflag, options[10];
  idxtype *perm, *iperm;
/*  idxtype *vwgt; */
  clock_t begin, end;
  double time_elapsed ;
/*
  vwgt = idxmalloc(nvtxs, "vwgt");
  for (i=0; i<nvtxs; i++)
    vwgt[i] = 1+RandomInRange(10);
*/
  perm = idxmalloc(nvtxs, "perm");
  iperm = idxmalloc(nvtxs, "iperm");

  /*=================================================================*/

  printf("\nTesting METIS_NodeND ----------------------------\n\n");
  fflush(stdout);
  tstnum = 1;

/**/
  numflag = 0; 
  options[0] = 0;
  options[OPTION_DBGLVL] = 1;

  /* print option values if desired */
  int debug = 0; /* switch to 1 for debugging */
  if(debug==1){
    printf("options[OPTION_PTYPE] = %d\n", options[OPTION_PTYPE]);
    printf("options[OPTION_CTYPE] = %d\n", options[OPTION_CTYPE]);
    printf("options[OPTION_ITYPE] = %d\n", options[OPTION_ITYPE]);
    printf("options[OPTION_RTYPE] = %d\n", options[OPTION_RTYPE]);
    printf("options[OPTION_DBGLVL] = %d\n", options[OPTION_DBGLVL]);
    printf("options[OPTION_OFLAGS] = %d\n", options[OPTION_OFLAGS]);
    printf("options[OPTION_PFACTOR] = %d\n", options[OPTION_PFACTOR]);
    printf("options[OPTION_NSEPS] = %d\n", options[OPTION_NSEPS]);
  }

  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);


  numflag = 0; 
  options[0] = 1; options[1] = 1; options[2] = 1; 
  options[3] = 1; options[4] = 0;
  options[5] = 0; options[6] = 0; options[7] = 1;

  if(debug==1){
    printf("options[OPTION_PTYPE] = %d\n", options[OPTION_PTYPE]);
    printf("options[OPTION_CTYPE] = %d\n", options[OPTION_CTYPE]);
    printf("options[OPTION_ITYPE] = %d\n", options[OPTION_ITYPE]);
    printf("options[OPTION_RTYPE] = %d\n", options[OPTION_RTYPE]);
    printf("options[OPTION_DBGLVL] = %d\n", options[OPTION_DBGLVL]);
    printf("options[OPTION_OFLAGS] = %d\n", options[OPTION_OFLAGS]);
    printf("options[OPTION_PFACTOR] = %d\n", options[OPTION_PFACTOR]);
    printf("options[OPTION_NSEPS] = %d\n", options[OPTION_NSEPS]);
  }

  options[OPTION_PTYPE] = 1;
  options[OPTION_CTYPE] = 1;
  options[OPTION_ITYPE] = 1;
  options[OPTION_RTYPE] = 1;
  options[OPTION_DBGLVL]  = 0;
  options[OPTION_OFLAGS]  = 0;
  options[OPTION_PFACTOR] = 0;
  options[OPTION_NSEPS] = 1;

  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 2; options[2] = 1; 
  options[3] = 1; options[4] = 0; 
  options[5] = 0; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 1; options[4] = 0;
  options[5] = 0; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 2; 
  options[3] = 1; options[4] = 0;
  options[5] = 0; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 0; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 1; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; options[3] = 2; 
  options[4] = 0;
  options[5] = 2; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 3; options[6] = 0; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 3; options[6] = 40; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 3; options[6] = 20; options[7] = 1;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 3; options[6] = 20; options[7] = 2;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);

  numflag = 0; 
  options[0] = 1; options[1] = 3; options[2] = 1; 
  options[3] = 2; options[4] = 0;
  options[5] = 0; options[6] = 0; options[7] = 2;
  begin = clock();
  METIS_NodeND(&nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
  end = clock();
  time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Total execution time: %f seconds ", time_elapsed);

  if ((rcode = VerifyND(nvtxs, perm, iperm)) == 0)
    printf("[%d:ok]\n", tstnum++);
  else
    printf("[%d:err-%d]\n", tstnum++, rcode);
  fflush(stdout);
  GK_free(&perm, &iperm, LTERM);
/*  GK_free(&vwgt, &perm, &iperm, LTERM); */
}

/*************************************************************************
* This function verifies that the partitioning was computed correctly
**************************************************************************/
int VerifyND(int nvtxs, idxtype *perm, idxtype *iperm)
{
  int i, rcode=0;

  for (i=0; i<nvtxs; i++) {
    if (perm[i]<0){
      rcode = 1;
      return rcode;
    }
  }

  for (i=0; i<nvtxs; i++) {
    if (iperm[i]<0){
      rcode = 2;
      return rcode;
    }
  }

  for (i=0; i<nvtxs; i++) {
    if (i != perm[iperm[i]]){
      rcode = 3;
      return rcode;
    }
  }

  for (i=0; i<nvtxs; i++) {
    if (i != iperm[perm[i]]){
      rcode = 4;
      return rcode;
    }
  }

  MALLOC_CHECK(NULL);

  return rcode;
}
