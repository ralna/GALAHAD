/*
 * Copyright 1994-2011, Regents of the University of Minnesota
 *
 * ndmetis.c
 *
 * Driver programs for nested disection ordering
 *
 * Started 8/28/94
 * George
 *
 * $Id: ndmetis.c 13900 2013-03-24 15:27:07Z karypis $
 *
 */

#include "metisbin_51.h"
#include <time.h>

/*********************************************************
* tests the the node nested-disection partitioning routine
**********************************************************/
int main()
{
  idx_t options[METIS_NOPTIONS];
  idx_t *perm, *iperm;
  int status=0;

  idx_t n = 1000000;
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

  perm  = imalloc(n, "main: perm");
  iperm = imalloc(n, "main: iperm");

  METIS_SetDefaultOptions(options);

/*
  options[METIS_OPTION_CTYPE]    = params->ctype;
  options[METIS_OPTION_IPTYPE]   = params->iptype;
  options[METIS_OPTION_RTYPE]    = params->rtype;
  options[METIS_OPTION_DBGLVL]   = params->dbglvl;
  options[METIS_OPTION_UFACTOR]  = params->ufactor;
  options[METIS_OPTION_NO2HOP]   = params->no2hop;
  options[METIS_OPTION_COMPRESS] = params->compress;
  options[METIS_OPTION_CCORDER]  = params->ccorder;
  options[METIS_OPTION_SEED]     = params->seed;
  options[METIS_OPTION_NITER]    = params->niter;
  options[METIS_OPTION_NSEPS]    = params->nseps;
  options[METIS_OPTION_PFACTOR]  = params->pfactor;

  options[METIS_OPTION_CTYPE]    = 1;
  options[METIS_OPTION_IPTYPE]   = 2;
  options[METIS_OPTION_RTYPE]    = 3;
  options[METIS_OPTION_DBGLVL]   = 0;
  options[METIS_OPTION_UFACTOR]  = 200;
  options[METIS_OPTION_NO2HOP]   = 0;
  options[METIS_OPTION_COMPRESS] = 1;
  options[METIS_OPTION_CCORDER]  = 0;
  options[METIS_OPTION_SEED]     = -1;
  options[METIS_OPTION_NITER]    = 10;
  options[METIS_OPTION_NSEPS]    = 1;
  options[METIS_OPTION_PFACTOR]  = 0;

*/
  options[METIS_OPTION_UFACTOR]  = 1;

  /* print option values if desired */
  int debug = 0; /* switch to 1 for debugging */
  if(debug==1){
    options[METIS_OPTION_DBGLVL] = 2;
    printf("options[METIS_OPTION_OBJTYPE] = %d\n",
            options[METIS_OPTION_OBJTYPE]);
    printf("options[METIS_OPTION_RTYPE] = %d\n",
            options[METIS_OPTION_RTYPE]);
    printf("options[METIS_OPTION_IPTYPE] = %d\n",
            options[METIS_OPTION_IPTYPE]);
    printf("options[METIS_OPTION_NSEPS] = %d\n",
            options[METIS_OPTION_NSEPS]);
    printf("options[METIS_OPTION_NITER] = %d\n",
            options[METIS_OPTION_NITER]);
    printf("options[METIS_OPTION_UFACTOR] = %d\n", 
            options[METIS_OPTION_UFACTOR]);
    printf("options[METIS_OPTION_COMPRESS] = %d\n", 
            options[METIS_OPTION_COMPRESS]);
    printf("options[METIS_OPTION_CCORDER] = %d\n", 
            options[METIS_OPTION_CCORDER]);
    printf("options[METIS_OPTION_CTYPE] = %d\n",
            options[METIS_OPTION_CTYPE]);
    printf("options[METIS_OPTION_NO2HOP] = %d\n",
            options[METIS_OPTION_NO2HOP]);
    /*  printf("options[METIS_OPTION_ONDISK] = %d\n",
               options[METIS_OPTION_ONDISK]);*/
    printf("options[METIS_OPTION_SEED] = %d\n",
            options[METIS_OPTION_SEED]);
    printf("options[METIS_OPTION_DBGLVL] = %d\n",
            options[METIS_OPTION_DBGLVL]);
    printf("options[METIS_OPTION_NUMBERING] = %d\n",
            options[METIS_OPTION_NUMBERING]);
    /*  printf("options[METIS_OPTION_DROPEDGES] = %d\n",
               options[METIS_OPTION_DROPEDGES]);*/
    printf("options[METIS_OPTION_PTYPE] = %d\n",
            options[METIS_OPTION_PTYPE]);
    printf("options[METIS_OPTION_NCUTS] = %d\n",
            options[METIS_OPTION_NCUTS]);
    printf("options[METIS_OPTION_MINCONN] = %d\n",
            options[METIS_OPTION_MINCONN]);
    printf("options[METIS_OPTION_CONTIG] = %d\n",
            options[METIS_OPTION_CONTIG]);
    printf("options[METIS_OPTION_PFACTOR] = %d\n",
            options[METIS_OPTION_PFACTOR]);
  }

  gk_malloc_init();

/*  status = METIS_NodeND(&n, adj, adjncy, NULL, options, perm, iperm);*/
  printf(" ** start metis nodend 5.1 test **\n");
  clock_t begin = clock();
  status = METIS_NodeND(&n, adj, adjncy, (void*)(idx_t)0, options, perm, iperm);
  clock_t end = clock();
  double time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
  printf(" ** total execution time: %f seconds\n", time_elapsed);
  printf(" ** end test **\n");

  if (gk_GetCurMemoryUsed() != 0)
    printf("***Metis did not free all of its memory! Report this.\n");
  gk_malloc_cleanup(0);


  if (status != METIS_OK) {
    printf("\n***Metis returned with an error.\n");
  }

  gk_free((void **)&perm, &iperm, LTERM);
  gk_free((void **)&adj, &adjncy, LTERM);
}

