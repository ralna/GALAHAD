/* Copyright (C) 2020 COIN-OR
 * All Rights Reserved.
 * This file is distributed under the Eclipse Public License.
 * Updated 2025-02-18 to override inappropriate MeTiS 5 ufactor value
 */

#ifdef INTEGER_64
#define galahad_metis5_adapter galahad_metis5_adapter_64
#define METIS_NodeND METIS_NodeND_64
#define METIS_SetDefaultOptions METIS_SetDefaultOptions_64
#endif

#include "galahad_metis.h" /* from MeTiS 5.2 */

#include "stdio.h"

/* name change to avoid any possible conflicts */

void galahad_metis5_adapter(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy,
                            idx_t* numflag, idx_t* n_options, idx_t* options, 
                            idx_t* perm, idx_t* iperm){
    idx_t options5[METIS_NOPTIONS];
    int debug = 0;

    /* Handle MA57 pathological case */
    if(*nvtxs == 1){
        /* MA57 seems to call metis with a graph containing 1 vertex and
         * a self-loop. Metis5 prints an error for this.
         */
        perm[0] = *numflag;
        iperm[0] = *numflag;
        return;
    }

    /* Set default MeTiS 5 options */
    METIS_SetDefaultOptions(options5);
    options5[METIS_OPTION_NUMBERING] = *numflag;

    /* print values if desired */
    if(debug==1){
      printf("Default values:\n");
      printf("options5[METIS_OPTION_OBJTYPE] = %d\n", 
                       options5[METIS_OPTION_OBJTYPE]);
      printf("options5[METIS_OPTION_RTYPE] = %d\n", 
                       options5[METIS_OPTION_RTYPE]);
      printf("options5[METIS_OPTION_IPTYPE] = %d\n", 
                       options5[METIS_OPTION_IPTYPE]);
      printf("options5[METIS_OPTION_NSEPS] = %d\n", 
                       options5[METIS_OPTION_NSEPS]);
      printf("options5[METIS_OPTION_NITER] = %d\n", 
                       options5[METIS_OPTION_NITER]);
      printf("options5[METIS_OPTION_UFACTOR] = %d\n", 
                       options5[METIS_OPTION_UFACTOR]);
      printf("options5[METIS_OPTION_COMPRESS] = %d\n", 
                       options5[METIS_OPTION_COMPRESS]);
      printf("options5[METIS_OPTION_CCORDER] = %d\n", 
                       options5[METIS_OPTION_CCORDER]);
      printf("options5[METIS_OPTION_CTYPE] = %d\n",
                       options5[METIS_OPTION_CTYPE]);
      printf("options5[METIS_OPTION_NO2HOP] = %d\n", 
                       options5[METIS_OPTION_NO2HOP]);
      printf("options5[METIS_OPTION_ONDISK] = %d\n", 
                       options5[METIS_OPTION_ONDISK]);
      printf("options5[METIS_OPTION_SEED] = %d\n", 
                       options5[METIS_OPTION_SEED]);
      printf("options5[METIS_OPTION_DBGLVL] = %d\n", 
                       options5[METIS_OPTION_DBGLVL]);
      printf("options5[METIS_OPTION_NUMBERING] = %d\n", 
                       options5[METIS_OPTION_NUMBERING]);
      printf("options5[METIS_OPTION_DROPEDGES] = %d\n", 
                       options5[METIS_OPTION_DROPEDGES]);
      printf("options5[METIS_OPTION_PTYPE] = %d\n",
                       options5[METIS_OPTION_PTYPE]);
      printf("options5[METIS_OPTION_NCUTS] = %d\n",
                       options5[METIS_OPTION_NCUTS]);
      printf("options5[METIS_OPTION_MINCONN] = %d\n",
                       options5[METIS_OPTION_MINCONN]);
      printf("options5[METIS_OPTION_CONTIG] = %d\n",
                       options5[METIS_OPTION_CONTIG]);
      printf("options5[METIS_OPTION_PFACTOR] = %d\n",
                       options5[METIS_OPTION_PFACTOR]);
      printf("options5[METIS_OPTION_NIPARTS] = %d\n", 
                       options5[METIS_OPTION_NIPARTS]);
      printf("options5[METIS_OPTION_TWOHOP] = %d\n", 
                       options5[METIS_OPTION_TWOHOP]);
      printf("options5[METIS_OPTION_FAST] = %d\n", 
                       options5[METIS_OPTION_FAST]);
    }
 
    /* Translate MeTiS 4 options MeTiS 5 options */
    if(options[0] != 0){
        if(options[1] == 1) /* random matching */
            options5[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
        else /* heavy-edge or sorted heavy-edge matching; map both to shem,
                as heave-edge matching not available in metis5 */
            options5[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;

        if(options[2] == 1) /* edge-based region-growing */
            options5[METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;
        else /* node-based region-growing */
            options5[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;

        if(options[3] == 1) /* two-sided node FM refinement */
            options5[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP2SIDED;
        else  /* one-sided node FM refinement */
            options5[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;

        options5[METIS_OPTION_DBGLVL] = options[4];

        switch(options[5]){
            case 0:  /* do not try to compress or order connected components */
                options5[METIS_OPTION_COMPRESS] = 0;
                options5[METIS_OPTION_CCORDER] = 0;
                break;
            case 1:  /* try to compress graph */
                options5[METIS_OPTION_COMPRESS] = 1;
                options5[METIS_OPTION_CCORDER] = 0;
                break;
            case 2:  /* order each component separately */
                options5[METIS_OPTION_COMPRESS] = 0;
                options5[METIS_OPTION_CCORDER] = 1;
                break;
            case 3:  /* try to compress and order components */
                options5[METIS_OPTION_COMPRESS] = 1;
                options5[METIS_OPTION_CCORDER] = 1;
                break;
        }

        options5[METIS_OPTION_PFACTOR] = options[6];

        options5[METIS_OPTION_NSEPS] = options[7];

        if(*n_options >= 9) /* override inappropriate MeTiS 5 ufactor option */
          options5[METIS_OPTION_UFACTOR] = options[8];
    }

/*
    options5[METIS_OPTION_OBJTYPE] =  METIS_OBJTYPE_NODE;
    options5[METIS_OPTION_RTYPE] =    METIS_RTYPE_SEP1SIDED;
    options5[METIS_OPTION_IPTYPE] =   METIS_IPTYPE_EDGE;
    options5[METIS_OPTION_NSEPS] =    1;
    options5[METIS_OPTION_NITER] =    10;
    options5[METIS_OPTION_COMPRESS] = 1;
    options5[METIS_OPTION_CCORDER] =  0;
    options5[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options5[METIS_OPTION_NO2HOP] = 0;
    options5[METIS_OPTION_ONDISK] = 0;
    options5[METIS_OPTION_SEED] = -1;
    options5[METIS_OPTION_DBGLVL] = 0;
    options5[METIS_OPTION_NUMBERING] = 0;
    options5[METIS_OPTION_DROPEDGES] = 0;
    options5[METIS_OPTION_UFACTOR] = 200;
    options5[METIS_OPTION_DBGLVL] = 1;
    options5[METIS_OPTION_NUMBERING] = *numflag;
    options5[METIS_OPTION_UFACTOR] = 1;
    options5[METIS_OPTION_NUMBERING] = -1;
    options5[METIS_OPTION_PTYPE] = -1;
    options5[METIS_OPTION_NCUTS] = -1;
    options5[METIS_OPTION_MINCONN] = 1;
    options5[METIS_OPTION_CONTIG] = -1;
    options5[METIS_OPTION_PFACTOR] = 0;
*/

    /* print values if desired */
    if(debug==1){
      printf("Reset values:\n");
      printf("options5[METIS_OPTION_OBJTYPE] = %d\n", 
                       options5[METIS_OPTION_OBJTYPE]);
      printf("options5[METIS_OPTION_RTYPE] = %d\n", 
                       options5[METIS_OPTION_RTYPE]);
      printf("options5[METIS_OPTION_IPTYPE] = %d\n", 
                       options5[METIS_OPTION_IPTYPE]);
      printf("options5[METIS_OPTION_NSEPS] = %d\n", 
                       options5[METIS_OPTION_NSEPS]);
      printf("options5[METIS_OPTION_NITER] = %d\n", 
                       options5[METIS_OPTION_NITER]);
      printf("options5[METIS_OPTION_UFACTOR] = %d\n", 
                       options5[METIS_OPTION_UFACTOR]);
      printf("options5[METIS_OPTION_COMPRESS] = %d\n", 
                       options5[METIS_OPTION_COMPRESS]);
      printf("options5[METIS_OPTION_CCORDER] = %d\n", 
                       options5[METIS_OPTION_CCORDER]);
      printf("options5[METIS_OPTION_CTYPE] = %d\n",
                       options5[METIS_OPTION_CTYPE]);
      printf("options5[METIS_OPTION_NO2HOP] = %d\n", 
                       options5[METIS_OPTION_NO2HOP]);
      printf("options5[METIS_OPTION_ONDISK] = %d\n", 
                       options5[METIS_OPTION_ONDISK]);
      printf("options5[METIS_OPTION_SEED] = %d\n", 
                       options5[METIS_OPTION_SEED]);
      printf("options5[METIS_OPTION_DBGLVL] = %d\n", 
                       options5[METIS_OPTION_DBGLVL]);
      printf("options5[METIS_OPTION_NUMBERING] = %d\n", 
                       options5[METIS_OPTION_NUMBERING]);
      printf("options5[METIS_OPTION_DROPEDGES] = %d\n", 
                       options5[METIS_OPTION_DROPEDGES]);
      printf("options5[METIS_OPTION_PTYPE] = %d\n",
                       options5[METIS_OPTION_PTYPE]);
      printf("options5[METIS_OPTION_NCUTS] = %d\n",
                       options5[METIS_OPTION_NCUTS]);
      printf("options5[METIS_OPTION_MINCONN] = %d\n",
                       options5[METIS_OPTION_MINCONN]);
      printf("options5[METIS_OPTION_CONTIG] = %d\n",
                       options5[METIS_OPTION_CONTIG]);
      printf("options5[METIS_OPTION_PFACTOR] = %d\n",
                       options5[METIS_OPTION_PFACTOR]);
      printf("options5[METIS_OPTION_NIPARTS] = %d\n", 
                       options5[METIS_OPTION_NIPARTS]);
      printf("options5[METIS_OPTION_TWOHOP] = %d\n", 
                       options5[METIS_OPTION_TWOHOP]);
      printf("options5[METIS_OPTION_FAST] = %d\n", 
                       options5[METIS_OPTION_FAST]);
    }

    /* Call MeTiS 5 to get ordering */
    METIS_NodeND(nvtxs, xadj, adjncy, NULL, options5, perm, iperm);
}
