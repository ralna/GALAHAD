/* Copyright (C) 2020 COIN-OR
 * All Rights Reserved.
 * This file is distributed under the Eclipse Public License.
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
                            idx_t* numflag, idx_t* options, idx_t* perm,
                            idx_t* iperm){
    idx_t options5[METIS_NOPTIONS];

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
    // options5[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_NODE;

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
    }

    /* Call MeTiS 5 to get ordering */
    /* METIS_NodeND(nvtxs, xadj, adjncy, (void*)0, options5, perm, iperm); */
    METIS_NodeND(nvtxs, xadj, adjncy, NULL, options5, perm, iperm);
}
