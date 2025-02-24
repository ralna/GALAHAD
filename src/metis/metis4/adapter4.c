/* THIS VERSION: GALAHAD 5.2 - 2025-02-21 AT 12:50 GMT.
 */

#ifdef INTEGER_64
#define galahad_metis5_adapter galahad_metis5_adapter_64
#define METIS_NodeND METIS_NodeND_64
#define METIS_SetDefaultOptions METIS_SetDefaultOptions_64
#endif

/*#include "galahad_metis4.h"*/
#include "metis.h" /* from MeTiS 4 */
#include "stdio.h"

/* name change to avoid any possible conflicts */

/*
void galahad_metis4_adapter(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy,
                            idx_t* numflag, idx_t* options, idx_t* perm,
                            idx_t* iperm){
*/

void galahad_metis4_adapter(int* nvtxs, idxtype* xadj, idxtype* adjncy,
                            int* numflag, int* options, idxtype* perm,
                            idxtype* iperm){

    /* Call MeTiS 4 to get ordering */
    METIS_NodeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
