/* THIS VERSION: GALAHAD 5.2 - 2025-03-11 AT 09:30 GMT */

#include <stddef.h>
#include <inttypes.h>

#ifdef INTEGER_64
typedef int64_t idx_t;
#define galahad_nodend4_adapter galahad_nodend4_adapter_64
#define METIS_NodeND_4 METIS_NodeND_4_64
#define galahad_nodend51_adapter galahad_nodend51_adapter_64
#define METIS_NodeND_51 METIS_NodeND_51_64
#define galahad_nodend52_adapter galahad_nodend52_adapter_64
#define METIS_NodeND_52 METIS_NodeND_52_64
#else
typedef int32_t idx_t;
#endif

/*--------------------
 * Function prototypes 
 *--------------------*/

#ifdef _WINDLL
#define METIS_API(type) __declspec(dllexport) type __cdecl
#elif defined(__cdecl)
#define METIS_API(type) type __cdecl
#else
#define METIS_API(type) type
#endif

METIS_API(int) METIS_NodeND_4(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, 
                              idx_t *numflag,  idx_t *options, 
                              idx_t *perm, idx_t *iperm);

METIS_API(int) METIS_NodeND_51(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, 
                               idx_t *vwgt,  idx_t *options, 
                               idx_t *perm, idx_t *iperm);

METIS_API(int) METIS_NodeND_52(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, 
                               idx_t *vwgt,  idx_t *options, 
                               idx_t *perm, idx_t *iperm);

/* name change to avoid any possible conflicts */

void galahad_nodend4_adapter(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy,
                            idx_t* options, idx_t* perm, idx_t* iperm){
    int numflag = 1;

    /* Call MeTiS 4 to get ordering */
    METIS_NodeND_4(nvtxs, xadj, adjncy, &numflag, options, perm, iperm);
}

void galahad_nodend51_adapter(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy,
                             idx_t* options, idx_t* perm, idx_t* iperm){

    /* Handle MA57 pathological case */
    if(*nvtxs == 1){
        /* MA57 seems to call metis with a graph containing 1 vertex and
         * a self-loop. Metis5 prints an error for this.
         */
        perm[0] = 1;
        iperm[0] = 1;
        return;
    }

    /* Call MeTiS 5.1 to get ordering */
    METIS_NodeND_51(nvtxs, xadj, adjncy, NULL, options, perm, iperm);
}

void galahad_nodend52_adapter(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy,
                             idx_t* options, idx_t* perm, idx_t* iperm){

    /* Handle MA57 pathological case */
    if(*nvtxs == 1){
        /* MA57 seems to call metis with a graph containing 1 vertex and
         * a self-loop. Metis5 prints an error for this.
         */
        perm[0] = 1;
        iperm[0] = 1;
        return;
    }

    /* Call MeTiS 5.2 to get ordering */
    METIS_NodeND_52(nvtxs, xadj, adjncy, NULL, options, perm, iperm);
}
