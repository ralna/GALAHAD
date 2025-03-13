/*
 * Dummy version of
 *
 *  ometis.c
 *
 * that returns an impossible value of the permutation to signal
 * the absence of the proper (licence-restricted) MeTiS 4 codes
 *
 * If you wish to enable the proper version, and qualify under the
 * licence conditons specified in the LICENSE file, please refer to
 * the AVAILABILITY file.
 *
 * Nick Gould, for GALAHAD productions
 * This version: GALAHAD 5.2 - 2025-03-02
 *
 */

#include "metis.h"

void METIS_NodeND(int *nvtxs, idxtype *xadj, idxtype *adjncy, 
                  int *numflag, int *options, 
                  idxtype *perm, idxtype *iperm) 
{
/* set impossible values */
  perm[0] = - 1;
  iperm[0] = - 1;
}
