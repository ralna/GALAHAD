// include guard
#ifndef GALAHAD_CALLBACKS_H
#define GALAHAD_CALLBACKS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// ARC, BGO, DGO, TRB, TRU
typedef ipc_ galahad_f( ipc_ n, const rpc_ x[], rpc_ *f, const void *userdata );
typedef ipc_ galahad_g( ipc_ n, const rpc_ x[], rpc_ g[], const void *userdata );
typedef ipc_ galahad_h( ipc_ n, ipc_ ne, const rpc_ x[], rpc_ h[], const void *userdata );
typedef ipc_ galahad_prec( ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[], const void *userdata );
typedef ipc_ galahad_hprod( ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[], bool got_h, const void *userdata );
typedef ipc_ galahad_shprod( ipc_ n, const rpc_ x[], ipc_ nnz_v, const ipc_ index_nz_v[], const rpc_ v[], ipc_ *nnz_u, ipc_ index_nz_u[], rpc_ u[], bool got_h, const void *userdata );

// BLLS, SLLS
typedef ipc_ galahad_constant_prec( ipc_ n, const rpc_ v[], rpc_ p[], const void *userdata );

// NLS
typedef ipc_ galahad_r( ipc_ n, ipc_ m, const rpc_ x[], rpc_ r[], const void *userdata );
typedef ipc_ galahad_jr( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jr[], const void *userdata );
typedef ipc_ galahad_hr( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[], rpc_ hr[], const void *userdata );
typedef ipc_ galahad_jrprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose, rpc_ u[], const rpc_ v[], bool got_j, const void *userdata );
typedef ipc_ galahad_hrprod( ipc_ n, ipc_ m, const rpc_ x[], const rpc_ y[], rpc_ u[], const rpc_ v[], bool got_h, const void *userdata );
typedef ipc_ galahad_shrprod( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[], const rpc_ v[], rpc_ pval[], bool got_h, const void *userdata );

// EXPO
typedef ipc_ galahad_fc( ipc_ n, ipc_ m, const rpc_ x[], rpc_ *f, rpc_ c[], const void *userdata );
typedef ipc_ galahad_gj( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[], rpc_ j[], const void *userdata );
typedef ipc_ galahad_hl( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[], rpc_ h[], const void *userdata );

// UGO
typedef ipc_ galahad_fgh( rpc_ x, rpc_ *f, rpc_ *g, rpc_*h, const void *userdata );

#ifdef __cplusplus
}
#endif

// end include guard
#endif
