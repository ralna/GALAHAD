/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 10:30 GMT
 *
 *  \brief Defines C++ interface to routines from spral_ssids_contrib and
 *         spral_ssids_contrib_free modules.
 */

#include "ssids_rip.hxx"

#ifndef SPRAL_SSIDS_CONTRIB_H
#define SPRAL_SSIDS_CONTRIB_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef REAL_32
void spral_ssids_contrib_get_data_single(const void *const contrib,
      ipc_ *const n, const float* *const val, ipc_ *const ldval,
      const ipc_* *const rlist, ipc_ *const ndelay, 
      const ipc_* *const delay_perm,
      const float* *const delay_val, ipc_ *const lddelay);
void spral_ssids_contrib_free_sgl(void *const contrib);
#elif REAL_128
void spral_ssids_contrib_get_data_quadruple(const void *const contrib,
      ipc_ *const n, const __float128* *const val, ipc_ *const ldval,
      const ipc_* *const rlist, ipc_ *const ndelay, 
      const ipc_* *const delay_perm,
      const __float128* *const delay_val, ipc_ *const lddelay);
void spral_ssids_contrib_free_qul(void *const contrib);
#else
void spral_ssids_contrib_get_data_double(const void *const contrib,
      ipc_ *const n, const double* *const val, ipc_ *const ldval,
      const ipc_* *const rlist, ipc_ *const ndelay, 
      const ipc_* *const delay_perm,
      const double* *const delay_val, ipc_ *const lddelay);
void spral_ssids_contrib_free_dbl(void *const contrib);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPRAL_SSIDS_CONTRIB_H */
