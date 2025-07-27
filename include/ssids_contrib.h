/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 10:30 GMT
 *
 *  \brief Defines C++ interface to routines from spral_ssids_contrib and
 *         spral_ssids_contrib_free modules.
 */

#include "spral_procedures.h"
#include "ssids_rip.hxx"

#ifndef SPRAL_SSIDS_CONTRIB_H
#define SPRAL_SSIDS_CONTRIB_H

#ifdef __cplusplus
extern "C" {
#endif

void spral_ssids_contrib_get_data(const void *const contrib,
      ipc_ *const n, const rpc_* *const val, ipc_ *const ldval,
      const ipc_* *const rlist, ipc_ *const ndelay, 
      const ipc_* *const delay_perm,
      const rpc_* *const delay_val, ipc_ *const lddelay);

void spral_ssids_contrib_free(void *const contrib);

#ifdef __cplusplus
}
#endif

#endif /* SPRAL_SSIDS_CONTRIB_H */
