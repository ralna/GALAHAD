/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-19 AT 09:30 GMT
 *
 *  \brief Defines C++ interface to routines from galahad_ssids_contrib and
 *         galahad_ssids_contrib_free modules.
 */

#include "galahad_modules.h"
#include "galahad_precision.h"

#ifndef GALAHAD_SSIDS_CONTRIB_H
#define GALAHAD_SSIDS_CONTRIB_H

#ifdef __cplusplus
extern "C" {
#endif

void galahad_ssids_contrib_get_data(const void *const contrib,
      ipc_ *const n, const rpc_* *const val, ipc_ *const ldval,
      const ipc_* *const rlist, ipc_ *const ndelay,
      const ipc_* *const delay_perm,
      const rpc_* *const delay_val, ipc_ *const lddelay);

void galahad_ssids_contrib_free(void *const contrib);

#ifdef __cplusplus
}
#endif

#endif /* GALAHAD_SSIDS_CONTRIB_H */
