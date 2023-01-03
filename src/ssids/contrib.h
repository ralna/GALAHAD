/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 *  \brief Defines C++ interface to routines from spral_ssids_contrib and
 *         spral_ssids_contrib_free modules.
 */
#ifndef SPRAL_SSIDS_CONTRIB_H
#define SPRAL_SSIDS_CONTRIB_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SPRAL_SINGLE
void spral_ssids_contrib_get_data_single(const void *const contrib, 
      int *const n, const float* *const val, int *const ldval, 
      const int* *const rlist, int *const ndelay, const int* *const delay_perm,
      const float* *const delay_val, int *const lddelay);
void spral_ssids_contrib_free_sgl(void *const contrib);
#else
void spral_ssids_contrib_get_data_double(const void *const contrib, 
      int *const n, const double* *const val, int *const ldval, 
      const int* *const rlist, int *const ndelay, const int* *const delay_perm,
      const double* *const delay_val, int *const lddelay);
void spral_ssids_contrib_free_dbl(void *const contrib);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPRAL_SSIDS_CONTRIB_H */
