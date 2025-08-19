/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#define MAX_CUDA_BLOCKS 65535

#include "ssids_rip.hxx"

namespace galahad { namespace ssids { namespace gpu {

/** \brief Represents work for a a node to be factorized
 *         (as part of a batched call).
 */
struct multinode_fact_type {
  ipc_ nrows; ///< number of rows in node
  ipc_ ncols; ///< number of columns in node
  rpc_ *lval; ///< pointer to factors L
  rpc_ *ldval; ///< pointer to workspace for storing L*D
  rpc_ *dval; ///< pointer to factors D
  ipc_ offp; ///< offset into permutation vector for this node
  ipc_ ib; ///< ???
  ipc_ jb; ///< ???
  ipc_ done; ///< number of columns sucessfully factorized?
  ipc_ rght; ///< ???
  ipc_ lbuf; ///< ???
};

/** \brief Statistics to be returned to user. */
struct cuda_stats {
  ipc_ num_two; ///< Number of 2x2 pivots
  ipc_ num_neg; ///< Number of negative pivots
  ipc_ num_zero; ///< Number of zero pivots
};

}}} /* namespace galahad::ssids::gpu */
