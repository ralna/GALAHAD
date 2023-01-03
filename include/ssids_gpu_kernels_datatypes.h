#define MAX_CUDA_BLOCKS 65535

namespace spral { namespace ssids { namespace gpu {

/** \brief Represents work for a a node to be factorized
 *         (as part of a batched call).
 */
struct multinode_fact_type {
  int nrows; ///< number of rows in node
  int ncols; ///< number of columns in node
  double *lval; ///< pointer to factors L
  double *ldval; ///< pointer to workspace for storing L*D
  double *dval; ///< pointer to factors D
  int offp; ///< offset into permutation vector for this node
  int ib; ///< ???
  int jb; ///< ???
  int done; ///< number of columns sucessfully factorized?
  int rght; ///< ???
  int lbuf; ///< ???
};

/** \brief Statistics to be returned to user. */
struct cuda_stats {
  int num_two; ///< Number of 2x2 pivots
  int num_neg; ///< Number of negative pivots
  int num_zero; ///< Number of zero pivots
};

}}} /* namespace spral::ssids::gpu */
