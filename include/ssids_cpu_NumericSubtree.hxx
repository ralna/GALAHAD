/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.2 - 2025-01-24 AT 14:00 GMT
 */

#pragma once

#include "ssids_routines.h"
#include "ssids_rip.hxx"
#include "ssids_profile.hxx"
#include "ssids_cpu_cpu_iface.hxx"
#include "ssids_cpu_factor.hxx"
#include "ssids_cpu_BuddyAllocator.hxx"
#include "ssids_cpu_NumericNode.hxx"
#include "ssids_cpu_SymbolicSubtree.hxx"
#include "ssids_cpu_SmallLeafNumericSubtree.hxx"
#include "ssids_cpu_ThreadStats.hxx"

namespace spral { namespace ssids { namespace cpu {

/** \brief Represents a submatrix (subtree) factorized on the CPU.
 *
 * \tparam posdef true for Cholesky factorization, false for indefinite LDL^T
 * \tparam T underlying numerical type e.g. double
 * \tparam PAGE_SIZE initial size to be used for thread Workspace
 * \tparam FactorAllocator allocator to be used for factor storage. It must
 *         zero memory upon allocation (eg through calloc or memset).
 * */
template <bool posdef, //< true for Cholesky factoriztion, false for indefinte
          typename T,
          size_t PAGE_SIZE,
          typename FactorAllocator
          >
class NumericSubtree {
   typedef BuddyAllocator<T,std::allocator<T>> PoolAllocator;
   //typedef SimpleAlignedAllocator<T> PoolAllocator;
   typedef SmallLeafNumericSubtree<posdef, T, FactorAllocator, PoolAllocator> SLNS;
public:
   /* Delete copy constructors for safety re allocated memory */
   NumericSubtree(const NumericSubtree&) =delete;
   NumericSubtree& operator=(const NumericSubtree&) =delete;
   /** \brief Construct factors associated with specified symbolic subtree by
    *         performing factorization.
    *  \param symbolic_subtree symbolic factorization of subtree to factorize
    *  \param aval pointer to user's a value array (references entire matrix)
    *  \param scaling pointer to optional scaling vector to be applied
    *         (references entire matrix). No scaling applied if null.
    *  \param child_contrib array of pointers to contributions from child
    *         subtrees. Information to be extracted by call to Fortran routine
    *         spral_ssids_contrib_get_data_double() or _single().
    *  \param options user-supplied options controlling execution.
    *  \param stats collection of statistics for return to user.
    */
   NumericSubtree(
         SymbolicSubtree const& symbolic_subtree,
         T const* aval,
         T const* scaling,
         void** child_contrib,
         struct cpu_factor_options const& options,
         ThreadStats& stats)
   : symb_(symbolic_subtree),
     factor_alloc_(symbolic_subtree.get_factor_mem_est(options.multiplier)),
     pool_alloc_(symbolic_subtree.get_pool_size<T>()),
     small_leafs_(static_cast<SLNS*>(::operator new[](symb_.small_leafs_.size()*sizeof(SLNS))))
   {
      /* Associate symbolic nodes to numeric ones; copy tree structure */
      nodes_.reserve(symbolic_subtree.nnodes_+1);
      for(ipc_ ni=0; ni<symb_.nnodes_+1; ++ni) {
         nodes_.emplace_back(symbolic_subtree[ni], pool_alloc_);
         auto* fc = symbolic_subtree[ni].first_child;
         nodes_[ni].first_child = fc ? &nodes_[fc->idx] : nullptr;
         auto* nc = symbolic_subtree[ni].next_child;
         nodes_[ni].next_child = nc ? &nodes_[nc->idx] :  nullptr;
      }

      /* Allocate workspaces */
      int num_threads = omp_get_num_threads();
      std::vector<ThreadStats> thread_stats(num_threads);
      std::vector<Workspace> work;
      work.reserve(num_threads);
      for(ipc_ i=0; i<num_threads; ++i)
         work.emplace_back(PAGE_SIZE);

       // initialise stats already so we can safely early-return in case of
       // failure if not compiled with OpenMP (instead of omp cancel)
       stats = ThreadStats();

      // Each node is depend(inout) on itself and depend(in) on its parent.
      // Whilst this isn't really what's happening it does ensure our
      // ordering is correct: each node cannot be scheduled until all its
      // children are done, but its children to run in any order.
      bool abort;
      #pragma omp atomic write
      abort = false; // Set to true to abort remaining tasks
      #pragma omp taskgroup
      {
         /* Loop over small leaf subtrees */
         for(uipc_ si=0; si<symb_.small_leafs_.size(); ++si) {
            auto* parent_lcol = nodes_.data() + symb_.small_leafs_[si].get_parent();
            #pragma omp task default(none) \
               firstprivate(si) \
               shared(aval, abort, options, scaling, thread_stats, work) \
               depend(in: parent_lcol[0:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
               // #pragma omp cancellation point taskgroup
               try {
                  int this_thread = omp_get_thread_num();
#ifdef PROFILE
                  Profile::Task task_subtree("TA_SUBTREE");
#endif
                  auto const& leaf = symb_.small_leafs_[si];
                  new (&small_leafs_[si]) SLNS(leaf, nodes_, aval, scaling,
                        factor_alloc_, pool_alloc_, work,
                        options, thread_stats[this_thread]);
                  if(thread_stats[this_thread].flag<Flag::SUCCESS) {
#ifdef _OPENMP
                     #pragma omp atomic write
                     abort = true;
                     #pragma omp cancel taskgroup
#else
                     stats += thread_stats[this_thread];
                     return;
#endif /* _OPENMP */
                  }
#ifdef PROFILE
                  task_subtree.done();
#endif
               } catch (std::bad_alloc const&) {
                  int thread_num = omp_get_thread_num();
                  thread_stats[thread_num].flag =
                     Flag::ERROR_ALLOCATION;
#ifdef _OPENMP
                  #pragma omp atomic write
                  abort = true;
                  #pragma omp cancel taskgroup
#else
                  stats += thread_stats[0];
                  return;
#endif /* _OPENMP */
               } catch (SingularError const&) {
                  int thread_num = omp_get_thread_num();
                  thread_stats[thread_num].flag =
                     Flag::ERROR_SINGULAR;
#ifdef _OPENMP
                  #pragma omp atomic write
                  abort = true;
                  #pragma omp cancel taskgroup
#else
                  stats += thread_stats[0];
                  return;
#endif /* _OPENMP */
               }
            } } // task/abort
         }

         /* Loop over singleton nodes in order */
         for(ipc_ ni=0; ni<symb_.nnodes_; ++ni) {
            if(symb_[ni].insmallleaf) continue; // already handled
            auto* this_lcol = &nodes_[ni]; // for depend
            auto* parent_lcol = nodes_.data() + symb_[ni].parent; // for depend
            #pragma omp task default(none) \
               firstprivate(ni) \
               shared(aval, abort, child_contrib, options, scaling, \
                      thread_stats, work) \
               depend(inout: this_lcol[0:1]) \
               depend(in: parent_lcol[0:1])
            {
              bool my_abort;
              #pragma omp atomic read
              my_abort = abort;
              if (!my_abort) {
               // #pragma omp cancellation point taskgroup
               try {
                  // printf("%d: Node %d parent %d (of %d) size %d x %d\n",
                  //       omp_get_thread_num(), ni, symb_[ni].parent,
                  //       symb_.nnodes_, symb_[ni].nrow, symb_[ni].ncol);
                  int this_thread = omp_get_thread_num();
                  // Assembly of node (not of contribution block)
                  assemble_pre
                     (posdef, symb_.n, symb_[ni], child_contrib, nodes_[ni],
                      factor_alloc_, pool_alloc_, work, aval, scaling);
                  // Update stats
                  ipc_ nrow = symb_[ni].nrow + nodes_[ni].ndelay_in;
                  thread_stats[this_thread].maxfront =
                     std::max(thread_stats[this_thread].maxfront, nrow);
                  ipc_ ncol = symb_[ni].ncol + nodes_[ni].ndelay_in;
                  thread_stats[this_thread].maxsupernode =
                     std::max(thread_stats[this_thread].maxsupernode, ncol);

                  // Factorization
                  factor_node<posdef>
                     (ni, symb_[ni], nodes_[ni], options,
                      thread_stats[this_thread], work,
                      pool_alloc_);
                  if(thread_stats[this_thread].flag<Flag::SUCCESS) {
#ifdef _OPENMP
                     #pragma omp atomic write
                     abort = true;
                     #pragma omp cancel taskgroup
#else
                     stats += thread_stats[0];
                     return;
#endif /* _OPENMP */
                  }

                  // Assemble children into contribution block
                  #pragma omp atomic read
                  my_abort = abort;
                  if (!my_abort)
                     assemble_post(symb_.n, symb_[ni], child_contrib,
                           nodes_[ni], pool_alloc_, work);
               } catch (std::bad_alloc const&) {
                  int thread_num = omp_get_thread_num();
                  thread_stats[thread_num].flag =
                     Flag::ERROR_ALLOCATION;
#ifdef _OPENMP
                  #pragma omp atomic write
                  abort = true;
                  #pragma omp cancel taskgroup
#else
                  stats += thread_stats[0];
                  return;
#endif /* _OPENMP */
               } catch (SingularError const&) {
                  int thread_num = omp_get_thread_num();
                  thread_stats[thread_num].flag =
                     Flag::ERROR_SINGULAR;
#ifdef _OPENMP
                  #pragma omp atomic write
                  abort = true;
                  #pragma omp cancel taskgroup
#else
                  stats += thread_stats[0];
                  return;
#endif /* _OPENMP */
               }
            } } // task/abort
         }
      } // taskgroup

      // Reduce thread_stats (stats already initialised above)
      for(auto tstats : thread_stats)
         stats += tstats;
      if(stats.flag < 0) return;

      // Count stats
      // FIXME: Do this as we go along...
      if(posdef) {
         // all stats remain zero
      } else { // indefinite
         for(ipc_ ni=0; ni<symb_.nnodes_; ni++) {
            ipc_ m = symb_[ni].nrow + nodes_[ni].ndelay_in;
            ipc_ n = symb_[ni].ncol + nodes_[ni].ndelay_in;
            ipc_ ldl = align_lda<T>(m);
            T *d = nodes_[ni].lcol + n*ldl;
            for(ipc_ i=0; i<nodes_[ni].nelim; ) {
               T a11 = d[2*i];
               T a21 = d[2*i+1];
               if(i+1==nodes_[ni].nelim || 
#ifdef REAL_128
                  std::isfinite(static_cast<double>(d[2*i+2]))) {
#else
                  std::isfinite(d[2*i+2])) {
#endif
                  // 1x1 pivot (or zero)
                  if(a11 == 0.0) {
                     // NB: If we reach this stage, options.action must be true.
                     stats.flag = Flag::WARNING_FACT_SINGULAR;
                     stats.num_zero++;
                  }
                  if(a11 < 0.0) stats.num_neg++;
                  i++;
               } else {
                  // 2x2 pivot
                  T a22 = d[2*i+3];
                  stats.num_two++;
                  T det = a11*a22 - a21*a21; // product of evals
                  T trace = a11 + a22; // sum of evals
                  if(det < 0) stats.num_neg++;
                  else if(trace < 0) stats.num_neg+=2;
                  i+=2;
               }
            }
         }
      }
   }
   ~NumericSubtree() {
      delete[] small_leafs_;
   }

   void solve_fwd(ipc_ nrhs, T* x, ipc_ ldx) const {
      /* Allocate memory */
      T* xlocal = new T[nrhs*symb_.n];
      ipc_* map_alloc = (!posdef) ? new ipc_[symb_.n] : nullptr; // only indef

      /* Main loop */
      for(ipc_ ni=0; ni<symb_.nnodes_; ++ni) {
         ipc_ m = symb_[ni].nrow;
         ipc_ n = symb_[ni].ncol;
         ipc_ nelim = (posdef) ? n
                              : nodes_[ni].nelim;
         ipc_ ndin = (posdef) ? 0
                             : nodes_[ni].ndelay_in;
         ipc_ ldl = align_lda<T>(m+ndin);

         /* Build map (indef only) */
         ipc_ const *map;
         if(!posdef) {
            // indef need to allow for permutation and/or delays
            for(ipc_ i=0; i<n+ndin; ++i)
               map_alloc[i] = nodes_[ni].perm[i];
            for(ipc_ i=n; i<m; ++i)
               map_alloc[i+ndin] = symb_[ni].rlist[i];
            map = map_alloc;
         } else {
            // posdef there is no permutation
            map = symb_[ni].rlist;
         }

         /* Gather into dense vector xlocal */
         // FIXME: don't bother copying elements of x > m, just use beta=0
         //        in dgemm call and then add as we scatter
         for(ipc_ r=0; r<nrhs; ++r)
         for(ipc_ i=0; i<m+ndin; ++i)
            xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1]; // Fortran indexed

         /* Perform dense solve */
         if(posdef) {
            cholesky_solve_fwd(m, n, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n);
         } else { /* indef */
            ldlt_app_solve_fwd(m+ndin, nelim, nodes_[ni].lcol, ldl, nrhs,
                  xlocal, symb_.n);
         }

         /* Scatter result */
         for(ipc_ r=0; r<nrhs; ++r)
         for(ipc_ i=0; i<m+ndin; ++i)
            x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
      }

      /* Cleanup memory */
      if(!posdef) delete[] map_alloc; // only used in indef case
      delete[] xlocal;
   }

   template <bool do_diag, bool do_bwd>
   void solve_diag_bwd_inner(ipc_ nrhs, T* x, ipc_ ldx) const {
      if(posdef && !do_bwd) return; // diagonal solve is a no-op for posdef

      /* Allocate memory - map only needed for indef bwd/diag_bwd solve */
      T* xlocal = new T[nrhs*symb_.n];
      ipc_* map_alloc = (!posdef && do_bwd) ? new ipc_[symb_.n]
                                           : nullptr;

      /* Perform solve */
      for(ipc_ ni=symb_.nnodes_-1; ni>=0; --ni) {
         ipc_ m = symb_[ni].nrow;
         ipc_ n = symb_[ni].ncol;
         ipc_ nelim = (posdef) ? n
                              : nodes_[ni].nelim;
         ipc_ ndin = (posdef) ? 0
                             : nodes_[ni].ndelay_in;

         /* Build map (indef only) */
         ipc_ const *map;
         if(!posdef) {
            // indef need to allow for permutation and/or delays
            if(do_bwd) {
               for(ipc_ i=0; i<n+ndin; ++i)
                  map_alloc[i] = nodes_[ni].perm[i];
               for(ipc_ i=n; i<m; ++i)
                  map_alloc[i+ndin] = symb_[ni].rlist[i];
               map = map_alloc;
            } else { // if only doing diagonal, only need first nelim<=n+ndin
               map = nodes_[ni].perm;
            }
         } else {
            // posdef there is no permutation
            map = symb_[ni].rlist;
         }

         /* Gather into dense vector xlocal */
         ipc_ blkm = (do_bwd) ? m+ndin
                             : nelim;
         ipc_ ldl = align_lda<T>(m+ndin);
         for(ipc_ r=0; r<nrhs; ++r)
         for(ipc_ i=0; i<blkm; ++i)
            xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1];

         /* Perform dense solve */
         if(posdef) {
            cholesky_solve_bwd(m, n, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n);
         } else {
            if(do_diag) ldlt_app_solve_diag(
                  nelim, &nodes_[ni].lcol[(n+ndin)*ldl], nrhs, xlocal, symb_.n
                  );
            if(do_bwd) ldlt_app_solve_bwd(
                  m+ndin, nelim, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n
                  );
         }

         /* Scatter result (only first nelim entries have changed) */
         for(ipc_ r=0; r<nrhs; ++r)
         for(ipc_ i=0; i<nelim; ++i)
            x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
      }

      /* Cleanup memory */
      if(!posdef && do_bwd) delete[] map_alloc; // only used in indef case
      delete[] xlocal;
   }

   void solve_diag(ipc_ nrhs, T* x, ipc_ ldx) const {
      solve_diag_bwd_inner<true, false>(nrhs, x, ldx);
   }

   void solve_diag_bwd(ipc_ nrhs, T* x, ipc_ ldx) const {
      solve_diag_bwd_inner<true, true>(nrhs, x, ldx);
   }

   void solve_bwd(ipc_ nrhs, T* x, ipc_ ldx) const {
      solve_diag_bwd_inner<false, true>(nrhs, x, ldx);
   }

   /** Returns information on diagonal entries and/or pivot order.
    * Note that piv_order is only set in indefinite case.
    * One of piv_order or d may be null in indefinite case.
    */
   void enquire(ipc_ *piv_order, T* d) const {
      if(posdef) {
         for(ipc_ ni=0; ni<symb_.nnodes_; ++ni) {
            ipc_ blkm = symb_[ni].nrow;
            ipc_ nelim = symb_[ni].ncol;
            ipc_ ldl = align_lda<T>(blkm);
            for(ipc_ i=0; i<nelim; ++i)
               *(d++) = nodes_[ni].lcol[i*(ldl+1)];
         }
      } else { /*indef*/
         for(ipc_ ni=0, piv=0; ni<symb_.nnodes_; ++ni) {
            ipc_ blkm = symb_[ni].nrow + nodes_[ni].ndelay_in;
            ipc_ blkn = symb_[ni].ncol + nodes_[ni].ndelay_in;
            ipc_ ldl = align_lda<T>(blkm);
            ipc_ nelim = nodes_[ni].nelim;
            T const* dptr = &nodes_[ni].lcol[blkn*ldl];
//            if (d) {
//              printf("d01 = %.1f %.1f\n", dptr[0], dptr[1]);
//              printf("d23 = %.1f %.1f\n", dptr[2], dptr[3]);
//              printf("d45 = %.1f %.1f\n", dptr[4], dptr[5]);
//              printf("d67 = %.1f %.1f\n", dptr[6], dptr[7]);
//            }
//          printf("ni = %i, nelim = %i\n", ni+1, nelim);
//printf(" print from ssids_cpu_NumericSubtree.hxx:\n");
            for(ipc_ i=0; i<nelim; ) {
//             bool a=i+1==nelim ;
//             bool b=(std::isfinite(dptr[2*i+2]));
//             printf(" i = %d a = %d b = %d\n", i, a, b );
               if(i+1==nelim || 
#ifdef REAL_128
                  std::isfinite(static_cast<double>(dptr[2*i+2]))) {
#else
                  std::isfinite(dptr[2*i+2])) {
#endif
                  /* 1x1 pivot */
                  if(piv_order) {
                     piv_order[nodes_[ni].perm[i]-1] = (piv++);
//                   printf(" 1x1 pivot order %d = %d\n", nodes_[ni].perm[i]-1,
//                            piv_order[nodes_[ni].perm[i]-1]);
                  }
                  if(d) {
//                   printf("in = %i d(1,1) = %.1f\n", 2*i+0, dptr[2*i+0]);
//                   printf("in = %i d(2,1) = %.1f\n", 2*i+1, dptr[2*i+1]);
                     *(d++) = dptr[2*i+0];
                     *(d++) = 0.0;
                  }
                  i+=1;
               } else {
                  /* 2x2 pivot */
                  if(piv_order) {
                     piv_order[nodes_[ni].perm[i]-1] = -(piv++);
//                   printf(" 2x2 pivot order %d = %d\n", nodes_[ni].perm[i]-1,
//                            piv_order[nodes_[ni].perm[i]-1]);
                     piv_order[nodes_[ni].perm[i+1]-1] = -(piv++);
//                  printf(" 2x2 pivot order %d = %d\n", nodes_[ni].perm[i+1]-1,
//                         piv_order[nodes_[ni].perm[i+1]-1]);
                  }
                  if(d) {
//                    printf("in = %i d(1,1) = %.1f\n", 2*i+0, dptr[2*i+0]);
//                    printf("in = %i d(2,1) = %.1f\n", 2*i+1, dptr[2*i+1]);
//                    printf("in = %i d(1,2) = %.1f\n", 2*i+2, dptr[2*i+2]);
//                    printf("in = %i d(2,2) = %.1f\n", 2*i+3, dptr[2*i+3]);
                     *(d++) = dptr[2*i+0];
                     *(d++) = dptr[2*i+1];
                     *(d++) = dptr[2*i+3]; /* not 2*i+2 as stated ?? */
                     *(d++) = 0.0;
                  }
                  i+=2;
               }
            }

         }
//            printf("piv_order: ");
//            for(ipc_ i=0; i<4; i++) {
//              printf("  %d", piv_order[i]);
//            }
//            printf("\n");
      }
   }

   /** Allows user to alter D values, indef case only. */
   void alter(T const* d) {
      for(ipc_ ni=0; ni<symb_.nnodes_; ++ni) {
         ipc_ blkm = symb_[ni].nrow + nodes_[ni].ndelay_in;
         ipc_ blkn = symb_[ni].ncol + nodes_[ni].ndelay_in;
         ipc_ ldl = align_lda<T>(blkm);
         ipc_ nelim = nodes_[ni].nelim;
         T* dptr = &nodes_[ni].lcol[blkn*ldl];

         for(ipc_ i=0; i<nelim; ) {
#ifdef REAL_128
            if(i+1==nelim || std::isfinite(static_cast<double>(dptr[2*i+2]))) {
#else
            if(i+1==nelim || std::isfinite(dptr[2*i+2])) {
#endif
               /* 1x1 pivot */
               dptr[2*i+0] = *(d++);
               d++; /* bug fix - correct increment */
//             printf("in = %i d(1,1) = %.1f\n", 2*i+0, dptr[2*i+0]);
               i+=1;
            } else {
               /* 2x2 pivot */
               dptr[2*i+0] = *(d++);
               dptr[2*i+1] = *(d++);
               dptr[2*i+3] = *(d++);
               d++; /* bug fix - correct increment */
//             printf("in = %i d(1,1) = %.1f\n", 2*i+0, dptr[2*i+0]);
//             printf("in = %i d(2,1) = %.1f\n", 2*i+1, dptr[2*i+1]);
//             printf("in = %i d(1,2) = %.1f\n", 2*i+2, dptr[2*i+2]);
//             printf("in = %i d(2,2) = %.1f\n", 2*i+3, dptr[2*i+3]);
               i+=2;
            }
         }
//       for(ipc_ i=0; i<nelim; ++i) {
//          dptr[2*i+0] = *(d++);
//          dptr[2*i+1] = *(d++);
//       }
      }
   }

	void print() const {
		for(ipc_ node=0; node<symb_.nnodes_; node++) {
#ifdef INTEGER_64
			printf("== Node %ld ==\n", node);
#else
			printf("== Node %d ==\n", node);
#endif
			ipc_ m = symb_[node].nrow + nodes_[node].ndelay_in;
			ipc_ n = symb_[node].ncol + nodes_[node].ndelay_in;
         ipc_ ldl = align_lda<T>(m);
         ipc_ nelim = nodes_[node].nelim;
			ipc_ const* rlist = &symb_[node].rlist[ symb_[node].ncol ];
			for(ipc_ i=0; i<m; ++i) {
#ifdef INTEGER_64
				if(i<n) printf("%ld%s:", nodes_[node].perm[i], 
                                               (i<nelim)?"X":"D");
				else    printf("%ld:", rlist[i-n]);
#else
				if(i<n) printf("%d%s:", nodes_[node].perm[i], 
                                               (i<nelim)?"X":"D");
				else    printf("%d:", rlist[i-n]);
#endif
#ifdef REAL_128
				for(ipc_ j=0; j<n; j++) {
                                  char buf1[128];
                                  int n1 = quadmath_snprintf(buf1, sizeof buf1,
                                      "%+-#*.20Qe", nodes_[node].lcol[j*ldl+i]);
                                  if ((size_t) n1 < sizeof buf1)
                                     printf( "%s", buf1);
//                                   printf(" %10.2Qe", 
//                                     nodes_[node].lcol[j*ldl+i]);
                                 }
#else
				for(ipc_ j=0; j<n; j++) printf(" %10.2e", 
                                   nodes_[node].lcol[j*ldl+i]);
#endif
            T const* d = &nodes_[node].lcol[n*ldl];
            if(!posdef && i<nelim){
#ifdef REAL_128
               char buf1[128], buf2[128];
               int n1 = quadmath_snprintf(buf1, sizeof buf1, 
                                           "%+-#*.20Qe", d[2*i+0]);
               int n2 = quadmath_snprintf(buf2, sizeof buf2, 
                                           "%+-#*.20Qe", d[2*i+1]);
               if ((size_t) n1 < sizeof buf1 && 
                   (size_t) n2 < sizeof buf2)
                  printf( "  d: %s %s", buf1, buf2);
//               printf("  d: %10.2Qe %10.2Qe", d[2*i+0], d[2*i+1]);
#else
               printf("  d: %10.2e %10.2e", d[2*i+0], d[2*i+1]);
#endif
            }
	    printf("\n");
	  }
	 }
	}

   /** Return contribution block from subtree (if not a real root) */
   void get_contrib(ipc_& n, T const*& val, ipc_& ldval, ipc_ const*& rlist,
         ipc_& ndelay, ipc_ const*& delay_perm, T const*& delay_val,
         ipc_& lddelay) const {
      auto& root = *nodes_.back().first_child;
      n = root.symb.nrow - root.symb.ncol;
      val = root.contrib;
      ldval = n;
      rlist = &root.symb.rlist[root.symb.ncol];
      ndelay = root.ndelay_out;
      delay_perm = (ndelay>0) ? &root.perm[root.nelim]
                              : nullptr;
      lddelay = align_lda<T>(root.symb.nrow + root.ndelay_in);
      delay_val = (ndelay>0) ? &root.lcol[root.nelim*(lddelay+1)]
                             : nullptr;
   }

   /** Frees root's contribution block */
   void free_contrib() {
      nodes_.back().first_child->free_contrib();
   }

   SymbolicSubtree const& get_symbolic_subtree() { return symb_; }

private:
   SymbolicSubtree const& symb_;
   FactorAllocator factor_alloc_;
   PoolAllocator pool_alloc_;
   std::vector<NumericNode<T,PoolAllocator>> nodes_;
   SLNS *small_leafs_; // Apparently emplace_back isn't threadsafe, so
      // std::vector is out. So we use placement new instead.
};

}}} /* end of namespace spral::ssids::cpu */
