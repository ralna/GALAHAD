/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids/cpu/kernels/ldlt_nopiv.hxx"

namespace spral { namespace ssids { namespace cpu {

/* We perform a 2x2 blocked LDL^T factorization of an m x n matrix.
 * This is not pivoted, and assumes we're doing this instead of a cholesky
 * factorization.
 * By doing a 2x2 blocked factor we only need n/2 divides, rather than the
 * n divides + n sqrts of the Cholesky factorization, which is the throughput
 * limiting operation for small matrices (at the cost of some additional
 * multiplies).
 *
 * m - number of rows
 * n - number of cols (n<=m)
 * a[n*lda] - matrix to factor on input; factors on output
 * lda - leading dimension of a (lda>=m)
 * work[2*m] - workspace for internal use
 *
 * Returns -1 on success, otherwise location of negative or zero pivot.
 * */
int ldlt_nopiv_factor(int m, int n, double* a, int lda, double* work) {
   for(int j=0; j<n-1; j+=2) {
      /* Setup shortcut pointers to make code easier to read */
      double *a1 = &a[j*lda], *a2 = &a[(j+1)*lda];
      double *work1 = &work[0], *work2 = &work[m];
      /* Invert 2x2 diagonal block (j,j+1) */
      double a11 = a1[j];
      double a21 = a1[j+1];
      double a22 = a2[j+1];
      double det = a11*a22 - a21*a21;
      if(det <= 0.0)
         return (a11 <= 0.0) ? j : j+1; /* Matrix is not +ive definite */
      det = 1/det;
      double l11 = a22 * det;    a1[j]   = l11;
      double l21 = -a21 * det;   a1[j+1] = l21;
      double l22 = a11 * det;    a2[j+1] = l22;
      /* Apply to block below diagonal */
      for(int i=j+2; i<m; ++i) {
         double x1 = a1[i]; work1[i] = x1;
         double x2 = a2[i]; work2[i] = x2;
         a1[i] = l11*x1 + l21*x2;
         a2[i] = l21*x1 + l22*x2;
      }
      /* Apply schur complement update */
      for(int k=j+2; k<n; ++k)
      for(int i=j+2; i<m; ++i)
         a[k*lda+i] -= a1[i]*work1[k] + a2[i]*work2[k];
   }

   if(n%2!=0) {
      /* n is odd, last column can't use a 2x2 pivot, so use a 1x1 */
      int j = n-1;
      double *a1 = &a[j*lda];
      if(a1[j] <= 0.0) return j; /* matrix not posdef */
      double l11 = 1/a1[j]; a1[j] = l11;
      for(int i=j+1; i<m; ++i)
         a1[i] *= l11;
   }

   return -1; /* success */
}

/* Corresponding forward solve to ldlt_nopiv_factor() */
void ldlt_nopiv_solve_fwd(int m, int n, double const* a, int lda, double *x) {
   for(int j=0; j<n-1; j+=2) {
      for(int i=j+2; i<m; ++i)
         x[i] -= a[j*lda+i]*x[j] + a[(j+1)*lda+i]*x[j+1];
   }
   if(n%2!=0) {
      // n is odd, handle last column as 1x1 pivot
      int j = n-1;
      for(int i=n; i<m; ++i)
         x[i] -= a[j*lda+i]*x[j];
   }
}

/* Corresponding diagonal solve to ldlt_nopiv_factor() */
void ldlt_nopiv_solve_diag(int m, int n, double const* a, int lda, double *x) {
   for(int j=0; j<n-1; j+=2) {
      double x1 = x[j];
      double x2 = x[j+1];
      x[j]   = a[j*lda+j  ]*x1 + a[    j*lda+j+1]*x2;
      x[j+1] = a[j*lda+j+1]*x1 + a[(j+1)*lda+j+1]*x2;
   }
   if(n%2!=0) {
      // n is odd, handle last column as 1x1 pivot
      int j = n-1;
      x[j] *= a[j*lda+j];
   }
}

/* Corresponding backward solve to ldlt_nopiv_factor() */
void ldlt_nopiv_solve_bwd(int m, int n, double const* a, int lda, double *x) {
   if(n%2!=0) {
      // n is odd, handle last column as 1x1 pivot
      int j = n-1;
      for(int i=n; i<m; ++i)
         x[j] -= a[j*lda+i]*x[i];
      n--;
   }
   for(int j=n-2; j>=0; j-=2) {
      for(int i=j+2; i<m; ++i) {
         x[j] -= a[j*lda+i] * x[i];
         x[j+1] -= a[(j+1)*lda+i] * x[i];
      }
   }
}

}}} /* namespaces spral::ssids::cpu */
