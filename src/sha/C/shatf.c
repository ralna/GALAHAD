/* shatf.c */
/* Full test for the SHA C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_sha.h"

int main(void) {

    // Derived types
    void *data;
    struct sha_control_type control;
    struct sha_inform_type inform;

    // Set problem data
    int n = 5; // dimension of H
    int ne = 9; // number of entries in upper triangle of H
    int m_max = 9; // upper bound on max # differences required
    int row[] = {1, 1, 1, 1, 1, 2, 3, 4, 5}; // row indices, NB upper triangle
    int col[] = {1, 2, 3, 4, 5, 2, 3, 4, 5}; // column indices
    real_wp_ val[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // values
    int i, j, k, l, m, status;
    real_wp_ rr, v;
    real_wp_ val_est[ne];
    int ls1 = m_max, ls2 = n; // dimensions of s
    int ly1 = m_max, ly2 = n; // dimensions of y
    real_wp_ strans[ls1][ls2];
    real_wp_ ytrans[ly1][ly2];
    int order[m_max]; // diffference precedence order

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( int algorithm=1; algorithm <= 5; algorithm++){

        // Initialize SHA - use the sytr solver
        sha_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        control.approximation_algorithm = algorithm;
        control.extra_differences = 1;

        // analyse the matrix and discover the number of differences needed
        sha_analyse_matrix( &control, &data, &status, n, ne, row, col, &m );
        printf(" Algorithm %i - %i differences required,"
               " one extra might help\n", algorithm, m);
        m = m + control.extra_differences;

        srand(1);
        for( int k = 0; k < m; k++) {
          // set up random differences strans
          for( int i = 0; i < n; i++) {
            rr = ((real_wp_) rand()/ RAND_MAX);
            strans[k][i] = -1.0 + 2.0 * rr;
            ytrans[k][i] = 0.0; // initialize ytrans as the zero matrix
          }
          // form y = H s
          for( int l = 0; l < ne; l++) {
            i = row[l]-1;
            j = col[l]-1;
            v = val[l];
            ytrans[k][i] = ytrans[k][i] + v * strans[k][j];
            if ( i != j ) ytrans[k][j] = ytrans[k][j] + v * strans[k][i];
          }
          order[k] = m - k; // pick the (s,y) vectors in reverse order
        }

        // recover the matrix
        sha_recover_matrix( &data, &status, ne, m, ls1, ls2, strans,
                            ly1, ly2, ytrans, val_est, order );
        //                  ly1, ly2, ytrans, val_est, NULL );
        //                  if the natural order is ok

        printf(" H from %i differences:\n", m);
        for( int i = 0; i < ne; i++) printf(" %1.2f", val_est[i]);

        sha_information( &data, &inform, &status );

        // Delete internal workspace
        sha_terminate( &data, &control, &inform );
        printf("\n\n");
    }
}

