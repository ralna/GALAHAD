/* bsct.c */
/* Full test for the BSC C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bsc.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct bsc_control_type control;
    struct bsc_inform_type inform;

    // Set problem data
    ipc_ m = 3; // row dimension of A
    ipc_ n = 4; // column dimension of A
    ipc_ A_ne = 6; // nonzeros in lower triangle of A
    ipc_ A_dense_ne = 12; // positions in lower triangle of A
    ipc_ A_row[] = {0, 0, 1, 1, 2, 2}; // row indices
    ipc_ A_col[] = {0, 1, 2, 3, 0, 3}; // column indices
    ipc_ A_ptr[] = {0, 2, 4, 6}; // row pointers
    rpc_ A_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // values
    rpc_ A_dense[] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0, 1.0, 1.0, 0.0, 0.0, 1.0}; // dense values
    rpc_ D[] = {1.0, 2.0, 3.0, 4.0}; // diagonals of D

    // Set output storage
    char st = ' ';
    ipc_ status, S_ne;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

      for( ipc_ d=1; d <= 3; d++){

        // Initialize BSC
        bsc_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                bsc_import( &control, &data, &status, m, n,
                           "coordinate", A_ne, A_row, A_col, NULL, &S_ne );
                break;
            case 2: // sparse by rows
                st = 'R';
                bsc_import( &control, &data, &status, m, n,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr, &S_ne );
                break;
            case 3: // dense
                st = 'D';
                bsc_import( &control, &data, &status, m, n,
                            "dense", A_dense_ne, NULL, NULL, NULL, &S_ne );
                break;
            }

        ipc_ S_row[S_ne], S_col[S_ne], S_ptr[m+1];
        rpc_ S_val[S_ne];

        for( ipc_ ptr=0; ptr <= 1; ptr++){

          if(ptr == 0){
            switch(d){
                case 3: // dense
                    bsc_form_s( &data, &status, m, n, A_dense_ne, A_dense, 
                                S_ne, S_row, S_col, NULL, S_val, NULL );
                    break;
                default:
                    bsc_form_s( &data, &status, m, n, A_ne, A_val, 
                                S_ne, S_row, S_col, NULL, S_val, NULL );
                }
          } else {
            switch(d){
                case 3: // dense
                    bsc_form_s( &data, &status, m, n, A_dense_ne, A_dense, 
                                S_ne, S_row, S_col, S_ptr, S_val, D );
                    break;
                default:
                    bsc_form_s( &data, &status, m, n, A_ne, A_val, 
                                S_ne, S_row, S_col, S_ptr, S_val, D );
                }
          }

          bsc_information( &data, &inform, &status );

          if(inform.status == 0){
#ifdef REAL_128
            printf(" format %c: status = %1" d_ipc_ "\n", st, inform.status);
#else
            printf(" format %c: status = %1" d_ipc_ "\n", st, inform.status);
#endif
          }else{
              printf(" format %c: BSC_solve exit status = %1" d_ipc_ "\n", 
                     st, inform.status);
          }

          printf("S_row: ");
          for( ipc_ i = 0; i < S_ne; i++) printf("%1" d_ipc_ " ", S_row[i]);
          printf("\n");
          printf("S_col: ");
          for( ipc_ i = 0; i < S_ne; i++) printf("%1" d_ipc_ " ", S_col[i]);
          printf("\n");
          printf("S_val: ");
#ifdef REAL_128
          for( ipc_ i = 0; i < S_ne; i++) printf("%.2f ", (double)S_val[i]);
#else
          for( ipc_ i = 0; i < S_ne; i++) printf("%.2f ", S_val[i]);
#endif
          printf("\n");
          if(ptr == 1){
            printf("S_ptr: ");
            for( ipc_ i = 0; i < m + 1; i++) printf("%1" d_ipc_ " ", S_ptr[i]);
            printf("\n");
          }
        }

        // Delete internal workspace
        bsc_terminate( &data, &control, &inform );
    }
    printf("Tests complete\n");
}
