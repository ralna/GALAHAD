/* nrektf.c */
/* Full test for the NREK C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_nrek.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct nrek_control_type control;
    struct nrek_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension of H
    ipc_ H_ne = 4; // number of elements of H
    ipc_ S_ne = 3; // number of elements of M
    ipc_ H_dense_ne = 6; // number of elements of H
    ipc_ S_dense_ne = 6; // number of elements of M
    ipc_ H_row[] = {1, 2, 3, 3}; // row indices, NB lower triangle
    ipc_ H_col[] = {1, 2, 3, 1};
    ipc_ H_ptr[] = {1, 2, 3, 5};
    ipc_ S_row[] = {1, 2, 3}; // row indices, NB lower triangle
    ipc_ S_col[] = {1, 2, 3};
    ipc_ S_ptr[] = {1, 2, 3, 4};
    rpc_ H_val[] = {1.0, 2.0, 3.0, 4.0};
    rpc_ S_val[] = {1.0, 2.0, 1.0};
    rpc_ H_dense[] = {1.0, 0.0, 2.0, 4.0, 0.0, 3.0};
    rpc_ S_dense[] = {1.0, 0.0, 2.0, 0.0, 0.0, 1.0};
    rpc_ H_diag[] = {1.0, 0.0, 2.0};
    rpc_ S_diag[] = {1.0, 2.0, 1.0};
    rpc_ power = 3.0;
    rpc_ weight = 1.0;
    rpc_ c[] = {0.0, 2.0, 0.0};
    char st = ' ';
    ipc_ status;
    rpc_ x[n];
    char sr[3];

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ s_is=0; s_is <= 1; s_is++){ // include a scaling matrix?
      for( ipc_ storage_type=1; storage_type <= 4; storage_type++){

        // Initialize NREK
        nrek_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        switch(storage_type){
            case 1: // sparse co-ordinate storage
                st = 'C';
                // import the control parameters and structural data
                nrek_import( &control, &data, &status, n,
                             "coordinate", H_ne, H_row, H_col, NULL );
                if (s_is == 1) {
                  nrek_s_import( &data, &status, n,
                                "coordinate", S_ne, S_row, S_col, NULL );
                }
                break;
            case 2: // sparse by rows
                st = 'R';
                // import the control parameters and structural data
                nrek_import( &control, &data, &status, n,
                             "sparse_by_rows", H_ne, NULL, H_col, H_ptr );
                if (s_is == 1) {
                  nrek_s_import( &data, &status, n,
                                 "sparse_by_rows", S_ne, NULL, S_col, S_ptr );
                }
                break;
            case 3: // dense
                st = 'D';
                // import the control parameters and structural data
                nrek_import( &control, &data, &status, n,
                            "dense", H_ne, NULL, NULL, NULL );
                if (s_is == 1) {
                  nrek_s_import( &data, &status, n,
                                 "dense", S_ne, NULL, NULL, NULL );
                }
                break;
            case 4: // diagonal
                st = 'G';
                // import the control parameters and structural data
                nrek_import( &control, &data, &status, n,
                             "diagonal", H_ne, NULL, NULL, NULL );
                if (s_is == 1) {
                  nrek_s_import( &data, &status, n,
                                 "diagonal", S_ne, NULL, NULL, NULL );
                }
                break;
            }
        for( ipc_ w_is=1; w_is <= 2; w_is++){ // original or larger weight
        
          if (w_is == 1) { // use the original weight
             weight = 1.0;
          }
          else { // use the larger weight
             weight = inform.next_weight;
             control.new_weight = true;
             nrek_reset_control( &control, &data, &status );
          }

          if (w_is == 2 && s_is == 1 ) {
            strcpy(sr, "S+");
          }
          else if (w_is == 2) {
            strcpy(sr, "+ ");
          }
          else if (s_is == 1) {
            strcpy(sr, "S ");
          }
          else {
            strcpy(sr, "  ");
          }

          // solve the problem
          switch(storage_type){
              case 1: // sparse co-ordinate storage
                  if (s_is == 1) {
                    nrek_solve_problem( &data, &status, n, H_ne, H_val, c, 
                                        power, weight, x, S_ne, S_val );
                  }
                  else {
                    nrek_solve_problem( &data, &status, n, H_ne, H_val, c, 
                                        power, weight, x, 0, NULL );
                  }
                  break;
              case 2: // sparse by rows
                  if (s_is == 1) {
                    nrek_solve_problem( &data, &status, n, H_ne, H_val, c, 
                                        power, weight, x, S_ne, S_val );
                  }
                  else {
                    nrek_solve_problem( &data, &status, n, H_ne, H_val, c, 
                                        power, weight, x, 0, NULL );
                  }
                  break;
              case 3: // dense
                  if (s_is == 1) {
                    nrek_solve_problem( &data, &status, n, H_dense_ne, 
                                        H_dense, c, power, weight, x,
                                        S_dense_ne, S_dense );
                  }
                  else {
                    nrek_solve_problem( &data, &status, n, H_dense_ne, 
                                        H_dense, c, power, weight, x, 0, NULL );
                  }
                  break;
              case 4: // diagonal
                  if (s_is == 1) {
                    nrek_solve_problem( &data, &status, n, n, H_diag, c,
                                        power, weight, x, n, S_diag );
                  }
                  else {
                    nrek_solve_problem( &data, &status, n, n, H_diag, c,
                                        power, weight, x, 0, NULL );
                  }
                  break;
              }

          nrek_information( &data, &inform, &status );
#ifdef REAL_128
          printf("format %c%s: NREK_solve_problem exit status = %1" d_ipc_ 
                 ", f = %.2f\n", st, sr, inform.status, (double)inform.obj );
#else
          printf("format %c%s: NREK_solve_problem exit status = %1" d_ipc_ 
                 ", f = %.2f\n", st, sr, inform.status, inform.obj );
#endif
          //printf("x: ");
          //for( ipc_ i = 0; i < n+m; i++) printf("%f ", x[i]);

        }
        // Delete internal workspace
        nrek_terminate( &data, &control, &inform );

      }
    }
  }

