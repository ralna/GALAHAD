/* trstf.c */
/* Full test for the TRS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_trs.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct trs_control_type control;
    struct trs_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension of H
    ipc_ m = 1; // dimension of A
    ipc_ H_ne = 4; // number of elements of H
    ipc_ M_ne = 3; // number of elements of M
    ipc_ A_ne = 3; // number of elements of A
    ipc_ H_dense_ne = 6; // number of elements of H
    ipc_ M_dense_ne = 6; // number of elements of M
    ipc_ H_row[] = {1, 2, 3, 3}; // row indices, NB lower triangle
    ipc_ H_col[] = {1, 2, 3, 1};
    ipc_ H_ptr[] = {1, 2, 3, 5};
    ipc_ M_row[] = {1, 2, 3}; // row indices, NB lower triangle
    ipc_ M_col[] = {1, 2, 3};
    ipc_ M_ptr[] = {1, 2, 3, 4};
    ipc_ A_row[] = {1, 1, 1} ;
    ipc_ A_col[] = {1, 2, 3};
    ipc_ A_ptr[] = {1, 4};
    rpc_ H_val[] = {1.0, 2.0, 3.0, 4.0};
    rpc_ M_val[] = {1.0, 2.0, 1.0};
    rpc_ A_val[] = {1.0, 1.0, 1.0};
    rpc_ H_dense[] = {1.0, 0.0, 2.0, 4.0, 0.0, 3.0};
    rpc_ M_dense[] = {1.0, 0.0, 2.0, 0.0, 0.0, 1.0};
    rpc_ H_diag[] = {1.0, 0.0, 2.0};
    rpc_ M_diag[] = {1.0, 2.0, 1.0};
    rpc_ f = 0.96;
    rpc_ radius = 1.0;
    rpc_ c[] = {0.0, 2.0, 0.0};

    char st = ' ';
    ipc_ status;
    rpc_ x[n];
    char ma[3];

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ a_is=0; a_is <= 1; a_is++){ // add a linear constraint?
      for( ipc_ m_is=0; m_is <= 1; m_is++){ // include a scaling matrix?

        if (a_is == 1 && m_is == 1 ) {
          strcpy(ma, "MA");
        }
        else if (a_is == 1) {
          strcpy(ma, "A ");
        }
        else if (m_is == 1) {
          strcpy(ma, "M ");
        }
        else {
          strcpy(ma, "  ");
        }

        for( ipc_ storage_type=1; storage_type <= 4; storage_type++){

          // Initialize TRS
          trs_initialize( &data, &control, &status );

          // Set user-defined control options
          control.f_indexing = true; // fortran sparse matrix indexing

          switch(storage_type){
              case 1: // sparse co-ordinate storage
                  st = 'C';
                  // import the control parameters and structural data
                  trs_import( &control, &data, &status, n,
                             "coordinate", H_ne, H_row, H_col, NULL );
                  if (m_is == 1) {
                    trs_import_m( &data, &status, n,
                                  "coordinate", M_ne, M_row, M_col, NULL );
                  }
                  if (a_is == 1) {
                    trs_import_a( &data, &status, m,
                                  "coordinate", A_ne, A_row, A_col, NULL );
                  }
                  // solve the problem
                  if (a_is == 1 && m_is == 1 ) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       M_ne, M_val, m, A_ne, A_val, NULL );
                  }
                  else if (a_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       0, NULL, m, A_ne, A_val, NULL );
                  }
                  else if (m_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       M_ne, M_val, 0, 0, NULL, NULL );
                  }
                  else {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       0, NULL, 0, 0, NULL, NULL );
                  }
                  break;
              case 2: // sparse by rows
                  st = 'R';
                  // import the control parameters and structural data
                  trs_import( &control, &data, &status, n,
                              "sparse_by_rows", H_ne, NULL, H_col, H_ptr );
                  if (m_is == 1) {
                    trs_import_m( &data, &status, n,
                                  "sparse_by_rows", M_ne, NULL, M_col, M_ptr );
                  }
                  if (a_is == 1) {
                    trs_import_a( &data, &status, m,
                                 "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                  }
                  // solve the problem
                  if (a_is == 1 && m_is == 1 ) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       M_ne, M_val, m, A_ne, A_val, NULL );
                  }
                  else if (a_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       0, NULL, m, A_ne, A_val, NULL );
                  }
                  else if (m_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       M_ne, M_val, 0, 0, NULL, NULL );
                  }
                  else {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_ne, H_val, x,
                                       0, NULL, 0, 0, NULL, NULL );
                  }
                  break;
              case 3: // dense
                  st = 'D';
                  // import the control parameters and structural data
                  trs_import( &control, &data, &status, n,
                              "dense", H_ne, NULL, NULL, NULL );
                  if (m_is == 1) {
                    trs_import_m( &data, &status, n,
                                 "dense", M_ne, NULL, NULL, NULL );
                  }
                  if (a_is == 1) {
                    trs_import_a( &data, &status, m,
                                 "dense", A_ne, NULL, NULL, NULL );
                  }
                  // solve the problem
                  if (a_is == 1 && m_is == 1 ) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_dense_ne, H_dense, x,
                                       M_dense_ne, M_dense, m, A_ne, A_val,
                                       NULL );
                  }
                  else if (a_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_dense_ne, H_dense, x,
                                       0, NULL, m, A_ne, A_val, NULL );
                  }
                  else if (m_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_dense_ne, H_dense, x,
                                       M_dense_ne, M_dense, 0, 0, NULL, NULL );
                  }
                  else {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, H_dense_ne, H_dense, x,
                                       0, NULL, 0, 0, NULL, NULL );
                  }
                  break;
              case 4: // diagonal
                  st = 'L';
                  // import the control parameters and structural data
                  trs_import( &control, &data, &status, n,
                              "diagonal", H_ne, NULL, NULL, NULL );
                  if (m_is == 1) {
                    trs_import_m( &data, &status, n,
                                 "diagonal", M_ne, NULL, NULL, NULL );
                  }
                  if (a_is == 1) {
                    trs_import_a( &data, &status, m,
                                 "dense", A_ne, NULL, NULL, NULL );
                  }
                  // solve the problem
                  if (a_is == 1 && m_is == 1 ) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, n, H_diag, x,
                                       n, M_diag, m, A_ne, A_val, NULL );
                  }
                  else if (a_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, n, H_diag, x,
                                       0, NULL, m, A_ne, A_val, NULL );
                  }
                  else if (m_is == 1) {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, n, H_diag, x,
                                       n, M_diag, 0, 0, NULL, NULL );
                  }
                  else {
                    trs_solve_problem( &data, &status, n,
                                       radius, f, c, n, H_diag, x,
                                       0, NULL, 0, 0, NULL, NULL );
                  }
                  break;
              }

          trs_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_trs.h
#include "galahad_pquad_trs.h"
#else
          printf("format %c%s: TRS_solve_problem exit status = %1" i_ipc_ 
                 ", f = %.2f\n", st, ma, inform.status, inform.obj );
#endif
          //printf("x: ");
          //for( ipc_ i = 0; i < n+m; i++) printf("%f ", x[i]);

          // Delete internal workspace
          trs_terminate( &data, &control, &inform );
       }
     }
   }
}

