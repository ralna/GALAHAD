/* nlst.c */
/* Full test for the NLS C interface using C sparse matrix indexing */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2021 */

#include <stdio.h>
#include <math.h>
#include "nls.h"

#define max(a,b)  \
({  __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

// Custom userdata struct
struct userdata_type {
   double p;
};

// Function prototypes

int res( int n, int m, const double x[], double c[], const void * );
int jac( int n, int m, int jne, const double x[], double jval[], const void * );
int hess( int n, int m, int hne, const double x[], const double y[], 
          double hval[], const void * );
int jacprod( int n, int m, const double x[], const bool transpose, double u[], 
             const double v[], bool got_j, const void * );
int hessprod( int n, int m, const double x[], const double y[], double u[], 
              const double v[], bool got_h, const void * );
int rhessprods( int n, int m, int pne, const double x[], const double v[],
                double pval[], bool got_h, const void * );
int scale( int n, int m, const double x[], double u[], 
           const double v[], const void * );
int jac_dense( int n, int m, int jne, const double x[], double jval[], 
               const void * );
int hess_dense( int n, int m, int hne, const double x[], const double y[],
                double hval[], const void * );
int rhessprods_dense( int n, int m, int pne, const double x[], 
                      const double v[], double pval[], bool got_h,
                      const void * );

int main(void) {

    // Derived types
    void *data;
    struct nls_control_type control;
    struct nls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 1.0;

    // Set problem data
    int n = 2; // # variables
    int m = 3; // # residuals
    int j_ne = 5; // Jacobian elements
    int h_ne = 2; // Hesssian elements
    int p_ne = 2; // residual-Hessians-vector products elements
    int J_row[] = {0, 1, 1, 2, 2}; // Jacobian J
    int J_col[] = {0, 0, 1, 0, 1}; // 
    int J_ptr[] = {0, 1, 3, 5};    // row pointers
    int H_row[] = {0, 1};          // Hessian H
    int H_col[] = {0, 1};          // NB lower triangle
    int H_ptr[] = {0, 1, 2};       // row pointers
    int P_row[] = {0, 1};          // residual-Hessians-vector product matrix
    int P_ptr[] = {0, 1, 2, 2};    // column pointers

    // Set storage
    double g[n]; // gradient
    double c[m]; // residual
    double y[m]; // multipliers
    double W[] = {1.0, 1.0, 1.0}; // weights
    char st;
    int status;

    printf(" C sparse matrix indexing\n\n");

    printf(" tests options for all-in-one storage format\n\n");

    for( int d=1; d <= 5; d++){
//  for( int d=5; d <= 5; d++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = 6;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                nls_import( &control, &data, &status, n, m, 
                            "coordinate", j_ne, J_row, J_col, NULL,
                            "coordinate", h_ne, H_row, H_col, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac, 
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 2: // sparse by rows  
                st = 'R';
                nls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac, 
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 3: // dense
                st = 'D';
                nls_import( &control, &data, &status, n, m, 
                            "dense", j_ne, NULL, NULL, NULL,
                            "dense", h_ne, NULL, NULL, NULL,
                            "dense", p_ne, NULL, NULL, NULL, W ); 
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac_dense, 
                                    h_ne, hess_dense, p_ne, rhessprods_dense );
                break;
            case 4: // diagonal
                st = 'I';
                nls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "diagonal", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac, 
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 5: // access by products
                st = 'P';
                nls_import( &control, &data, &status, n, m, 
                            "absent", j_ne, NULL, NULL, NULL,
                            "absent", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                nls_solve_without_mat( &data, &userdata, &status,
                                       n, m, x, c, g, res, jacprod, 
                                       hessprod, p_ne, rhessprods );
                break;
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: NLS_solve exit status = %1i\n", st, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    int eval_status;
    double u[max(m,n)], v[max(m,n)];
    double J_val[j_ne], J_dense[m*n];
    double H_val[h_ne], H_dense[n*(n+1)/2], H_diag[n];
    double P_val[p_ne], P_dense[m*n];
    bool transpose;
    bool got_j = false;
    bool got_h = false;

    for( int d=1; d <= 5; d++){
//  for( int d=1; d <= 4; d++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = 6;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                nls_import( &control, &data, &status, n, m, 
                            "coordinate", j_ne, J_row, J_col, NULL,
                            "coordinate", h_ne, H_row, H_col, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y, 
                                              h_ne, H_val, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1i of status should not occur\n", 
                        status);
                      break;
                  }
                }
                break;
            case 2: // sparse by rows  
                st = 'R';
                nls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y, 
                                              h_ne, H_val, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1i of status should not occur\n", 
                        status);
                      break;
                  }
                }
                break;
            case 3: // dense
                st = 'D';
                nls_import( &control, &data, &status, n, m, 
                            "dense", j_ne, NULL, NULL, NULL,
                            "dense", h_ne, NULL, NULL, NULL,
                            "dense", p_ne, NULL, NULL, NULL, W ); 
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_dense, y, 
                                              h_ne, H_dense, v, p_ne, P_dense );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac_dense( n, m, j_ne, x, J_dense, 
                                               &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess_dense( n, m, h_ne, x, y, H_dense, 
                                                &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods_dense( n, m, p_ne, x, v, P_dense,
                                                      got_h, &userdata );
                  }else{
                      printf(" the value %1i of status should not occur\n", 
                        status);
                      break;
                  }
                }
                break;
            case 4: // diagonal
                st = 'I';
                nls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "diagonal", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y, 
                                              h_ne, H_diag, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_diag, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1i of status should not occur\n", 
                        status);
                      break;
                  }
                }
                break;
            case 5: // access by products
                st = 'P';
//              control.print_level = 1;
                nls_import( &control, &data, &status, n, m, 
                            "absent", j_ne, NULL, NULL, NULL,
                            "absent", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
                while(true){ // reverse-communication loop
                  nls_solve_reverse_without_mat( &data, &status, &eval_status,
                                                 n, m, x, c, g, &transpose, 
                                                 u, v, y, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 5){ // evaluate u + J v or u + J'v
                      eval_status = jacprod( n, m, x, transpose, u, v, got_j, 
                                             &userdata );
                  }else if(status == 6){ // evaluate u + H v
                      eval_status = hessprod( n, m, x, y, u, v, got_h, 
                                              &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1i of status should not occur\n", 
                        status);
                      break;
                  }
                }
                break;
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: NLS_solve exit status = %1i\n", st, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, direct access\n\n");

    for( int model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = model;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m, 
                    "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
        nls_solve_with_mat( &data, &userdata, &status,
                            n, m, x, c, g, res, j_ne, jac, 
                            h_ne, hess, p_ne, rhessprods );

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf(" %1i:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   model, inform.iter, inform.obj, inform.status);
        }else{
            printf(" %i: NLS_solve exit status = %1i\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, access by products\n\n");

    for( int model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = model;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m, 
                    "absent", j_ne, NULL, NULL, NULL,
                    "absent", h_ne, NULL, NULL, NULL,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
        nls_solve_without_mat( &data, &userdata, &status,
                               n, m, x, c, g, res, jacprod, 
                               hessprod, p_ne, rhessprods );
        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   model, inform.iter, inform.obj, inform.status);
        }else{
            printf("P%i: NLS_solve exit status = %1i\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, reverse access\n\n");

    for( int model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = model;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m, 
                    "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
        while(true){ // reverse-communication loop
          nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                      n, m, x, c, g, j_ne, J_val, y, 
                                      h_ne, H_val, v, p_ne, P_val );
          if(status == 0){ // successful termination
                break;
          }else if(status < 0){ // error exit
              break;
          }else if(status == 2){ // evaluate c
              eval_status = res( n, m, x, c, &userdata );
          }else if(status == 3){ // evaluate J
              eval_status = jac( n, m, j_ne, x, J_val, &userdata );
          }else if(status == 4){ // evaluate H
              eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
          }else if(status == 7){ // evaluate P
              eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                        got_h, &userdata );
          }else{
              printf(" the value %1i of status should not occur\n", 
                status);
              break;
          }
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   model, inform.iter, inform.obj, inform.status);
        }else{
            printf(" %i: NLS_solve exit status = %1i\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, reverse access by products\n\n");

    for( int model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2; 
        control.hessian_available = 2;
        control.model = model;
        double x[] = {1.5,1.5}; // starting point
        double W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m, 
                    "absent", j_ne, NULL, NULL, NULL,
                    "absent", h_ne, NULL, NULL, NULL,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W ); 
        while(true){ // reverse-communication loop
          nls_solve_reverse_without_mat( &data, &status, &eval_status,
                                         n, m, x, c, g, &transpose, 
                                         u, v, y, p_ne, P_val );
          if(status == 0){ // successful termination
                break;
          }else if(status < 0){ // error exit
              break;
          }else if(status == 2){ // evaluate c
              eval_status = res( n, m, x, c, &userdata );
          }else if(status == 5){ // evaluate u + J v or u + J'v
              eval_status = jacprod( n, m, x, transpose, u, v, got_j, 
                                     &userdata );
          }else if(status == 6){ // evaluate u + H v
              eval_status = hessprod( n, m, x, y, u, v, got_h, 
                                      &userdata );
          }else if(status == 7){ // evaluate P
              eval_status = rhessprods( n, m, p_ne, x, v, P_val, 
                                        got_h, &userdata );
          }else{
              printf(" the value %1i of status should not occur\n", 
                status);
              break;
          }
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n", 
                   model, inform.iter, inform.obj, inform.status);
        }else{
            printf("P%i: NLS_solve exit status = %1i\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }
}

// compute the residuals
int res( int n, int m, const double x[], double c[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    double p = myuserdata->p;
    c[0] = pow(x[0],2.0) + p;
    c[1] = x[0] + pow(x[1],2.0);
    c[2] = x[0] - x[1];
    return 0;
}

// compute the Jacobian
int jac( int n, int m, int jne, const double x[], double jval[], 
         const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    jval[0] = 2.0 * x[0];
    jval[1] = 1.0;
    jval[2] = 2.0 * x[1];
    jval[3] = 1.0;
    jval[4] = - 1.0;
    return 0;
}

// compute the Hessian
int hess( int n, int m, int hne, const double x[], const double y[], 
           double hval[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    hval[0] = 2.0 * y[0];
    hval[1] = 2.0 * y[1];
    return 0;
}

// compute Jacobian-vector products
int jacprod( int n, int m, const double x[], const bool transpose, double u[], 
             const double v[], bool got_j, const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    if (transpose) {
      u[0] = u[0] + 2.0 * x[0] * v[0] + v[1] + v[2];
      u[1] = u[1] + 2.0 * x[1] * v[1] - v[2];
    }else{
      u[0] = u[0] + 2.0 * x[0] * v[0];
      u[1] = u[1] + v[0]  + 2.0 * x[1] * v[1];
      u[2] = u[2] + v[0] - v[1];
    }
    return 0;
}

// compute Hessian-vector products
int hessprod( int n, int m, const double x[], const double y[], double u[], 
              const double v[], bool got_h, const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    u[0] = u[0] + 2.0 * y[0] * v[0];
    u[1] = u[1] + 2.0 * y[1] * v[1];
    return 0;
}

// compute residual-Hessians-vector products
int rhessprods( int n, int m, int pne, const double x[], const double v[],
                double pval[], bool got_h, const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    pval[0] = 2.0 * v[0];
    pval[1] = 2.0 * v[1];
    return 0;
}

// scale v
int scale( int n, int m, const double x[], double u[], 
           const double v[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    u[0] = v[0];
    u[1] = v[1];
    return 0;
}

// compute the dense Jacobian
int jac_dense( int n, int m, int jne, const double x[], double jval[], 
               const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    jval[0] = 2.0 * x[0];
    jval[1] = 0.0;
    jval[2] = 1.0;
    jval[3] = 2.0 * x[1];
    jval[4] = 1.0;
    jval[5] = - 1.0;
    return 0;
}

// compute the dense Hessian
int hess_dense( int n, int m, int hne, const double x[], const double y[],
                double hval[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    hval[0] = 2.0 * y[0];
    hval[1] = 0.0;
    hval[2] = 2.0 * y[1];
    return 0;
}

// compute dense residual-Hessians-vector products
int rhessprods_dense( int n, int m, int pne, const double x[], 
                      const double v[], double pval[], bool got_h,
                      const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    pval[0] = 2.0 * v[0];
    pval[1] = 0.0;
    pval[2] = 0.0;
    pval[3] = 2.0 * v[1];
    pval[4] = 0.0;
    pval[5] = 0.0;
    return 0;
}























