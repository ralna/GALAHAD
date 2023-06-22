/* nlss.c */
/* Examples for the NLS C interface using C sparse matrix indexing */
/* Basic tests of models used, direct access */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2021 */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_nls.h"

#define max(a,b)  \
({  __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

// Custom userdata struct
struct userdata_type {
   real_wp_ p;
};

// Function prototypes

int res( int n, int m, const real_wp_ x[], real_wp_ c[], const void * );
int jac( int n, int m, int jne, const real_wp_ x[], real_wp_ jval[], 
         const void * );
int hess( int n, int m, int hne, const real_wp_ x[], const real_wp_ y[], 
          real_wp_ hval[], const void * );
int jacprod( int n, int m, const real_wp_ x[], const bool transpose,  
             real_wp_ u[], const real_wp_ v[], bool got_j, const void * );
int hessprod( int n, int m, const real_wp_ x[], const real_wp_ y[], 
              real_wp_ u[], const real_wp_ v[], bool got_h, const void * );
int rhessprods( int n, int m, int pne, const real_wp_ x[], const real_wp_ v[],
                real_wp_ pval[], bool got_h, const void * );
int scale( int n, int m, const real_wp_ x[], real_wp_ u[], 
           const real_wp_ v[], const void * );
int jac_dense( int n, int m, int jne, const real_wp_ x[], real_wp_ jval[], 
               const void * );
int hess_dense( int n, int m, int hne, const real_wp_ x[], const real_wp_ y[],
                real_wp_ hval[], const void * );
int rhessprods_dense( int n, int m, int pne, const real_wp_ x[], 
                      const real_wp_ v[], real_wp_ pval[], bool got_h,
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
    real_wp_ x[n]; // variables
    real_wp_ g[n]; // gradient
    real_wp_ c[m]; // residual
    real_wp_ y[m]; // multipliers
    real_wp_ W[] = {1.0, 1.0, 1.0}; // weights
    char st;
    int status;

    printf(" C sparse matrix indexing\n");

    printf("basic tests of models used, direct access\n\n");

    // ============== Gauss-Newton model ==================

    // Initialize NLS
    nls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2; 
    control.hessian_available = 2;
    control.model = 3;
    x[0] = x[1] = x[2] = 1.5; // starting point

    nls_import( &control, &data, &status, n, m, 
                "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                "absent", h_ne, NULL, NULL, NULL,
                "absent", p_ne, NULL, NULL, NULL, W ); 
    nls_solve_with_mat( &data, &userdata, &status,
                        n, m, x, c, g, res, j_ne, jac, 
                        h_ne, NULL, p_ne, NULL );

    nls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %i Gauss-Newton iterations. Optimal objective value = %5.2f"
               " status = %1i\n", 
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" NLS_solve exit status = %1i\n", inform.status);
    }
    // Delete internal workspace
    nls_terminate( &data, &control, &inform );

    // ================= Newton model =====================

    // Initialize NLS
    nls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2; 
    control.hessian_available = 2;
    control.model = 4;
    x[0] = x[1] = x[2] = 1.5; // starting point

    nls_import( &control, &data, &status, n, m, 
                "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                "absent", p_ne, NULL, NULL, NULL, W ); 
    nls_solve_with_mat( &data, &userdata, &status,
                        n, m, x, c, g, res, j_ne, jac, 
                        h_ne, hess, p_ne, NULL );

    nls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %i Newton iterations. Optimal objective value = %5.2f"
               " status = %1i\n", 
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" NLS_solve exit status = %1i\n", inform.status);
    }
    // Delete internal workspace
    nls_terminate( &data, &control, &inform );

    // ============== tensor-Newton model =================

    // Initialize NLS
    nls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2; 
    control.hessian_available = 2;
    control.model = 6;
    x[0] = x[1] = x[2] = 1.5; // starting point

    nls_import( &control, &data, &status, n, m, 
                "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
    nls_solve_with_mat( &data, &userdata, &status,
                        n, m, x, c, g, res, j_ne, jac, 
                        h_ne, hess, p_ne, rhessprods );

    nls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %i tensor-Newton iterations. Optimal objective value = %5.2f"
               " status = %1i\n", 
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" NLS_solve exit status = %1i\n", inform.status);
    }
    // Delete internal workspace
    nls_terminate( &data, &control, &inform );
}

// compute the residuals
int res( int n, int m, const real_wp_ x[], real_wp_ c[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    real_wp_ p = myuserdata->p;
    c[0] = pow(x[0],2.0) + p;
    c[1] = x[0] + pow(x[1],2.0);
    c[2] = x[0] - x[1];
    return 0;
}

// compute the Jacobian
int jac( int n, int m, int jne, const real_wp_ x[], real_wp_ jval[], 
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
int hess( int n, int m, int hne, const real_wp_ x[], const real_wp_ y[], 
           real_wp_ hval[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    hval[0] = 2.0 * y[0];
    hval[1] = 2.0 * y[1];
    return 0;
}

// compute Jacobian-vector products
int jacprod( int n, int m, const real_wp_ x[], const bool transpose, 
             real_wp_ u[], const real_wp_ v[], bool got_j, 
             const void *userdata ){
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
int hessprod( int n, int m, const real_wp_ x[], const real_wp_ y[], 
              real_wp_ u[], const real_wp_ v[], bool got_h, 
              const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    u[0] = u[0] + 2.0 * y[0] * v[0];
    u[1] = u[1] + 2.0 * y[1] * v[1];
    return 0;
}

// compute residual-Hessians-vector products
int rhessprods( int n, int m, int pne, const real_wp_ x[], const real_wp_ v[],
                real_wp_ pval[], bool got_h, const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    pval[0] = 2.0 * v[0];
    pval[1] = 2.0 * v[1];
    return 0;
}






















