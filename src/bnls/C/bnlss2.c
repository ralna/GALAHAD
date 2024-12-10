/* bnlss2.c */
/* Examples for the BNLS C interface using C sparse matrix indexing */
/* Basic tests of models used, access via products */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2024 */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_bnls.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

// Custom userdata struct
struct userdata_type {
   rpc_ p;
};

// Function prototypes

ipc_ res( ipc_ n, ipc_ m, const rpc_ x[], rpc_ c[], const void * );
ipc_ jac( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
         const void * );
ipc_ hess( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
          rpc_ hval[], const void * );
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose,
             rpc_ u[], const rpc_ v[], bool got_j, const void * );
ipc_ hessprod( ipc_ n, ipc_ m, const rpc_ x[], const rpc_ y[],
              rpc_ u[], const rpc_ v[], bool got_h, const void * );
ipc_ rhessprods( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[], const rpc_ v[],
                rpc_ pval[], bool got_h, const void * );
ipc_ scale( ipc_ n, ipc_ m, const rpc_ x[], rpc_ u[],
           const rpc_ v[], const void * );
ipc_ jac_dense( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
               const void * );
ipc_ hess_dense( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
                rpc_ hval[], const void * );
ipc_ rhessprods_dense( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[],
                      const rpc_ v[], rpc_ pval[], bool got_h,
                      const void * );

int main(void) {

    // Derived types
    void *data;
    struct bnls_control_type control;
    struct bnls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 1.0;

    // Set problem data
    ipc_ n = 2; // # variables
    ipc_ m = 3; // # residuals
    ipc_ j_ne = 5; // Jacobian elements
    ipc_ h_ne = 2; // Hesssian elements
    ipc_ p_ne = 2; // residual-Hessians-vector products elements
    ipc_ J_row[] = {0, 1, 1, 2, 2}; // Jacobian J
    ipc_ J_col[] = {0, 0, 1, 0, 1}; //
    ipc_ J_ptr[] = {0, 1, 3, 5};    // row pointers
    ipc_ H_row[] = {0, 1};          // Hessian H
    ipc_ H_col[] = {0, 1};          // NB lower triangle
    ipc_ H_ptr[] = {0, 1, 2};       // row pointers
    ipc_ P_row[] = {0, 1};          // residual-Hessians-vector product matrix
    ipc_ P_ptr[] = {0, 1, 2, 2};    // column pointers

    // Set storage
    rpc_ x[n]; // variables
    rpc_ g[n]; // gradient
    rpc_ c[m]; // residual
    rpc_ y[m]; // multipliers
    rpc_ W[] = {1.0, 1.0, 1.0}; // weights
    char st = ' ';
    ipc_ status;

    printf(" C sparse matrix indexing\n");

    printf("basic tests of models used, access via products\n\n");

    // ============== Gauss-Newton model ==================

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2;
    control.hessian_available = 2;
    control.model = 3;
    x[0] = x[1] = x[2] = 1.5; // starting point

    bnls_import( &control, &data, &status, n, m,
                "absent", j_ne, NULL, NULL, NULL,
                "absent", h_ne, NULL, NULL, NULL,
                "absent", p_ne, NULL, NULL, NULL, W );
    bnls_solve_without_mat( &data, &userdata, &status,
                           n, m, x, c, g, res, jacprod,
                           NULL, p_ne, NULL );

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %" i_ipc_ " Gauss-Newton iterations. Optimal objective value = %5.2f"
               " status = %1" i_ipc_ "\n",
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" BNLS_solve exit status = %1" i_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );

    // ================= Newton model =====================

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2;
    control.hessian_available = 2;
    control.model = 4;
    x[0] = x[1] = x[2] = 1.5; // starting point

    bnls_import( &control, &data, &status, n, m,
                "absent", j_ne, NULL, NULL, NULL,
                "absent", h_ne, NULL, NULL, NULL,
                "absent", p_ne, NULL, NULL, NULL, W );
    bnls_solve_without_mat( &data, &userdata, &status,
                           n, m, x, c, g, res, jacprod,
                           hessprod, p_ne, NULL );

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %" i_ipc_ " Newton iterations. Optimal objective value = %5.2f"
               " status = %1" i_ipc_ "\n",
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" BNLS_solve exit status = %1" i_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );

    // ============== tensor-Newton model =================

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2;
    control.hessian_available = 2;
    control.model = 6;
    x[0] = x[1] = x[2] = 1.5; // starting point

    bnls_import( &control, &data, &status, n, m,
                "absent", j_ne, NULL, NULL, NULL,
                "absent", h_ne, NULL, NULL, NULL,
                "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
    bnls_solve_without_mat( &data, &userdata, &status,
                           n, m, x, c, g, res, jacprod,
                           hessprod, p_ne, rhessprods );

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" %" i_ipc_ " tensor-Newton iterations. Optimal objective value = %5.2f"
               " status = %1" i_ipc_ "\n",
               inform.iter, inform.obj, inform.status);
    }else{
        printf(" BNLS_solve exit status = %1" i_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );
}

// compute the residuals
ipc_ res( ipc_ n, ipc_ m, const rpc_ x[], rpc_ c[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    rpc_ p = myuserdata->p;
    c[0] = pow(x[0],2.0) + p;
    c[1] = x[0] + pow(x[1],2.0);
    c[2] = x[0] - x[1];
    return 0;
}

// compute the Jacobian
ipc_ jac( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
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
ipc_ hess( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
           rpc_ hval[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    hval[0] = 2.0 * y[0];
    hval[1] = 2.0 * y[1];
    return 0;
}

// compute Jacobian-vector products
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose, rpc_ u[],
             const rpc_ v[], bool got_j, const void *userdata ){
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
ipc_ hessprod( ipc_ n, ipc_ m, const rpc_ x[], const rpc_ y[],
              rpc_ u[], const rpc_ v[], bool got_h,
              const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    u[0] = u[0] + 2.0 * y[0] * v[0];
    u[1] = u[1] + 2.0 * y[1] * v[1];
    return 0;
}

// compute residual-Hessians-vector products
ipc_ rhessprods( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[], const rpc_ v[],
                rpc_ pval[], bool got_h, const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    pval[0] = 2.0 * v[0];
    pval[1] = 2.0 * v[1];
    return 0;
}






















