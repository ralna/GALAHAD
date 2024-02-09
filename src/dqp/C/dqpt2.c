/* dqpt.c */
/* Full test for the DQP C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_dqp.h"

int main(void) {

    // Derived types
    void *data;
    struct dqp_control_type control;
    struct dqp_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ m = 2; // number of general constraints
    ipc_ H_ne = 4; // Hesssian elements
    ipc_ H_row[] = {0, 1, 2, 2 };   // row indices, NB lower triangle
    ipc_ H_col[] = {0, 1, 1, 2};    // column indices, NB lower triangle
    real_wp_ H_val[] = {1.0, 2.0, 1.0, 3.0 };   // values
    real_wp_ g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    real_wp_ f = 1.0;  // constant term in the objective
    ipc_ A_ne = 4; // Jacobian elements
    ipc_ A_row[] = {0, 0, 1, 1}; // row indices
    ipc_ A_col[] = {0, 1, 1, 2}; // column indices
    real_wp_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
    real_wp_ c_l[] = {1.0, 2.0};   // constraint lower bound
    real_wp_ c_u[] = {2.0, 2.0};   // constraint upper bound
    real_wp_ x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    real_wp_ x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound

    // Set output storage
    real_wp_ c[m]; // constraint values
    ipc_ x_stat[n]; // variable status
    ipc_ c_stat[m]; // constraint status
    char st;
    ipc_ status;

    printf(" C sparse matrix indexing\n\n");

    // Initialize DQP
    dqp_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    strcpy(control.symmetric_linear_solver, "sytr ") ;
    strcpy(control.definite_linear_solver, "sytr ") ;

    // Start from 0
    real_wp_ x[] = {0.0,0.0,0.0};
    real_wp_ y[] = {0.0,0.0};
    real_wp_ z[] = {0.0,0.0,0.0};

    st = 'C';
    dqp_import( &control, &data, &status, n, m,
               "coordinate", H_ne, H_row, H_col, NULL,
               "coordinate", A_ne, A_row, A_col, NULL );
    dqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                  x_stat, c_stat );
    dqp_information( &data, &inform, &status );
    printf("status %i\n", inform.status);
    printf("alloc_status %i\n", inform.alloc_status);
    printf("bad_alloc %s\n", inform.bad_alloc);
    printf("fdc status %i\n", inform.fdc_inform.status);
    printf("fdc pivot %f\n", inform.fdc_inform.non_negligible_pivot);
    printf("fdc sls status %i\n", inform.fdc_inform.sls_inform.status);
    printf("fdc sls nodes %i\n", inform.fdc_inform.sls_inform.nodes_assembly_tree);
    printf("fdc sls flops %li\n", inform.fdc_inform.sls_inform.flops_blas);
    printf("fdc sls alt %d\n", inform.fdc_inform.sls_inform.alternative);

    printf("fdc sls ma97 %i\n", inform.fdc_inform.sls_inform.ma97_info.flag);

//    printf("fdc sls ssids %i\n", inform.fdc_inform.sls_inform.ssids_inform.flag);
      printf("fdc sls mc61(0) %i\n", inform.fdc_inform.sls_inform.mc61_info[0]);
      printf("fdc sls mc61(1) %i\n", inform.fdc_inform.sls_inform.mc61_info[1]);
      printf("fdc sls mc64 %i\n", inform.fdc_inform.sls_inform.mc64_info.flag);
//    printf("fdc sls mc64 %i\n", inform.fdc_inform.sls_inform.mc64_info.more);
//    printf("fdc sls mc64 %i\n", inform.fdc_inform.sls_inform.mc64_info.strucrank);
//    printf("fdc sls mc64 %i\n", inform.fdc_inform.sls_inform.mc64_info.stat);
      printf("fdc sls mc68 %i\n", inform.fdc_inform.sls_inform.mc68_info.flag);

    printf("fdc sls mumps %i\n", inform.fdc_inform.sls_inform.mumps_error);
//    printf("fdc sls pard %i\n", inform.fdc_inform.sls_inform.pardiso_error);
    printf("fdc sls wsmp %i\n", inform.fdc_inform.sls_inform.wsmp_error);
    printf("fdc sls pastix %i\n", inform.fdc_inform.sls_inform.pastix_info);
    printf("fdc sls mpi %i\n", inform.fdc_inform.sls_inform.mpi_ierr);
    printf("fdc sls lapack %i\n", inform.fdc_inform.sls_inform.lapack_error);





    printf("fdc uls status %i\n", inform.fdc_inform.uls_inform.status);




    printf("sls status %i\n", inform.sls_inform.status);
    printf("sbls status %i\n", inform.sbls_inform.status);
    printf("sbls alloc_status %i\n", inform.sbls_inform.alloc_status);
    printf("sbls bad_alloc %s\n", inform.sbls_inform.bad_alloc);

    if(inform.status == 0){
        printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
               st, inform.iter, inform.obj, inform.status);
    }else{
        printf("%c: DQP_solve exit status = %1i\n", st, inform.status);
    }
    //printf("x: ");
    //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
    //printf("\n");
    //printf("gradient: ");
    //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
    //printf("\n");

    // Delete internal workspace
    dqp_terminate( &data, &control, &inform );
}

