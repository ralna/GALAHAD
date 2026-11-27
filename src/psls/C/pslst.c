/* pslst.c */
/* Full test for the PSLS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_psls.h"

int main(void) {

    // Derived types
    void *data;
    struct psls_control_type control;
    struct psls_inform_type inform;

    // Set problem data
    ipc_ n = 5; // dimension of A
    ipc_ ne = 7; // number of elements of A
    ipc_ dense_ne = n * ( n + 1 ) / 2; // number of elements of dense A

    ipc_ row[] = {0, 1, 1, 2, 2, 3, 4}; // A indices & values, NB lower triangle
    ipc_ col[] = {0, 0, 4, 1, 2, 2, 4};
    ipc_ ptr[] = {0, 1, 3, 5, 6, 7};
    rpc_ val[] = {2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0};
    rpc_ dense[] = {2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                      0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0};
    char st = ' ';
    ipc_ status;
    ipc_ status_apply;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ d=1; d <= 3; d++){

        // Initialize PSLS
        psls_initialize( &data, &control, &status );
        control.preconditioner = 2; // band preconditioner
        control.semi_bandwidth = 1; // semibandwidth
        strcpy( control.symmetric_linear_solver, "sils" );
        strcpy( control.definite_linear_solver, "sils" );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                psls_import( &control, &data, &status, n,
                             "coordinate", ne, row, col, NULL );
                psls_form_preconditioner( &data, &status, ne, val );
                break;
            printf(" case %1" d_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                psls_import( &control, &data, &status, n,
                             "sparse_by_rows", ne, NULL, col, ptr );
                psls_form_preconditioner( &data, &status, ne, val );
                break;
            case 3: // dense
                st = 'D';
                psls_import( &control, &data, &status, n,
                             "dense", ne, NULL, NULL, NULL );
                psls_form_preconditioner( &data, &status, dense_ne, dense );
                break;
            }

        // Set right-hand side b in x
        rpc_ x[] = {8.0, 45.0, 31.0, 15.0, 17.0};   // values

        if(status == 0){
          psls_information( &data, &inform, &status );
          psls_apply_preconditioner( &data, &status_apply, n, x );
        }else{
          status_apply = - 1;
        }

        printf("%c storage: status from form & factorize = %" d_ipc_ 
               " apply = %" d_ipc_ "\n", st, status, status_apply );

        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);

        // Delete internal workspace
        psls_terminate( &data, &control, &inform );
    }

    // subset preconditioner: the diagonal of the rows (and columns) 0, 2
    // and 4 of A (indices from 0), which are 2, 1 and 1, and then with
    // row 0 removed; components outside the subset are not altered, while
    // those removed are set to zero

    ipc_ failures = 0;
    ipc_ n_sub = 3;
    ipc_ sub[] = {0, 2, 4};
    ipc_ fix[] = {0};
    rpc_ x_sub[] = {4.0, 45.0, 31.0, 15.0, 17.0};
    rpc_ x_fix[] = {0.0, 45.0, 31.0, 15.0, 17.0};
    psls_initialize( &data, &control, &status );
    control.preconditioner = 1; // diagonal preconditioner
    control.f_indexing = false;
    psls_import( &control, &data, &status, n,
                 "coordinate", ne, row, col, NULL );
    psls_form_subset_preconditioner( &data, &status, ne, val, n_sub, sub );
    rpc_ x[] = {8.0, 45.0, 31.0, 15.0, 17.0};
    if(status == 0) psls_apply_preconditioner( &data, &status_apply, n, x );
    if(status != 0 || status_apply != 0){
        printf(" subset: status = %" d_ipc_ ", apply = %" d_ipc_ "\n",
               status, status_apply);
        failures++;
    }else{
        for( ipc_ i = 0; i < n; i++) if(fabs(x[i] - x_sub[i]) > 1.0e-12){
            printf(" subset: x[%" d_ipc_ "] = %f, not %f\n", i, x[i], x_sub[i]);
            failures++;
        }
    }
    psls_update_preconditioner( &data, &status, 1, fix );
    rpc_ y[] = {8.0, 45.0, 31.0, 15.0, 17.0};
    if(status == 0) psls_apply_preconditioner( &data, &status_apply, n, y );
    if(status != 0 || status_apply != 0){
        printf(" update: status = %" d_ipc_ ", apply = %" d_ipc_ "\n",
               status, status_apply);
        failures++;
    }else{
        for( ipc_ i = 0; i < n; i++) if(fabs(y[i] - x_fix[i]) > 1.0e-12){
            printf(" update: x[%" d_ipc_ "] = %f, not %f\n", i, y[i], x_fix[i]);
            failures++;
        }
    }
    psls_terminate( &data, &control, &inform );
    if(failures > 0){
        printf(" %" d_ipc_ " subset check(s) failed\n", failures);
        return 1;
    }
    printf(" subset checks ok\n");
    return 0;
}
