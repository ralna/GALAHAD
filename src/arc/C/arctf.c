/* arctf.c */
/* Full test for the ARC C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_arc.h"

// Custom userdata struct
struct userdata_type {
   real_wp_ p;
};

// Function prototypes
int fun( int n, const real_wp_ x[], real_wp_ *f, const void * );
int grad( int n, const real_wp_ x[], real_wp_ g[], const void * );
int hess( int n, int ne, const real_wp_ x[], real_wp_ hval[], const void * );
int hess_dense( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
                const void * );
int hessprod( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
              bool got_h, const void * );
int prec( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
          const void * );
int fun_diag( int n, const real_wp_ x[], real_wp_ *f, const void * );
int grad_diag( int n, const real_wp_ x[], real_wp_ g[], const void * );
int hess_diag( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
               const void * );
int hessprod_diag( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
                   bool got_h, const void * );

int main(void) {

    // Derived types
    void *data;
    struct arc_control_type control;
    struct arc_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    int H_row[] = {1, 2, 3, 3, 3}; // Hessian H
    int H_col[] = {1, 2, 1, 2, 3}; // NB lower triangle
    int H_ptr[] = {1, 2, 3, 6};    // row pointers

    // Set storage
    real_wp_ g[n]; // gradient
    char st;
    int status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" tests options for all-in-one storage format\n\n");

    for( int d=1; d <= 5; d++){

        // Initialize ARC
        arc_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        //control.print_level = 1;

        // Start from 1.5
        real_wp_ x[] = {1.5,1.5,1.5}; 

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                arc_import( &control, &data, &status, n, "coordinate", 
                            ne, H_row, H_col, NULL );
                arc_solve_with_mat( &data, &userdata, &status,
                                    n, x, g, ne, fun, grad, hess, prec );
                break;
            case 2: // sparse by rows  
                st = 'R';
                arc_import( &control, &data, &status, n, "sparse_by_rows", 
                           ne, NULL, H_col, H_ptr);
                arc_solve_with_mat( &data, &userdata, &status,
                                    n, x, g, ne, fun, grad, hess, prec );
                break;
            case 3: // dense
                st = 'D';
                arc_import( &control, &data, &status, n, "dense", 
                           ne, NULL, NULL, NULL );
                arc_solve_with_mat( &data, &userdata, &status,
                                    n, x, g, ne, fun, grad, hess_dense, prec );
                break;
            case 4: // diagonal
                st = 'I';
                arc_import( &control, &data, &status, n, "diagonal", 
                           ne, NULL, NULL, NULL );
                arc_solve_with_mat( &data, &userdata, &status, n, x, g, 
                                    ne, fun_diag, grad_diag, hess_diag, prec) ;
                break;
            case 5: // access by products
                st = 'P';
                arc_import( &control, &data, &status, n, "absent", 
                           ne, NULL, NULL, NULL );
                arc_solve_without_mat( &data, &userdata, &status,
                                       n, x, g, fun, grad, hessprod, prec );
                break;
        }
        arc_information( &data, &inform, &status);

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", 
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: ARC_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        arc_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    int eval_status;
    real_wp_ f = 0.0;
    real_wp_ u[n], v[n];
    int index_nz_u[n], index_nz_v[n];
    real_wp_ H_val[ne], H_dense[n*(n+1)/2], H_diag[n];
 
    for( int d=1; d <= 5; d++){

        // Initialize ARC
        arc_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;

        // Start from 1.5
        real_wp_ x[] = {1.5,1.5,1.5}; 

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                arc_import( &control, &data, &status, n, "coordinate", 
                           ne, H_row, H_col, NULL );
                while(true){ // reverse-communication loop
                    arc_solve_reverse_with_mat( &data, &status, &eval_status, 
                                                n, x, f, g, ne, H_val, u, v );
                    if(status == 0){ // successful termination
                        break;
                    }else if(status < 0){ // error exit
                        break;
                    }else if(status == 2){ // evaluate f
                        eval_status = fun( n, x, &f, &userdata );
                    }else if(status == 3){ // evaluate g
                        eval_status = grad( n, x, g, &userdata );
                    }else if(status == 4){ // evaluate H
                        eval_status = hess( n, ne, x, H_val, &userdata ); 
                    }else if(status == 6){ // evaluate the product with P
                        eval_status = prec( n, x, u, v, &userdata );
                    }else{
                        printf(" the value %1i of status should not occur\n", 
                          status);
                        break;
                    }
                }
                break;
            case 2: // sparse by rows  
                st = 'R';
                arc_import( &control, &data, &status, n, "sparse_by_rows", ne, 
                           NULL, H_col, H_ptr);
                while(true){ // reverse-communication loop
                    arc_solve_reverse_with_mat( &data, &status, &eval_status, 
                                                n, x, f, g, ne, H_val, u, v );
                    if(status == 0){ // successful termination
                        break;
                    }else if(status < 0){ // error exit
                        break;
                    }else if(status == 2){ // evaluate f
                        eval_status = fun( n, x, &f, &userdata );
                    }else if(status == 3){ // evaluate g
                        eval_status = grad( n, x, g, &userdata );
                    }else if(status == 4){ // evaluate H
                        eval_status = hess( n, ne, x, H_val, &userdata ); 
                    }else if(status == 6){ // evaluate the product with P
                        eval_status = prec( n, x, u, v, &userdata );
                    }else{
                        printf(" the value %1i of status should not occur\n", 
                          status);
                        break;
                    }
                }
                break;
            case 3: // dense
                st = 'D';
                arc_import( &control, &data, &status, n, "dense", 
                           ne, NULL, NULL, NULL );
                while(true){ // reverse-communication loop
                    arc_solve_reverse_with_mat( &data, &status, &eval_status, 
                                         n, x, f, g, n*(n+1)/2, H_dense, u, v );
                    if(status == 0){ // successful termination
                        break;
                    }else if(status < 0){ // error exit
                        break;
                    }else if(status == 2){ // evaluate f
                        eval_status = fun( n, x, &f, &userdata );
                    }else if(status == 3){ // evaluate g
                        eval_status = grad( n, x, g, &userdata );
                    }else if(status == 4){ // evaluate H
                        eval_status = hess_dense( n, n*(n+1)/2, x, H_dense, 
                                                 &userdata ); 
                    }else if(status == 6){ // evaluate the product with P
                        eval_status = prec( n, x, u, v, &userdata );
                    }else{
                        printf(" the value %1i of status should not occur\n", 
                          status);
                        break;
                    }
                }
                break;
            case 4: // diagonal
                st = 'I';
                arc_import( &control, &data, &status, n, "diagonal", 
                           ne, NULL, NULL, NULL );
                while(true){ // reverse-communication loop
                    arc_solve_reverse_with_mat( &data, &status, &eval_status, 
                                                n, x, f, g, n, H_diag, u, v );
                    if(status == 0){ // successful termination
                        break;
                    }else if(status < 0){ // error exit
                        break;
                    }else if(status == 2){ // evaluate f
                        eval_status = fun_diag( n, x, &f, &userdata );
                    }else if(status == 3){ // evaluate g
                        eval_status = grad_diag( n, x, g, &userdata );
                    }else if(status == 4){ // evaluate H
                        eval_status = hess_diag( n, n, x, H_diag, &userdata ); 
                    }else if(status == 6){ // evaluate the product with P
                        eval_status = prec( n, x, u, v, &userdata );
                    }else{
                        printf(" the value %1i of status should not occur\n", 
                          status);
                        break;
                    }
                }
                break;
            case 5: // access by products
                st = 'P';
                arc_import( &control, &data, &status, n, "absent", 
                           ne, NULL, NULL, NULL );
                while(true){ // reverse-communication loop
                    arc_solve_reverse_without_mat( &data, &status, &eval_status,
                                                   n, x, f, g, u, v );
                    if(status == 0){ // successful termination
                        break;
                    }else if(status < 0){ // error exit
                        break;
                    }else if(status == 2){ // evaluate f
                        eval_status = fun( n, x, &f, &userdata );
                    }else if(status == 3){ // evaluate g
                        eval_status = grad( n, x, g, &userdata );
                    }else if(status == 5){ // evaluate H
                        eval_status = hessprod( n, x, u, v, false, &userdata );
                    }else if(status == 6){ // evaluate the product with P
                        eval_status = prec( n, x, u, v, &userdata );
                    }else{
                        printf(" the value %1i of status should not occur\n", 
                          status);
                        break;
                    }
                }
                break;
        }
        arc_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", 
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: ARC_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        arc_terminate( &data, &control, &inform );
    }

}

// Objective function 
int fun( int n, const real_wp_ x[], real_wp_ *f, const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
    return 0;
}

// Gradient of the objective
int grad( int n, const real_wp_ x[], real_wp_ g[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - sin(x[0]);
    g[1] = 2.0 * ( x[1] + x[2] );
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] );
    return 0;
}

// Hessian of the objective
int hess( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
         const void *userdata ){
    hval[0] = 2.0 - cos(x[0]);
    hval[1] = 2.0;
    hval[2] = 2.0;
    hval[3] = 2.0;
    hval[4] = 4.0;
    return 0;
}

// Dense Hessian
int hess_dense( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
                const void *userdata ){ 
    hval[0] = 2.0 - cos(x[0]);
    hval[1] = 0.0;
    hval[2] = 2.0;
    hval[3] = 2.0;
    hval[4] = 2.0;
    hval[5] = 4.0;
    return 0;
}

// Hessian-vector product
int hessprod( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
              bool got_h, const void *userdata ){
    u[0] = u[0] + 2.0 * ( v[0] + v[2] ) - cos(x[0]) * v[0];
    u[1] = u[1] + 2.0 * ( v[1] + v[2] );
    u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
    return 0;
}

// Apply preconditioner
int prec( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
          const void *userdata ){
   u[0] = 0.5 * v[0];
   u[1] = 0.5 * v[1];
   u[2] = 0.25 * v[2];
   return 0;
}

 // Objective function 
int fun_diag( int n, const real_wp_ x[], real_wp_ *f, const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    *f = pow(x[2] + p, 2) + pow(x[1], 2) + cos(x[0]);
    return 0;
}

// Gradient of the objective
int grad_diag( int n, const real_wp_ x[], real_wp_ g[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    g[0] = -sin(x[0]);
    g[1] = 2.0 * x[1];
    g[2] = 2.0 * ( x[2] + p );
    return 0;
}

// Hessian of the objective
int hess_diag( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
               const void *userdata ){
    hval[0] = -cos(x[0]);
    hval[1] = 2.0;
    hval[2] = 2.0;
    return 0;
}  

// Hessian-vector product
int hessprod_diag( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[], 
                   bool got_h, const void *userdata ){
    u[0] = u[0] + - cos(x[0]) * v[0];
    u[1] = u[1] + 2.0 * v[1];
    u[2] = u[2] + 2.0 * v[2];
    return 0;
}
