/* snlstf.c */
/* Full test for the SNLS C interface using fortran sparse matrix indexing */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2026 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_snls.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

// Define imax
ipc_ imax(ipc_ a, ipc_ b) {
    return (a > b) ? a : b;
};

// Custom userdata struct
struct userdata_type {
   rpc_ p;
};

// Function prototypes

ipc_ res( ipc_ n, ipc_ m_r, const rpc_ x[], rpc_ r[], const void * );
ipc_ jac( ipc_ n, ipc_ m_r, ipc_ jr_ne, const rpc_ x[], 
          rpc_ jr_val[], const void * );
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose, 
              const rpc_ v[], rpc_ p[], bool got_jr, const void * );
ipc_ jaccol( ipc_ n, ipc_ m_r, const rpc_ x[], ipc_ index,
             rpc_ val[], ipc_ row[], ipc_ nz, bool got_jr, const void * );
ipc_ sjacprod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
               const rpc_ v[], rpc_ p[], const ipc_ free[],
               ipc_ n_free, bool got_jr, const void * );

int main(void) {

    // Derived types
    void *data;
    struct snls_control_type control;
    struct snls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    ipc_ n = 5; // # variables
    ipc_ m_r = 4; // # observations
    ipc_ m_c = 2; // # number of cohorts
    ipc_ cohort[] = {1, 2, 0, 1, 2}; // cohorts
    rpc_ w[] = {1.0, 1.0, 1.0, 1.0}; // weights
    ipc_ jr_ne = 8; // Jacobian elements
    ipc_ Jr_row[] = {1, 1, 2, 2, 3, 3, 4, 4}; // Jacobian J
    ipc_ Jr_col[] = {1, 2, 2, 3, 3, 4, 4, 5};
    rpc_ Jr_val[jr_ne];

    // Set storage
    rpc_ x[n]; // variables
    rpc_ y[m_c]; // multipliers
    rpc_ z[n]; // dual variables
    rpc_ r[m_r]; // residual
    rpc_ g[n]; // gradient
    ipc_ x_stat[n]; // variable status
    ipc_ status;

    printf(" fortran sparse matrix indexing\n\n");

    // solve when Jacobian is available via function calls

    // Initialize SNLS
    snls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing
    // control.print_level = 1;
    control.jacobian_available = 2;
    control.stop_pg_absolute = 0.00001;
    strcpy(control.slls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.slls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    snls_import( &control, &data, &status, n, m_r, m_c,
                 "coordinate", jr_ne, Jr_row, Jr_col, 0, NULL, cohort );
    snls_solve_with_jac( &data, &userdata, &status, n, m_r, m_c, 
                         x, y, z, r, g, x_stat, res, jr_ne, jac, w );

    snls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" SNLS(JF):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" SNLS(JF): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    snls_terminate( &data, &control, &inform );

    // solve when Jacobian products are available via function calls

    // Initialize SNLS
    snls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing
    // control.print_level = 1;
    // control.slls_control.print_level = 1;
    control.jacobian_available = 1;
    control.stop_pg_absolute = 0.00001;
    // control.maxit = 1;
    // control.slls_control.maxit = 5;
    strcpy(control.slls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.slls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    snls_import_without_jac( &control, &data, &status, n, m_r, m_c, cohort );
    snls_solve_with_jacprod( &data, &userdata, &status,
                             n, m_r, m_c, x, y, z, r, g, x_stat, 
                             res, jacprod, jaccol, sjacprod, w );
    snls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" SNLS(PF):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" SNLS(PF): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    snls_terminate( &data, &control, &inform );

    // reverse-communication input/output
    ipc_ mnm, lp;
    mnm = imax( m_r, n );
    lp = 0;
    ipc_ eval_status, lvl, lvu, index;
    ipc_ iv[mnm], ip[m_r];
    rpc_ v[mnm], p[mnm];
    bool got_jr;

    // solve when Jacobian is available via reverse access

    // Initialize SNLS
    snls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing
    //control.print_level = 1;
    control.jacobian_available = 2;
    control.stop_pg_absolute = 0.00001;
    strcpy(control.slls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.slls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    snls_import( &control, &data, &status, n, m_r, m_c,
                "coordinate", jr_ne, Jr_row, Jr_col, 0, NULL, cohort );
    while(true){ // reverse-communication loop
      snls_solve_reverse_with_jac( &data, &status, &eval_status,
                                  n, m_r, m_c, x, y, z, r, g, x_stat, 
                                  jr_ne, Jr_val, w );
      if(status == 0){ // successful termination
            break;
      }else if(status < 0){ // error exit
          break;
      }else if(status == 2){ // evaluate r
          eval_status = res( n, m_r, x, r, &userdata );
      }else if(status == 3){ // evaluate Jr
          eval_status = jac( n, m_r, jr_ne, x, Jr_val, &userdata );
      }else{
          printf(" the value %1" d_ipc_ " of status should not occur\n",
            status);
          break;
      }
    }

    snls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" SNLS(JR):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" SNLS(JR): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    snls_terminate( &data, &control, &inform );

    // solve when Jacobian products are available via reverse access

    // Initialize SNLS
    snls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing
    // control.print_level = 1;
    // control.slls_control.print_level = 1;
    control.jacobian_available = 1;
    control.stop_pg_absolute = 0.00001;
    strcpy(control.slls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.slls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    snls_import_without_jac( &control, &data, &status, n, m_r, m_c, cohort );
    while(true){ // reverse-communication loop
      snls_solve_reverse_with_jacprod( &data, &status, &eval_status,
                                       n, m_r, m_c, x, y, z, r, g, x_stat, v, 
                                       iv, &lvl, &lvu, &index, p, ip, lp, w );
      if(status == 0){ // successful termination
            break;
      }else if(status < 0){ // error exit
          break;
      }else if(status == 2){ // evaluate r
          eval_status = res( n, m_r, x, r, &userdata );
          got_jr = false;
      }else if(status == 4){ // evaluate p = Jr v 
          eval_status = jacprod( n, m_r, x, false, v, p, got_jr, &userdata );
      }else if(status == 5){ // evaluate p = Jr' v
          eval_status = jacprod( n, m_r, x, true, v, p, got_jr, &userdata );
      }else if(status == 6){ // find the index-th column of Jr
          eval_status = jaccol( n, m_r, x, index, p, ip, lp, 
                                got_jr, &userdata );
      }else if(status == 7){ // evaluate p = J_o sparse(v)
          eval_status = sjacprod( n, m_r, x, false, v, p, iv, lvu,
                                  got_jr, &userdata );
      }else if(status == 8){ // evaluate p = sparse(Jr' v)
          eval_status = sjacprod( n, m_r, x, true, v, p, iv, lvu,
                                  got_jr, &userdata );
      }else{
          printf(" the value %1" d_ipc_ " of status should not occur\n",
            status);
          break;
      }
    }

    snls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" SNLS(PR):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    } else {
        printf(" SNLS(PR): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    snls_terminate( &data, &control, &inform );
}

// compute the residuals
ipc_ res( ipc_ n, ipc_ m_r, const rpc_ x[], rpc_ r[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    rpc_ p = myuserdata->p;
    r[0] = x[0] * x[1] - p;
    r[1] = x[1] * x[2] - 1.0;
    r[2] = x[2] * x[3] - 1.0;
    r[3] = x[3] * x[4] - 1.0;
    return 0;
}

// compute the Jacobian
ipc_ jac( ipc_ n, ipc_ m_r, ipc_ jne, const rpc_ x[], rpc_ jr_val[],
         const void *userdata ){
    jr_val[0] = x[1];
    jr_val[1] = x[0];
    jr_val[2] = x[2];
    jr_val[3] = x[1];
    jr_val[4] = x[3];
    jr_val[5] = x[2];
    jr_val[6] = x[4];
    jr_val[7] = x[3];
    return 0;
}

// compute Jacobian-vector products
ipc_ jacprod( ipc_ n, ipc_ m_r, const rpc_ x[], const bool transpose, 
             const rpc_ v[], rpc_ p[], bool got_jr, const void *userdata ){
    if (transpose) {
       p[0] = x[1] * v[0];
       p[1] = x[2] * v[1] + x[0] * v[0];
       p[2] = x[3] * v[2] + x[1] * v[1];
       p[3] = x[4] * v[3] + x[2] * v[2];
       p[4] = x[3] * v[3];
    } else {
       p[0] = x[1] * v[0] + x[0] * v[1];
       p[1] = x[2] * v[1] + x[1] * v[2];
       p[2] = x[3] * v[2] + x[2] * v[3];
       p[3] = x[4] * v[3] + x[3] * v[4];
    }
    got_jr = true;
    return 0;
}

// compute the index-th column of the Jacobian
ipc_ jaccol( ipc_ n, ipc_ m_r, const rpc_ x[], ipc_ index,
             rpc_ val[], ipc_ row[], ipc_ nz, bool got_jr,
             const void *userdata ) {
    if (index == 1){
      val[0] = x[1];
      row[0] = 0;
      nz = 1;
    } else if (index == n) {
      val[0] = x[n-2];
      row[0] = n-1;
      nz = 1;
    } else {
      val[0] = x[index-2];
      row[0] = index-1;
      val[1] = x[index];
      row[1] = index;
      nz = 2;
    }
    got_jr = true;
    return 0;
}


// compute a sparse product with the Jacobian
ipc_ sjacprod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
               const rpc_ v[], rpc_ p[], const ipc_ free[], ipc_ n_free, 
               bool got_jr, const void *userdata ) {
    ipc_ j;
    rpc_ val;
    if (transpose) {
      for( ipc_ i = 0; i < n_free; i++) {
        j = free[i]-1;
        if (j == 0) {
          p[0] = x[1] * v[0];
        } else if (j == n-1) {
          p[n-1] = x[m_r-1] * v[m_r-1];
        } else {
          p[j] = x[j-1] * v[j-1] + x[j+1] * v[j];
        }
      }
    } else {
      for( ipc_ i = 0; i < m_r; i++) p[i] = 0.0;
      for( ipc_ i = 0; i < n_free; i++) {
        j = free[i]-1;
        val = v[j];
        if (j == 0) {
          p[0] = p[0] + x[1] * val;
        } else if (j == n-1) {
          p[m_r-1] = p[m_r-1] + x[m_r-1] * val;
        } else {
          p[j-1] = p[j-1] + x[j-1] * val;
          p[j] = p[j] + x[j+1] * val;
        }
      }
    }
    got_jr = true;
    return 0;
}
