/* bnlstf.c */
/* Full test for the BNLS C interface using C sparse matrix indexing */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2026 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bnls.h"
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
   ipc_ *flag;
   ipc_ *flags;
};

// Function prototypes

ipc_ res( ipc_ n, ipc_ m_r, const rpc_ x[], rpc_ r[], const void * );
ipc_ jac( ipc_ n, ipc_ m_r, ipc_ jr_ne, const rpc_ x[], 
          rpc_ jr_val[], const void * );
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose, 
              const rpc_ v[], rpc_ p[], bool got_jr, const void * );
ipc_ jacprods( ipc_ n, ipc_ m_r, const rpc_ x[], const rpc_ v[], 
               rpc_ p[], const ipc_ iv[], ipc_ lvl, ipc_ lvu, 
               ipc_ ip[], ipc_ *lp, bool got_jr, const void *userdata );
ipc_ sjacprod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
               const rpc_ v[], rpc_ p[], const ipc_ free[],
               ipc_ n_free, bool got_jr, const void * );

int main(void) {

    // Derived types
    void *data;
    struct bnls_control_type control;
    struct bnls_inform_type inform;

    // Set problem data
    ipc_ n = 5; // # variables
    ipc_ m_r = 4; // # observations
    rpc_ w[] = {1.0, 1.0, 1.0, 1.0}; // weights
    ipc_ jr_ne = 8; // Jacobian elements
    ipc_ Jr_row[] = {0, 0, 1, 1, 2, 2, 3, 3}; // Jacobian J
    ipc_ Jr_col[] = {0, 1, 1, 2, 2, 3, 3, 4};
    rpc_ Jr_val[jr_ne];

    // Set storage
    rpc_ x_l[n]; // lower bounds
    rpc_ x_u[n]; // upper bounds
    rpc_ x[n]; // variables
    rpc_ z[n]; // dual variables
    rpc_ r[m_r]; // residual
    rpc_ g[n]; // gradient
    ipc_ x_stat[n]; // variable status
    ipc_ status;

    // set variable bounds
    for( ipc_ i = 0; i < n; i++) x_l[i] = 0.0; // lower bounds
    for( ipc_ i = 0; i < n; i++) x_u[i] = 1.0; // upper bounds

    // set up array to flag current nonzeros in a  vector
    ipc_ flag = 0; // current flag value
    ipc_ flags[m_r]; // array of flags
    for( ipc_ i = 0; i < m_r; i++) flags[i] = 0;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;
    userdata.flag = &flag;
    userdata.flags = flags;

    printf(" C sparse matrix indexing\n\n");
//if(false){
    // solve when Jacobian is available via function calls

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    //control.print_level = 1;
    //control.blls_control.print_level = 1;
    control.jacobian_available = 2;
#ifdef REAL_32
    control.stop_pg_absolute = 0.0001;
#else
    control.stop_pg_absolute = 0.00001;
#endif
    strcpy(control.blls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.blls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    bnls_import( &control, &data, &status, n, m_r, 
                 "coordinate", jr_ne, Jr_row, Jr_col, 0, NULL );
    bnls_solve_with_jac( &data, &userdata, &status, n, m_r, x_l, x_u, 
                         x, z, r, g, x_stat, res, jr_ne, jac, w );

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" BNLS(JF):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" BNLS(JF): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );
//}
    // solve when Jacobian products are available via function calls

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // fortran sparse matrix indexing
    //control.print_level = 1;
    //control.blls_control.print_level = 1;
    control.jacobian_available = 1;
#ifdef REAL_32
    control.stop_pg_absolute = 0.005;
#else
    control.stop_pg_absolute = 0.00001;
#endif
    // control.maxit = 10;
    // control.blls_control.maxit = 10;
    // control.blls_control.maxit = 5;
    strcpy(control.blls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.blls_control.sbls_control.symmetric_linear_solver, "sytr ");

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    bnls_import_without_jac( &control, &data, &status, n, m_r );
    bnls_solve_with_jacprod( &data, &userdata, &status,
                             n, m_r, x_l, x_u, x, z, r, g, x_stat, 
                             res, jacprod, jacprods, sjacprod, w );
    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" BNLS(PF):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" BNLS(PF): exit status = %1" d_ipc_ "\n", inform.status);
    }

    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );
//if(false){
    // reverse-communication input/output
    ipc_ mnm, lp;
    mnm = imax( m_r, n );
    lp = 0;
    ipc_ eval_status, lvl, lvu;
    ipc_ iv[mnm], ip[mnm];
    rpc_ v[mnm], p[mnm];
    bool got_jr;

    // solve when Jacobian is available via reverse access

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // fortran sparse matrix indexing
    //control.print_level = 1;
    //control.blls_control.print_level = 1;
    control.jacobian_available = 2;
#ifdef REAL_32
    control.stop_pg_absolute = 0.0001;
#else
    control.stop_pg_absolute = 0.00001;
#endif
    strcpy(control.blls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.blls_control.sbls_control.symmetric_linear_solver, "sytr ");
    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    bnls_import( &control, &data, &status, n, m_r, 
                "coordinate", jr_ne, Jr_row, Jr_col, 0, NULL );
    while(true){ // reverse-communication loop
      bnls_solve_reverse_with_jac( &data, &status, &eval_status,
                                  n, m_r, x_l, x_u, x, z, r, g, x_stat, 
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

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" BNLS(JR):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    }else{
        printf(" BNLS(JR): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );

    // solve when Jacobian products are available via reverse access

    // Initialize BNLS
    bnls_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // fortran sparse matrix indexing
    //control.print_level = 1;
    //control.blls_control.print_level = 1;
    control.jacobian_available = 1;
#ifdef REAL_32
    control.stop_pg_absolute = 0.0001;
#else
    control.stop_pg_absolute = 0.00001;
#endif
    strcpy(control.blls_control.sbls_control.definite_linear_solver, "potr ");
    strcpy(control.blls_control.sbls_control.symmetric_linear_solver, "sytr ");

lp = mnm;
for( ipc_ i = 0; i < mnm; i++) ip[i] =i;

    for( ipc_ i = 0; i < n; i++) x[i] = 0.5; // starting point
    bnls_import_without_jac( &control, &data, &status, n, m_r );
    while(true){ // reverse-communication loop
//printf(" bnlst status in  = %1" d_ipc_ "\n", status);
      bnls_solve_reverse_with_jacprod( &data, &status, &eval_status,
                                       n, m_r, x_l, x_u, x, z, r, g, x_stat,
                                       v, iv, &lvl, &lvu, p, ip, lp, w );
//printf(" bnlst status out = %1" d_ipc_ "\n", status);
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
      }else if(status == 6){ // evaluate p = Jr * sparse v
          eval_status = jacprods( n, m_r, x, v, p, iv, lvl, lvu, NULL, NULL,
                                  got_jr, &userdata );
      }else if(status == 7){ // evaluate p = sparse( Jr(x) * sparse v )
          eval_status = jacprods( n, m_r, x, v, p, iv, lvl, lvu, ip, &lp,
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

    bnls_information( &data, &inform, &status );

    if(inform.status == 0){
        printf(" BNLS(PR):%6" d_ipc_ " iterations. Optimal objective value"
               " = %5.2f status = %1" d_ipc_ "\n",
               inform.iter, (double)inform.obj, inform.status);
    } else {
        printf(" BNLS(PR): exit status = %1" d_ipc_ "\n", inform.status);
    }
    // Delete internal workspace
    bnls_terminate( &data, &control, &inform );

//}

    printf(" BNLS tests complete\n");
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

// compute a sparse product with the Jacobian
ipc_ jacprods( ipc_ n, ipc_ m_r, const rpc_ x[], const rpc_ v[], 
               rpc_ p[], const ipc_ iv[], ipc_ lvl, ipc_ lvu, 
               ipc_ ip[], ipc_ *lp, bool got_jr,
               const void *userdata ) {
    ipc_ i, j;
    rpc_ val;
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    ipc_ flag = *(myuserdata->flag);
    ipc_ *flags = myuserdata->flags;
//printf(" bnlst  lvl, lvu = %1" d_ipc_ " %1" d_ipc_ "\n", lvl, lvu);
    if (ip != NULL && lp != NULL) {
      flag = flag+1;
//printf(" bnlst  flag = %1" d_ipc_ "\n", flag);
      *lp = 0;
      for( ipc_ l = lvl; l <= lvu; l++){
        j = iv[l];
//printf(" bnlst  j = %1" d_ipc_ "\n", j);
        val = v[j];
        if (j == 0){
          i = 0;
//printf(" bnlst  flags[%1" d_ipc_ "] = %1" d_ipc_ "\n", i,flags[i]);
          if (flags[i] < flag) {
            flags[i] = flag;
            p[i] = x[i+1] * val;
            ip[*lp] = i;
            *lp = *lp + 1;
          } else {
            p[i] = p[i] + x[i+1] * val;
          }
        } else if (j == n-1) {
          i = m_r-1;
          if (flags[i] < flag) {
            flags[i] = flag;
            p[i] = x[i] * val;
            ip[*lp] = i;
            *lp = *lp + 1;
          } else {
            p[i] = p[i] + x[i] * val;
          }
        } else {
          i = j - 1;
          if (flags[i] < flag) {
            flags[i] = flag;
            p[i] = x[i] * val;
            ip[*lp] = i;
            *lp = *lp + 1;
          } else {
            p[i] = p[i] + x[i] * val;
          }
          i = j;
          if (flags[i] < flag) {
            flags[i] = flag;
            p[i] = x[i+1] * val;
            ip[*lp] = i;
            *lp = *lp + 1;
          } else {
            p[i] = p[i] + x[i+1] * val;
          }
        }
//printf(" bnlst  lp = %1" d_ipc_ "\n", *lp);
      for( ipc_ i = 0; i < *lp; i++) flags[ip[i]] = 0;
      }
    } else {
      for( ipc_ i = 0; i < m_r; i++) p[i] = 0.0;
      for( ipc_ l = lvl; l <= lvu; l++){
        j = iv[l];
        val = v[j];
        if (j == 0) {
          i = 0;
          p[i] = p[i] + x[i+1] * val;
        } else if (j == n-1) {
          i = m_r - 1;
          p[i] = p[i] + x[i] * val;
        } else {
          i = j - 1;
          p[i] = p[i] + x[i] * val;
          i = j;
          p[i] = p[i] + x[i+1] * val;
        }
      }
    }
    got_jr = true;
    return 0;
}

// compute a sparse product with the Jacobian or its transpose
ipc_ sjacprod( ipc_ n, ipc_ m_r, const rpc_ x[], bool transpose,
               const rpc_ v[], rpc_ p[], const ipc_ free[], ipc_ n_free, 
               bool got_jr, const void *userdata ) {
    ipc_ j;
    rpc_ val;
    if (transpose) {
      for( ipc_ i = 0; i < n_free; i++) {
        j = free[i];
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
        j = free[i];
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

