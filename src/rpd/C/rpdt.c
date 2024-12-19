/* rpdt.c */
/* Full test for the RPD C interface using C sparse matrix indexing */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_rpd.h"
#ifdef REAL_128
#include <quadmath.h>
#endif
#define BUFSIZE 1000

int main(void) {

    // Derived types
    void *data;
    struct rpd_control_type control;
    struct rpd_inform_type inform;

    char qplib_file[BUFSIZE];
    char *galahad = "GALAHAD";
    ipc_ qplib_file_len;
#ifdef REAL_128
//    char buf0[128], buf1[128], buf2[128], buf3[128], buf4[128];
//    int n0, n1, n2, n3, n4;
#endif

    // make sure the GALAHAD environment variable actually exists
    if(!getenv(galahad)){
      fprintf(stderr, " The environment variable %s was not found.\n", galahad);
      exit( 1 );
    }

    // make sure the buffer is large enough to hold the environment variable
    // value, and if so, copy it into qplib_file
    if( snprintf( qplib_file, BUFSIZE, "%s", getenv(galahad) ) >= BUFSIZE){
      fprintf (stderr, " BUFSIZE of %d was too small. Aborting\n", BUFSIZE );
      exit( 1 );
    }
    // extend the qplib_file string to include the actual position of the
    // provided ALLINIT.qplib example file provided as part GALAHAD
    char source[] = "/examples/ALLINIT.qplib";
    strcat( qplib_file, source );
    // compute the length of the string
    qplib_file_len = strlen( qplib_file );

    printf( " QPLIB file: %s\n", qplib_file );

    ipc_ status;
    ipc_ n;
    ipc_ m;
    ipc_ h_ne;
    ipc_ a_ne;
    ipc_ h_c_ne;
    char p_type[4];

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    // Initialize RPD */
    rpd_initialize( &data, &control, &status );

    // Set user-defined control options */
    control.f_indexing = false; // C sparse matrix indexing

    // Recover vital statistics from the QPLIB file
    rpd_get_stats( qplib_file, qplib_file_len, &control, &data, &status,
                   p_type, &n, &m, &h_ne, &a_ne, &h_c_ne );
    printf( " QPLIB file is of type %s\n", p_type );
    printf( " n = %" i_ipc_ ", m = %" i_ipc_ ", h_ne = %" i_ipc_ 
            ", a_ne = %" i_ipc_ ", h_c_ne = %" i_ipc_ "\n",
            n, m, h_ne, a_ne, h_c_ne );

    // Recover g
    rpc_ g[n];
    rpd_get_g( &data, &status, n, g );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", g[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", g[1]);
//    n2 = quadmath_snprintf(buf2, sizeof buf2, "%*.1Qf", g[2]);
//    n3 = quadmath_snprintf(buf3, sizeof buf3, "%*.1Qf", g[3]);
//    n4 = quadmath_snprintf(buf4, sizeof buf4, "%*.1Qf", g[4]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1 &&
//        (size_t) n2 < sizeof buf2 &&
//        (size_t) n3 < sizeof buf3 &&
//        (size_t) n4 < sizeof buf4)
 //     printf( " g = %s %s %s %s %s\n", buf0, buf1, buf2, buf3, buf4);
    printf( " g = %.1f %.1f %.1f %.1f %.1f\n",(double)g[0], (double)g[1], 
            (double)g[2], (double)g[3], (double)g[4]);
#else
    printf( " g = %.1f %.1f %.1f %.1f %.1f\n",g[0], g[1], g[2], g[3], g[4]);
#endif

    // Recover f
    rpc_ f;
    rpd_get_f( &data, &status, &f );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", f);
//    if ((size_t) n0 < sizeof buf0)
//      printf( " f = %s\n", buf0 );
    printf( " f = %.1f\n", (double)f );
#else
    printf( " f = %.1f\n", f );
#endif

    // Recover xlu
    rpc_ x_l[n];
    rpc_ x_u[n];
    rpd_get_xlu( &data, &status, n, x_l, x_u );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", x_l[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", x_l[1]);
//    n2 = quadmath_snprintf(buf2, sizeof buf2, "%*.1Qf", x_l[2]);
//    n3 = quadmath_snprintf(buf3, sizeof buf3, "%*.1Qf", x_l[3]);
//    n4 = quadmath_snprintf(buf4, sizeof buf4, "%*.1Qf", x_l[4]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1 &&
//        (size_t) n2 < sizeof buf2 &&
//        (size_t) n3 < sizeof buf3 &&
//        (size_t) n4 < sizeof buf4)
//      printf( " x_l = %s %s %s %s %s\n", buf0, buf1, buf2, buf3, buf4);
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", x_u[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", x_u[1]);
//    n2 = quadmath_snprintf(buf2, sizeof buf2, "%*.1Qf", x_u[2]);
//    n3 = quadmath_snprintf(buf3, sizeof buf3, "%*.1Qf", x_u[3]);
//    n4 = quadmath_snprintf(buf4, sizeof buf4, "%*.1Qf", x_u[4]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1 &&
//        (size_t) n2 < sizeof buf2 &&
//        (size_t) n3 < sizeof buf3 &&
//        (size_t) n4 < sizeof buf4)
//      printf( " x_u = %s %s %s %s %s\n", buf0, buf1, buf2, buf3, buf4);
    printf( " x_l = %.1f %.1f %.1f %.1f %.1f\n", (double)x_l[0], 
             (double)x_l[1], (double)x_l[2],
             (double)x_l[3], (double)x_l[4]);
    printf( " x_u = %.1f %.1f %.1f %.1f %.1f\n", (double)x_u[0], 
             (double)x_u[1], (double)x_u[2],
             (double)x_u[3], (double)x_u[4]);
#else
    printf( " x_l = %.1f %.1f %.1f %.1f %.1f\n", x_l[0], x_l[1], x_l[2],
             x_l[3], x_l[4]);
    printf( " x_u = %.1f %.1f %.1f %.1f %.1f\n", x_u[0], x_u[1], x_u[2],
             x_u[3], x_u[4]);
#endif

    // Recover clu
    rpc_ c_l[m];
    rpc_ c_u[m];
    rpd_get_clu( &data, &status, m, c_l, c_u );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", c_l[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", c_l[1]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1)
//      printf( " c_l = %s %s\n", buf0, buf1);
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", c_u[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", c_u[1]);
//   if ((size_t) n0 < sizeof buf0 &&
//       (size_t) n1 < sizeof buf1)
//     printf( " c_u = %s %s\n", buf0, buf1);
    printf( " c_l = %.1f %.1f\n", (double)c_l[0], (double)c_l[1] );
    printf( " c_u = %.1e %.1e\n", (double)c_u[0], (double)c_u[1] );
#else
    printf( " c_l = %.1f %.1f\n", c_l[0], c_l[1] );
    printf( " c_u = %.1f %.1f\n", c_u[0], c_u[1] );
#endif

    // Recover H
    ipc_ h_row[h_ne];
    ipc_ h_col[h_ne];
    rpc_ h_val[h_ne];
    rpd_get_h( &data, &status, h_ne, h_row, h_col, h_val );
    printf( " h_row, h_col, h_val =\n");
    for( ipc_ i = 0; i < h_ne; i++) {
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf",  h_val[i]);
//    if ((size_t) n0 < sizeof buf0)
//      printf("   %" i_ipc_ " %" i_ipc_ " %s\n", h_row[i], h_col[i], buf0);
      printf("   %" i_ipc_ " %" i_ipc_ " %.1f\n", h_row[i], h_col[i], 
             (double)h_val[i]);
#else
      printf("   %" i_ipc_ " %" i_ipc_ " %.1f\n", h_row[i], h_col[i], h_val[i]);
#endif
    }

    // Recover A
    ipc_ a_row[a_ne];
    ipc_ a_col[a_ne];
    rpc_ a_val[a_ne];
    rpd_get_a( &data, &status, a_ne, a_row, a_col, a_val );
    printf( " a_row, a_col, a_val =\n");
    for( ipc_ i = 0; i < a_ne; i++) {
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf",  a_val[i]);
//    if ((size_t) n0 < sizeof buf0)
//      printf("   %" i_ipc_ " %" i_ipc_ " %s\n", a_row[i], a_col[i], buf0);
      printf("   %" i_ipc_ " %" i_ipc_ " %.1f\n", a_row[i], a_col[i], 
             (double)a_val[i]);
#else
      printf("   %" i_ipc_ " %" i_ipc_ " %.1f\n", a_row[i], a_col[i], a_val[i]);
#endif
    }

    // Recover H_c
    ipc_ h_c_ptr[h_c_ne];
    ipc_ h_c_row[h_c_ne];
    ipc_ h_c_col[h_c_ne];
    rpc_ h_c_val[h_c_ne];
    rpd_get_h_c( &data, &status, h_c_ne, h_c_ptr, h_c_row, h_c_col, h_c_val );
    printf( " h_c_row, h_c_col, h_c_val =\n");
    for( ipc_ i = 0; i < h_c_ne; i++) {
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf",  h_c_val[i]);
//    if ((size_t) n0 < sizeof buf0)
//      printf("   %" i_ipc_ " %" i_ipc_ " %s\n", h_c_row[i], h_c_col[i], buf0);
      printf("   %" i_ipc_ " %" i_ipc_ " %" i_ipc_ " %.1f\n", 
             h_c_ptr[i], h_c_row[i], h_c_col[i], (double)h_c_val[i]);
#else
      printf("   %" i_ipc_ " %" i_ipc_ " %" i_ipc_ " %.1f\n", 
             h_c_ptr[i], h_c_row[i], h_c_col[i], h_c_val[i]);
#endif
    }
    // Recover x_type
    ipc_ x_type[n];
    rpd_get_x_type( &data, &status, n, x_type );
    printf( " x_type = %" i_ipc_ " %" i_ipc_ " %" i_ipc_ " %" i_ipc_ 
            " %" i_ipc_ "\n", x_type[0], x_type[1], x_type[2],
            x_type[3], x_type[4] );

    // Recover x
    rpc_ x[n];
    rpd_get_x( &data, &status, n, x );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", x[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", x[1]);
//    n2 = quadmath_snprintf(buf2, sizeof buf2, "%*.1Qf", x[2]);
//    n3 = quadmath_snprintf(buf3, sizeof buf3, "%*.1Qf", x[3]);
//    n4 = quadmath_snprintf(buf4, sizeof buf4, "%*.1Qf", x[4]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1 &&
//        (size_t) n2 < sizeof buf2 &&
//        (size_t) n3 < sizeof buf3 &&
//        (size_t) n4 < sizeof buf4)
//      printf( " x = %s %s %s %s %s\n", buf0, buf1, buf2, buf3, buf4);
    printf( " x = %.1f %.1f %.1f %.1f %.1f\n",(double)x[0], (double)x[1], 
            (double)x[2], (double)x[3], (double)x[4]);
#else
    printf( " x = %.1f %.1f %.1f %.1f %.1f\n",x[0], x[1], x[2], x[3], x[4]);
#endif

    // Recover y
    rpc_ y[m];
    rpd_get_y( &data, &status, m, y );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", y[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", y[1]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1)
//      printf( " y = %s %s\n", buf0, buf1);
    printf( " y = %.1f %.1f\n",(double)y[0], (double)y[1]);
#else
    printf( " y = %.1f %.1f\n",y[0], y[1]);
#endif

    // Recover z
    rpc_ z[n];
    rpd_get_z( &data, &status, n, z );
#ifdef REAL_128
//    n0 = quadmath_snprintf(buf0, sizeof buf0, "%*.1Qf", z[0]);
//    n1 = quadmath_snprintf(buf1, sizeof buf1, "%*.1Qf", z[1]);
//    n2 = quadmath_snprintf(buf2, sizeof buf2, "%*.1Qf", z[2]);
//    n3 = quadmath_snprintf(buf3, sizeof buf3, "%*.1Qf", z[3]);
//    n4 = quadmath_snprintf(buf4, sizeof buf4, "%*.1Qf", z[4]);
//    if ((size_t) n0 < sizeof buf0 &&
//        (size_t) n1 < sizeof buf1 &&
//        (size_t) n2 < sizeof buf2 &&
//        (size_t) n3 < sizeof buf3 &&
//        (size_t) n4 < sizeof buf4)
//      printf( " z = %s %s %s %s %s\n", buf0, buf1, buf2, buf3, buf4);
    printf( " z = %.1f %.1f %.1f %.1f %.1f\n",(double)z[0], (double)z[1], 
           (double)z[2], (double)z[3], (double)z[4]);
#else
    printf( " z = %.1f %.1f %.1f %.1f %.1f\n",z[0], z[1], z[2], z[3], z[4]);
#endif

    // Delete internal workspace
    rpd_terminate( &data, &control, &inform );
}
