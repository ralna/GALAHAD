/* rpdtf.c */
/* Full test for the RPD C interface using Fortran sparse matrix indexing */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_rpd.h"
#define BUFSIZE 1000

ipc_ main(void) {

    // Derived types
    void *data;
    struct rpd_control_type control;
    struct rpd_inform_type inform;

    char qplib_file[BUFSIZE];
    char *galahad = "GALAHAD";
    ipc_ qplib_file_len;

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

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    // Initialize RPD */
    rpd_initialize( &data, &control, &status );

    // Set user-defined control options */
    control.f_indexing = true; // fortran sparse matrix indexing

    // Recover vital statistics from the QPLIB file
    rpd_get_stats( qplib_file, qplib_file_len, &control, &data, &status,
                   p_type, &n, &m, &h_ne, &a_ne, &h_c_ne );
    printf( " QPLIB file is of type %s\n", p_type );
    printf( " n = %i, m = %i, h_ne = %i, a_ne = %i, h_c_ne = %i\n",
            n, m, h_ne, a_ne, h_c_ne );

    // Recover g
    real_wp_ g[n];
    rpd_get_g( &data, &status, n, g );
    printf( " g = %.1f %.1f %.1f %.1f %.1f\n",g[0], g[1], g[2], g[3], g[4]);

    // Recover f
    real_wp_ f;
    rpd_get_f( &data, &status, &f );
    printf( " f = %.1f\n", f );

    // Recover xlu
    real_wp_ x_l[n];
    real_wp_ x_u[n];
    rpd_get_xlu( &data, &status, n, x_l, x_u );
    printf( " x_l = %.1f %.1f %.1f %.1f %.1f\n", x_l[0], x_l[1], x_l[2],
             x_l[3], x_l[4]);
    printf( " x_u = %.1f %.1f %.1f %.1f %.1f\n", x_u[0], x_u[1], x_u[2],
             x_u[3], x_u[4]);

    // Recover clu
    real_wp_ c_l[m];
    real_wp_ c_u[m];
    rpd_get_clu( &data, &status, m, c_l, c_u );
    printf( " c_l = %.1f %.1f\n", c_l[0], c_l[1] );
    printf( " c_u = %.1f %.1f\n", c_u[0], c_u[1] );

    // Recover H
    ipc_ h_row[h_ne];
    ipc_ h_col[h_ne];
    real_wp_ h_val[h_ne];
    rpd_get_h( &data, &status, h_ne, h_row, h_col, h_val );
    printf( " h_row, h_col, h_val =\n");
    for( ipc_ i = 0; i < h_ne; i++) printf("   %i %i %.1f\n",
         h_row[i], h_col[i], h_val[i]);

    // Recover A
    ipc_ a_row[a_ne];
    ipc_ a_col[a_ne];
    real_wp_ a_val[a_ne];
    rpd_get_a( &data, &status, a_ne, a_row, a_col, a_val );
    printf( " a_row, a_col, a_val =\n");
    for( ipc_ i = 0; i < a_ne; i++) printf("   %i %i %.1f\n",
         a_row[i], a_col[i], a_val[i]);

    // Recover H_c
    ipc_ h_c_ptr[h_c_ne];
    ipc_ h_c_row[h_c_ne];
    ipc_ h_c_col[h_c_ne];
    real_wp_ h_c_val[h_c_ne];
    rpd_get_h_c( &data, &status, h_c_ne, h_c_ptr, h_c_row, h_c_col, h_c_val );
    printf( " h_c_row, h_c_col, h_c_val =\n");
    for( ipc_ i = 0; i < h_c_ne; i++) printf("   %i %i %i %.1f\n",
         h_c_ptr[i], h_c_row[i], h_c_col[i], h_c_val[i]);

    // Recover x_type
    ipc_ x_type[n];
    rpd_get_x_type( &data, &status, n, x_type );
    printf( " x_type = %i %i %i %i %i\n", x_type[0], x_type[1], x_type[2],
            x_type[3], x_type[4] );

    // Recover x
    real_wp_ x[n];
    rpd_get_x( &data, &status, n, x );
    printf( " x = %.1f %.1f %.1f %.1f %.1f\n",x[0], x[1], x[2], x[3], x[4]);

    // Recover y
    real_wp_ y[m];
    rpd_get_y( &data, &status, m, y );
    printf( " y = %.1f %.1f\n",y[0], y[1]);

    // Recover z
    real_wp_ z[n];
    rpd_get_z( &data, &status, n, z );
    printf( " z = %.1f %.1f %.1f %.1f %.1f\n",z[0], z[1], z[2], z[3], z[4]);

    // Delete internal workspace
    rpd_terminate( &data, &control, &inform );
}
