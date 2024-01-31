! THIS VERSION: GALAHAD 4.3 - 2024-01-31 AT 08:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_RPD_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_RPD_precision
   IMPLICIT NONE
   TYPE ( RPD_control_type ) :: control
   TYPE ( RPD_inform_type ) :: inform
   TYPE ( RPD_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, a_ne, h_ne, h_c_ne
   INTEGER ( KIND = ip_ ) :: status
   INTEGER :: length
   REAL ( KIND = rp_ ) :: f
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, G
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_col
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_c_row, H_c_col
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_c_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val, H_c_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_type
   CHARACTER ( LEN = 3 ) :: p_type
   INTEGER ( KIND = ip_ ) :: i, qplib_unit = 21
   CHARACTER ( LEN = 8 ) :: galahad_var = 'GALAHAD'
   CHARACTER( LEN = : ), ALLOCATABLE :: galahad
!  open the QPLIB file ALLINIT.qplib for reading on unit 21
!  CALL GET_ENVIRONMENT_VARIABLE( galahad_var, status = status )
!  WRITE( 6, "( ' status of environment variable ', A, ' = ', I0 )" )          &
!    galahad_var, status
   CALL GET_ENVIRONMENT_VARIABLE( galahad_var, length = length )
!  WRITE( 6, "( ' length of environment variable ', A, ' = ', I0 )" )          &
!    galahad_var, length
   ALLOCATE( CHARACTER( LEN = length ):: galahad )
   CALL GET_ENVIRONMENT_VARIABLE( galahad_var, value = galahad )
!  WRITE( 6, "( ' environment variable = ', A )" ) galahad
   OPEN( qplib_unit, file = galahad // "/examples/ALLINIT.qplib",              &
         FORM = 'FORMATTED', STATUS = 'OLD' )
   CALL RPD_initialize( data, control, inform )
   control%qplib = qplib_unit
!  collect the problem statistics
   CALL RPD_get_stats( control, data, status, p_type,                          &
                       n, m, h_ne, a_ne, h_c_ne )
   WRITE( 6, "( ' get stats status = ', I0 )" ) status
   WRITE( 6, "( ' qplib example ALLINIT type = ', A )" ) p_type
   WRITE( 6, "( ' n, m, h_ne, a_ne, h_c_ne =', 5I3 )" )                        &
     n, m, h_ne, a_ne, h_c_ne
!  close the QPLIB file after reading
   CLOSE( qplib_unit )
!  allocate space to hold the problem data
   ALLOCATE( X( n ), G( n ), Z( n ), X_l( n ), X_u( n ), X_type( n ) )
   ALLOCATE( Y( m ), C_l( m ), C_u( m ) )
   ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ) )
   ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_col( h_ne ) )
   ALLOCATE( H_c_val( h_c_ne ), H_c_row( h_c_ne ) )
   ALLOCATE( H_c_col( h_c_ne ), H_c_ptr( h_c_ne ) )
!  investigate the problem data
   CALL RPD_get_g( data, status, G )
   WRITE( 6, "( ' G =', 5F5.1 )" ) G
   CALL RPD_get_f( data, status, f )
   WRITE( 6, "( ' f =', F5.1 )" ) f
   CALL RPD_get_xlu( data, status, X_l, X_u )
   WRITE( 6, "( ' X_l =', 5F4.1 )" ) X_l
   WRITE( 6, "( ' X_u =', 5F4.1 )" ) X_u
   CALL RPD_get_clu( data, status, C_l, C_u )
   WRITE( 6, "( ' C_l =', 2F4.1 )" ) C_l
   WRITE( 6, "( ' C_u =', 2ES8.1 )" ) C_u
   CALL RPD_get_H( data, status, H_row, H_col, H_val )
   DO i = 1, h_ne
     WRITE( 6, "( ' H(row, col, val) =', 2I3, F5.1 )" )                        &
       H_row( i ), H_col( i ), H_val( i )
   END DO
   CALL RPD_get_A( data, status, A_row, A_col, A_val )
   DO i = 1, a_ne
     WRITE( 6, "( ' A(row, col, val) =', 2I3, F5.1 )" )                        &
       A_row( i ), A_col( i ), A_val( i )
   END DO
   CALL RPD_get_H_c( data, status, H_c_ptr, H_c_row, H_c_col, H_c_val )
   DO i = 1, h_c_ne
     WRITE( 6, "( ' H_c(ptr, row, col, val) =', 3I3, F5.1 )" )                 &
       H_c_ptr( i ), H_c_row( i ), H_c_col( i ), H_c_val( i )
   END DO
   CALL RPD_get_x_type( data, status, X_type )
   WRITE( 6, "( ' X_type =', 5I2 )" ) X_type
   CALL RPD_get_x( data, status, X )
   WRITE( 6, "( ' X =', 5F4.1 )" ) X
   CALL RPD_get_y( data, status, Y )
   WRITE( 6, "( ' Y =', 2F4.1 )" ) Y
   CALL RPD_get_z( data, status, Z )
   WRITE( 6, "( ' Z =', 5F4.1 )" ) Z
!  deallocate internal array space
   CALL RPD_terminate( data, control, inform )
   DEALLOCATE( X, G, Y, Z, x_l, X_u, C_l, C_u, X_type, galahad )
   DEALLOCATE( A_val, A_row, A_col, H_val, H_row, H_col )
   DEALLOCATE( H_c_val, H_c_row, H_c_col, H_c_ptr )

   END PROGRAM GALAHAD_RPD_interface_test
