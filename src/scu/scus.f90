! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_SCU_EXAMPLE 
   USE GALAHAD_SCU_DOUBLE                    ! double precision version
   IMPLICIT NONE 
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER :: i, row_del, col_del, status 
   INTEGER, PARAMETER :: n = 5, m = 2, mmax = m + 1 
   TYPE ( SCU_matrix_type ) :: mat 
   TYPE ( SCU_data_type ) :: data 
   TYPE ( SCU_inform_type ) :: info 
   REAL ( KIND = wp ) :: X1( n + m ), RHS1( n + m )
   REAL ( KIND = wp ) :: X2( n + m + 1 ), RHS2( n + m + 1 )
   REAL ( KIND = wp ) :: X3( n + m ), RHS3( n + m ), VECTOR( n ) 
   mat%m_max = mmax ; mat%class = 1 
   mat%n = n ; mat%m = m 
   ALLOCATE ( mat%BD_val( 15 ), mat%BD_row( 15 ), mat%BD_col_start( mmax+1 ),& 
              mat%CD_val( 13 ), mat%CD_col( 13 ), mat%CD_row_start( mmax+1 ) ) 
   mat%BD_col_start( : 3 ) = (/1, 7, 10/) 
   mat%CD_row_start( : 3 ) = (/1, 6, 10/) 
   mat%BD_row( : 9 ) = (/1, 2, 3, 4, 5, 6, 5, 6, 7/) 
   mat%BD_val( : 9 ) = (/1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, & 
                     1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp, 4.0_wp/) 
   mat%CD_col( : 9 ) = (/1, 2, 3, 4, 5, 1, 3, 5, 6/) 
   mat%CD_val( : 9 ) = (/1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, & 
                       1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 3.0_wp/) 
   RHS1 = (/2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 7.0_wp, 8.0_wp, 10.0_wp/) 
   RHS2 = (/5.0_wp, 5.0_wp, 4.0_wp, 5.0_wp, 7.0_wp, 12.0_wp, 12.0_wp, 4.0_wp/) 
   RHS3 = (/3.0_wp, 5.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 6.0_wp, 2.0_wp/) 
!  First system 
   status = 1 
   DO 
     CALL SCU_factorize( mat, data, VECTOR, status, info ) 
     IF ( status <= 0 ) EXIT 
     DO i = 1, n 
        VECTOR( i ) = VECTOR( i ) / DBLE( FLOAT( i ) ) 
     END DO 
   END DO 
   WRITE( 6, "( /, ' On exit from SCU_factorize,  status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   status = 1 
   DO 
     CALL SCU_solve( mat, data, RHS1, X1, VECTOR, status ) 
     IF ( status <= 0 ) EXIT 
     DO i = 1, n        !  multiply by the inverse of A 
        VECTOR( i ) = VECTOR( i ) / DBLE( FLOAT( i ) ) 
     END DO 
   END DO 
   WRITE( 6, "( ' On exit from SCU_solve,      status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   WRITE( 6, "( /, ' Solution (first system)', /, ( 8ES9.2 ) )" ) X1( : ) 
!  Second system 
   mat%BD_row( 10 : 12 ) = (/ 1, 6, 8 /) 
   mat%BD_val( 10 : 12 ) = (/ 1.0_wp, 1.0_wp, 1.0_wp /) 
   mat%BD_col_start( 4 ) = 13 
   mat%CD_col( 10 ) = 1 
   mat%CD_val( 10 ) = 1.0_wp 
   mat%CD_row_start( 4 ) = 11 
   status = 1 
   DO 
     CALL SCU_append( mat, data, VECTOR, status, info ) 
     IF ( status <= 0 ) EXIT 
     DO i = 1, n 
        VECTOR( i ) = VECTOR( i ) / DBLE( FLOAT( i ) ) 
     END DO 
   END DO 
   WRITE( 6, "( /, ' On exit from SCU_append,     status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   status = 1 
   DO 
     CALL SCU_solve( mat, data, RHS2, X2, VECTOR, status ) 
     IF ( status <= 0 ) EXIT 
     DO i = 1, n 
        VECTOR( i ) = VECTOR( i ) / DBLE( FLOAT( i ) ) 
     END DO 
   END DO 
   WRITE( 6, "( ' On exit from SCU_solve,      status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   WRITE( 6, "( /, ' Solution (second system)', /, ( 8ES9.2 ) )" ) X2 
!  Third system 
   row_del = 1 
   col_del = 2 
   status = 1 
   CALL SCU_delete( mat, data, VECTOR, status, info, col_del, & 
                     row_del = row_del ) 
   WRITE( 6, "( /, ' On exit from SCU_delete,     status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   status = 1 
   DO 
     CALL SCU_solve( mat, data, RHS3, X3, VECTOR, status ) 
     IF ( status <= 0 ) EXIT 
       DO i = 1, n 
        VECTOR( i ) = VECTOR( i ) / DBLE( FLOAT( i ) ) 
     END DO 
   END DO 
   WRITE( 6, "( ' On exit from SCU_solve,      status = ', I3 )" ) status 
   IF ( status < 0 ) STOP 
   WRITE( 6, "( /, ' Solution (third system)', /, ( 8ES9.2 ) )" ) X3 
   CALL SCU_terminate( data, status, info ) 
   WRITE( 6, "( /, ' On exit from SCU_terminate,  status = ', I3 )" ) status 
   END PROGRAM GALAHAD_SCU_EXAMPLE 



