! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_SCU_testdeck
   USE GALAHAD_SCU_DOUBLE                    ! double precision version
   IMPLICIT NONE 
   TYPE ( SCU_matrix_type ) :: mat, mat2 
   TYPE ( SCU_data_type ) :: data
   TYPE ( SCU_info_type ) :: info
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 5, m = 2, mmax = m + 1
   INTEGER, PARAMETER :: lcd = 10, lbd = 11, lcd2 = 13, lbd2 = 14
   INTEGER, PARAMETER :: mp1 = m + 1, mp2 = mp1 + 1, mpn = m + n
   INTEGER, PARAMETER :: mpnp1 = mp1 + n
   REAL ( KIND = wp ) :: X1( n + m ), RHS1( n + m )
   REAL ( KIND = wp ) :: X2( n + m + 1 ), RHS2( n + m + 1 )
   REAL ( KIND = wp ) :: X3( n + m ), RHS3( n + m ), VECTOR( n ) 
   REAL ( KIND = wp ) :: epsqrt
   INTEGER :: i, class, status, row_del, col_del

   ALLOCATE( mat%BD_val( lbd ), mat%BD_row( lbd ) )
   ALLOCATE( mat%BD_col_start( mmax + 1 ) )
   ALLOCATE( mat%CD_val( lcd ), mat%CD_col( lcd ) )
   ALLOCATE( mat%CD_row_start( mmax + 1 ) )
   
   mat%BD_col_start = (/ 1, 7, 10, 12 /)
   mat%BD_row = (/ 1, 2, 3, 4, 5, 6, 5, 6, 7, 2, 8 /)
   mat%BD_val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 8.0_wp, 1.0_wp,     &
                   3.0_wp, 1.0_wp, 1.0_wp, 12.0_wp /)
   mat%CD_row_start = (/ 1, 6, 10, 11 /)
   mat%CD_col = (/ 1, 2, 3, 4, 5, 1, 3, 5, 6, 1 /)
   mat%CD_val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,     &
                   1.0_wp, 2.0_wp, 1.0_wp/)
   RHS1 = (/ 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 7.0_wp, 16.0_wp, 6.0_wp /)
   RHS2 = (/ 2.0_wp, 4.0_wp, 4.0_wp, 5.0_wp, 7.0_wp, 16.0_wp, 6.0_wp, 13.0_wp /)
   RHS3 = (/ 2.0_wp, 4.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 5.0_wp, 13.0_wp/)

   NULLIFY( mat2%BD_val, mat2%BD_row, mat2%BD_col_start )
   NULLIFY( mat2%CD_val, mat2%CD_col, mat2%CD_row_start )

   epsqrt = SQRT( EPSILON( 1.0_wp ) )

!  Calls to provoke unsuccessful returns

!  Error exit: status = - 1

   WRITE ( 6, 2300 ) - 1
   mat%n = n ; mat%m = - 1 ; mat%class = 0 ; mat%m_max = mmax 

   status = 1
   CALL SCU_factorize( mat, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

   status = 1
   CALL SCU_append( mat, data, VECTOR, status, info )
   WRITE ( 6, 2020 ) status

   status = 1 ; row_del = 1 ; col_del = 1
   CALL SCU_delete( mat, data, VECTOR, status, info, col_del, row_del )
   WRITE ( 6, 2030 ) status

!  Error exit: status = - 2

   mat%n = n ; mat%m = m ; mat%class = 1 ; mat%m_max = mmax 
   status = - 1
   WRITE ( 6, 2300 ) - 2
   CALL SCU_factorize( mat, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

   status = - 1
   CALL SCU_solve( mat, data, RHS1, X1, VECTOR, status )
   WRITE ( 6, 2010 ) status

   status = - 1
   CALL SCU_append( mat, data, VECTOR, status, info )
   WRITE ( 6, 2020 ) status

!  Error exit: status = - 3

   WRITE ( 6, 2300 ) - 3
   mat%class = 1
   status = 1
   CALL SCU_solve( mat, data, RHS1, X1, VECTOR, status )
   WRITE ( 6, 2010 ) status

!  Error exit: status = - 4

   mat2%n = n ; mat2%m = m ; mat2%class = 1 ; mat2%m_max = mmax 
   status = 1
   WRITE ( 6, 2300 ) - 4
   CALL SCU_factorize( mat2, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 5

   ALLOCATE( mat2%BD_val( lbd2 ), mat2%BD_row( lbd2 ) )
   ALLOCATE( mat2%BD_col_start( m ) )
   status = 1
   WRITE ( 6, 2300 ) - 5
   CALL SCU_factorize( mat2, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 6

   ALLOCATE( mat2%CD_val( lcd2 ), mat2%CD_col( lcd2 ) )
   ALLOCATE( mat2%CD_row_start( m ) )
   status = 1
   WRITE ( 6, 2300 ) - 6
   CALL SCU_factorize( mat2, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 7

   DEALLOCATE( mat2%BD_col_start )
   ALLOCATE( mat2%BD_col_start( mmax + 1 ) )
   mat2%BD_col_start( mat2%m + 1 ) =                                           &
     MIN( SIZE( mat2%BD_val ), SIZE( mat2%BD_row ) ) + 1

   status = 1
   WRITE ( 6, 2300 ) - 7
   CALL SCU_factorize( mat2, data, VECTOR, status, info )
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 8

   WRITE ( 6, 2300 ) - 8
   DEALLOCATE( mat2%CD_row_start )
   ALLOCATE( mat2%CD_row_start( mmax + 1 ) )
   mat2%CD_row_start( mat2%m + 1 ) =                                           &
     MIN( SIZE( mat2%CD_val ), SIZE( mat2%CD_col ) ) + 1

   status = 1
   DO
     CALL SCU_factorize( mat, data, VECTOR, status, info )
     IF ( status <= 0 ) EXIT
     DO i = 1, n
       VECTOR( i ) = VECTOR( i ) / i
     END DO
   END DO

   mat%m = 1
   status = 1
   CALL SCU_solve( mat, data, RHS1, X1, VECTOR, status )
   WRITE ( 6, 2010 ) status

   status = 1
   CALL SCU_append( mat, data, VECTOR, status, info )
   WRITE ( 6, 2020 ) status

   row_del = 1 ; col_del = 1
   status = 1
   CALL SCU_delete( mat, data, VECTOR, status, info, col_del, row_del )
   WRITE ( 6, 2030 ) status
   
!  Error exit: status = - 9

   mat2%n = n ; mat2%m = 2 ; mat2%class = 1 ; mat2%m_max = mmax 
   mat2%BD_col_start( : mat2%m + 1 ) = (/ 1, 2, 3 /)
   mat2%BD_row( : mat2%BD_col_start( mat2%m + 1 ) - 1 ) = (/ 1, 1 /)
   mat2%BD_val( : mat2%BD_col_start( mat2%m + 1 ) - 1 ) = (/ 1.0_wp, 2.0_wp /)
   mat2%CD_row_start( : mat2%m + 1 ) = (/ 1, 2, 3 /)
   mat2%CD_col( : mat2%CD_row_start( mat2%m + 1 ) - 1 ) = (/ 1, 2 /)
   mat2%CD_val( : mat2%CD_row_start( mat2%m + 1 ) - 1 ) = (/ 1.0_wp, 2.0_wp /)

   WRITE ( 6, 2300 ) - 9
   status = 1
   DO
     CALL SCU_factorize( mat2, data, VECTOR, status, info )
     IF ( status <= 0 ) EXIT
     DO i = 1, n
       VECTOR( i ) = VECTOR( i ) / i
     END DO
   END DO
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 10

   mat%n = n ; mat%m = m ; mat%class = 3 ; mat%m_max = mmax 
   WRITE ( 6, 2300 ) - 10
   status = 1
   DO
     CALL SCU_factorize( mat, data, VECTOR, status, info )
     IF ( status <= 0 ) EXIT
     DO i = 1, n
       VECTOR( i ) = VECTOR( i ) / i
     END DO
   END DO
   WRITE ( 6, 2000 ) status

!  Error exit: status = - 11

   mat%n = n ; mat%m = m ; mat%class = 4 ; mat%m_max = mmax 
   WRITE ( 6, 2300 ) - 11
   status = 1
   DO
     CALL SCU_factorize( mat, data, VECTOR, status, info )
     IF ( status <= 0 ) EXIT
     DO i = 1, n
       VECTOR( i ) = VECTOR( i ) / i
     END DO
   END DO
   WRITE ( 6, 2000 ) status

!  Calls to solve linear systems

   DO class = 0, 4
     mat%n = n ; mat%m = m ; mat%class = class ; mat%m_max = mmax 
     IF ( class == 0 ) mat%class = 1
     WRITE ( 6, 2200 ) mat%class
     IF ( class == 0 )                                                         &
       RHS3 = (/ 1.0_wp, 3.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 8.0_wp, 13.0_wp/)
     IF ( class == 1 )                                                         &
       RHS3 = (/ 2.0_wp, 4.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 5.0_wp, 13.0_wp/)
     IF ( mat%class == 2 ) THEN
       mat%BD_val( 8 ) = 2.0_wp
       RHS1( 6 ) = 15.0_wp
       RHS1( 7 ) = 4.0_wp
       RHS2( 6 ) = 15.0_wp
       RHS2( 7 ) = 4.0_wp
       RHS3( 6 ) = 13.0_wp
     END IF
   
     IF ( mat%class == 4 ) THEN
       mat%BD_val( 6 ) = 0.0_wp
       mat%BD_val( 8 ) = 0.0_wp
       mat%BD_val( 9 ) = 0.0_wp
       mat%BD_val( 11 ) = 0.0_wp
       RHS1( 6 ) = 5.0_wp
       RHS1( 7 ) = 1.0_wp
       RHS2( 6 ) = 5.0_wp
       RHS2( 7 ) = 1.0_wp
       RHS2( 8 ) = 1.0_wp
       RHS3( 6 ) = 5.0_wp
       RHS3( 7 ) = 1.0_wp
     END IF
   
     status = 1
     DO
       CALL SCU_factorize( mat, data, VECTOR, status, info )
       IF ( status <= 0 ) EXIT
       DO i = 1, n
         VECTOR( i ) = VECTOR( i ) / i
       END DO
     END DO
   
     WRITE ( 6, 2000 ) status
     IF ( status < 0 ) CYCLE
   
     status = 1
     DO 
       CALL SCU_solve( mat, data, RHS1, X1, VECTOR, status )
       IF ( status <= 0 ) EXIT
       DO i = 1, n
         VECTOR( i ) = VECTOR( i ) / i
       END DO
     END DO
   
     WRITE ( 6, 2010 ) status
     IF ( status < 0 ) CYCLE
     DO I = 1, mpn
       IF ( ABS( X1( i ) - 1.0 ) > epsqrt ) THEN
         WRITE ( 6, 2100 )
         GO TO 60
       END IF
     END DO
     WRITE ( 6, 2110 )
 60  CONTINUE

     status = 1
     DO
       CALL SCU_append( mat, data, VECTOR, status, info )
       IF ( status <= 0 ) EXIT
       DO i = 1, n
         VECTOR( i ) = VECTOR( i ) / i
       END DO
     END DO
     
     WRITE ( 6, 2020 ) status
     IF ( status < 0 ) CYCLE
     
     status = 1
     DO
       CALL SCU_solve( mat, data, RHS2, X2, VECTOR, status )
       IF ( status <= 0 ) EXIT
       DO i = 1, n
         VECTOR( I ) = VECTOR( I ) / i
       END DO
     END DO
     
     WRITE ( 6, 2010 ) status
     DO I = 1, mpnp1
       IF ( ABS( X2( i ) - 1.0 ) > epsqrt ) THEN
         WRITE ( 6, 2120 )
         GO TO 160
       END IF
     END DO
     
     WRITE ( 6, 2130 )
 160 CONTINUE
     IF ( status < 0 ) CYCLE
     
     mat2%n = mat%n ; mat2%m = mat%m
     mat2%m_max = mat%m_max ; mat2%class = mat%class

     DO i = 1, mp2
       mat2%BD_col_start( i ) = mat%BD_col_start( i )
       mat2%CD_row_start( i ) = mat%CD_row_start( i )
     END DO
     
     DO i = 1, lbd
       mat2%BD_row( i ) = mat%BD_row( i )
       mat2%BD_val( i ) = mat%BD_val( i )
     END DO
     
     DO i = 1, lcd
       mat2%CD_col( i ) = mat%CD_col( i )
       mat2%CD_val( i ) = mat%CD_val( i )
     END DO
     
     IF ( class == 0 ) THEN 
       row_del = 2 ; col_del = 1
     ELSE 
       row_del = 1 ; col_del = 2
     END IF
     status = 1
     CALL SCU_delete( mat2, data, VECTOR, status, info, col_del, row_del )
     WRITE ( 6, 2030 ) status
     IF ( status < 0 ) CYCLE
     
     status = 1
     DO
       CALL SCU_solve( mat2, data, RHS3, X3, VECTOR, status )
       IF ( status <= 0 ) EXIT
       DO i = 1, n
         VECTOR( i ) = VECTOR( i ) / i
       END DO
     END DO
     WRITE ( 6, 2010 ) status
     IF ( status < 0 ) CYCLE
     
     DO I = 1, MPN
       IF ( ABS( X3(I) - 1.0 ) > epsqrt ) THEN
         WRITE ( 6, 2140 )
         GO TO 260
       END IF
     END DO
     WRITE ( 6, 2150 )
 260 CONTINUE
   END DO
   CALL SCU_terminate( data, status, info )

   STOP

 2000 FORMAT( '  On exit from SCU_factorize,    status = ', I3 )
 2010 FORMAT( '  On exit from SCU_solve,        status = ', I3 )
 2020 FORMAT( '  On exit from SCU_append,       status = ', I3 )
 2030 FORMAT( '  On exit from SCU_delete,       status = ', I3 )
 2100 FORMAT( '  Solution (original equations)             incorrect' )
 2110 FORMAT( '  Solution (original equations)             correct' )
 2120 FORMAT( '  Solution (equations with added row/col)   incorrect' )
 2130 FORMAT( '  Solution (equations with added row/col)   correct' )
 2140 FORMAT( '  Solution (equations with removed row/col) incorrect' )
 2150 FORMAT( '  Solution (equations with removed row/col) correct' )
 2200 FORMAT( /,' *** Problem class ', I2, ' *** ',/ )
 2300 FORMAT( /,' *** Test to provoke return with status = ',I3 )

   END PROGRAM GALAHAD_SCU_testdeck
