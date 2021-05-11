   PROGRAM GALAHAD_RQS_EXAMPLE2  !  GALAHAD 2.3 - 29/01/2009 AT 10:30 GMT.
   USE GALAHAD_RQS_DOUBLE                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   INTEGER, PARAMETER :: n = 10000              ! problem dimension
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X
   TYPE ( SMT_type ) :: H, A
   TYPE ( RQS_data_type ) :: data
   TYPE ( RQS_control_type ) :: control        
   TYPE ( RQS_inform_type ) :: inform
   REAL ( KIND = wp ) :: f = 1.0_wp             ! constant term, f
   REAL ( KIND = wp ) :: sigma = 10.0_wp        ! regularisation weight
   REAL ( KIND = wp ) :: p = 3.0_wp             ! regularisation order
   INTEGER :: i, s
   C = 1.0_wp
   CALL SMT_put( H%type, 'COORDINATE', s )      ! Specify co-ordinate for H
   H%ne = 2 * n - 1
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   DO i = 1, n
    H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_wp 
   END DO
   DO i = 1, n - 1
    H%row( n + i ) = i + 1 ; H%col( n + i ) = i ; H%val( n + i ) = 1.0_wp 
   END DO
   CALL SMT_put( A%type, 'DENSE', s )           ! Specify 1 by n matrix A
   ALLOCATE( A%val( n ) ) ; A%val = 1.0_wp ; A%m = 1 ; A%n = n
   DO i = 1, n
    A%val( i ) = REAL( i, KIND = wp )
   END DO
   CALL RQS_initialize( data, control, inform )  ! Initialize control parameters
   CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, A = A ) 
   IF ( inform%status == 0 ) THEN !  Successful return
    WRITE( 6, "( 1X, I0, ' factorizations. Objective and Lagrange multiplier',&
   &  ' =',  2ES12.4 )" ) inform%factorizations, inform%obj, inform%multiplier
   ELSE  !  Error returns
    WRITE( 6, "( ' RQS_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL RQS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( H%row, H%col, H%val, A%val )
   END PROGRAM GALAHAD_RQS_EXAMPLE2
