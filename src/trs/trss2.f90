   PROGRAM GALAHAD_TRS_EXAMPLE2  !  GALAHAD 2.3 - 29/01/2009 AT 10:30 GMT.
   USE GALAHAD_TRS_DOUBLE                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   INTEGER, PARAMETER :: n = 10000              ! problem dimension
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X
   TYPE ( SMT_type ) :: H, A
   TYPE ( TRS_data_type ) :: data
   TYPE ( TRS_control_type ) :: control        
   TYPE ( TRS_inform_type ) :: inform
   REAL ( KIND = wp ) :: f = 1.0_wp             ! constant term, f
   REAL ( KIND = wp ) :: radius = 10.0_wp       ! trust-region radius
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
   ALLOCATE( A%val( n ) ) ; A%val = 1.0_wp ; A%m = 1
   CALL TRS_initialize( data, control, inform )  ! Initialize control parameters
   CALL TRS_solve( n, radius, f, C, H, X, data, control, inform, A = A )  ! Solve
   IF ( inform%status == 0 ) THEN !  Successful return
    WRITE( 6, "( 1X, I0, ' factorizations. Objective and Lagrange multiplier =',&
   &    2ES12.4 )" ) inform%factorizations, inform%obj, inform%multiplier
   ELSE  !  Error returns
    WRITE( 6, "( ' TRS_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL TRS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( H%row, H%col, H%val, A%val )
   END PROGRAM GALAHAD_TRS_EXAMPLE2
