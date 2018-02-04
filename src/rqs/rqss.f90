   PROGRAM GALAHAD_RQS_EXAMPLE  !  GALAHAD 2.4 - 14/05/2010 AT 14:30 GMT.
   USE GALAHAD_RQS_DOUBLE                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   INTEGER, PARAMETER :: n = 10000              ! problem dimension
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X
   TYPE ( SMT_type ) :: H, M
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
   CALL SMT_put( M%type, 'DIAGONAL', s )         ! Specify diagonal for M
   ALLOCATE( M%val( n ) ) ; M%val = 2.0_wp
   CALL RQS_initialize( data, control, inform )  ! Initialize control parameters
   CALL RQS_solve( n, p, sigma, f, C, H, X, data, control, inform, M = M ) ! Solve
   IF ( inform%status == 0 ) THEN !  Successful return
    WRITE( 6, "( 1X, I0, ' factorizations. Objective and Lagrange multiplier =',&
   &    2ES12.4 )" ) inform%factorizations, inform%obj, inform%multiplier
   ELSE  !  Error returns
    WRITE( 6, "( ' RQS_solve exit status = ', I0 ) " ) inform%status
   END IF
   CALL RQS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( H%row, H%col, H%val, M%val )
   END PROGRAM GALAHAD_RQS_EXAMPLE
