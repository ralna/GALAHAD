   PROGRAM GALAHAD_DPS_EXAMPLE   !  GALAHAD 3.0 - 23/03/2018 AT 07:30 GMT.
   USE GALAHAD_DPS_DOUBLE                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   INTEGER, PARAMETER :: n = 10                 ! problem dimension
   INTEGER :: i
   REAL ( KIND = wp ) :: f, delta
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X
   TYPE ( DPS_data_type ) :: data
   TYPE ( DPS_control_type ) :: control
   TYPE ( DPS_inform_type ) :: inform
   INTEGER :: s
   H%ne = 2 * n - 1                             ! set up problem
   CALL SMT_put( H%type, 'COORDINATE', s )      ! specify co-ordinate for H
   ALLOCATE( H%row( H%ne ), H%col( H%ne ), H%val( H%ne ), STAT = i )
   DO i = 1, n - 1
      H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_wp
      H%row( n + i ) = i + 1; H%col( n + i ) = i ; H%val( n + i ) = 1.0_wp
   END DO
   H%row( n ) = n ; H%col( n ) = n ; H%val( n ) = -2.0_wp
   C = 1.0_wp ; f = 0.0_wp ; delta = 1.0_wp
   CALL DPS_initialize( data, control, inform )  ! initialize control parameters
   CALL DPS_solve( n, H, C, f, X, data, control, inform, delta = delta )
   WRITE( 6, "( / A, ES12.4, A, / ( 5ES12.4 ) )" )                             &
             ' optimal f =', inform%obj, ', optimal x = ', X
   C( 1 ) = 2.0_wp                       ! change the first component of C to 2
   CALL DPS_resolve( n, X, data, control, inform, C = C, delta = delta )
   WRITE( 6, "( / A, /, A, ES12.4, A, / ( 5ES12.4 ) )" )                       &
     ' change C:', ' optimal f =', inform%obj, ', optimal x = ', X
   delta = 10.0_wp                       !  increase the radius
   CALL DPS_resolve( n, X, data, control, inform, delta = delta )
   WRITE( 6, "( / A, /, A, ES12.4, A, / ( 5ES12.4 ) )" )                       &
     ' increase delta:', ' optimal f =', inform%obj, ', optimal x = ', X
   CALL DPS_terminate( data, control, inform )  ! Deallocate arrays
   END PROGRAM GALAHAD_DPS_EXAMPLE
