   PROGRAM GALAHAD_APS_EXAMPLE
   USE GALAHAD_APS_DOUBLE                         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )      ! set precision
   INTEGER, PARAMETER :: n = 10                   ! problem dimension
   INTEGER :: i
   REAL ( KIND = wp ) :: f, delta
   LOGICAL :: new_H = .TRUE.
   TYPE ( SMT_type ) :: H
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X
   TYPE ( APS_data_type ) :: data
   TYPE ( APS_control_type ) :: control
   TYPE ( APS_inform_type ) :: inform
   INTEGER :: s
   CALL SMT_put( H%type, 'COORDINATE', s )       ! Specify co-ordinate for H
   H%ne = 2 * n - 1                              ! set up problem
   ALLOCATE( H%row( H%ne ), H%col( H%ne ), H%val( H%ne ), STAT = i )
   DO i = 1, n - 1
      H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_wp
      H%row( n + i ) = i + 1; H%col( n + i ) = i ; H%val( n + i ) = 1.0_wp
   END DO
   H%row( n ) = n ; H%col( n ) = n ; H%val( n ) = -2.0_wp
   delta = 1.0_wp ; C = 1.0_wp

   CALL APS_initialize( data, control, inform )  ! Initialize control parameters

   CALL APS_solve( n, delta, C, H, new_H, f, X, data, control, inform )
   WRITE( 6, "( / A, ES12.4, A, / ( 5ES12.4 ) )" )                             &
             ' optimal f =', f, ' optimal x = ', X

   C( 1 ) = 2.0_wp                        ! Change the first component of C to 2
   CALL APS_resolve( n, delta, X, f, data, control, inform, C = C )
   WRITE( 6, "( / A, ES12.4, A, / ( 5ES12.4 ) )" )                             &
             ' optimal f =', f, ' optimal x = ', X

   delta = 10.0_wp                       !  increase the radius
   CALL APS_resolve( n, delta, X, f, data, control, inform )
   WRITE( 6, "( / A, ES12.4, A, / ( 5ES12.4 ) )" )                             &
             ' optimal f =', f, ' optimal x = ', X

   CALL APS_terminate( data, control, inform )  ! Deallocate arrays
   END PROGRAM GALAHAD_APS_EXAMPLE
