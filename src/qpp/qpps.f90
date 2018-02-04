! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_QPP_EXAMPLE
   USE GALAHAD_QPP_double                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPP_dims_type ) :: d
   TYPE ( QPP_map_type ) :: map
   TYPE ( QPP_control_type ) :: control        
   TYPE ( QPP_inform_type ) :: info
   TYPE ( QPT_problem_type ) :: p
   INTEGER :: i, j, s
   INTEGER, PARAMETER :: n = 4, m = 2, h_ne = 5, a_ne = 5
   REAL ( KIND = wp ) :: X_orig( n )
! sparse co-ordinate storage format
   CALL SMT_put( p%H%type, 'COORDINATE', s )  ! Specify co-ordinate 
   CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for H and A
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   p%H%val = (/ 1.0_wp, 2.0_wp, -1.0_wp, 5.0_wp, 4.0_wp /) ! Hessian H
   p%H%row = (/ 1, 2, 3, 3, 4 /)                           ! NB lower triangle
   p%H%col = (/ 1, 2, 2, 3, 1 /) ; p%H%ne = h_ne
   p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)  ! Jacobian A
   p%A%row = (/ 1, 1, 2, 2, 2 /)
   p%A%col = (/ 1, 2, 2, 3, 4 /) ; p%A%ne = a_ne
! sparse row-wise storage format
!  CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', s )  ! Specify sparse-by-rows
!  CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', s )  ! storage for H and A
!  ALLOCATE( p%H%val( h_ne ), p%H%col( h_ne ), p%H%ptr( n + 1 ) )
!  ALLOCATE( p%A%val( a_ne ), p%A%col( a_ne ), p%A%ptr( m + 1 ) )
!  p%H%val = (/ 1.0_wp, 2.0_wp, -1.0_wp, 5.0_wp, 4.0_wp /) ! Hessian H
!  p%H%col = (/ 1, 2, 2, 3, 1 /)                           ! NB lower triangular
!  p%H%ptr = (/ 1, 2, 3, 5, 6 /)                           ! Set row pointers
!  p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)  ! Jacobian A
!  p%A%col = (/ 1, 2, 2, 3, 4 /)
!  p%A%ptr = (/ 1, 3, 6 /)                                 ! Set row pointers  
! dense storage format
!  CALL SMT_put( p%H%type, 'DENSE', s )  ! Specify dense
!  CALL SMT_put( p%A%type, 'DENSE', s )  ! storage for H and A
!  ALLOCATE( p%H%val( n * ( n + 1 ) / 2) )
!  ALLOCATE( p%A%val( n * m ) )
!  p%H%val = (/ 1.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, -1.0_wp, 5.0_wp,              &
!               4.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /)          ! Hessian
!  p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp,               &
!               1.0_wp, 1.0_wp /)                          ! Jacobian
! arrays complete
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%n = n ; p%m = m ; p%f = 1.0_wp           ! dimensions & objective constant
   p%Hessian_kind = - 1 ; p%gradient_kind = - 1 ! generic quadratic program
   p%G = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! objective gradient
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, 1.0_wp, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 1.0_wp, 2.0_wp /)         ! variable upper bound
   CALL QPP_initialize( map, control )          ! Initialize control parameters
   control%infinity = infinity                   ! Set infinity
! reorder problem
   CALL QPP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
   IF ( info%status /= 0 ) & !  Error returns
     WRITE( 6, "( ' QPP_solve exit status = ', I6 ) " ) info%status
   WRITE( 6, "( ' problem now involves ', I1, ' variables and ',               &
  &       I1, ' constraints. f is now', ES12.4 )" ) p%n, p%m, p%f
! re-ordered variables
   WRITE( 6, "( /, 5X, 'i', 6x, 'v', 11X, 'l', 11X, 'u', 11X, 'z', 11X,        &
  &            'g', 6X, 'type' )" )
   DO i = 1, d%x_free                      ! free variables
     WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), '  '
   END DO
   DO i = d%x_free + 1, d%x_l_start - 1    ! non-negativities
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), '0< '
   END DO
   DO i = d%x_l_start, d%x_u_start - 1     ! lower-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), 'l< '
   END DO
   DO i = d%x_u_start, d%x_l_end           ! range-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), 'l<u'
   END DO
   DO i = d%x_l_end + 1, d%x_u_end         ! upper-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), ' <u'
   END DO
   DO i = d%x_u_end + 1, p%n                ! non-positivities
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), p%G( i ), ' <0'
   END DO
! re-ordered constraints
   WRITE( 6, "( /, 5X,'i', 5x, 'A*v', 10X, 'l', 11X, 'u', 11X, 'y',            &
  &                6X, 'type' )" )
   DO i = 1, d%c_l_start - 1                ! equality constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l=u'
   END DO
   DO i = d%c_l_start, d%c_u_start - 1      ! lower-bounded constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l< '
   END DO
   DO i = d%c_u_start, d%c_l_end            ! range-bounded constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l<u'
   END DO
   DO i = d%c_l_end + 1, d%c_u_end          ! upper-bounded constraints
     WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), ' <u'
   END DO
! re-ordered matrices
   WRITE( 6, 30 ) 'Hessian ', ( ( 'H', i, p%H%col( j ), p%H%val( j ),          &
     j = p%H%ptr( i ), p%H%ptr( i + 1 ) - 1 ), i = 1, p%n ) ! Hessian
   WRITE( 6, 30 ) 'Jacobian', ( ( 'A', i, p%A%col( j ), p%A%val( j ),          &
     j = p%A%ptr( i ), p%A%ptr( i + 1 ) - 1 ), i = 1, p%m )  ! Jacobian
   p%X( : 3 ) = (/ 1.6_wp, 0.2_wp, -0.6_wp /)
   CALL QPP_get_values( map, info, p, X_val = X_orig )
   WRITE( 6, "( /, ' solution = ', ( 4ES12.4 ) )" ) X_orig( : n )
! recover constraint bounds
   CALL QPP_restore( map, info, p, get_c_bounds = .TRUE. )
! change upper bound
   p%C_u( 1 ) = 3.0_wp
! reorder new problem
   CALL QPP_apply( map, info, p, get_c_bounds = .TRUE. ) 
! re-ordered new constraints
   WRITE( 6, "( /, 5X,'i', 5x, 'A*v', 10X, 'l', 11X, 'u', 11X, 'y',            &
  &                6X, 'type' )" )
   DO i = 1, d%c_l_start - 1                ! equality constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l=u'
   END DO
   DO i = d%c_l_start, d%c_u_start - 1      ! lower-bounded constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l< '
   END DO
   DO i = d%c_u_start, d%c_l_end            ! range-bounded constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l<u'
   END DO
   DO i = d%c_l_end + 1, d%c_u_end          ! upper-bounded constraints
    WRITE( 6, 20 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), ' <u'
   END DO
   CALL QPP_terminate( map, control, info ) !  delete internal workspace
10 FORMAT( I6, 5ES12.4, 2X, A3 )
20 FORMAT( I6, 4ES12.4, 2X, A3 )
30 FORMAT( /, 1X, A8, /, ( :, 3 ( 1X, A1, '(', 2I2, ') =', ES12.4, : ) ) )
   END PROGRAM GALAHAD_QPP_EXAMPLE

