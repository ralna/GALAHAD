! THIS VERSION: GALAHAD 4.1 - 2022-09-07 AT 10:25 GMT.
   PROGRAM GALAHAD_LSP_EXAMPLE
   USE GALAHAD_LSP_double                            ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_dimensions_type ) :: d
   TYPE ( LSP_map_type ) :: map
   TYPE ( LSP_control_type ) :: control
   TYPE ( LSP_inform_type ) :: info
   TYPE ( QPT_problem_type ) :: p
   INTEGER :: i, j, s
   INTEGER, PARAMETER :: coordinate = 1, sparse_by_rows = 2
   INTEGER, PARAMETER :: sparse_by_columns = 3, dense = 4
   INTEGER, PARAMETER :: type = 1
   INTEGER, PARAMETER :: n = 4, m = 2, o = 7, a_ne = 16, l_ne = 5
   REAL ( KIND = wp ) :: X_orig( n )
! sparse co-ordinate storage format
   IF ( type == coordinate ) THEN
     WRITE( 6, "( ' co-ordinate storage' )" )
     CALL SMT_put( p%Ao%type, 'COORDINATE', s )  ! Specify co-ordinate
     CALL SMT_put( p%A%type, 'COORDINATE', s )  ! storage for A and L
     ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( a_ne ) )
     ALLOCATE( p%A%val( l_ne ), p%A%row( l_ne ), p%A%col( l_ne ) )
     p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                             &
                   2.0_wp, 1.0_wp, 1.0_wp, 5.0_wp,                             &
                   3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp,                             &
                   3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp /)  ! Jacobian A_o
     p%Ao%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /)
     p%Ao%col = (/ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 /)
     p%Ao%ne = a_ne
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)  ! Jacobian L
     p%A%row = (/ 1, 1, 2, 2, 2 /)
     p%A%col = (/ 1, 2, 2, 3, 4 /)
     p%A%ne = l_ne
! sparse row-wise storage format
   ELSE IF ( type == sparse_by_rows ) THEN
     WRITE( 6, "( ' sparse by rows storage' )" )
     CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', s )  ! Specify sparse-by-rows
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', s )  ! storage for A and L
     ALLOCATE( p%Ao%val( a_ne ), p%Ao%col( a_ne ), p%Ao%ptr( o + 1 ) )
     ALLOCATE( p%A%val( l_ne ), p%A%col( l_ne ), p%A%ptr( m + 1 ) )
     p%Ao%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 1.0_wp, 3.0_wp,             &
                   1.0_wp, 1.0_wp, 1.0_wp, 4.0_wp, 5.0_wp, 1.0_wp,             &
                   1.0_wp, 6.0_wp, 1.0_wp, 7.0_wp /)          ! Jacobian A_o
     p%Ao%col = (/ 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4 /)
     p%Ao%ptr = (/ 1, 2, 4, 7, 11, 14, 16, 17 /)              ! Set row pointers
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)  ! Jacobian L
     p%A%col = (/ 1, 2, 2, 3, 4 /)
     p%A%ptr = (/ 1, 3, 6 /)                                 ! Set row pointers
! sparse colum-wise storage format
   ELSE IF ( type == sparse_by_columns ) THEN
     WRITE( 6, "( ' sparse by columns storage' )" )
     CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s ) !Specify sparse-by-column
     CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', s ) ! storage for A and L
     ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%ptr( n + 1 ) )
     ALLOCATE( p%A%val( l_ne ), p%A%row( l_ne ), p%A%ptr( n + 1 ) )
     p%Ao%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp, 1.0_wp,             &
                   1.0_wp, 5.0_wp, 3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp,             &
                   4.0_wp, 1.0_wp, 1.0_wp, 7.0_wp /)        ! Jacobian A_o
     p%Ao%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /)
     p%Ao%ptr = (/ 1, 5, 9, 13, 17 /)                       ! Set column pointer
     p%A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian L
     p%A%row = (/ 1, 1, 2, 2, 2 /)
     p%A%ptr = (/ 1, 2, 4, 5, 6 /)                          ! Set column pointer
! dense storage format
   ELSE IF ( type == dense ) THEN
     WRITE( 6, "( ' dense (by rows) storage' )" )
     CALL SMT_put( p%Ao%type, 'DENSE', s )  ! Specify dense (by rows)
     CALL SMT_put( p%A%type, 'DENSE', s )  ! storage for A and L
     ALLOCATE( p%Ao%val( n * o ) )
     ALLOCATE( p%A%val( n * m ) )
     p%Ao%val = (/ 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                             &
                   1.0_wp, 2.0_wp, 0.0_wp, 0.0_wp,                             &
                   1.0_wp, 1.0_wp, 3.0_wp, 0.0_wp,                             &
                   1.0_wp, 1.0_wp, 1.0_wp, 4.0_wp,                             &
                   0.0_wp, 5.0_wp, 1.0_wp, 1.0_wp,                             &
                   0.0_wp, 0.0_wp, 6.0_wp, 1.0_wp,                             &
                   0.0_wp, 0.0_wp, 0.0_wp, 7.0_wp /)         ! Jacobian A_o
     p%A%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp,              &
                  1.0_wp, 1.0_wp /)                          ! Jacobian L
   END IF
! matrices complete, initialize arrays
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   p%n = n ; p%m = m ; p%o = o                ! dimensions
   p%B = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 7.0_wp /) ! obs
   p%C_l = (/ 1.0_wp, 2.0_wp /)               ! constraint lower bound
   p%C_u = (/ 2.0_wp, 2.0_wp /)               ! constraint upper bound
   p%X_l = (/ - 1.0_wp, - infinity, 1.0_wp, - infinity /) ! variable lower bound
   p%X_u = (/ 1.0_wp, infinity, 1.0_wp, 2.0_wp /)         ! variable upper bound
!  WRITE( 6, "( /, 5X,'i', 7X, 'l', 7X, 'u' )" )
!  DO i = 1, p%m
!   WRITE( 6, "( I6, 2ES9.1 )" ) i, p%C_l( i ), p%C_u( i )
!  END DO
   CALL LSP_initialize( map, control )          ! Initialize control parameters
   control%infinity = infinity                  ! Set infinity
! reorder problem
   CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
   IF ( info%status /= 0 ) & !  Error returns
     WRITE( 6, "( ' LSP_solve exit status = ', I6 ) " ) info%status
   WRITE( 6, "( ' reordered problem now involves ', I0, ' variables and ',     &
  &       I0, ' constraints' )" ) p%n, p%m
! re-ordered variables
   WRITE( 6, "( /, 5X, 'i', 6x, 'v', 6X, 'l', 8X, 'u', 8X, 'z', 5X, 'type' )" )
   DO i = 1, d%x_free                      ! free variables
     WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), '  '
   END DO
   DO i = d%x_free + 1, d%x_l_start - 1    ! non-negativities
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), '0< '
   END DO
   DO i = d%x_l_start, d%x_u_start - 1     ! lower-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), 'l< '
   END DO
   DO i = d%x_u_start, d%x_l_end           ! range-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), 'l<u'
   END DO
   DO i = d%x_l_end + 1, d%x_u_end         ! upper-bounded variables
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), ' <u'
   END DO
   DO i = d%x_u_end + 1, p%n                ! non-positivities
    WRITE( 6, 10 ) i, p%X( i ), p%X_l( i ), p%X_u( i ), p%Z( i ), ' <0'
   END DO
! re-ordered constraints
   WRITE( 6, "( /, 5X,'i', 5x, 'L*v', 7X, 'l', 8X, 'u', 8X, 'y', 3X, 'type' )" )
   DO i = 1, d%c_l_start - 1                ! equality constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l=u'
   END DO
   DO i = d%c_l_start, d%c_u_start - 1      ! lower-bounded constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l< '
   END DO
   DO i = d%c_u_start, d%c_l_end            ! range-bounded constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l<u'
   END DO
   DO i = d%c_l_end + 1, d%c_u_end          ! upper-bounded constraints
     WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), ' <u'
   END DO
! re-ordered matrices
   WRITE( 6, "( /, ' Observations B', /, 7ES8.1 )" ) p%B( : p%o )
   WRITE( 6, 20 ) 'Objective Jacobian A', ( ( 'A', i, p%Ao%row( j ),           &
     p%Ao%val( j ), j = p%Ao%ptr( i ), p%Ao%ptr( i + 1 ) - 1 ), i = 1, p%n )
   WRITE( 6, 20 ) 'Constraint Jacobian L', ( ( 'L', i, p%A%col( j ),           &
     p%A%val( j ), j = p%A%ptr( i ), p%A%ptr( i + 1 ) - 1 ), i = 1, p%m )
   p%X( : 3 ) = (/ 1.6_wp, 0.2_wp, -0.6_wp /)
   CALL LSP_get_values( map, info, p, X_val = X_orig )
   WRITE( 6, "( /, ' solution = ', ( 4ES9.1 ) )" ) X_orig( : n )
! recover observations and constraint bounds
   CALL LSP_restore( map, info, p, get_b = .TRUE., get_c_bounds = .TRUE. )

!  WRITE( 6, "( /, 5X,'i', 7X, 'l', 7X, 'u' )" )
!  DO i = 1, p%m
!   WRITE( 6, "( I6, 2ES9.1 )" ) i, p%C_l( i ), p%C_u( i )
!  END DO
!  WRITE( 6, "( /, 5X,'i', 7X, 'b' )" )
!  DO i = 1, p%o
!   WRITE( 6, "( I6, ES9.1 )" ) i, p%B( i )
!  END DO

   WRITE( 6, "( /, ' modified problem now involves ', I0, ' variables and ',   &
  &       I0, ' constraints' )" ) p%n, p%m
! change upper bound
   p%C_u( 1 ) = 3.0_wp
! reorder new problem
   CALL LSP_apply( map, info, p, get_c_bounds = .TRUE. )
! re-ordered new constraints
   WRITE( 6, "( /, 5X,'i', 5x, 'A*v', 7X, 'l', 8X, 'u', 8X, 'y', 3X, 'type' )" )
   DO i = 1, d%c_l_start - 1                ! equality constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l=u'
   END DO
   DO i = d%c_l_start, d%c_u_start - 1      ! lower-bounded constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l< '
   END DO
   DO i = d%c_u_start, d%c_l_end            ! range-bounded constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), 'l<u'
   END DO
   DO i = d%c_l_end + 1, d%c_u_end          ! upper-bounded constraints
    WRITE( 6, 10 ) i, p%C( i ), p%C_l( i ), p%C_u( i ), p%Y( i ), ' <u'
   END DO
   CALL LSP_terminate( map, control, info ) !  delete internal workspace
10 FORMAT( I6, 4ES9.1, 2X, A3 )
20 FORMAT( /, 1X, A, /, ( :, 3 ( 1X, A1, '(', 2I2, ') =', ES8.1, : ) ) )
   END PROGRAM GALAHAD_LSP_EXAMPLE
