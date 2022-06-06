! THIS VERSION: GALAHAD 4.1 - 2022-06-03 AT 11:40 GMT
! Used to test components of package
   PROGRAM GALAHAD_SLLS_EXAMPLE3
   USE GALAHAD_SLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
!  TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: status, segment, max_steps
   LOGICAL :: advance
   REAL ( KIND = wp ) :: f_opt, t_opt, t_0, t_max, beta, eta, v_j
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status, FREE
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D, S, AD, AE, R, B, R_t
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr
   TYPE ( SLLS_search_data_type ) :: data
   TYPE ( SLLS_reverse_type ) :: reverse
   REAL ( KIND = wp ) :: scale, scale2, flipflop, x_j, d_j, a
   INTEGER :: i, j, l, n_free
!  INTEGER, PARAMETER :: n = 100
   INTEGER, PARAMETER :: n = 10
!!    INTEGER, PARAMETER :: n = 4
   INTEGER, PARAMETER :: m = n + 1
   INTEGER, PARAMETER :: a_ne = 2 * n
   ALLOCATE( p%X( n ), D( n ), FREE( n ), X_status( n ), AD( m ), AE( m )
   ALLOCATE( R( m ) ), B( m ), A_val( a_ne ), A_row( a_ne ), A_ptr( n + 1 ) )
!  p%X = (/ 0.01_wp, 0.01_wp, 0.05_wp, 0.93_wp /)
!  D = (/ -10.0_wp, - 1.0_wp, -10.0_wp, 21.0_wp /)
   scale = REAL( n, KIND = wp )
   scale2 = REAL( n * ( n + 1 ) / 2, KIND = wp )
   flipflop = 1.0_wp
   DO j = 1, n
     p%X( j ) = 1.0_wp / scale
     D( j ) = flipflop * REAL( j, KIND = wp ) / scale2
     flipflop = - 1.0_wp * flipflop
   END DO
!  p%X = 0.0_wp
!  p%X( n-1 ) = 1.0_wp
!  CALL RANDOM_NUMBER( D )
!  D = 2.0_wp * D - 1.0_wp
   D = 1.0_wp ;  D(1) = - 1.0_wp ;  D(2) = 0.0_wp
   l = 1
   DO j = 1, n
     A_ptr( j ) = l
     A_row( l ) = j ; A_val( l ) = REAL( j, KIND = wp )
     l = l + 1
     A_row( l ) = j + 1; A_val( l ) = 1.0_wp
     l = l + 1
   END DO
   A_ptr( n + 1 ) = l

   B = 0.0_wp
   DO j = 1, n
     DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
       B( A_row( l ) ) = B( A_row( l ) ) + 2.0_wp * A_val( l )
     END DO
   END DO

!  initialise the vectors r = A x - b, A e and A s

    AD = 0.0_wp ; AE = 0.0_wp ; R = - B
    DO j = 1, n
      x_j = p%X( j ) ; d_j = D( j )
      DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
        i = A_row( l ) ; a = A_val( l )
        R( i ) = R( i ) + a * x_j
        AE( i ) = AE( i ) + a
        AD( i ) = AD( i ) + a * d_j
      END DO
    END DO

!  available A version

   WRITE( 6, "( ' exact search, A given' )" )
   status = 0
   CALL SLLS_exact_arc_search( n, m, 6, .TRUE., .FALSE.,                       &
                               status, p%X, R, D, AD, AE, segment,             &
                               n_free, FREE, X_status, data, f_opt, t_opt,     &
                               A_val = A_val, A_row = A_row, A_ptr = A_ptr )


   D = 1.0_wp ; D(1) = - 1.0_wp ; D(2) = 0.0_wp

!  initialise the vectors r = A x - b, A e and A s

    AD = 0.0_wp ; AE = 0.0_wp ; R = - B
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
      x_j = p%X( j ) ; d_j = D( j )
      DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
        i = A_row( l ) ; a = A_val( l )
        R( i ) = R( i ) + a * x_j
        AE( i ) = AE( i ) + a
        AD( i ) = AD( i ) + a * d_j
      END DO
    END DO

!  reverse communication version

   WRITE( 6, "( ' exact search, reverse A' )" )
   ALLOCATE( reverse%NZ_out( m ), reverse%P( m ) )
   status = 0
   DO
    CALL SLLS_exact_arc_search( n, m, 6, .TRUE., .FALSE.,                      &
                                 status, p%X, R, D, AD, AE, segment,           &
                                 n_free, FREE, X_status, data, f_opt, t_opt,   &
                                 reverse = reverse )
     IF ( status <= 0 ) EXIT
     reverse%nz_out_end = 0
     DO l = A_ptr( status ), A_ptr( status + 1 ) - 1
       reverse%nz_out_end = reverse%nz_out_end + 1
       reverse%NZ_out( reverse%nz_out_end ) = A_row( l )
       reverse%P( reverse%nz_out_end ) = A_val( l )
     END DO
   END DO

!  test the inexact search

   t_0 = 1.0_wp
   t_max = 1000.0_wp
   beta = 0.5_wp
   eta = 0.1_wp
   advance = .TRUE.
   status = 0

   ALLOCATE( S( n ), R_t( m ) )
   D = 1.0_wp ; D(1) = - 1.0_wp ; D(2) = 0.0_wp

!  initialise the vector r = A x - b

    R = - B
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
      x_j = p%X( j )
      DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
        i = A_row( l ) ; a = A_val( l )
        R( i ) = R( i ) + a * x_j
      END DO
    END DO

!  available A version

   WRITE( 6, "( ' inexact search, A given' )" )
   CALL SLLS_inexact_arc_search( n, m, 6, .TRUE., .FALSE.,                     &
                                 status, p%X, R, D, S, R_t, t_0, t_max,        &
                                 beta, eta, max_steps, advance,                &
                                 n_free, FREE, data, f_opt, t_opt,             &
                                 A_val = A_val, A_row = A_row, A_ptr = A_ptr )

!  initialise the vector r = A x - b

    R = - B
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
      x_j = p%X( j )
      DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
        i = A_row( l ) ; a = A_val( l )
        R( i ) = R( i ) + a * x_j
      END DO
    END DO

!  reverse communication version

   WRITE( 6, "( ' inexact search, reverse A' )" )
   ALLOCATE( reverse%V( n ) )
   status = 0
   DO
     CALL SLLS_inexact_arc_search( n, m, 6, .TRUE., .FALSE.,                   &
                                   status, p%X, R, D, S, R_t, t_0, t_max,      &
                                   beta, eta, max_steps, advance,              &
                                   n_free, FREE, data, f_opt, t_opt,           &
                                   reverse = reverse )
     IF ( status <= 0 ) EXIT
     DO j = 1, n
       v_j = reverse%V( j )
       DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
         i = A_row( l )
         reverse%P( i ) = reverse%P( i ) + A_val( l ) * v_j
       END DO
     END DO
   END DO

!  CALL SLLS_simplex_projection_path( n, p%X, D, status )

!   scale = REAL( n * ( n + 1 ) / 2, KIND = wp )
!   DO j = 1, n
!     p%X( j ) = - REAL( j, KIND = wp ) / scale
!   END DO
!   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
!   WRITE( 6, "( ' status = ', I0 )" ) status
!   WRITE( 6, "( 8X, '        x          p(x)' )" )
!   DO j = 1, n
!     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
!   END DO
!   scale = REAL( n * ( n + 1 ) / 2, KIND = wp )
!   flipflop = 1.0_wp
!   DO j = 1, n
!     p%X( j ) = flipflop * REAL( j, KIND = wp ) / scale
!     flipflop = - 1.0_wp * flipflop
!   END DO
!   CALL SLLS_project_onto_simplex( n, p%X, X_proj, status )
!   WRITE( 6, "( ' status = ', I0 )" ) status
!   WRITE( 6, "( 8X, '        x          p(x)' )" )
!   DO j = 1, n
!     WRITE(6, "( I8, 2ES12.4 )" ) j, p%X( j ), X_proj( j )
!   END DO
!   DEALLOCATE( p%X, X_proj )
   END PROGRAM GALAHAD_SLLS_EXAMPLE3
