! THIS VERSION: GALAHAD 4.1 - 2022-06-03 AT 11:40 GMT
! Used to test components of package
   PROGRAM GALAHAD_SLLS_EXAMPLE3
   USE GALAHAD_SLLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: status, segment, max_steps
   LOGICAL :: advance
   REAL ( KIND = wp ) :: f_opt, t_opt, t_0, t_max, beta, eta, v_j
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: FREE, X_stat
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D, S, AD, AE, R, R_t, DIAG
   TYPE ( SLLS_search_data_type ) :: search_data
   TYPE ( SLLS_subproblem_data_type ) :: subproblem_data
   TYPE ( SLLS_reverse_type ) :: reverse
   REAL ( KIND = wp ) :: scale, scale2, flipflop, x_j, d_j, a, val, f
   INTEGER :: i, j, l, n_free, alloc_status
   CHARACTER ( LEN = 80 ) :: bad_alloc
   INTEGER, PARAMETER :: n = 100
!  INTEGER, PARAMETER :: n = 10
!!    INTEGER, PARAMETER :: n = 4
   INTEGER, PARAMETER :: m = n + 1
   INTEGER, PARAMETER :: a_ne = 2 * n
   ALLOCATE( p%X( n ), p%Z( n ), D( n ), DIAG( n ), FREE( n ), X_stat( n ) )
   ALLOCATE( AD( m ), AE( m ), R( m ), p%B( m ), p%C( m ), p%G( n ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%ptr( n + 1 ) )
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

   CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', status )
   p%m = m ; p%n = n
   p%A%m = m ; p%A%n = n
   DO j = 1, n
     p%A%ptr( j ) = l
     p%A%row( l ) = j ; p%A%val( l ) = REAL( j, KIND = wp )
     l = l + 1
     p%A%row( l ) = j + 1; p%A%val( l ) = 1.0_wp
     l = l + 1
   END DO
   p%A%ptr( n + 1 ) = l

   p%B = 0.0_wp
   DO j = 1, n / 2
     DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
       p%B( p%A%row( l ) ) = p%B( p%A%row( l ) ) + 2.0_wp * p%A%val( l )
     END DO
   END DO
   DO j = n / 2 + 1, n
     DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
       p%B( p%A%row( l ) ) = p%B( p%A%row( l ) ) + p%A%val( l ) / scale
     END DO
   END DO

!  initialise the vectors r = A x - b, A e and A s

    AD = 0.0_wp ; AE = 0.0_wp ; R = - p%B
    DO j = 1, n
      x_j = p%X( j ) ; d_j = D( j )
      DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
        i = p%A%row( l ) ; a = p%A%val( l )
        R( i ) = R( i ) + a * x_j
        AE( i ) = AE( i ) + a
        AD( i ) = AD( i ) + a * d_j
      END DO
    END DO

GO TO 10

!  available A version

   WRITE( 6, "( ' exact search, A given' )" )
   status = 0
   CALL SLLS_exact_arc_search( n, m, 6, .TRUE., .FALSE., '',                   &
                               status, p%X, R, D, AD, AE, segment,             &
                               n_free, FREE, search_data, f_opt, t_opt,        &
                               A_val = p%A%val, A_row = p%A%row,               &
                               A_ptr = p%A%ptr )

   D = 1.0_wp ; D(1) = - 1.0_wp ; D(2) = 0.0_wp

!  initialise the vectors r = A x - b, A e and A s

    AD = 0.0_wp ; AE = 0.0_wp ; R = - p%B
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
      x_j = p%X( j ) ; d_j = D( j )
      DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
        i = p%A%row( l ) ; a = p%A%val( l )
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
    CALL SLLS_exact_arc_search( n, m, 6, .TRUE., .FALSE., '',                  &
                                status, p%X, R, D, AD, AE, segment,            &
                                n_free, FREE, search_data, f_opt, t_opt,       &
                                reverse = reverse )
     IF ( status <= 0 ) EXIT
     reverse%nz_out_end = 0
     DO l = p%A%ptr( status ), p%A%ptr( status + 1 ) - 1
       reverse%nz_out_end = reverse%nz_out_end + 1
       reverse%NZ_out( reverse%nz_out_end ) = p%A%row( l )
       reverse%P( reverse%nz_out_end ) = p%A%val( l )
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

    R = - p%B
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
      x_j = p%X( j )
      DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
        i = p%A%row( l ) ; a = p%A%val( l )
        R( i ) = R( i ) + a * x_j
      END DO
    END DO

!  available A version

   WRITE( 6, "( ' inexact search, A given' )" )
   CALL SLLS_inexact_arc_search( n, m, 6, .TRUE., .FALSE., '',                 &
                                 status, p%X, R, D, S, R_t, t_0, t_max,        &
                                 beta, eta, max_steps, advance,                &
                                 n_free, FREE, search_data, f_opt, t_opt,      &
                                 A_val = p%A%val, A_row = p%A%row,             &
                                 A_ptr = p%A%ptr )

    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
    END DO

!  reverse communication version

   WRITE( 6, "( ' inexact search, reverse A' )" )
   ALLOCATE( reverse%V( n ) )
   status = 0
   DO
     CALL SLLS_inexact_arc_search( n, m, 6, .TRUE., .FALSE., '',               &
                                   status, p%X, R, D, S, R_t, t_0, t_max,      &
                                   beta, eta, max_steps, advance,              &
                                   n_free, FREE, search_data, f_opt, t_opt,    &
                                   reverse = reverse )
     IF ( status <= 0 ) EXIT
     DO j = 1, n
       v_j = reverse%V( j )
       DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
         i = p%A%row( l )
         reverse%P( i ) = reverse%P( i ) + p%A%val( l ) * v_j
       END DO
     END DO
   END DO

 10 CONTINUE
!   GO TO 20

    R = - p%B
    DO j = 1, n
      x_j = 1.0_wp / scale
      p%X( j ) = x_j
      val = 0.0_wp
      DO l = p%A%ptr( j ), p%A%ptr( j + 1 ) - 1
        i = p%A%row( l ) ; a = p%A%val( l )
        R( i ) = R( i ) + a * x_j
        val = val + a ** 2
      END DO
      DIAG( j ) = val
    END DO

   control%stop_cg_relative = 1.0D-14
   control%stop_cg_absolute = EPSILON( 1.0_wp )
   control%cg_maxit = n
!  n_free = 5 
!  FREE( : n_free ) = (/ 46, 47, 48, 49, 50 /)
   n_free = n / 2
   FREE( : n_free ) = (/ ( j, j = 1, n/2 ) /)

  CALL SLLS_cgls( m, n, n_free, 0.0_wp, 6, .TRUE., .FALSE., '', f, p%X, R,     &
                  FREE, control%stop_cg_relative, control%stop_cg_absolute,    &
                  i, control%cg_maxit, subproblem_data, userdata, status,      &
                  alloc_status, bad_alloc,                                     &
                  A_ptr = p%A%ptr, A_row = p%A%row, A_val = p%A%val,           &
                  DPREC = DIAG, preconditioned = .TRUE. )
   GO TO 30

!  solve the problem

 20 CONTINUE
    DO j = 1, n
      p%X( j ) = 1.0_wp / scale
    END DO

   CALL SLLS_initialize( data, control, inform )
   control%print_level = 1
   control%maxit = 10
   control%exact_arc_search = .FALSE.
   control%direct_subproblem_solve = .FALSE.
   control%preconditioner = 1
   control%stop_cg_relative = 1.0D-14
   control%stop_cg_absolute = EPSILON( 1.0_wp )

   CALL SLLS_solve( p, X_stat, data, control, inform, userdata )
   IF ( inform%status == 0 ) THEN             !  Successful return
      WRITE( 6, "( /, ' SLLS: ', I0, ' iterations  ', /,                       &
     &     ' Optimal objective value =', ES16.8 )" ) inform%iter, inform%obj
!     WRITE( 6, "( ' Optimal solution = ', /, ( 5ES12.4 ) )" ) p%X
    ELSE                                       ! Error returns
      WRITE( 6, "( /, ' SLLS_solve exit status = ', I0 ) " ) inform%status
!     WRITE( 6, * ) inform%alloc_status, inform%bad_alloc
    END IF
   CALL SLLS_terminate( data, control, inform )  !  delete workspace

30 CONTINUE
   DEALLOCATE( p%B, p%X, p%Z, p%C, p%G, X_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%ptr, p%A%type )
   DEALLOCATE( D, DIAG, FREE, AD, AE, r )

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
