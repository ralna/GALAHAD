   PROGRAM GALAHAD_SNLS_EXAMPLE4 !  GALAHAD 5.5 - 2026-02-22 AT 14:30 GMT
   USE GALAHAD_SNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   TYPE ( REVERSE_type ) :: reverse
   INTEGER :: i, j, s
   INTEGER, PARAMETER :: n = 5, m_r = 4, m_c = 2
   REAL ( KIND = wp ) :: val
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp  ! parameter p
! start problem data
   nlp%n = n ; nlp%m_r = m_r ; nlp%m_c = m_c ! dimensions
   ALLOCATE( nlp%COHORT( n ), nlp%X( n ) )
   nlp%COHORT = [ 1, 2, 0, 1, 2 ]
   nlp%X = [ 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp ]
   ALLOCATE( userdata%real( 1 ), userdata%integer( 2 ) ) ! space for parameters
   userdata%real( 1 ) = p                        ! record parameter, p
   userdata%integer( 1 ) = n                     ! record parameter, n
   userdata%integer( 2 ) = m_r                   ! record parameter, o
! problem data complete ; solve using a Gauss-Newton model
   CALL SNLS_initialize( data, control, inform ) ! initialize control params
   control%jacobian_available = 1                ! jacobian by products
   control%print_level = 1
   control%print_obj = .TRUE.
   control%subproblem_solver = 2 ! use internal slls
!  control%SLLS_control%print_level = 1
   control%SLLS_control%SBLS_control%definite_linear_solver = 'potr '
   control%SLLS_control%SBLS_control%symmetric_linear_solver = 'sytr '
   inform%status = 1                             ! set for initial entry
   DO
     CALL SNLS_solve( nlp, control, inform, data, userdata, reverse = reverse )
     SELECT CASE( inform%status )
     CASE ( 0 ) ! successful return
       WRITE( 6, "( ' SNLS: ', I0, ' iterations -',                            &
      &     ' optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
       inform%iter, inform%obj, nlp%X
       EXIT
     CASE( 2 ) ! evaluate residual
       nlp%R( 1 ) = nlp%X( 1 ) * nlp%X( 2 ) - p
       nlp%R( 2 ) = nlp%X( 2 ) * nlp%X( 3 ) - 1.0_wp
       nlp%R( 3 ) = nlp%X( 3 ) * nlp%X( 4 ) - 1.0_wp
       nlp%R( 4 ) = nlp%X( 4 ) * nlp%X( 5 ) - 1.0_wp
       reverse%eval_status = 0
     CASE( 4 ) ! evaluate Jr(x) * v
       reverse%P( 1 )                                                          &
         = nlp%X( 2 ) * reverse%V( 1 ) + nlp%X( 1 ) * reverse%V( 2 )
       reverse%P( 2 )                                                          &
         = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 2 ) * reverse%V( 3 )
       reverse%P( 3 )                                                          &
         = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 3 ) * reverse%V( 4 )
       reverse%P( 4 )                                                          &
         = nlp%X( 5 ) * reverse%V( 4 ) + nlp%X( 4 ) * reverse%V( 5 )
       reverse%eval_status = 0
     CASE( 5 ) ! evaluate Jr^T(x) * v
       reverse%P( 1 ) = nlp%X( 2 ) * reverse%V( 1 )
       reverse%P( 2 )                                                          &
         = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 1 ) * reverse%V( 1 )
       reverse%P( 3 )                                                          &
         = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 2 ) * reverse%V( 2 )
       reverse%P( 4 )                                                          &
         = nlp%X( 5 ) * reverse%V( 4 ) + nlp%X( 3 ) * reverse%V( 3 )
       reverse%P( 5 ) = nlp%X( 4 ) * reverse%V( 4 )
       reverse%eval_status = 0
     CASE( 6 ) ! evaluate column index of J(x)
       IF ( reverse%index == 1 ) THEN
         reverse%P( 1 ) = nlp%X( 2 )
         reverse%IP( 1 ) = 1
         reverse%lp = 1
       ELSE IF ( reverse%index == n ) THEN
         reverse%P( 1 ) = nlp%X( n - 1 )
         reverse%IP( 1 ) = n - 1
         reverse%lp = 1
       ELSE
         reverse%P( 1 ) = nlp%X( reverse%index - 1 )
         reverse%IP( 1 ) = reverse%index - 1
         reverse%P( 2 ) = nlp%X( reverse%index + 1 )
         reverse%IP( 2 ) = reverse%index
         reverse%lp = 2
       END IF
       reverse%eval_status = 0
     CASE( 7 ) ! evaluate Jr(x) * v for sparse v
       reverse%P( : m_r ) = 0.0_wp
       DO i = reverse%lvl, reverse%lvu
         j = reverse%IV( i )
         val = reverse%V( j )
         IF ( j == 1 ) THEN
           reverse%P( 1 ) = reverse%P( 1 ) + nlp%X( 2 ) * val
         ELSE IF ( j == n ) THEN
           reverse%P( m_r ) = reverse%P( m_r ) + nlp%X( m_r ) * val
         ELSE
           reverse%P( j - 1 ) = reverse%P( j - 1 ) + nlp%X( j - 1 ) * val 
           reverse%P( j ) = reverse%P( j ) + nlp%X( j + 1 ) * val 
         END IF
       END DO
       reverse%eval_status = 0
     CASE( 8 ) ! evaluate sparse Jr^T(x) * v 
       DO i = reverse%lvl, reverse%lvu
         j = reverse%IV( i )
         IF ( j == 1 ) THEN
           reverse%P( 1 ) = nlp%X( 2 ) * reverse%V( 1 )
         ELSE IF ( j == n ) THEN
           reverse%P( n ) = nlp%X( m_r ) * reverse%V( m_r )
         ELSE
           reverse%P( j ) = nlp%X( j - 1 ) * reverse%V( j - 1 )                &
                            + nlp%X( j + 1 ) * reverse%V( j )
         END IF
       END DO
       reverse%eval_status = 0
     CASE DEFAULT ! error returns
       WRITE( 6, "( ' SNLS_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END SELECT
   END DO
   DEALLOCATE( nlp%X, nlp%G, nlp%R, nlp%COHORT )
   DEALLOCATE( userdata%real, userdata%integer )
   END PROGRAM GALAHAD_SNLS_EXAMPLE4

