   PROGRAM GALAHAD_SNLS_EXAMPLE3  ! GALAHAD 5.5 - 2026-03-07 AT 15:00 GMT
   USE GALAHAD_SNLS_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   EXTERNAL :: EVALR, EVALJr_prod, EVALJr_scol, EVALJr_sprod
   INTEGER :: s
   INTEGER, PARAMETER :: n = 5, m_r = 4, m_c = 2
   REAL ( KIND = wp ), PARAMETER :: p = 4.0_wp ! parameter p
! start problem data
   nlp%n = n ; nlp%m_r = m_r ; nlp%m_c = m_c ! dimensions
   ALLOCATE( nlp%COHORT( n ), nlp%X( n ) )
   nlp%COHORT = [ 1, 2, 0, 1, 2 ]
   nlp%X = [ 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp, 0.5_wp ]
   ALLOCATE( userdata%real( 1 ), userdata%integer( 2 ) ) ! space for parameters
   userdata%real( 1 ) = p                        ! record parameter, p
   userdata%integer( 1 ) = n                     ! record parameter, n
   userdata%integer( 2 ) = m_r                   ! record parameter, m_r
! problem data complete ; solve using a Gauss-Newton model
   CALL SNLS_initialize( data, control, inform ) ! initialize control params
   control%jacobian_available = 1                ! jacobian by products
   control%print_level = 1
   control%print_obj = .TRUE.
   control%subproblem_solver = 2 ! use internal slls (1 for sllsb)
!  control%SLLS_control%print_level = 1
   control%SLLS_control%SBLS_control%definite_linear_solver = 'potr '
   control%SLLS_control%SBLS_control%symmetric_linear_solver = 'sytr '
   inform%status = 1                             ! set for initial entry
   CALL SNLS_solve( nlp, control, inform, data, userdata,                      &
                    eval_R = EVALR, eval_Jr_prod = EVALJr_prod,                &
                    eval_Jr_scol = EVALJr_scol, eval_Jr_sprod = EVALJr_sprod )
   IF ( inform%status == 0 ) THEN                ! successful return
     WRITE( 6, "( ' SNLS: ', I0, ' iterations -',                              &
    &     ' optimal objective value =',                                        &
    &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )                &
     inform%iter, inform%obj, nlp%X
   ELSE                                          ! Error returns
     WRITE( 6, "( ' SNLS_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL SNLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%R, nlp%COHORT )
   DEALLOCATE( userdata%real, userdata%integer )
   END PROGRAM GALAHAD_SNLS_EXAMPLE3

   SUBROUTINE EVALR( status, X, userdata, R ) ! residual
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
   REAL ( KIND = wp ), DIMENSION( : ),INTENT( OUT ) :: R
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp ) :: p
   p = userdata%real( 1 )
   R( 1 ) = X( 1 ) * X( 2 ) - p
   R( 2 ) = X( 2 ) * X( 3 ) - 1.0_wp
   R( 3 ) = X( 3 ) * X( 4 ) - 1.0_wp
   R( 4 ) = X( 4 ) * X( 5 ) - 1.0_wp
   status = 0
   RETURN
   END SUBROUTINE EVALR

   SUBROUTINE EVALJr_prod( status, X, userdata, transpose, V, P, got_jr )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
   IF ( transpose ) THEN
     P( 1 ) = X( 2 ) * V( 1 )
     P( 2 ) = X( 3 ) * V( 2 ) + X( 1 ) * V( 1 )
     P( 3 ) = X( 4 ) * V( 3 ) + X( 2 ) * V( 2 )
     P( 4 ) = X( 5 ) * V( 4 ) + X( 3 ) * V( 3 )
     P( 5 ) = X( 4 ) * V( 4 )
   ELSE
     P( 1 ) = X( 2 ) * V( 1 ) + X( 1 ) * V( 2 )
     P( 2 ) = X( 3 ) * V( 2 ) + X( 2 ) * V( 3 )
     P( 3 ) = X( 4 ) * V( 3 ) + X( 3 ) * V( 4 )
     P( 4 ) = X( 5 ) * V( 4 ) + X( 4 ) * V( 5 )
   END IF
   status = 0
   RETURN
   END SUBROUTINE EVALJr_prod

   SUBROUTINE EVALJr_scol( status, X, userdata, index, VAL, ROW, nz, got_jr )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   INTEGER, INTENT( IN ) :: index
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: VAL
   INTEGER, DIMENSION( : ), INTENT( INOUT ) :: ROW
   INTEGER, INTENT( INOUT ) :: nz
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
   INTEGER :: n
   n = userdata%integer( 1 ) 
   IF ( index == 1 ) THEN
     VAL( 1 ) = X( 2 )
     ROW( 1 ) = 1
     nz = 1
   ELSE IF ( index == n ) THEN
     VAL( 1 ) = X( n - 1 )
     ROW( 1 ) = n - 1
     nz = 1
   ELSE
     VAL( 1 ) = X( index - 1 )
     ROW( 1 ) = index - 1
     VAL( 2 ) = X( index + 1 )
     ROW( 2 ) = index
     nz = 2
   END IF
   status = 0
   RETURN
   END SUBROUTINE EVALJr_scol

   SUBROUTINE EVALJr_sprod( status, X, userdata, transpose, V, P, FREE,        &
                            n_free, got_jr )
   USE GALAHAD_USERDATA_double
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, INTENT( OUT ) :: status
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
   INTEGER, INTENT( IN ) :: n_free
   LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
   INTEGER :: i, j, n, m_r
   REAL ( KIND = wp ) :: val
   n = userdata%integer( 1 ) 
   m_r = userdata%integer( 2 )
   IF ( transpose ) THEN
     DO i = 1, n_free
       j = FREE( i )
       IF ( j == 1 ) THEN
         P( 1 ) = X( 2 ) * V( 1 )
       ELSE IF ( j == n ) THEN
         P( n ) = X( m_r ) * V( m_r )
       ELSE
         P( j ) = X( j - 1 ) * V( j - 1 ) + X( j + 1 ) * V( j )
       END IF
     END DO
   ELSE
     P( : m_r ) = 0.0_wp
     DO i = 1, n_free
       j = FREE( i )
       val = V( j )
       IF ( j == 1 ) THEN
         P( 1 ) = P( 1 ) + X( 2 ) * val
       ELSE IF ( j == n ) THEN
         P( m_r ) = P( m_r ) + X( m_r ) * val
       ELSE
         P( j - 1 ) = P( j - 1 ) + X( j - 1 ) * val 
         P( j ) = P( j ) + X( j + 1 ) * val 
       END IF
     END DO
   END IF
   status = 0
   RETURN
   END SUBROUTINE EVALJr_sprod

