! THIS VERSION: GALAHAD 5.5 - 2026-02-23 AT 15:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SNLS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SNLS_precision
   IMPLICIT NONE
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_full_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
!  EXTERNAL :: EVALR, EVALJr, EVALJr_prod, EVALJr_scol, EVALJr_sprod
   INTEGER ( KIND = ip_ ) :: i, j, mnn, solver, status, eval_status
   INTEGER, PARAMETER :: n = 5, m_r = 4, m_c = 2, Jr_ne = 8
   REAL ( KIND = rp_ ) :: val
   REAL ( KIND = rp_ ), PARAMETER :: p_val = 4.0_rp_
   CHARACTER ( LEN = 2 ) :: c_solver
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Y, Z, R, G, W
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Jr_row, Jr_col, Jr_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Jr_val
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat, COHORT
   INTEGER ( KIND = ip_ ) :: lvl, lvu, lp, index
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IV, IP
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V, P

   WRITE( 6, "( /, ' SNLS - test of interface modes', / )" )

   ALLOCATE( X( n ), Y( m_c ), Z( n ), G( n ), R( m_r ), W( m_r ) )
   ALLOCATE( X_stat( n ), COHORT( n ) )
   COHORT = [ 1, 2, 0, 1, 2 ] ; W = 1.0_rp_
   ALLOCATE( userdata%real( 1 ), userdata%integer( 2 ) )
   userdata%real( 1 ) = p_val
   userdata%integer( 1 ) = n ; userdata%integer( 2 ) = m_r

!  DO solver = 2, 2
   DO solver = 1, 4
     CALL SNLS_initialize( data, control, inform )
!    control%print_level = 1
!    control%SLLS_control%print_level = 1
!    control%maxit = 1
!    control%SLLS_control%maxit = 5
     control%stop_pg_absolute = 0.00001_rp_
     CALL WHICH_sls( control )

     X = [ 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_ ]
     SELECT CASE( solver )
     CASE( 1 ) ! jacobian is available via function calls
       c_solver = 'JF'
       ALLOCATE( Jr_val( jr_ne ), Jr_row( jr_ne ), Jr_col( jr_ne ) )
       Jr_row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
       Jr_col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
       control%jacobian_available = 2                
       control%subproblem_solver = 1
       CALL SNLS_import( control, data, status, n, m_r, m_c, 'coordinate',     &
                         Jr_ne, Jr_row, Jr_col, Jr_ptr, COHORT )
       CALL SNLS_solve_with_jac( data, userdata, status, X, Y, Z, R, G,        &
                                 X_stat, EVALR, EVALJr, W )

     CASE( 2 )  ! jacobian products are available via function calls
       c_solver = 'PF'
       control%jacobian_available = 1
       control%subproblem_solver = 2
       CALL SNLS_import_without_jac( control, data, status, n, m_r, m_c,       &
                                     COHORT )
       CALL SNLS_solve_with_jacprod( data, userdata, status, X, Y, Z, R, G,    &
                                     X_stat, EVALR, EVALJr_PROD,               &
                                     EVALJR_SCOL, EVALJr_SPROD, W )

     CASE( 3 )  ! jacobian is available via reverse communication
       c_solver = 'JR'
       ALLOCATE( Jr_val( jr_ne ), Jr_row( jr_ne ), Jr_col( jr_ne ) )
       Jr_row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
       Jr_col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
       control%jacobian_available = 2                
       control%subproblem_solver = 1
       CALL SNLS_import( control, data, status, n, m_r, m_c, 'coordinate',     &
                         Jr_ne, Jr_row, Jr_col, Jr_ptr, COHORT )
       DO
         CALL SNLS_solve_reverse_with_jac( data, status, eval_status, X, Y,    &
                                           Z, R, G, X_stat, Jr_val, W )
         SELECT CASE( status )
         CASE ( 0 ) ! successful return
           EXIT
         CASE( 2 ) ! evaluate residual
           R( 1 ) = X( 1 ) * X( 2 ) - p_val
           R( 2 ) = X( 2 ) * X( 3 ) - 1.0_rp_
           R( 3 ) = X( 3 ) * X( 4 ) - 1.0_rp_
           R( 4 ) = X( 4 ) * X( 5 ) - 1.0_rp_
           eval_status = 0
         CASE( 3 ) ! evaluate Jacobian
           Jr_val( 1 ) = X( 2 )
           Jr_val( 2 ) = X( 1 )
           Jr_val( 3 ) = X( 3 )
           Jr_val( 4 ) = X( 2 )
           Jr_val( 5 ) = X( 4 )
           Jr_val( 6 ) = X( 3 )
           Jr_val( 7 ) = X( 5 )
           Jr_val( 8 ) = X( 4 )
           eval_status = 0
         CASE DEFAULT ! error returns
           EXIT
         END SELECT
       END DO

     CASE( 4 ) ! jacobian products are available via reverse communication
       c_solver = 'PR'
       mnn = MAX( n, m_r )
       ALLOCATE( IV( mnn ), IP( m_r ), V( mnn ), P( mnn ) )
       control%jacobian_available = 1
       control%subproblem_solver = 2
       CALL SNLS_import_without_jac( control, data, status, n, m_r, m_c,       &
                                     COHORT )
       DO
         CALL SNLS_solve_reverse_with_jacprod( data, status, eval_status,      &
                                               X, Y, Z, R, G, X_stat,          &
                                               V, IV, lvl, lvu, index,         &
                                               P, IP, lp, W )
         SELECT CASE( status )
         CASE ( 0 ) ! successful return
           EXIT
         CASE( 2 ) ! evaluate residual
           R( 1 ) = X( 1 ) * X( 2 ) - p_val
           R( 2 ) = X( 2 ) * X( 3 ) - 1.0_rp_
           R( 3 ) = X( 3 ) * X( 4 ) - 1.0_rp_
           R( 4 ) = X( 4 ) * X( 5 ) - 1.0_rp_
!write(6,"( ' r : ', 4F6.2 )" ) R( : 4 )
           eval_status = 0
         CASE( 4 ) ! evaluate Jr(x) * v
           P( 1 )  = X( 2 ) * V( 1 ) + X( 1 ) * V( 2 )
           P( 2 )  = X( 3 ) * V( 2 ) + X( 2 ) * V( 3 )
           P( 3 ) = X( 4 ) * V( 3 ) + X( 3 ) * V( 4 )
           P( 4 ) = X( 5 ) * V( 4 ) + X( 4 ) * V( 5 )
           eval_status = 0
!write(6,"( ' p : ', 5F6.2 )" ) P( : 4 )
         CASE( 5 ) ! evaluate Jr^T(x) * v
           P( 1 ) = X( 2 ) * V( 1 )
           P( 2 ) = X( 3 ) * V( 2 ) + X( 1 ) * V( 1 )
           P( 3 ) = X( 4 ) * V( 3 ) + X( 2 ) * V( 2 )
           P( 4 ) = X( 5 ) * V( 4 ) + X( 3 ) * V( 3 )
           P( 5 ) = X( 4 ) * V( 4 )
           eval_status = 0
!write(6,"( ' pt: ', 5F6.2 )" ) P( : 5 )
         CASE( 6 ) ! evaluate column index of J(x)
!write(6,"( ' index ', I0 )" ) index
           IF ( index == 1 ) THEN
             P( 1 ) = X( 2 )
             IP( 1 ) = 1
             lp = 1
           ELSE IF ( index == n ) THEN
             P( 1 ) = X( n - 1 )
             IP( 1 ) = n - 1
             lp = 1
           ELSE
             P( 1 ) = X( index - 1 )
             IP( 1 ) = index - 1
             P( 2 ) = X( index + 1 )
             IP( 2 ) = index
             lp = 2
           END IF
           eval_status = 0
         CASE( 7 ) ! evaluate Jr(x) * v for sparse v
           P( : m_r ) = 0.0_rp_
           DO i = lvl, lvu
             j = IV( i )
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
           eval_status = 0
!write(6,"( ' j : ', 5( F12.8, 2X ) )" ) P( : m_r )
         CASE( 8 ) ! evaluate sparse Jr^T(x) * v 
           DO i = lvl, lvu
             j = IV( i )
             IF ( j == 1 ) THEN
               P( 1 ) = X( 2 ) * V( 1 )
             ELSE IF ( j == n ) THEN
               P( n ) = X( m_r ) * V( m_r )
             ELSE
               P( j ) = X( j - 1 ) * V( j - 1 )  + X( j + 1 ) * V( j )
             END IF
           END DO
           eval_status = 0
!write(6,"( ' jt: ', 5( F6.2, I2 ) )" ) ( P( IV(i) ), IV(i), i = lvl, lvu )
         CASE DEFAULT ! error returns
           EXIT
         END SELECT
       END DO
       DEALLOCATE( IV, IP, V, P )
     END SELECT
     CALL SNLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( ' SNLS(', A2, '): ', I2, ' iterations -',                  &
      &             ' optimal objective value =', ES12.4 )" )                  &
           c_solver, inform%iter, inform%obj
     ELSE
       WRITE( 6, "( ' SNLS(', A2, '): exit status = ', I6 ) " )                &
           c_solver, inform%status
     END IF
     CALL SNLS_terminate( data, control, inform )
     SELECT CASE( solver )
     CASE( 1, 3 )
       DEALLOCATE( Jr_val, Jr_row, Jr_col )
     END SELECT
   END DO ! solver loop
   IF ( ALLOCATED( COHORT ) ) DEALLOCATE( COHORT )
   DEALLOCATE( X, G, R, userdata%real, userdata%integer )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

     SUBROUTINE WHICH_sls( control )
     TYPE ( SNLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%SLLS_control%SBLS_control%definite_linear_solver                  &
       = definite_linear_solver
     control%SLLS_control%SBLS_control%symmetric_linear_solver                 &
       = symmetric_linear_solver
     END SUBROUTINE WHICH_sls

     SUBROUTINE EVALR( status, X, userdata, R ) ! residual
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ) :: p
     p = userdata%real( 1 )
     R( 1 ) = X( 1 ) * X( 2 ) - p
     R( 2 ) = X( 2 ) * X( 3 ) - 1.0_rp_
     R( 3 ) = X( 3 ) * X( 4 ) - 1.0_rp_
     R( 4 ) = X( 4 ) * X( 5 ) - 1.0_rp_
!write(6,"( ' r : ', 4F6.2 )" ) R( : 4 )
     status = 0
     RETURN
     END SUBROUTINE EVALR

     SUBROUTINE EVALJr( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ) :: p
     p = userdata%real( 1 )
     Jr_val( 1 ) = X( 2 )
     Jr_val( 2 ) = X( 1 )
     Jr_val( 3 ) = X( 3 )
     Jr_val( 4 ) = X( 2 )
     Jr_val( 5 ) = X( 4 )
     Jr_val( 6 ) = X( 3 )
     Jr_val( 7 ) = X( 5 )
     Jr_val( 8 ) = X( 4 )
     status = 0
     RETURN
     END SUBROUTINE EVALJr

     SUBROUTINE EVALJr_prod( status, X, userdata, transpose, V, P, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
     IF ( transpose ) THEN
       P( 1 ) = X( 2 ) * V( 1 )
       P( 2 ) = X( 3 ) * V( 2 ) + X( 1 ) * V( 1 )
       P( 3 ) = X( 4 ) * V( 3 ) + X( 2 ) * V( 2 )
       P( 4 ) = X( 5 ) * V( 4 ) + X( 3 ) * V( 3 )
       P( 5 ) = X( 4 ) * V( 4 )
!write(6,"( ' pt: ', 5F6.2 )" ) P( : 5 )
     ELSE
       P( 1 ) = X( 2 ) * V( 1 ) + X( 1 ) * V( 2 )
       P( 2 ) = X( 3 ) * V( 2 ) + X( 2 ) * V( 3 )
       P( 3 ) = X( 4 ) * V( 3 ) + X( 3 ) * V( 4 )
       P( 4 ) = X( 5 ) * V( 4 ) + X( 4 ) * V( 5 )
!write(6,"( ' p : ', 5F6.2 )" ) P( : 4 )
     END IF
     status = 0
     RETURN
     END SUBROUTINE EVALJr_prod

     SUBROUTINE EVALJr_scol( status, X, userdata, index, VAL, ROW, nz, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     INTEGER, INTENT( IN ) :: index
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: VAL
     INTEGER, DIMENSION( : ), INTENT( INOUT ) :: ROW
     INTEGER, INTENT( INOUT ) :: nz
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
     INTEGER :: n
     n = userdata%integer( 1 ) 
!write(6,"( ' index ', I0 )" ) index
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

     SUBROUTINE EVALJr_sprod( status, X, userdata, transpose, V, P, FREE,      &
                              n_free, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
     INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
     INTEGER, INTENT( IN ) :: n_free
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
     INTEGER :: i, j, n, m_r
     REAL ( KIND = rp_ ) :: val
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
!write(6,"( ' jt: ', 5( F6.2, I2 ) )" ) ( P( FREE(i) ), FREE(i), i = 1, n_free )
     ELSE
       P( : m_r ) = 0.0_rp_
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
!write(6,"( ' j : ', 5( F12.8, 2X ) )" ) P( : m_r )
     END IF
     status = 0
     RETURN
     END SUBROUTINE EVALJr_sprod

   END PROGRAM GALAHAD_SNLS_interface_test
