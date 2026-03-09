! THIS VERSION: GALAHAD 5.5 - 2026-02-22 AT 15:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SNLS_TESTS
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SNLS_precision
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( SNLS_control_type ) :: control
   TYPE ( SNLS_inform_type ) :: inform
   TYPE ( SNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   TYPE ( REVERSE_type ) :: reverse
!  EXTERNAL :: EVALR, EVALJr, EVALJr_prod, EVALJr_scol, EVALJr_sprod
   INTEGER ( KIND = ip_ ) :: i, j, s, error, cohort, solver
   INTEGER, PARAMETER :: n = 5, m_r = 4, jr_ne = 8
   REAL ( KIND = rp_ ) :: val
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   CHARACTER ( LEN = 3 ) :: c_solver

   WRITE( 6, "( /, ' SNLS - error tests', / )" )

   DO error = 1, 8
     CALL SNLS_initialize( data, control, inform )
!    control%error = 0
     SELECT CASE( error )
     CASE( 1 )
       nlp%n = 0 ; nlp%m_r = 1
     CASE( 2 )
       nlp%n = 1 ; nlp%m_r = 0
     CASE( 3 )
       nlp%n = 1 ; nlp%m_r = 1
       control%jacobian_available = 0
     CASE( 4 )
       nlp%n = 1 ; nlp%m_r = 1
       ALLOCATE( nlp%W( nlp%m_r ) )
       nlp%W( : nlp%m_r ) = 0.0_rp_
     CASE( 5 )
       nlp%n = 1 ; nlp%m_r = 1
       control%jacobian_available = 1
     CASE( 6 )
       nlp%n = 1 ; nlp%m_r = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'UNKNOWN', s )
     CASE ( 7 )
       nlp%n = 1 ; nlp%m_r = 1 ; nlp%m_c = 1 ; nlp%Jr%ne = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( 1 ), nlp%Jr%row( 1 ), nlp%Jr%col( 1 ) )
       nlp%Jr%row( 1 ) = 1 ; nlp%Jr%col( 1 ) = 1
       ALLOCATE( nlp%COHORT( nlp%n ) )
       nlp%COHORT( 1 ) = 2
     CASE ( 8 )
       nlp%n = 1 ; nlp%m_r = 1 ; nlp%m_c = 1 ; nlp%Jr%ne = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( 1 ), nlp%Jr%row( 1 ), nlp%Jr%col( 1 ) )
       nlp%Jr%row( 1 ) = 1 ; nlp%Jr%col( 1 ) = 1
       ALLOCATE( nlp%X( 1 ) )
       nlp%X( 1 ) = 1.0_rp_
     END SELECT
     inform%status = 1
     CALL SNLS_solve( nlp, control, inform, data, userdata,                    &
                      eval_R = EVALR_error, eval_Jr = EVALJr_error )
     WRITE( 6, "( ' SNLS(error test ', I0, '): exit status = ', I0 ) " )       &
        error, inform%status
     CALL SNLS_terminate( data, control, inform )
     SELECT CASE( error )
     CASE( 4 )
       DEALLOCATE( nlp%W )
     CASE( 6 )
       DEALLOCATE( nlp%Jr%type )
     CASE( 7 )
       DEALLOCATE( nlp%Jr%type, nlp%COHORT, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     CASE( 8 )
       DEALLOCATE( nlp%X, nlp%Jr%type,  nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     END SELECT
   END DO

   DO error = 9, 9
     CALL SNLS_initialize( data, control, inform )
     SELECT CASE( error )
     CASE ( 9 )
       nlp%n = 2 ; nlp%m_r = 1 ; nlp%m_c = 1 ; nlp%Jr%ne = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( 1 ), nlp%Jr%row( 1 ), nlp%Jr%col( 1 ) )
       nlp%Jr%row( 1 ) = 1 ; nlp%Jr%col( 1 ) = 1
       ALLOCATE( nlp%X( 2 ) )
       nlp%X = 0.0_rp_
       control%maxit = 1
     END SELECT
     inform%status = 1
     CALL SNLS_solve( nlp, control, inform, data, userdata,                    &
                      eval_R = EVALR_simple, eval_Jr = EVALJr_simple )
     WRITE( 6, "( ' SNLS(error test ', I0, '): exit status = ', I0 ) " )       &
        error, inform%status
     CALL SNLS_terminate( data, control, inform )
     SELECT CASE( error )
     CASE( 9 )
       DEALLOCATE( nlp%X, nlp%Jr%type,  nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     END SELECT
   END DO

   WRITE( 6, "( /, ' SNLS - test of input modes and options', / )" )

   nlp%n = n ; nlp%m_r = m_r ; nlp%Jr%ne = jr_ne
   ALLOCATE( nlp%X( n ) )
   ALLOCATE( userdata%real( 1 ), userdata%integer( 2 ) )
   userdata%real( 1 ) = p
   userdata%integer( 1 ) = n
   userdata%integer( 2 ) = m_r 

!  DO cohort = 0, 0
   DO cohort = 0, 2
     SELECT CASE( cohort )
     CASE( 0 )
       nlp%m_c = 1
     CASE( 1 )
       ALLOCATE( nlp%COHORT( n ) )
       nlp%m_c = 1
       nlp%COHORT = [ 1, 1, 1, 1, 1 ]
     CASE( 2 )
       ALLOCATE( nlp%COHORT( n ) )
       nlp%m_c = 2
       nlp%COHORT = [ 1, 2, 0, 1, 2 ]
     END SELECT

!    DO solver = 1, 1
     DO solver = 1, 6
       CALL SNLS_initialize( data, control, inform )
!      control%print_level = 1
!      control%SLLS_control%print_level = 1
!      control%SLLSB_control%print_level = 1
       control%stop_pg_absolute = 0.00001_rp_
       CALL WHICH_sls( control )

       nlp%X = [ 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_ ]
       SELECT CASE( solver )
       CASE( 1 ) ! jacobian is available via function calls, slls sub-solver
         c_solver = 'JF '
         CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
         ALLOCATE( nlp%Jr%val( jr_ne ) )
         ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
         nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
         nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
         control%jacobian_available = 2                
         control%subproblem_solver = 1
         inform%status = 1
         CALL SNLS_solve( nlp, control, inform, data, userdata,                &
                          eval_R = EVALR, eval_Jr = EVALJr )

       CASE( 2 ) ! jacobian is available via function calls, sllsb sub-solver
         c_solver = 'JFB'
         CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
         ALLOCATE( nlp%Jr%val( jr_ne ) )
         ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
         nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
         nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
         control%jacobian_available = 2                
         control%subproblem_solver = 2
         inform%status = 1
         CALL SNLS_solve( nlp, control, inform, data, userdata,                &
                          eval_R = EVALR, eval_Jr = EVALJr )

       CASE( 3 )  ! jacobian is available via reverse communication, slls
         c_solver = 'JR '
         CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
         ALLOCATE( nlp%Jr%val( jr_ne ) )
         ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
         nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
         nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
         control%jacobian_available = 2                
         control%subproblem_solver = 1
         inform%status = 1
         DO
           CALL SNLS_solve( nlp, control, inform, data, userdata,              &
                            reverse = reverse )
           SELECT CASE( inform%status )
           CASE ( 0 ) ! successful return
             EXIT
           CASE( 2 ) ! evaluate residual
             nlp%R( 1 ) = nlp%X( 1 ) * nlp%X( 2 ) - p
             nlp%R( 2 ) = nlp%X( 2 ) * nlp%X( 3 ) - 1.0_rp_
             nlp%R( 3 ) = nlp%X( 3 ) * nlp%X( 4 ) - 1.0_rp_
             nlp%R( 4 ) = nlp%X( 4 ) * nlp%X( 5 ) - 1.0_rp_
             reverse%eval_status = 0
           CASE( 3 ) ! evaluate Jacobian
             nlp%Jr%val( 1 ) = nlp%X( 2 )
             nlp%Jr%val( 2 ) = nlp%X( 1 )
             nlp%Jr%val( 3 ) = nlp%X( 3 )
             nlp%Jr%val( 4 ) = nlp%X( 2 )
             nlp%Jr%val( 5 ) = nlp%X( 4 )
             nlp%Jr%val( 6 ) = nlp%X( 3 )
             nlp%Jr%val( 7 ) = nlp%X( 5 )
             nlp%Jr%val( 8 ) = nlp%X( 4 )
             reverse%eval_status = 0
           CASE DEFAULT ! error returns
             EXIT
           END SELECT
         END DO

       CASE( 4 )  ! jacobian is available via reverse communication, sllsb
         c_solver = 'JRB'
         CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
         ALLOCATE( nlp%Jr%val( jr_ne ) )
         ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
         nlp%Jr%row = (/ 1, 1, 2, 2, 3, 3, 4, 4 /) 
         nlp%Jr%col = (/ 1, 2, 2, 3, 3, 4, 4, 5 /)
         control%jacobian_available = 2                
         control%subproblem_solver = 2
         inform%status = 1
         DO
           CALL SNLS_solve( nlp, control, inform, data, userdata,              &
                            reverse = reverse )
           SELECT CASE( inform%status )
           CASE ( 0 ) ! successful return
             EXIT
           CASE( 2 ) ! evaluate residual
             nlp%R( 1 ) = nlp%X( 1 ) * nlp%X( 2 ) - p
             nlp%R( 2 ) = nlp%X( 2 ) * nlp%X( 3 ) - 1.0_rp_
             nlp%R( 3 ) = nlp%X( 3 ) * nlp%X( 4 ) - 1.0_rp_
             nlp%R( 4 ) = nlp%X( 4 ) * nlp%X( 5 ) - 1.0_rp_
             reverse%eval_status = 0
           CASE( 3 ) ! evaluate Jacobian
             nlp%Jr%val( 1 ) = nlp%X( 2 )
             nlp%Jr%val( 2 ) = nlp%X( 1 )
             nlp%Jr%val( 3 ) = nlp%X( 3 )
             nlp%Jr%val( 4 ) = nlp%X( 2 )
             nlp%Jr%val( 5 ) = nlp%X( 4 )
             nlp%Jr%val( 6 ) = nlp%X( 3 )
             nlp%Jr%val( 7 ) = nlp%X( 5 )
             nlp%Jr%val( 8 ) = nlp%X( 4 )
             reverse%eval_status = 0
           CASE DEFAULT ! error returns
             EXIT
           END SELECT
         END DO

       CASE( 5 )  ! jacobian products are available via function calls
         c_solver = 'PF '
         control%jacobian_available = 1
         control%subproblem_solver = 2
         inform%status = 1
         CALL SNLS_solve( nlp, control, inform, data, userdata,                &
                          eval_R = EVALR, eval_Jr_prod = EVALJr_prod,          &
                          eval_Jr_scol = EVALJr_scol,                          &
                          eval_Jr_sprod = EVALJr_sprod )

       CASE( 6 ) ! jacobian products are available via reverse communication
         c_solver = 'PR '
         control%jacobian_available = 1
         control%subproblem_solver = 2
         inform%status = 1
         DO
           CALL SNLS_solve( nlp, control, inform, data, userdata,              &
                            reverse = reverse )
           SELECT CASE( inform%status )
           CASE ( 0 ) ! successful return
             EXIT
           CASE( 2 ) ! evaluate residual
             nlp%R( 1 ) = nlp%X( 1 ) * nlp%X( 2 ) - p
             nlp%R( 2 ) = nlp%X( 2 ) * nlp%X( 3 ) - 1.0_rp_
             nlp%R( 3 ) = nlp%X( 3 ) * nlp%X( 4 ) - 1.0_rp_
             nlp%R( 4 ) = nlp%X( 4 ) * nlp%X( 5 ) - 1.0_rp_
             reverse%eval_status = 0
           CASE( 4 ) ! evaluate Jr(x) * v
             reverse%P( 1 )                                                    &
               = nlp%X( 2 ) * reverse%V( 1 ) + nlp%X( 1 ) * reverse%V( 2 )
             reverse%P( 2 )                                                    &
               = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 2 ) * reverse%V( 3 )
             reverse%P( 3 )                                                    &
               = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 3 ) * reverse%V( 4 )
             reverse%P( 4 )                                                    &
               = nlp%X( 5 ) * reverse%V( 4 ) + nlp%X( 4 ) * reverse%V( 5 )
             reverse%eval_status = 0
           CASE( 5 ) ! evaluate Jr^T(x) * v
             reverse%P( 1 ) = nlp%X( 2 ) * reverse%V( 1 )
             reverse%P( 2 )                                                    &
               = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 1 ) * reverse%V( 1 )
             reverse%P( 3 )                                                    &
               = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 2 ) * reverse%V( 2 )
             reverse%P( 4 )                                                    &
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
             reverse%P( : m_r ) = 0.0_rp_
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
                 reverse%P( j ) = nlp%X( j - 1 ) * reverse%V( j - 1 )          &
                                  + nlp%X( j + 1 ) * reverse%V( j )
               END IF
             END DO
             reverse%eval_status = 0
           CASE DEFAULT ! error returns
             EXIT
           END SELECT
         END DO
       END SELECT

       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( ' SNLS(', I1, A3, '): ', I2, ' iterations -',            &
        &             ' optimal objective value =', ES12.4 )" )                &
             cohort, c_solver, inform%iter, inform%obj
       ELSE
         WRITE( 6, "( ' SNLS(', I1, A3, '): exit status = ', I6 ) " )          &
             cohort, c_solver, inform%status
       END IF
       CALL SNLS_terminate( data, control, inform )
       SELECT CASE( solver )
       CASE( 1 : 4 )
         DEALLOCATE( nlp%Jr%type, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
       END SELECT
     END DO ! solver loop
     SELECT CASE( cohort )
     CASE( 1 : 2 )
       DEALLOCATE( nlp%COHORT )
     END SELECT
   END DO ! cohort loop
   IF ( ALLOCATED( nlp%COHORT ) ) DEALLOCATE( nlp%COHORT )
   DEALLOCATE( nlp%X, nlp%G, nlp%R, userdata%real, userdata%integer )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

     SUBROUTINE WHICH_sls( control )
     TYPE ( SNLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%SLLS_control%SBLS_control%definite_linear_solver                  &
       = definite_linear_solver
     control%SLLS_control%SBLS_control%symmetric_linear_solver                 &
       = symmetric_linear_solver
     control%SLLSB_control%symmetric_linear_solver                             &
       = symmetric_linear_solver
     control%SLLSB_control%FDC_control%symmetric_linear_solver                 &
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
     END IF
     status = 0
     RETURN
     END SUBROUTINE EVALJr_sprod

     SUBROUTINE EVALR_error( status, X, userdata, R ) ! residual
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     R( 1 ) = X( 1 ) - 1.0_rp_
     status = - 1
     RETURN
     END SUBROUTINE EVALR_error

     SUBROUTINE EVALJr_error( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = 1.0_rp_
     status = - 1
     RETURN
     END SUBROUTINE EVALJr_error

     SUBROUTINE EVALR_simple( status, X, userdata, R ) ! residual
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     R( 1 ) = X( 1 ) ** 3
     status = 0
     RETURN
     END SUBROUTINE EVALR_simple

     SUBROUTINE EVALJr_simple( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = 3.0_rp_ * X( 1 ) ** 2
     status = 0
     RETURN
     END SUBROUTINE EVALJr_simple

   END PROGRAM GALAHAD_SNLS_TESTS
