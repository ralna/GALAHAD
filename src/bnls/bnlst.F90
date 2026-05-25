! THIS VERSION: GALAHAD 5.5 - 2026-05-06 AT 11:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BNLS_TESTS
   USE GALAHAD_KINDS_precision
   USE GALAHAD_BNLS_precision
   IMPLICIT NONE
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( BNLS_control_type ) :: control
   TYPE ( BNLS_inform_type ) :: inform
   TYPE ( BNLS_data_type ) :: data
   TYPE ( USERDATA_type ) :: userdata
   TYPE ( REVERSE_type ) :: reverse
!  EXTERNAL :: EVALR, EVALJr, EVALJr_prod, EVALJr_prods, EVALJr_sprod
   INTEGER ( KIND = ip_ ) :: i, j, l, nf, s, error, solver, mode
   INTEGER ( KIND = ip_ ) :: mnm, nflag, st_flag, len_integer
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, m_r = 4, jr_ne = 8
   INTEGER ( KIND = ip_ ), DIMENSION( n ) :: FLAG
   REAL ( KIND = rp_ ) :: val
   REAL ( KIND = rp_ ), PARAMETER :: p = 4.0_rp_
   CHARACTER ( LEN = 3 ) :: c_solver

   WRITE( 6, "( /, ' BNLS - error tests', / )" )

   DO error = 1, 7
     CALL BNLS_initialize( data, control, inform )
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
       nlp%n = 1 ; nlp%m_r = 1 ; nlp%Jr%ne = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( 1 ), nlp%Jr%row( 1 ), nlp%Jr%col( 1 ) )
       nlp%Jr%row( 1 ) = 1 ; nlp%Jr%col( 1 ) = 1
       ALLOCATE( nlp%X( 1 ), nlp%X_l( 1 ), nlp%X_u( 1 ) )
       nlp%X( 1 ) = 1.0_rp_ ; nlp%X_l = 0.0_rp_ ; nlp%X_u = 1.0_rp_
     END SELECT
     inform%status = 1
     CALL BNLS_solve( nlp, control, inform, data, userdata,                    &
                      eval_R = EVALR_error, eval_Jr = EVALJr_error )
     WRITE( 6, "( ' BNLS(error test ', I0, '): exit status = ', I0 ) " )       &
        error, inform%status
     CALL BNLS_terminate( data, control, inform )
     SELECT CASE( error )
     CASE( 4 )
       DEALLOCATE( nlp%W )
     CASE( 6 )
       DEALLOCATE( nlp%Jr%type )
     CASE( 7 )
       DEALLOCATE( nlp%X, nlp%X_l, nlp%X_u )
       DEALLOCATE( nlp%Jr%type,  nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     END SELECT
   END DO

   DO error = 8, 8
     CALL BNLS_initialize( data, control, inform )
     SELECT CASE( error )
     CASE ( 8 )
       nlp%n = 2 ; nlp%m_r = 1 ; nlp%Jr%ne = 1
       control%jacobian_available = 2
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( 1 ), nlp%Jr%row( 1 ), nlp%Jr%col( 1 ) )
       nlp%Jr%row( 1 ) = 1 ; nlp%Jr%col( 1 ) = 1
       ALLOCATE( nlp%X( 2 ), nlp%X_l( 2 ), nlp%X_u( 2 ) )
       nlp%X = 0.5_rp_ ; nlp%X_l = - 1.0_rp_ ; nlp%X_u = 1.0_rp_
       control%maxit = 1
     END SELECT
     inform%status = 1
     CALL BNLS_solve( nlp, control, inform, data, userdata,                    &
                      eval_R = EVALR_simple, eval_Jr = EVALJr_simple )
     WRITE( 6, "( ' BNLS(error test ', I0, '): exit status = ', I0 ) " )       &
        error, inform%status
     CALL BNLS_terminate( data, control, inform )
     SELECT CASE( error )
     CASE( 8 )
       DEALLOCATE( nlp%X, nlp%X_l, nlp%X_u )
       DEALLOCATE( nlp%Jr%type,  nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     END SELECT
   END DO

   WRITE( 6, "( /, ' BNLS - test of storage formats', / )" )

   nlp%n = n ; nlp%m_r = m_r ; nlp%Jr%ne = jr_ne
   mnm = MAX( n, m_r ) ; nflag = 3 ; st_flag = 3 ; len_integer = st_flag + mnm
   ALLOCATE( nlp%X( nlp%n ), nlp%X_l( nlp%n ), nlp%X_u( nlp%n ) )
   ALLOCATE( userdata%real( 1 ), userdata%integer( len_integer ) )
   userdata%real( 1 ) = p
   userdata%integer( 1 ) = n ; userdata%integer( 2 ) = m_r
   userdata%integer( nflag ) = 0
   userdata%integer( st_flag + 1 : st_flag + mnm ) = 0

   DO mode = 1, 5
     CALL BNLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
!    control%print_level = 1                    ! print one line/iteration
     control%jacobian_available = 2                
     control%subproblem_solver = 1
     nlp%Jr%m = m_r ; nlp%Jr%n = n
     nlp%X = [ 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_ ]
     nlp%X_l = 0.0_rp_ ; nlp%X_u = 1.0_rp_
     SELECT CASE ( mode )
     CASE ( 1 ) ! A by columns
       CALL SMT_put( nlp%Jr%type, 'SPARSE_BY_COLUMNS', s )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%ptr( n + 1 ) )
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ]
       nlp%Jr%ptr = [ 1, 2, 4, 6, 8, 9 ]
     CASE ( 2 ) ! A by rows
       CALL SMT_put( nlp%Jr%type, 'SPARSE_BY_ROWS', s )
       ALLOCATE( nlp%Jr%col( jr_ne ), nlp%Jr%ptr( m_r + 1 ) )
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
       nlp%Jr%ptr( : m_r + 1 ) = [ 1, 3, 5, 7, 9 ]
     CASE ( 3 ) ! A coordinate
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
       nlp%Jr%ne = jr_ne
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] 
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
     CASE ( 4 ) ! A dense by columns
       CALL SMT_put( nlp%Jr%type, 'DENSE_BY_COLUMNS', s )
     CASE ( 5 ) ! A dense by rows
       CALL SMT_put( nlp%Jr%type, 'DENSE_BY_ROWS', s )
     END SELECT

     inform%status = 1
     SELECT CASE ( mode )
     CASE ( 1 : 3 ) ! A by columns
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr = EVALJr )
     CASE ( 4 ) ! A dense by columns
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr = EVALJr_dense_by_cols )
     CASE ( 5 ) ! A dense by rows
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr = EVALJr_dense_by_rows )
     END SELECT
    
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( ' BNLS(', I1, '): ', I2, ' iterations -',                  &
      &             ' optimal objective value =', ES12.4 )" )                  &
           mode, inform%iter, inform%obj
     ELSE
       WRITE( 6, "( ' BNLS(', I1, '): exit status = ', I6 ) " )                &
           mode, inform%status
     END IF
     CALL BNLS_terminate( data, control, inform )

     DEALLOCATE( nlp%Jr%type )
     IF ( ALLOCATED( nlp%Jr%val ) ) DEALLOCATE( nlp%Jr%val )
     IF ( ALLOCATED( nlp%Jr%row ) ) DEALLOCATE( nlp%Jr%row )
     IF ( ALLOCATED( nlp%Jr%col ) ) DEALLOCATE( nlp%Jr%col )
     IF ( ALLOCATED( nlp%Jr%ptr ) ) DEALLOCATE( nlp%Jr%ptr )
   END DO
   DEALLOCATE( nlp%X, nlp%X_l, nlp%X_u, nlp%Z )
   DEALLOCATE( nlp%G, nlp%R, nlp%X_status )

   WRITE( 6, "( /, ' BNLS - test of input modes and options', / )" )

   nlp%n = n ; nlp%m_r = m_r ; nlp%Jr%ne = jr_ne
   ALLOCATE( nlp%X( nlp%n ), nlp%X_l( nlp%n ), nlp%X_u( nlp%n ) )

!  DO solver = 1, 1
   DO solver = 1, 6
     CALL BNLS_initialize( data, control, inform )
!    control%print_level = 1
!    control%BLLS_control%print_level = 1
!    control%BLLSB_control%print_level = 1
#ifdef REAL_32
     control%stop_pg_absolute = 0.0001_rp_
#else
     control%stop_pg_absolute = 0.00001_rp_
#endif
     CALL WHICH_sls( control )

     nlp%X = [ 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_, 0.5_rp_ ]
     nlp%X_l = 0.0_rp_ ; nlp%X_u = 1.0_rp_
     SELECT CASE( solver )
     CASE( 1 ) ! jacobian is available via function calls, blls sub-solver
       c_solver = 'JF '
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( jr_ne ) )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] 
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
       control%jacobian_available = 2                
       control%subproblem_solver = 1
       inform%status = 1
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr = EVALJr )

     CASE( 2 ) ! jacobian is available via function calls, bllsb sub-solver
       c_solver = 'JFB'
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( jr_ne ) )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] 
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
       control%jacobian_available = 2                
       control%subproblem_solver = 2
       inform%status = 1
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr = EVALJr )

     CASE( 3 )  ! jacobian is available via reverse communication, blls
       c_solver = 'JR '
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( jr_ne ) )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] 
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
       control%jacobian_available = 2                
       control%subproblem_solver = 1
       inform%status = 1
       DO
         CALL BNLS_solve( nlp, control, inform, data, userdata,                &
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

     CASE( 4 )  ! jacobian is available via reverse communication, bllsb
       c_solver = 'JRB'
       CALL SMT_put( nlp%Jr%type, 'COORDINATE', s )
       ALLOCATE( nlp%Jr%val( jr_ne ) )
       ALLOCATE( nlp%Jr%row( jr_ne ), nlp%Jr%col( jr_ne ) )
       nlp%Jr%row = [ 1, 1, 2, 2, 3, 3, 4, 4 ] 
       nlp%Jr%col = [ 1, 2, 2, 3, 3, 4, 4, 5 ]
       control%jacobian_available = 2                
       control%subproblem_solver = 2
       inform%status = 1
       DO
         CALL BNLS_solve( nlp, control, inform, data, userdata,                &
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
       CALL BNLS_solve( nlp, control, inform, data, userdata,                  &
                        eval_R = EVALR, eval_Jr_prod = EVALJr_prod,            &
                        eval_Jr_prods = EVALJr_prods,                          &
                        eval_Jr_sprod = EVALJr_sprod )

     CASE( 6 ) ! jacobian products are available via reverse communication
       c_solver = 'PR '
       control%jacobian_available = 1
       control%subproblem_solver = 2
       nf = 0 ; FLAG = 0
       inform%status = 1
       DO
         CALL BNLS_solve( nlp, control, inform, data, userdata,                &
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
           reverse%P( 1 )                                                      &
             = nlp%X( 2 ) * reverse%V( 1 ) + nlp%X( 1 ) * reverse%V( 2 )
           reverse%P( 2 )                                                      &
             = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 2 ) * reverse%V( 3 )
           reverse%P( 3 )                                                      &
             = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 3 ) * reverse%V( 4 )
           reverse%P( 4 )                                                      &
             = nlp%X( 5 ) * reverse%V( 4 ) + nlp%X( 4 ) * reverse%V( 5 )
           reverse%eval_status = 0
         CASE( 5 ) ! evaluate Jr^T(x) * v
           reverse%P( 1 ) = nlp%X( 2 ) * reverse%V( 1 )
           reverse%P( 2 )                                                      &
             = nlp%X( 3 ) * reverse%V( 2 ) + nlp%X( 1 ) * reverse%V( 1 )
           reverse%P( 3 )                                                      &
             = nlp%X( 4 ) * reverse%V( 3 ) + nlp%X( 2 ) * reverse%V( 2 )
           reverse%P( 4 )                                                      &
             = nlp%X( 5 ) * reverse%V( 4 ) + nlp%X( 3 ) * reverse%V( 3 )
           reverse%P( 5 ) = nlp%X( 4 ) * reverse%V( 4 )
           reverse%eval_status = 0
         CASE( 6 ) ! evaluate Jr(x) * sparse v 
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
         CASE( 7 ) ! evaluate sparse( Jr(x) * sparse v )
           nf = nf + 1
           reverse%lp = 0
           DO l = reverse%lvl, reverse%lvu
             j = reverse%IV( l )
             val = reverse%V( j )
             IF ( j == 1 ) THEN
               i = 1
               IF ( FLAG( i ) < nf ) THEN
                 FLAG( i ) = nf
                 reverse%P( i ) = nlp%X( 2 ) * val
                 reverse%lp = reverse%lp + 1
                 reverse%IP( reverse%lp ) = i
               ELSE
                 reverse%P( i ) = reverse%P( i ) + nlp%X( 2 ) * val
               END IF
             ELSE IF ( j == n ) THEN
               i = n - 1
               IF ( FLAG( i ) < nf ) THEN
                 FLAG( i ) = nf
                 reverse%P( i ) = nlp%X( n - 1 ) * val
                 reverse%lp = reverse%lp + 1
                 reverse%IP( reverse%lp ) = i
               ELSE
                 reverse%P( i ) = reverse%P( i ) + nlp%X( n - 1 ) * val
               END IF
             ELSE
               i = j - 1
               IF ( FLAG( i ) < nf ) THEN
                 FLAG( i ) = nf
                 reverse%P( i ) = nlp%X( j - 1 ) * val
                 reverse%lp = reverse%lp + 1
                 reverse%IP( reverse%lp ) = i
               ELSE
                 reverse%P( i ) = reverse%P( i ) + nlp%X( j - 1 ) * val
               END IF
               i = j
               IF ( FLAG( i ) < nf ) THEN
                 FLAG( i ) = nf
                 reverse%P( i ) = nlp%X( j + 1 ) * val
                 reverse%lp = reverse%lp + 1
                 reverse%IP( reverse%lp ) = i
               ELSE
                 reverse%P( i ) = reverse%P( i ) + nlp%X( j + 1 ) * val
               END IF
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
               reverse%P( j ) = nlp%X( j - 1 ) * reverse%V( j - 1 )            &
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
       WRITE( 6, "( ' BNLS(', A3, '): ', I2, ' iterations -',                  &
      &             ' optimal objective value =', ES12.4 )" )                  &
           c_solver, inform%iter, inform%obj
     ELSE
       WRITE( 6, "( ' BNLS(', A3, '): exit status = ', I6 ) " )                &
           c_solver, inform%status
     END IF

     SELECT CASE( solver )
     CASE( 3, 4, 6 )
       CALL BNLS_terminate( data, control, inform, reverse = reverse )
     CASE DEFAULT
       CALL BNLS_terminate( data, control, inform )
     END SELECT

     SELECT CASE( solver )
     CASE( 1 : 4 )
       DEALLOCATE( nlp%Jr%type, nlp%Jr%val, nlp%Jr%row, nlp%Jr%col )
     END SELECT
   END DO ! solver loop
   DEALLOCATE( nlp%X, nlp%X_l, nlp%X_u, nlp%Z )
   DEALLOCATE( nlp%G, nlp%R, nlp%X_status )
   DEALLOCATE( userdata%real, userdata%integer )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

     SUBROUTINE WHICH_sls( control )
     TYPE ( BNLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%BLLS_control%SBLS_control%definite_linear_solver                  &
       = definite_linear_solver
     control%BLLS_control%SBLS_control%symmetric_linear_solver                 &
       = symmetric_linear_solver
     control%BLLSB_control%symmetric_linear_solver                             &
       = symmetric_linear_solver
     control%BLLSB_control%FDC_control%symmetric_linear_solver                 &
       = symmetric_linear_solver
     END SUBROUTINE WHICH_sls

     SUBROUTINE EVALR( status, X, userdata, R ) ! residual
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
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
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
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

     SUBROUTINE EVALJr_dense_by_rows( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = X( 2 )
     Jr_val( 2 ) = X( 1 )
     Jr_val( 3 ) = 0.0_rp_
     Jr_val( 4 ) = 0.0_rp_
     Jr_val( 5 ) = 0.0_rp_
     Jr_val( 6 ) = 0.0_rp_
     Jr_val( 7 ) = X( 3 )
     Jr_val( 8 ) = X( 2 )
     Jr_val( 9 ) = 0.0_rp_
     Jr_val( 10 ) = 0.0_rp_
     Jr_val( 11 ) = 0.0_rp_
     Jr_val( 12 ) = 0.0_rp_
     Jr_val( 13 ) = X( 4 )
     Jr_val( 14 ) = X( 3 )
     Jr_val( 15 ) = 0.0_rp_
     Jr_val( 16 ) = 0.0_rp_
     Jr_val( 17 ) = 0.0_rp_
     Jr_val( 18 ) = 0.0_rp_
     Jr_val( 19 ) = X( 5 )
     Jr_val( 20 ) = X( 4 )
     status = 0
     RETURN
     END SUBROUTINE EVALJr_dense_by_rows

     SUBROUTINE EVALJr_dense_by_cols( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = X( 2 )
     Jr_val( 2 ) = 0.0_rp_
     Jr_val( 3 ) = 0.0_rp_
     Jr_val( 4 ) = 0.0_rp_
     Jr_val( 5 ) = X( 1 )
     Jr_val( 6 ) = X( 3 )
     Jr_val( 7 ) = 0.0_rp_
     Jr_val( 8 ) = 0.0_rp_
     Jr_val( 9 ) = 0.0_rp_
     Jr_val( 10 ) = X( 2 )
     Jr_val( 11 ) = X( 4 )
     Jr_val( 12 ) = 0.0_rp_
     Jr_val( 13 ) = 0.0_rp_
     Jr_val( 14 ) = 0.0_rp_
     Jr_val( 15 ) = X( 3 )
     Jr_val( 16 ) = X( 5 )
     Jr_val( 17 ) = 0.0_rp_
     Jr_val( 18 ) = 0.0_rp_
     Jr_val( 19 ) = 0.0_rp_
     Jr_val( 20 ) = X( 4 )
     status = 0
     RETURN
     END SUBROUTINE EVALJr_dense_by_cols

     SUBROUTINE EVALJr_prod( status, X, userdata, transpose, V, P, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
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

     SUBROUTINE EVALJR_prods( status, X, userdata, V, P, IV, lvl, lvu,         &
                              IP, lp, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: lvl, lvu
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( OUT ) :: lp
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: IV
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: IP
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_jr
     INTEGER :: i, j, l, n, nflag, st_flag
     REAL ( KIND = rp_ ) :: val
     n = userdata%integer( 1 ) 
     nflag = 3
     st_flag = 3
     IF ( PRESENT( IP ) .AND. PRESENT( lp ) )  THEN
       userdata%integer( nflag ) = userdata%integer( nflag ) + 1
       lp = 0
       DO l = lvl, lvu
         j = IV( l )
         val = V( j )
         IF ( j == 1 ) THEN
           i = 1
           IF ( userdata%integer( st_flag + i )                                &
                  < userdata%integer( nflag ) ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = X( 2 ) * val
             lp = lp + 1
             IP( lp ) = i
           ELSE
             P( i ) = P( i ) + X( 2 ) * val
           END IF
         ELSE IF ( j == n ) THEN
           i = n - 1
           IF ( userdata%integer( st_flag + i )                                &
                  < userdata%integer( nflag ) ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = X( n - 1 ) * val
             lp = lp + 1
             IP( lp ) = i
           ELSE
             P( i ) = P( i ) + X( n - 1 ) * val
           END IF
         ELSE
           i = j - 1
           IF ( userdata%integer( st_flag + i )                                &
                  < userdata%integer( nflag ) ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = X( j - 1 ) * val
             lp = lp + 1
             IP( lp ) = i
           ELSE
             P( i ) = P( i ) + X( j - 1 ) * val
           END IF
           i = j
           IF ( userdata%integer( st_flag + i )                                &
                  < userdata%integer( nflag ) ) THEN
             userdata%integer( st_flag + i ) = userdata%integer( nflag )
             P( i ) = X( j + 1 ) * val
             lp = lp + 1
             IP( lp ) = i
           ELSE
             P( i ) = P( i ) + X( j + 1 ) * val
           END IF
         END IF
       END DO
     ELSE
       P = 0.0_rp_
       DO l = lvl, lvu
         j = IV( l )
         val = V( j )
         IF ( j == 1 ) THEN
           i = 1
           P( i ) = P( i ) + X( 2 ) * val
         ELSE IF ( j == n ) THEN
           i = n - 1
           P( i ) = P( i ) + X( n - 1 ) * val
         ELSE
           i = j - 1
           P( i ) = P( i ) + X( j - 1 ) * val
           i = j
           P( i ) = P( i ) + X( j + 1 ) * val
         END IF
       END DO
     END IF
     status = 0
     RETURN
     END SUBROUTINE EVALJR_prods

     SUBROUTINE EVALJr_sprod( status, X, userdata, transpose, V, P, FREE,      &
                              n_free, got_jr )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
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
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     R( 1 ) = X( 1 ) - 1.0_rp_
     status = - 1
     RETURN
     END SUBROUTINE EVALR_error

     SUBROUTINE EVALJr_error( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = 1.0_rp_
     status = - 1
     RETURN
     END SUBROUTINE EVALJr_error

     SUBROUTINE EVALR_simple( status, X, userdata, R ) ! residual
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( OUT ) :: R
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     R( 1 ) = X( 1 ) ** 3
     status = 0
     RETURN
     END SUBROUTINE EVALR_simple

     SUBROUTINE EVALJr_simple( status, X, userdata, Jr_val ) ! Jacobian
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Jr_val
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     Jr_val( 1 ) = 3.0_rp_ * X( 1 ) ** 2
     status = 0
     RETURN
     END SUBROUTINE EVALJr_simple

   END PROGRAM GALAHAD_BNLS_TESTS
