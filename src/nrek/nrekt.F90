! THIS VERSION: GALAHAD 5.4 - 2025-11-14 AT 15:00 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_NREK_test_program
   USE GALAHAD_KINDS_precision, ONLY: rp_, ip_
   USE GALAHAD_NREK_precision, ONLY: SMT_type, SMT_put, NREK_control_type,     &
          NREK_inform_type, NREK_data_type, NREK_initialize, NREK_solve,       &
          NREK_terminate
   USE GALAHAD_SYMBOLS, ONLY: GALAHAD_error_restrictions,                      &
          GALAHAD_error_preconditioner, GALAHAD_error_ill_conditioned,         &
          GALAHAD_error_max_iterations
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_, two = 2.0_rp_
   INTEGER ( KIND = ip_ ) :: i, smt_stat, n, nn, pass, h_ne, s_ne
   INTEGER ( KIND = ip_ ) :: id, iw, is, data_storage_type
   REAL ( KIND = rp_ ) :: power, weight
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, C
   TYPE ( SMT_type ) :: H, S
   TYPE ( NREK_data_type ) :: data
   TYPE ( NREK_control_type ) :: control
   TYPE ( NREK_inform_type ) :: inform

   CHARACTER ( LEN = 1 ) :: st, sa
   INTEGER ( KIND = ip_ ), PARAMETER :: n_errors = 5
   INTEGER ( KIND = ip_ ), DIMENSION( n_errors ) :: errors = [                 &
       GALAHAD_error_restrictions, GALAHAD_error_restrictions,                 &
       GALAHAD_error_restrictions, GALAHAD_error_restrictions,                 &
       GALAHAD_error_restrictions ]

! Initialize output unit

   OPEN( UNIT = 23, STATUS = 'SCRATCH' )
!  OPEN( UNIT = 23 )

!  =============
!  Error entries
!  =============

   n = 5 ; H%n = n
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )   ! Specify co-ordinate for H
   H%ne = 2 * n - 1
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   DO i = 1, n
    H%row( i ) = i ; H%col( i ) = i ; H%val( i ) = - 2.0_rp_
   END DO
   DO i = 1, n - 1
    H%row( n + i ) = i + 1 ; H%col( n + i ) = i ; H%val( n + i ) = 1.0_rp_
   END DO
   CALL SMT_put( S%type, 'DIAGONAL', smt_stat )     ! Specify diagonal for S
   ALLOCATE( S%val( n ) ) ; S%val = 2.0_rp_
   WRITE( 6, "( /, ' ==== error exits ===== ', / )" )
   IF ( .false. ) go to 10
!  IF ( .true. ) go to 10
! Initialize control parameters

!  DO i = 1, 1
   DO i = 1, n_errors
     pass = errors( i )
     nn = n
     power = 3.0_rp_
     weight = one
     CALL NREK_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%rqs_control%error = 23 ; control%rqs_control%out = 23
     control%sls_control%error = 23 ; control%sls_control%out = 23
     control%sls_control%warning = 23 ; control%sls_control%statistics = 23
!    control%error = 6 ; control%out = 6 ; control%print_level = 10
!    control%SLS_control%error = 6 ; control%SLS_control%out = 6
!    control%SLS_control%warning = 6 ; control%SLS_control%statistics = 6
!    control%SLS_control%print_level = 10
     IF ( pass == GALAHAD_error_restrictions ) THEN
       IF ( i == 1 ) THEN
         nn = 0
       ELSE IF ( i == 2 ) THEN
         nn = 1
       ELSE IF ( i == 3 ) THEN
         CALL SMT_put( H%type, 'WRONG', smt_stat )
       ELSE IF ( i == 4 ) THEN
         CALL SMT_put( S%type, 'ZERO', smt_stat )
       ELSE
         weight = - one
       END IF
     ELSE IF ( pass == GALAHAD_error_preconditioner ) THEN
       S%val( 1 ) = - one
     ELSE IF ( pass == GALAHAD_error_ill_conditioned ) THEN
       S%val( 1 ) = 2.0_rp_
       weight = 1000000.0_rp_
     ELSE IF ( pass == GALAHAD_error_max_iterations ) THEN
       control%eks_max = 1
     END IF
     ALLOCATE( X( nn ), C( nn ) )
     C = 1.0_rp_

!  Iteration to find the minimizer

     CALL NREK_solve( nn, H, C, power, weight, X, data, control, inform, S = S )

     WRITE( 6, "( ' pass  ', I3, ': NREK_solve exit status = ', I6 )" )        &
            pass, inform%status
     CALL NREK_terminate( data, control, inform ) !  delete internal workspace
     DEALLOCATE( X, C )
     CALL NREK_terminate( data, control, inform ) !  delete internal workspace
     IF ( i == 3 ) CALL SMT_put( H%type, 'COORDINATE', smt_stat )
   END DO
10 continue
   DEALLOCATE( H%row, H%col, H%val, S%val )

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' ==== basic tests of storage formats ===== ', / )" )

   power = 3.0_rp_
   n = 3 ; h_ne = 4 ; s_ne = 3 ; H%n = n ; S%n = n
   ALLOCATE( H%ptr( n + 1 ), S%ptr( n + 1 ), C( n ), X( n ) )

   C = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]

   DO data_storage_type = - 6, 0
!  DO data_storage_type = - 1, - 1
!  DO data_storage_type = - 3, - 2
!  DO data_storage_type = - 3, - 3
     CALL NREK_initialize( data, control, inform )
     CALL WHICH_sls( control )
!control%print_level = 1
!control%rqs_control%print_level = 1
!    control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%rqs_control%error = 23 ; control%rqs_control%out = 23
!    control%sls_control%error = 23 ; control%sls_control%out = 23
!    control%sls_control%warning = 23 ; control%sls_control%statistics = 23
!    control%eks_max = 4
     control%stop_check_all_orders = .TRUE.
     IF ( data_storage_type == 0 ) THEN           ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'COORDINATE', smt_stat )
       H%row = [ 1, 2, 3, 3 ]
       H%col = [ 1, 2, 3, 1 ] ; H%ne = h_ne
       ALLOCATE( S%val( s_ne ), S%row( s_ne ), S%col( s_ne ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'COORDINATE', smt_stat )
       S%row = [ 1, 2, 3 ]
       S%col = [ 1, 2, 3 ] ; S%ne = s_ne
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( H%val( h_ne ), H%row( 0 ), H%col( h_ne ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'SPARSE_BY_ROWS', smt_stat )
       H%col = [ 1, 2, 3, 1 ]
       H%ptr = [ 1, 2, 3, 5 ]
       ALLOCATE( S%val( s_ne ), S%row( 0 ), S%col( s_ne ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'SPARSE_BY_ROWS', smt_stat )
       S%col = [ 1, 2, 3 ]
       S%ptr = [ 1, 2, 3, 4 ]
     ELSE IF ( data_storage_type == - 2 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( H%val( n * ( n + 1 ) / 2 ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DENSE', smt_stat )
       ALLOCATE( S%val( n * ( n + 1 ) / 2 ), S%row( 0 ), S%col( 0 ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 3 ) THEN      ! diagonal storage
       st = 'G'
       ALLOCATE( H%val( n ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'DIAGONAL', smt_stat )
       ALLOCATE( S%val( n ), S%row( 0 ), S%col( 0 ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'DIAGONAL', smt_stat )
     ELSE IF ( data_storage_type == - 4 ) THEN      ! scaled identity H
       st = 'S'
       ALLOCATE( H%val( 1 ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'SCALED_IDENTITY', smt_stat )
       ALLOCATE( S%val( 0 ), S%row( 0 ), S%col( 0 ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'SCALED_IDENTITY', smt_stat )
     ELSE IF ( data_storage_type == - 5 ) THEN      ! identity H
       st = 'I'
       ALLOCATE( H%val( 0 ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'IDENTITY', smt_stat )
       ALLOCATE( S%val( 0 ), S%row( 0 ), S%col( 0 ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'IDENTITY', smt_stat )
     ELSE IF ( data_storage_type == - 6 ) THEN      ! no H
       st = 'Z'
       ALLOCATE( H%val( 0 ), H%row( 0 ), H%col( 0 ) )
       IF ( ALLOCATED( H%type ) ) DEALLOCATE( H%type )
       CALL SMT_put( H%type, 'ZERO', smt_stat )
       ALLOCATE( S%val( 0 ), S%row( 0 ), S%col( 0 ) )
       IF ( ALLOCATED( S%type ) ) DEALLOCATE( S%type )
       CALL SMT_put( S%type, 'IDENTITY', smt_stat )
     END IF

!  test with new and existing data

     IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
       H%val = [ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ ]
       S%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_ ]
     ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
       H%val = [ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ ]
       S%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_ ]
     ELSE IF ( data_storage_type == - 2 ) THEN    !  dense storage
       H%val = [ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_ ]
       S%val = [ 1.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_ ]
     ELSE IF ( data_storage_type == - 3 ) THEN    !  diagonal/dense storage
       H%val = [ 1.0_rp_, - 1.0_rp_, 2.0_rp_ ]
!      S%val = [ 1.0_rp_, 1.0_rp_, 1.0_rp_ ]
       S%val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_ ]
     ELSE IF ( data_storage_type == - 4 ) THEN    !  scaled identity storage
       H%val = [ 1.0_rp_ ]
       S%val = [ 1.0_rp_ ]
     END IF
!    control%error = 23 ; control%out = 23 ; control%print_level = 10
!    control%error = 6 ; control%out = 6 ; control%print_level = 1
     DO is = 0, 1 ! is S present?
!    DO is = 1, 1 ! is S present?
       IF ( is == 1 .AND. data_storage_type < - 3 ) CYCLE
       DO id = 0, 1 ! new data?
!      DO id = 0, 0 ! new data?
         control%new_values = id == 1
         DO iw = 1, 2 ! resolve with larger weight?
           control%new_weight = iw == 2
!        WRITE( 6, "( ' format ', A1, I1, A1, ':' )" ) st, iw, id, sa
!write(6,*) 'iw = ', iw, 'id = ', id, 'is = ', is, H%type, ' ', S%type
           IF ( control%new_weight ) THEN
             weight = inform%next_weight
           ELSE
             weight = 1.0_rp_
           END IF
           IF ( is == 0 ) THEN
             sa = ' '
             CALL NREK_solve( n, H, C, power, weight, X, data, control, inform )
           ELSE
             sa = 'S'
             CALL NREK_solve( n, H, C, power, weight, X, data, control,        &
                              inform, S = S )
           END IF
           WRITE( 6, "( ' format ', A1, I1, I1, A1, ':',                       &
        &    ' NREK_solve exit status = ', I4 )" ) st, iw, id, sa, inform%status
!stop
         END DO
       END DO
     END DO

     CALL NREK_terminate( data, control, inform ) !  delete internal workspace
     DEALLOCATE( H%val, H%row, H%col, H%type )
     DEALLOCATE( S%val, S%row, S%col, S%type )
!    STOP
   END DO
   DEALLOCATE( H%ptr, S%ptr, C, X )
!stop

!  ==============
!  Normal entries
!  ==============

   n = 3
   ALLOCATE( X( n ), C( n ) )
   CALL SMT_put( H%type, 'COORDINATE', smt_stat )    ! Specify co-ordinate for H
   H%ne = 4
   ALLOCATE( H%val( H%ne ), H%row( H%ne ), H%col( H%ne ) )
   H%row = [ 1, 2, 3, 3 ]
   H%col = [ 1, 2, 3, 1 ]
   H%val = [ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ ]
   CALL SMT_put( S%type, 'DIAGONAL', smt_stat )      ! Specify diagonal for S
   ALLOCATE( S%val( n ) ) ; S%val = 1.0_rp_

   WRITE( 6, "( /, ' ==== normal exits ===== ', / )" )

   DO pass = 1, 7
     C = [ 5.0_rp_, 0.0_rp_, 4.0_rp_ ]
     CALL NREK_initialize( data, control, inform )
     CALL WHICH_sls( control )
     control%error = 23 ; control%out = 23 ; control%print_level = 10
     control%rqs_control%error = 23 ; control%rqs_control%out = 23
     control%sls_control%error = 23 ; control%sls_control%out = 23
     control%sls_control%warning = 23 ; control%sls_control%statistics = 23
     weight = one
     IF ( pass == 2 ) weight = weight * two
     IF ( pass == 3 ) weight = 100000.0_rp_
     IF ( pass == 4 ) weight = 0.1_rp_
     IF ( pass == 5 ) C = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]
     IF ( pass == 6 ) C = [ 0.0_rp_, 2.0_rp_, 0.0001_rp_ ]
     IF ( pass == 7 ) C = [ 0.0_rp_, 0.0_rp_, 0.0_rp_ ]
     if(.false.)then
       control%error = 6 ; control%out = 6 ; control%print_level = 1
       control%rqs_control%error = 6 ; control%rqs_control%out = 6
       control%rqs_control%print_level = 1
       control%stop_check_all_orders = .TRUE.
     endif
     CALL NREK_solve( n, H, C, power, weight, X, data, control, inform, S = S )

     WRITE( 6, "( ' pass  ', I3, ': NREK_solve exit status = ', I6 )" )        &
            pass, inform%status
!    WRITE( 6, "( ' its, solution and Lagrange multiplier = ', I6, 2ES12.4 )") &
!              inform%iter + info%iter_pass2, f, info%multiplier
     CALL NREK_terminate( data, control, inform ) !  delete internal workspace
   END DO

   DEALLOCATE( X, C, H%row, H%col, H%val, H%type, S%val, S%type )
   CLOSE( unit = 23 )
   WRITE( 6, "( /, ' tests completed' )" )

   STOP

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( NREK_control_type ), INTENT( INOUT ) :: control
#include "galahad_sls_defaults_dls.h"
     control%linear_solver = definite_linear_solver
     control%linear_solver_for_S = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_NREK_test_program
