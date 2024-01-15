! THIS VERSION: GALAHAD 4.2 - 2023-08-10 AT 07:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_QPP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_QPP_precision
   USE GALAHAD_LMS_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infty = 10.0_rp_ ** 20
   TYPE ( QPT_dimensions_type ) :: d
   TYPE ( QPP_map_type ) :: map
   TYPE ( QPP_control_type ) :: control
   TYPE ( QPP_inform_type ) :: info
   TYPE ( QPT_problem_type ) :: p
   TYPE ( LMS_control_type ) :: LMS_control
   TYPE ( LMS_inform_type ) :: LMS_inform
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_orig, Y_orig, Z_orig
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S, Y
   INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, smt_stat
   INTEGER ( KIND = ip_ ) :: data_storage_type, i, j, status
   REAL ( KIND = rp_ ) :: delta
   CHARACTER ( len = 1 ) :: st
   CHARACTER ( len = 10 ) :: sname

!  GO TO 1
   n = 3 ; m = 2 ; h_ne = 4 ; a_ne = 4
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ), X_orig( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%DG( n ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )
   st = ' '

   p%Hessian_kind = - 1 ; p%gradient_kind = - 1
   p%n = n ; p%m = m ; p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_ /)
   p%DG = (/ 2.0_rp_, 0.0_rp_, 0.0_rp_ /)
   p%DC_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%DC_u = (/ 4.0_rp_, infty /)
   p%DX_l = (/ - 1.0_rp_, - infty, - infty /)
   p%DX_u = (/ 1.0_rp_, infty, 2.0_rp_ /)

   CALL QPP_initialize( map, control )
   control%infinity = infty

!  tests for status = - 1 ... - 8

!  DO status = 1, 0
   DO status = 1, 8
     IF ( status == 2 ) CYCLE
     IF ( status == 3 ) CYCLE
     ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
     CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
     p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
     p%H%col = (/ 1, 2, 3, 1 /)
     p%H%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

     IF ( status == 1 ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == 4 ) THEN
       p%H%col( 1 ) = 2
     ELSE IF ( status == 5 ) THEN
       p%X_u( 1 ) = - 2.0_rp_
     END IF
     IF ( status == 6 ) THEN
       CALL QPP_terminate( map, control, info )
       CALL QPP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL QPP_get_values( map, info, p, X_val = X_orig )
       sname = 'get_values'
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL QPP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
     ELSE IF ( status == 7 ) THEN
 ! reorder problem
       CALL QPP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL QPP_restore( map, info, p, get_all = .TRUE. )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       CALL QPP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
       CALL QPP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
     ELSE IF ( status == 8 ) THEN
       CALL QPP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL QPP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
       CALL QPP_restore( map, info, p, get_c = .TRUE. )
       WRITE( 6, 10 ) st, status, sname, info%status
       sname = 'apply     '
       CALL QPP_apply( map, info, p, get_c = .TRUE. )
     ELSE
 ! reorder problem
       CALL QPP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       sname = 'reorder   '
     END IF
     WRITE( 6, 10 ) st, status, sname, info%status
     DEALLOCATE( p%H%val, p%H%row, p%H%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( status == 1 ) THEN
       p%n = n ; p%m = m
     ELSE IF ( status == 2 ) THEN
     ELSE IF ( status == 3 ) THEN
     ELSE IF ( status == 5 ) THEN
       p%X_u( 1 ) = 1.0_rp_
     ELSE IF ( status == 6 ) THEN
       CALL QPP_initialize( map, control )
       control%infinity = infty
     END IF
   END DO

   CALL QPP_terminate( map, control, info )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u, p%H%ptr, p%A%ptr )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_orig )
   DEALLOCATE( p%DG, p%DX_l, p%DX_u, p%DC_l, p%DC_u )

!  =====================================
!  basic test of various storage formats
!  =====================================

!1 continue
   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 4 ; m = 2 ; h_ne = 5 ; a_ne = 6
   ALLOCATE( p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( MAX( m, n ) + 1 ) )
   ALLOCATE( X_orig( n ), Y_orig( m ), Z_orig( n ) )
   ALLOCATE( p%DG( n ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

   p%Hessian_kind = - 1 ; p%gradient_kind = - 1
   p%n = n ; p%m = m ; p%f = 0.96_rp_
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty /)
   p%X_l = (/ - 1.0_rp_, - infty, - infty, 1.0_rp_ /)
   p%X_u = (/ 1.0_rp_, infty, 2.0_rp_, 1.0_rp_ /)
   p%DG = (/ 0.0_rp_, -2.0_rp_, 0.0_rp_, 1.0_rp_ /)
   p%DC_l = (/ - 1.0_rp_, - 2.0_rp_ /)
   p%DC_u = (/ 4.0_rp_, 2.0_rp_ /)
   p%DX_l = (/  1.0_rp_, - 1.0_rp_, - 1.0_rp_, 1.0_rp_ /)
   p%DX_u = (/ 1.0_rp_, 3.0_rp_, 2.0_rp_, 1.0_rp_ /)

   DO data_storage_type = -10, 3
!  DO data_storage_type = -1, -2, -1
!  DO data_storage_type = -8, 3
!  DO data_storage_type = -8, -8
!  DO data_storage_type = -7, -7
     CALL QPP_initialize( map, control )
     control%infinity = infty
     IF ( data_storage_type == 3 ) THEN           ! weighted least-distance
       st = 'G'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       ALLOCATE( p%X0( n ), p%WEIGHT( n ) )
       p%Hessian_kind = 2 ; p%gradient_kind = 1
       p%A%row = (/ 1, 1, 2, 2, 1, 2 /)
       p%A%col = (/ 1, 2, 2, 3, 4, 4 /) ; p%A%ne = a_ne
       p%WEIGHT = (/ 1.0_rp_, 2.0_rp_, 2.0_rp_, 1.0_rp_ /)
       p%X0 = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == 2 ) THEN           ! weighted least-distance
       st = 'W'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       ALLOCATE( p%X0( n ), p%WEIGHT( n ) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = 2 ; p%gradient_kind = - 1
       p%A%row = (/ 1, 1, 2, 2, 1, 2 /)
       p%A%col = (/ 1, 2, 2, 3, 4, 4 /) ; p%A%ne = a_ne
       p%WEIGHT = (/ 1.0_rp_, 2.0_rp_, 2.0_rp_, 1.0_rp_ /)
       p%X0 = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == 1 ) THEN       ! least-distance
       st = 'L'
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       ALLOCATE( p%X0( n ) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = 1 ; p%gradient_kind = - 1
       p%A%row = (/ 1, 1, 2, 2, 1, 2 /)
       p%A%col = (/ 1, 2, 2, 3, 4, 4 /) ; p%A%ne = a_ne
       p%X0 = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == 0 ) THEN        ! sparse co-ordinate storage
       st = 'C'
       ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = - 1 ; p%gradient_kind = - 1
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
       p%H%row = (/ 1, 2, 3, 3, 4 /)
       p%H%col = (/ 1, 2, 3, 1, 4 /) ; p%H%ne = h_ne
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       p%A%row = (/ 1, 1, 2, 2, 1, 2 /)
       p%A%col = (/ 1, 2, 2, 3, 4, 4 /) ; p%A%ne = a_ne
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == - 1 ) THEN     ! sparse row-wise storage
       st = 'R'
       ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = - 1 ; p%gradient_kind = - 1
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
       p%H%col = (/ 1, 2, 3, 1, 4 /)
       p%H%ptr = (/ 1, 2, 3, 5, 6 /)
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       p%A%col = (/ 1, 2, 4, 2, 3, 4 /)
       p%A%ptr = (/ 1, 4, 7 /)
!      p%A%col = (/ 2, 4, 1, 2, 3, 4 /)
!      p%A%ptr = (/ 1, 3, 7 /)
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == - 2 ) THEN     ! sparse column-wise storage
       st = 'L'
       ALLOCATE( p%H%val( h_ne ), p%H%row( 0 ), p%H%col( h_ne ) )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( 0 ) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = - 1 ; p%gradient_kind = - 1
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SPARSE_BY_ROWS', smt_stat )
       p%H%col = (/ 1, 2, 3, 1, 4 /)
       p%H%ptr = (/ 1, 2, 3, 5, 6 /)
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', smt_stat )
       p%A%row = (/ 1, 1, 2, 2, 1, 2 /)
!      p%A%row = (/ 2, 1, 2, 2, 1, 2 /)
       p%A%ptr = (/ 1, 2, 4, 5, 7 /)
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == - 3 ) THEN      ! dense storage
       st = 'D'
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(n*(n+1)/2))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       ALLOCATE( p%G( n ) )
       p%Hessian_kind = - 1 ; p%gradient_kind = - 1
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 1.0_rp_ /)
     ELSE IF ( data_storage_type == - 4 ) THEN      ! dense storage
       st = 'Z'
       ALLOCATE( p%H%val(n*(n+1)/2), p%H%row(0), p%H%col(n*(n+1)/2))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DENSE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 5 ) THEN      ! diagonal storage (for H)
       st = 'I'
       ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'DIAGONAL', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 6 ) THEN      !scaled identity storage (H)
       st = 'C'
       ALLOCATE( p%H%val(1), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'SCALED_IDENTITY', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 7 ) THEN      ! identity storage (for H)
       st = 'I'
       ALLOCATE( p%H%val(0), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'IDENTITY', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 8 ) THEN      ! no Hessian storage (for H)
       st = 'N'
       ALLOCATE( p%H%val(0), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'NONE', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
     ELSE IF ( data_storage_type == - 9 ) THEN      ! limited-memory storage (H)
       st = 'B'
!      ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )

       CALL LMS_initialize( p%H_lm, LMS_control, LMS_inform )
       LMS_control%memory_length = 2
       LMS_control%method = 1
       CALL LMS_setup( n, p%H_lm, LMS_control, LMS_inform )
       ALLOCATE( S( p%n ), Y( p%n ) )
       DO i = 1, p%n + 2
         S = 1.0_rp_
         S( 1 ) = REAL( MOD( i, p%n ) + 1, KIND = rp_ )
         Y = S
         delta = 1.0_rp_ / S( 1 )
!         S = 0.0_rp_
!         S( MOD( i - 1, p%n ) + 1 ) = 1.0_rp_
!         Y = S
!         delta = REAL( MOD( i, 3 ) + 1, KIND = rp_ )
         CALL LMS_form( S, Y, delta, p%H_lm, LMS_control, LMS_inform )
       END DO
       DEALLOCATE( S, Y )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'LBFGS', smt_stat )
     ELSE IF ( data_storage_type == - 10 ) THEN     ! limited-memory storage (H)
       st = 'A'
!      ALLOCATE( p%H%val(n), p%H%row(0), p%H%col(n))
       ALLOCATE( p%A%val(n*m), p%A%row(0), p%A%col(n*m) )
       p%Hessian_kind = - 1 ; p%gradient_kind = 0
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )

       CALL LMS_initialize( p%H_lm, LMS_control, LMS_inform )
       LMS_control%memory_length = 2
       LMS_control%method = 1
       CALL LMS_setup( p%n + 1, p%H_lm, LMS_control, LMS_inform )
       ALLOCATE( S( p%n + 1 ), Y( p%n + 1 ) )
       DO i = 1, p%n + 2
         S = 1.0_rp_
         S( 1 ) = REAL( MOD( i, p%n ) + 1, KIND = rp_ )
         Y = S
         delta = 1.0_rp_ / S( 1 )
!         S = 0.0_rp_
!         S( MOD( i - 1, p%n ) + 1 ) = 1.0_rp_
!         Y = S
!         delta = REAL( MOD( i, 3 ) + 1, KIND = rp_ )
         CALL LMS_form( S, Y, delta, p%H_lm, LMS_control, LMS_inform )
       END DO
       DEALLOCATE( S, Y )
       ALLOCATE( p%H_lm%RESTRICTION( p%n ), map%W( p%n + 1 ) )
       IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
       CALL SMT_put( p%H%type, 'LBFGS', smt_stat )
     END IF

!  test with new and existing data

     DO i = 1, 2
!    DO i = 1, 1
       IF ( data_storage_type > 0 )                                            &
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       IF ( data_storage_type == 0 ) THEN          ! sparse co-ordinate storage
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 1 ) THEN    !  sparse row-wise storage
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 2 ) THEN    ! sparse column-wise storage
         p%H%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 3 .OR.                                 &
                 data_storage_type == - 4 ) THEN    !  dense storage
         p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_, 0.0_rp_, 3.0_rp_,    &
                    0.0_rp_, 1.0_rp_, 0.0_rp_, 5.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 5 ) THEN   !  dense/diagonal storage
         p%H%val = (/ 1.0_rp_, 0.0_rp_, 2.0_rp_, 4.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 6 ) THEN   !  dense/scaled id storage
         p%H%val( 1 )  = 4.0_rp_
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 7 ) THEN   !  dense/identity storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 8 ) THEN   !  dense/none storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 9 ) THEN   !  dense/diagonal storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == - 10 ) THEN   !  dense/diagonal storage
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,                      &
                    0.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
         p%H_lm%restricted = 1
         p%H_lm%n = p%n + 1
         p%H_lm%n_restriction = p%n
         DO j = 1, p%n
           p%H_lm%RESTRICTION( j ) = p%n + 1 - j
         END DO
       END IF
       p%X = MAX( p%X_l, MIN( 0.0_rp_, p%X_u ) ) ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col,         &
                p%A%ptr, p%X, p%C )
       CALL QPP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
       sname = 'reorder   '
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'restore   '
       CALL QPP_restore( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'apply     '
       CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col,         &
                p%A%ptr, p%X, p%C )
       CALL QPP_apply( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'get_values'
       CALL QPP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'restore   '
       CALL QPP_restore( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
     END DO
     CALL QPP_terminate( map, control, info )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( data_storage_type <= 0 ) THEN
       IF ( data_storage_type == - 9 .OR. data_storage_type == - 10 ) THEN
         CALL LMS_terminate( p%H_lm, LMS_control, LMS_inform )
       ELSE
         DEALLOCATE( p%H%val, p%H%row, p%H%col )
       END IF
     ELSE
       DEALLOCATE( p%X0 )
       IF ( data_storage_type > 1 )  DEALLOCATE( p%WEIGHT )
     END IF
     IF ( ALLOCATED( p%G ) ) DEALLOCATE( p%G )
   END DO
   DEALLOCATE( p%H%ptr, p%A%ptr )
   DEALLOCATE( p%DG, p%DX_l, p%DX_u, p%DC_l, p%DC_u )
   DEALLOCATE( X_orig, Y_orig, Z_orig, p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u, p%H%ptr, p%A%ptr, STAT = i )

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )
   st = ' '

   n = 14 ; m = 17 ; h_ne = 14 ; a_ne = 46
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_orig( n ), Y_orig( m ), Z_orig( n ) )
   ALLOCATE( p%H%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%H%val( h_ne ), p%H%row( h_ne ), p%H%col( h_ne ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%DG( n ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )
   p%Hessian_kind = - 1 ; p%gradient_kind = - 1
   IF ( ALLOCATED( p%H%type ) ) DEALLOCATE( p%H%type )
   CALL SMT_put( p%H%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%H%ne = h_ne ; p%A%ne = a_ne
   p%f = 1.0_rp_
   p%G = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_ /)
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - infty, - infty,                     &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              4.0_rp_, infty, 10.0_rp_, 2.0_rp_, infty,                        &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty,   &
              1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, - infty, - infty, - infty /)
   p%X_u = (/ 1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty,         &
              1.0_rp_, infty, infty, 3.0_rp_, 4.0_rp_, 0.0_rp_, infty /)
   p%DG = 1.0_rp_
   p%DC_l = - 1.0_rp_
   p%DC_u = 1.0_rp_
   p%DX_l = - 1.0_rp_
   p%DX_u = 1.0_rp_
   p%H%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,          &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_ /)
   p%H%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)
   p%H%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 /)
   p%A%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,          &
                1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                   &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_, 1.0_rp_, - 1.0_rp_,    &
                1.0_rp_, - 1.0_rp_ /)
   p%A%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%A%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /)

   CALL QPP_initialize( map, control )
   control%infinity = infty
   p%X = MAX( p%X_l, MIN( 0.0_rp_, p%X_u ) ) ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   sname = 'reorder   '
   CALL QPP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL QPP_restore( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'apply     '
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   CALL QPP_apply( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL QPP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL QPP_restore( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'apply     '
   CALL QPP_apply( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL QPP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL QPP_restore( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'apply     '
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   CALL QPP_apply( map, info, p, get_A = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_H = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_x = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_y = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_z = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_g = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_dg = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_c = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_x_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_dx_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_c_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_apply( map, info, p, get_dc_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL QPP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status

   sname = 'restore   '
   CALL QPP_restore( map, info, p, get_c = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_c_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_dc_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_x_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_dx_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_dg = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_g = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_A = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_H = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_x = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_y = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_restore( map, info, p, get_z = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL QPP_terminate( map, control, info )
   DEALLOCATE( p%H%val, p%H%row, p%H%col )
   DEALLOCATE( p%A%val, p%A%row, p%A%col )
   DEALLOCATE( p%G, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%DG, p%DX_l, p%DX_u, p%DC_l, p%DC_u )
   DEALLOCATE( X_orig, Y_orig, Z_orig )
   WRITE( 6, "( /, ' tests completed' )" )

10 FORMAT( A1, I1, ': QPP_', A10, ' exit status = ', I6 )
!30 FORMAT( A4, /, ( 6ES12.4 ) )
!40 FORMAT( A4, /, ( 3( 2I3, ES12.4 ) ) )

   CONTAINS

   SUBROUTINE AX(  m, n, a_type, a_ne, A_val, A_row, A_col, A_ptr, X, C )

   INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, a_ne
   CHARACTER, INTENT( IN ), DIMENSION( : ) :: a_type
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) ::  A_row, A_col
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: A_ptr
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: A_val
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( m ) :: C
   INTEGER ( KIND = ip_ ) :: i, l

   C = 0.0_rp_
   IF ( SMT_get( a_type ) == 'DENSE' ) THEN
     l = 0
     DO i = 1, m
       C( i ) = DOT_PRODUCT( A_val( l + 1 : l + n ), X )
       l = l + n
     END DO
   ELSE IF ( SMT_get( a_type ) == 'SPARSE_BY_ROWS' ) THEN
     DO i = 1, m
       DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
         C( i ) = C( i ) + A_val( l ) * X( A_col( l ) )
       END DO
     END DO
   ELSE IF ( SMT_get( a_type ) == 'SPARSE_BY_COLUMNS' ) THEN
     DO i = 1, n
       DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
         C( A_row( l ) ) = C( A_row( l ) ) + A_val( l ) * X( i )
       END DO
     END DO
   ELSE  ! sparse co-ordinate
     DO l = 1, a_ne
       i = A_row( l )
       C( i ) = C( i ) + A_val( l ) * X( A_col( l ) )
     END DO
   END IF

   RETURN
   END SUBROUTINE AX

   END PROGRAM GALAHAD_QPP_EXAMPLE

