! THIS VERSION: GALAHAD 4.2 - 2023-08-10 AT 07:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LSP_EXAMPLE
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LSP_precision
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPT_dimensions_type ) :: d
   TYPE ( LSP_map_type ) :: map
   TYPE ( LSP_control_type ) :: control
   TYPE ( LSP_inform_type ) :: info
   TYPE ( QPT_problem_type ) :: p
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_orig, Y_orig, Z_orig
   INTEGER ( KIND = ip_ ) :: n, m, o, a_ne, l_ne, smt_stat, data_storage_type
   INTEGER ( KIND = ip_ ) :: i, status
   CHARACTER ( len = 2 ) :: st
   CHARACTER ( len = 10 ) :: sname
   INTEGER ( KIND = ip_ ), PARAMETER :: coordinate = 1, sparse_by_rows = 2
   INTEGER ( KIND = ip_ ), PARAMETER :: sparse_by_columns = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: dense = 4, dense_by_columns = 5

!  GO TO 1
   n = 3 ; m = 2 ; o = 3 ; a_ne = 4 ; l_ne = 4
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ), X_orig( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )
   st = ' '

   p%n = n ; p%m = m ; p%o = o;
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infinity /)
   p%X_l = (/ - 1.0_rp_, - infinity, - infinity /)
   p%X_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)
   p%DB = (/ 2.0_rp_, 0.0_rp_, 0.0_rp_ /)
   p%DC_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%DC_u = (/ 4.0_rp_, infinity /)
   p%DX_l = (/ - 1.0_rp_, - infinity, - infinity /)
   p%DX_u = (/ 1.0_rp_, infinity, 2.0_rp_ /)

   CALL LSP_initialize( map, control )
   control%infinity = infinity

!  tests for status = - 1 ... - 8

!  DO status = 1, 0
   DO status = 1, 8
     ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( 0 ), p%Ao%col( a_ne ) )
     ALLOCATE( p%A%val( l_ne ), p%A%row( 0 ), p%A%col( l_ne ) )
     IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
     CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
     p%Ao%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
     p%Ao%col = (/ 1, 2, 3, 1 /)
     p%Ao%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
     p%A%col = (/ 1, 2, 2, 3 /)
     p%A%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_rp_ ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_

     IF ( status == 1 ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == 2 ) THEN
       p%X_u( 1 ) = - 2.0_rp_
     ELSE IF ( status == 3 ) THEN
       p%C_u( 1 ) = - 2.0_rp_
     END IF
     IF ( status == 4 ) THEN
       CALL LSP_terminate( map, control, info )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL LSP_get_values( map, info, p, X_val = X_orig )
       sname = 'get_values'
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
     ELSE IF ( status == 5 ) THEN
 ! reorder problem
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'DENSE', smt_stat )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
       CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
     ELSE IF ( status == 6 ) THEN
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
       CALL LSP_restore( map, info, p, get_c = .TRUE. )
       WRITE( 6, 10 ) st, status, sname, info%status
     ELSE IF ( status == 7 ) THEN
       sname = 'reorder   '
       DEALLOCATE( p%Z )
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
     ELSE IF ( status == 8 ) THEN
       sname = 'reorder   '
       DEALLOCATE( p%Y )
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
     ELSE
       sname = 'reorder   '
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
     END IF
     WRITE( 6, 10 ) st, status, sname, info%status
     DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col )
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     IF ( status == 1 ) THEN
       p%n = n ; p%m = m
     ELSE IF ( status == 2 ) THEN
       p%X_u( 1 ) = 1.0_rp_
     ELSE IF ( status == 3 ) THEN
       p%C_u( 1 ) = 4.0_rp_
     ELSE IF ( status == 4 ) THEN
       CALL LSP_initialize( map, control )
       control%infinity = infinity
     ELSE IF ( status == 7 ) THEN
       ALLOCATE( p%Z( n ) )
     ELSE IF ( status == 8 ) THEN
       ALLOCATE( p%Y( m ) )
     END IF
   END DO

   CALL LSP_terminate( map, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u, p%Ao%ptr, p%A%ptr )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_orig )
   DEALLOCATE( p%DB, p%DX_l, p%DX_u, p%DC_l, p%DC_u )

!  =====================================
!  basic test of various storage formats
!  =====================================

!1 continue
   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 4 ; m = 2 ; o = 7 ; a_ne = 16 ; l_ne = 5
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_orig( n ), Y_orig( m ), Z_orig( n ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

   p%n = n ; p%m = m ; p%o = o
   p%B = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_, 6.0_rp_, 7.0_rp_ /)
   p%C_l = (/ 1.0_rp_, 2.0_rp_ /)
   p%C_u = (/ 2.0_rp_, 2.0_rp_ /)
   p%X_l = (/ - 1.0_rp_, - infinity, 1.0_rp_, - infinity /)
   p%X_u = (/ 1.0_rp_, infinity, 1.0_rp_, 2.0_rp_ /)

   p%DB = (/ 7.0_rp_, 6.0_rp_, 5.0_rp_, 4.0_rp_, 3.0_rp_, 2.0_rp_, 1.0_rp_ /)
   p%DX_l = (/  1.0_rp_, - 1.0_rp_, - 1.0_rp_, 1.0_rp_ /)
   p%DX_u = (/ 1.0_rp_, 3.0_rp_, 2.0_rp_, 1.0_rp_ /)
   p%DC_l = (/ - 1.0_rp_, - 2.0_rp_ /)
   p%DC_u = (/ 4.0_rp_, 2.0_rp_ /)

   DO data_storage_type = 1, 5
     CALL LSP_initialize( map, control )
     control%infinity = infinity
     IF ( data_storage_type == coordinate ) THEN
       st = 'CO'
       CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( a_ne ) )
       ALLOCATE( p%A%val( l_ne ), p%A%row( l_ne ), p%A%col( l_ne ) )
       ALLOCATE( p%Ao%ptr( 0 ), p%A%ptr( 0 ) )
       p%Ao%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /)
       p%Ao%col = (/ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 /)
       p%Ao%ne = a_ne
       p%A%row = (/ 1, 1, 2, 2, 2 /)
       p%A%col = (/ 1, 2, 2, 3, 4 /)
       p%A%ne = l_ne
     ELSE IF ( data_storage_type == sparse_by_rows ) THEN
       st = 'SR'
       CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', smt_stat )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( 0 ), p%Ao%col( a_ne ) )
       ALLOCATE( p%A%val( l_ne ), p%A%row( 0 ), p%A%col( l_ne ) )
       ALLOCATE( p%Ao%ptr( o + 1 ), p%A%ptr( m + 1 ) )
       p%Ao%col = (/ 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4 /)
       p%Ao%ptr = (/ 1, 2, 4, 7, 11, 14, 16, 17 /)
       p%A%col = (/ 1, 2, 2, 3, 4 /)
       p%A%ptr = (/ 1, 3, 6 /)
     ELSE IF ( data_storage_type == sparse_by_columns ) THEN
       st = 'SC'
       CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', smt_stat )
       CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', smt_stat )
       ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( 0 ) )
       ALLOCATE( p%A%val( l_ne ), p%A%row( l_ne ), p%A%col( 0 ) )
       ALLOCATE( p%Ao%ptr( n + 1 ), p%A%ptr( n + 1 ) )
       p%Ao%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /)
       p%Ao%ptr = (/ 1, 5, 9, 13, 17 /)
       p%A%row = (/ 1, 1, 2, 2, 2 /)
       p%A%ptr = (/ 1, 2, 4, 5, 6 /)
     ELSE IF ( data_storage_type == dense ) THEN
       st = 'DR'
       CALL SMT_put( p%Ao%type, 'DENSE', smt_stat)  ! Specify dense (by rows)
       CALL SMT_put( p%A%type, 'DENSE', smt_stat)  ! storage for A and L
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), p%Ao%ptr( 0 ))
       ALLOCATE( p%A%val( n * m ), p%A%row( 0 ), p%A%col( 0 ), p%A%ptr( 0 ) )
     ELSE IF ( data_storage_type == dense_by_columns ) THEN
       st = 'DC'
       CALL SMT_put( p%Ao%type, 'DENSE_BY_COLUMNS', smt_stat )
       CALL SMT_put( p%A%type, 'DENSE_BY_COLUMNS', smt_stat )
       ALLOCATE( p%Ao%val( n * o ), p%Ao%row( 0 ), p%Ao%col( 0 ), p%Ao%ptr( 0 ))
       ALLOCATE( p%A%val( n * m ), p%A%row( 0 ), p%A%col( 0 ), p%A%ptr( 0 ) )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == coordinate ) THEN
         p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_,                     &
                       2.0_rp_, 1.0_rp_, 1.0_rp_, 5.0_rp_,                     &
                       3.0_rp_, 1.0_rp_, 1.0_rp_, 6.0_rp_,                     &
                       3.0_rp_, 1.0_rp_, 1.0_rp_, 6.0_rp_ /)
       p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == sparse_by_rows ) THEN
         p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_, 1.0_rp_, 3.0_rp_,   &
                       1.0_rp_, 1.0_rp_, 1.0_rp_, 4.0_rp_, 5.0_rp_, 1.0_rp_,   &
                       1.0_rp_, 6.0_rp_, 1.0_rp_, 7.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == sparse_by_columns ) THEN
         p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_,   &
                       1.0_rp_, 5.0_rp_, 3.0_rp_, 1.0_rp_, 1.0_rp_, 6.0_rp_,   &
                       4.0_rp_, 1.0_rp_, 1.0_rp_, 7.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == dense ) THEN
         p%Ao%val = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,                     &
                       1.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_,                     &
                       1.0_rp_, 1.0_rp_, 3.0_rp_, 0.0_rp_,                     &
                       1.0_rp_, 1.0_rp_, 1.0_rp_, 4.0_rp_,                     &
                       0.0_rp_, 5.0_rp_, 1.0_rp_, 1.0_rp_,                     &
                       0.0_rp_, 0.0_rp_, 6.0_rp_, 1.0_rp_,                     &
                       0.0_rp_, 0.0_rp_, 0.0_rp_, 7.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 1.0_rp_,    &
                      1.0_rp_, 1.0_rp_ /)
       ELSE IF ( data_storage_type == dense_by_columns ) THEN
         p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 2.0_rp_, 1.0_rp_, 1.0_rp_, 5.0_rp_,   &
                       0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 3.0_rp_, 1.0_rp_,   &
                       1.0_rp_, 6.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,   &
                       4.0_rp_, 1.0_rp_, 1.0_rp_, 7.0_rp_ /)
         p%A%val = (/ 2.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_, 0.0_rp_, 1.0_rp_,    &
                      0.0_rp_, 1.0_rp_ /)
       END IF
       p%X = MAX( p%X_l, MIN( 0.0_rp_, p%X_u ) ) ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
       CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col,         &
                p%A%ptr, p%X, p%C )
       CALL LSP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
       sname = 'reorder   '
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'restore   '
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'apply     '
       CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col,         &
                p%A%ptr, p%X, p%C )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'get_values'
       CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'restore   '
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
     END DO
     CALL LSP_terminate( map, control, info )

!  delate allocated matrix components

     DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%ptr )
     DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr )
   END DO

!  delate allocated array components

   DEALLOCATE( p%B, p%DB, p%DX_l, p%DX_u, p%DC_l, p%DC_u )
   DEALLOCATE( X_orig, Y_orig, Z_orig, p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u, p%Ao%ptr, p%A%ptr, STAT = i )

!stop

!  ============================
!  full test of generic problem
!  ============================

   WRITE( 6, "( /, ' full test of generic problems ', / )" )
   st = ' '

   n = 14 ; m = 17 ; o = 15 ; a_ne = 28 ; l_ne = 46
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_orig( n ), Y_orig( m ), Z_orig( n ) )
   ALLOCATE( p%Ao%ptr( n + 1 ), p%A%ptr( m + 1 ) )
   ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( a_ne ) )
   ALLOCATE( p%A%val( l_ne ), p%A%row( l_ne ), p%A%col( l_ne ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )
   IF ( ALLOCATED( p%Ao%type ) ) DEALLOCATE( p%Ao%type )
   CALL SMT_put( p%Ao%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%o = o ; p%Ao%ne = a_ne ; p%A%ne = l_ne
   p%B = (/ 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 2.0_rp_,     &
            1.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_, 0.0_rp_, 2.0_rp_, 0.0_rp_,     &
            2.0_rp_ /)
   p%C_l = (/ 4.0_rp_, 2.0_rp_, 6.0_rp_, - infinity, - infinity,               &
              4.0_rp_, 2.0_rp_, 6.0_rp_, - infinity, - infinity,               &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_, - 10.0_rp_,                  &
              - 10.0_rp_, - 10.0_rp_, - 10.0_rp_ /)
   p%C_u = (/ 4.0_rp_, infinity, 10.0_rp_, 2.0_rp_, infinity,                  &
              4.0_rp_, infinity, 10.0_rp_, 2.0_rp_, infinity,                  &
              10.0_rp_, 10.0_rp_, 10.0_rp_, 10.0_rp_,                          &
              10.0_rp_, 10.0_rp_, 10.0_rp_ /)
   p%X_l = (/ 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -infinity, -infinity,        &
              -infinity, 1.0_rp_, 0.0_rp_, 1.0_rp_, 2.0_rp_, -infinity,        &
              -infinity, -infinity /)
   p%X_u = (/ 1.0_rp_, infinity, infinity, 3.0_rp_, 4.0_rp_, 0.0_rp_,          &
              infinity, 1.0_rp_, infinity, infinity, 3.0_rp_, 4.0_rp_,         &
              0.0_rp_, infinity /)
   p%DB = 1.0_rp_ ;  p%DC_l = - 1.0_rp_ ;  p%DC_u = 1.0_rp_
   p%DX_l = - 1.0_rp_ ; p%DX_u = 1.0_rp_
   p%Ao%val = (/ 1.0_rp_, 1.0_rp_, 2.0_rp_, 2.0_rp_, 3.0_rp_, 3.0_rp_,         &
                4.0_rp_, 4.0_rp_, 5.0_rp_, 5.0_rp_, 6.0_rp_, 6.0_rp_,          &
                7.0_rp_, 7.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_,          &
                8.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_,          &
                8.0_rp_, 8.0_rp_, 8.0_rp_, 8.0_rp_  /)
   p%Ao%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                &
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15  /)
   p%Ao%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                     &
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 /)
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

!  WRITE( 6, "( ' A ', /, 5( 2I3, ES8.1 ) )" )                                 &
!    ( p%Ao%row( i ), p%Ao%col( i ), p%Ao%val( i ), i = 1, p%Ao%ne )
   CALL LSP_initialize( map, control )
   control%infinity = infinity
   p%X = MAX( p%X_l, MIN( 0.0_rp_, p%X_u ) ) ; p%Y = 0.0_rp_ ; p%Z = 0.0_rp_
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   sname = 'reorder   '
   CALL LSP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
!  WRITE( 6, "( ' A ', /, 5( 2I3, ES8.1 ) )" )                                 &
!    ( p%Ao%row( i ), p%Ao%col( i ), p%Ao%val( i ), i = 1, p%Ao%ne )

   sname = 'apply     '
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   CALL LSP_apply( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
!  WRITE( 6, "( ' A ', /, 5( 2I3, ES8.1 ) )" )                                 &
!    ( p%Ao%row( i ), p%Ao%col( i ), p%Ao%val( i ), i = 1, p%Ao%ne )
   sname = 'apply     '
   CALL LSP_apply( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
!  WRITE( 6, "( ' A ', /, 5( 2I3, ES8.1 ) )" )                                 &
!    ( p%Ao%row( i ), p%Ao%col( i ), p%Ao%val( i ), i = 1, p%Ao%ne )
   sname = 'apply     '
   CALL AX( p%m, p%n, p%A%type, p%A%ne, p%A%val, p%A%row, p%A%col, p%A%ptr,    &
            p%X, p%C )
   CALL LSP_apply( map, info, p, get_Ao = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_A = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_x = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_y = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_z = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_b = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_db = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_c = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_x_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_dx_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_c_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_dc_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status

   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_c = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_c_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_dc_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_x_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_dx_bounds = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_db = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_b = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_A = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_Ao = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_x = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_y = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_z = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_terminate( map, control, info )

   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, p%DB, p%DX_l, p%DX_u, p%DC_l, p%DC_u )
   DEALLOCATE( X_orig, Y_orig, Z_orig )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%type, STAT = smt_stat )
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type, STAT = smt_stat )

   WRITE( 6, "( /, ' tests completed' )" )

10 FORMAT( A2, I1, ': LSP_', A10, ' exit status = ', I6 )

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
   IF ( SMT_get( a_type ) == 'DENSE' .OR.                                      &
        SMT_get( a_type ) == 'DENSE_BY_ROWS' ) THEN
     l = 0
     DO i = 1, m
       C( i ) = DOT_PRODUCT( A_val( l + 1 : l + n ), X )
       l = l + n
     END DO
   ELSE IF ( SMT_get( a_type ) == 'DENSE_BY_COLUMNS' ) THEN
     l = 0 ; C = 0.0_rp_
     DO i = 1, n
       C = C + A_val( l + 1 : l + m ) * X( i )
       l = l + m
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

   END PROGRAM GALAHAD_LSP_EXAMPLE

