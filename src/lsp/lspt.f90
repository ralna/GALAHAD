! THIS VERSION: GALAHAD 4.1 - 2022-08-02 AT 15:25 GMT.
   PROGRAM GALAHAD_LSP_EXAMPLE
   USE GALAHAD_LSP_double                    ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_dimensions_type ) :: d
   TYPE ( LSP_map_type ) :: map
   TYPE ( LSP_control_type ) :: control        
   TYPE ( LSP_inform_type ) :: info
   TYPE ( QPT_problem_type ) :: p
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_orig, Y_orig, Z_orig
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S, Y
   INTEGER :: n, m, o, a_ne, l_ne, smt_stat
   INTEGER :: data_storage_type, i, j, status, iname = 0
   REAL ( KIND = wp ) :: delta
   CHARACTER ( len = 2 ) :: st
   CHARACTER ( len = 10 ) :: sname
   INTEGER, PARAMETER :: coordinate = 1, sparse_by_rows = 2
   INTEGER, PARAMETER :: sparse_by_columns = 3, dense = 4, dense_by_columns = 5

   GO TO 1
   n = 3 ; m = 2 ; o = 4 ; a_ne = 4 ; l_ne = 4 
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ), X_orig( n ) )
   ALLOCATE( p%A%ptr( n + 1 ), p%L%ptr( m + 1 ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )
   st = ' '

   p%n = n ; p%m = m ; p%f = 1.0_wp
   p%B = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 4.0_wp, infinity /)
   p%X_l = (/ - 1.0_wp, - infinity, - infinity /)
   p%X_u = (/ 1.0_wp, infinity, 2.0_wp /)
   p%DB = (/ 2.0_wp, 0.0_wp, 0.0_wp /)
   p%DC_l = (/ 1.0_wp, 2.0_wp /)
   p%DC_u = (/ 4.0_wp, infinity /)
   p%DX_l = (/ - 1.0_wp, - infinity, - infinity /)
   p%DX_u = (/ 1.0_wp, infinity, 2.0_wp /)

   CALL LSP_initialize( map, control )
   control%infinity = infinity

!  tests for status = - 1 ... - 8

!  DO status = 1, 0
   DO status = 1, 8
     IF ( status == 2 ) CYCLE
     IF ( status == 3 ) CYCLE
     ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
     ALLOCATE( p%L%val( l_ne ), p%L%row( 0 ), p%L%col( l_ne ) )
     IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
     CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     p%A%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
     p%A%col = (/ 1, 2, 3, 1 /)
     p%A%ptr = (/ 1, 2, 3, 5 /)
     IF ( ALLOCATED( p%L%type ) ) DEALLOCATE( p%L%type )
     CALL SMT_put( p%L%type, 'SPARSE_BY_ROWS', smt_stat ) 
     p%L%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
     p%L%col = (/ 1, 2, 2, 3 /)
     p%L%ptr = (/ 1, 3, 5 /)
     p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp

     IF ( status == 1 ) THEN
       p%n = 0 ; p%m = - 1
     ELSE IF ( status == 4 ) THEN 
       p%A%col( 1 ) = 2
     ELSE IF ( status == 5 ) THEN 
       p%X_u( 1 ) = - 2.0_wp
     END IF
     IF ( status == 6 ) THEN
       CALL LSP_terminate( map, control, info )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL LSP_get_values( map, info, p, X_val = X_orig )
       sname = 'get_values'
       WRITE( 6, 10 ) st, status, sname, info%status
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
     ELSE IF ( status == 7 ) THEN
 ! reorder problem
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       IF ( ALLOCATED( p%L%type ) ) DEALLOCATE( p%L%type )
       CALL SMT_put( p%L%type, 'DENSE', smt_stat )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       WRITE( 6, 10 ) st, status, sname, info%status
       IF ( ALLOCATED( p%L%type ) ) DEALLOCATE( p%L%type )
       CALL SMT_put( p%L%type, 'SPARSE_BY_ROWS', smt_stat )
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'DENSE', smt_stat )
       CALL LSP_apply( map, info, p, get_all = .TRUE. )
       sname = 'apply     '
       IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
     ELSE IF ( status == 8 ) THEN
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       sname = 'restore   '
       CALL LSP_restore( map, info, p, get_c = .TRUE. )
       WRITE( 6, 10 ) st, status, sname, info%status
       sname = 'apply     '
       CALL LSP_apply( map, info, p, get_c = .TRUE. )
     ELSE
 ! reorder problem
       CALL LSP_reorder( map, control, info, d, p, .TRUE., .TRUE., .TRUE. )
       sname = 'reorder   '
     END IF
     WRITE( 6, 10 ) st, status, sname, info%status
     DEALLOCATE( p%A%val, p%A%row, p%A%col )
     DEALLOCATE( p%L%val, p%L%row, p%L%col )
     IF ( status == 1 ) THEN
       p%n = n ; p%m = m
     ELSE IF ( status == 2 ) THEN
     ELSE IF ( status == 3 ) THEN
     ELSE IF ( status == 5 ) THEN
       p%X_u( 1 ) = 1.0_wp
     ELSE IF ( status == 6 ) THEN
       CALL LSP_initialize( map, control )
       control%infinity = infinity
     END IF
   END DO

   CALL LSP_terminate( map, control, info )
   DEALLOCATE( p%B, p%X_l, p%X_u, p%C_l, p%C_u, p%A%ptr, p%L%ptr )
   DEALLOCATE( p%X, p%Y, p%Z, p%C, X_orig )
   DEALLOCATE( p%DB, p%DX_l, p%DX_u, p%DC_l, p%DC_u )

!  =====================================
!  basic test of various storage formats
!  =====================================

1 continue
   WRITE( 6, "( /, ' basic tests of storage formats ', / )" )

   n = 4 ; m = 2 ; o = 7 ; a_ne = 16 ; l_ne = 5
   ALLOCATE( p%B( o ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%C( m ), p%C_l( m ), p%C_u( m ) )
   ALLOCATE( p%X( n ), p%Y( m ), p%Z( n ) )
   ALLOCATE( X_orig( n ), Y_orig( m ), Z_orig( n ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )

   p%n = n ; p%m = m ; p%o = o
   p%B = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp, 7.0_wp /)
   p%C_l = (/ 1.0_wp, 2.0_wp /)
   p%C_u = (/ 2.0_wp, 2.0_wp /)
   p%X_l = (/ - 1.0_wp, - infinity, 1.0_wp, - infinity /)
   p%X_u = (/ 1.0_wp, infinity, 1.0_wp, 2.0_wp /)

   p%DB = (/ 7.0_wp, 6.0_wp, 5.0_wp, 4.0_wp, 3.0_wp, 2.0_wp, 1.0_wp /)
   p%DX_l = (/  1.0_wp, - 1.0_wp, - 1.0_wp, 1.0_wp /)
   p%DX_u = (/ 1.0_wp, 3.0_wp, 2.0_wp, 1.0_wp /)
   p%DC_l = (/ - 1.0_wp, - 2.0_wp /)
   p%DC_u = (/ 4.0_wp, 2.0_wp /)

   DO data_storage_type = 1, 5
     CALL LSP_initialize( map, control )
     control%infinity = infinity
     IF ( data_storage_type == coordinate ) THEN
       st = 'CO'
       CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
       CALL SMT_put( p%L%type, 'COORDINATE', smt_stat )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
       ALLOCATE( p%L%val( l_ne ), p%L%row( l_ne ), p%L%col( l_ne ) )
       ALLOCATE( p%A%ptr( 0 ), p%L%ptr( 0 ) )
       p%A%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /)
       p%A%col = (/ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 /)
       p%A%ne = a_ne
       p%L%row = (/ 1, 1, 2, 2, 2 /)
       p%L%col = (/ 1, 2, 2, 3, 4 /)
       p%L%ne = l_ne
     ELSE IF ( data_storage_type == sparse_by_rows ) THEN
       st = 'SR'
       CALL SMT_put( p%A%type, 'SPARSE_BY_ROWS', smt_stat )
       CALL SMT_put( p%L%type, 'SPARSE_BY_ROWS', smt_stat )
       ALLOCATE( p%A%val( a_ne ), p%A%row( 0 ), p%A%col( a_ne ) )
       ALLOCATE( p%L%val( l_ne ), p%L%row( 0 ), p%L%col( l_ne ) )
       ALLOCATE( p%A%ptr( o + 1 ), p%L%ptr( m + 1 ) )
       p%A%col = (/ 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4 /) 
       p%A%ptr = (/ 1, 2, 4, 7, 11, 14, 16, 17 /)
       p%L%col = (/ 1, 2, 2, 3, 4 /)
       p%L%ptr = (/ 1, 3, 6 /)
     ELSE IF ( data_storage_type == sparse_by_columns ) THEN
       st = 'SC'
       CALL SMT_put( p%A%type, 'SPARSE_BY_COLUMNS', smt_stat )
       CALL SMT_put( p%L%type, 'SPARSE_BY_COLUMNS', smt_stat )
       ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( 0 ) )
       ALLOCATE( p%L%val( l_ne ), p%L%row( l_ne ), p%L%col( 0 ) )
       ALLOCATE( p%A%ptr( n + 1 ), p%L%ptr( n + 1 ) )
       p%A%row = (/ 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7 /) 
       p%A%ptr = (/ 1, 5, 9, 13, 17 /)
       p%L%row = (/ 1, 1, 2, 2, 2 /)
       p%L%ptr = (/ 1, 2, 4, 5, 6 /)
     ELSE IF ( data_storage_type == dense ) THEN
       st = 'DR'
       CALL SMT_put( p%A%type, 'DENSE', smt_stat)  ! Specify dense (by rows)
       CALL SMT_put( p%L%type, 'DENSE', smt_stat)  ! storage for A and L
       ALLOCATE( p%A%val( n * o ), p%A%row( 0 ), p%A%col( 0 ), p%A%ptr( 0 ) )
       ALLOCATE( p%L%val( n * m ), p%L%row( 0 ), p%L%col( 0 ), p%L%ptr( 0 ) )
     ELSE IF ( data_storage_type == dense_by_columns ) THEN
       st = 'DC'
       CALL SMT_put( p%A%type, 'DENSE_BY_COLUMNS', smt_stat )
       CALL SMT_put( p%L%type, 'DENSE_BY_COLUMNS', smt_stat )
       ALLOCATE( p%A%val( n * o ), p%A%row( 0 ), p%A%col( 0 ), p%A%ptr( 0 ) )
       ALLOCATE( p%L%val( n * m ), p%L%row( 0 ), p%L%col( 0 ), p%L%ptr( 0 ) )
     END IF

!  test with new and existing data

     DO i = 1, 2
       IF ( data_storage_type == coordinate ) THEN
         p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                          &
                      2.0_wp, 1.0_wp, 1.0_wp, 5.0_wp,                          &
                      3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp,                          &
                      3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp /)
       p%L%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == sparse_by_rows ) THEN
         p%A%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 1.0_wp, 1.0_wp, 3.0_wp,          &
                      1.0_wp, 1.0_wp, 1.0_wp, 4.0_wp, 5.0_wp, 1.0_wp,          &
                      1.0_wp, 6.0_wp, 1.0_wp, 7.0_wp /)
         p%L%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == sparse_by_columns ) THEN
         p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp, 1.0_wp,          &
                      1.0_wp, 5.0_wp, 3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp,          &
                      4.0_wp, 1.0_wp, 1.0_wp, 7.0_wp /)
         p%L%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == dense ) THEN
         p%A%val = (/ 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                         & 
                      1.0_wp, 2.0_wp, 0.0_wp, 0.0_wp,                         & 
                      1.0_wp, 1.0_wp, 3.0_wp, 0.0_wp,                         & 
                      1.0_wp, 1.0_wp, 1.0_wp, 4.0_wp,                         & 
                      0.0_wp, 5.0_wp, 1.0_wp, 1.0_wp,                         & 
                      0.0_wp, 0.0_wp, 6.0_wp, 1.0_wp,                         & 
                      0.0_wp, 0.0_wp, 0.0_wp, 7.0_wp /)
         p%L%val = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp,         &
                      1.0_wp, 1.0_wp /)
       ELSE IF ( data_storage_type == dense_by_columns ) THEN
         p%A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,  &
                      0.0_wp, 2.0_wp, 1.0_wp, 1.0_wp, 5.0_wp, 0.0_wp, 0.0_wp,  &
                      0.0_wp, 0.0_wp, 3.0_wp, 1.0_wp, 1.0_wp, 6.0_wp, 0.0_wp,  &
                      0.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp, 1.0_wp, 7.0_wp /)
         p%L%val = (/ 2.0_wp, 0.0_wp, 1.0_wp, 1.0_wp, 0.0_wp, 1.0_wp,          &
                      0.0_wp, 1.0_wp /)
       END IF
       p%X = MAX( p%X_l, MIN( 0.0_wp, p%X_u ) ) ; p%Y = 0.0_wp ; p%Z = 0.0_wp
       CALL LX( p%m, p%n, p%L%type, p%L%ne, p%L%val, p%L%row, p%L%col,         &
                p%L%ptr, p%X, p%C )
       iname = iname + 1
       CALL LSP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
       sname = 'reorder   '
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'restore   '
       CALL LSP_restore( map, info, p, get_all = .TRUE. )
       WRITE( 6, 10 ) st, i, sname, info%status
       sname = 'apply     '
       iname = iname + 1
       CALL LX( p%m, p%n, p%L%type, p%L%ne, p%L%val, p%L%row, p%L%col,         &
                p%L%ptr, p%X, p%C )
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

     DEALLOCATE( p%A%val, p%A%row, p%A%col, p%A%ptr )
     DEALLOCATE( p%L%val, p%L%row, p%L%col, p%L%ptr )
   END DO

!  delate allocated array components

   DEALLOCATE( p%B, p%DB, p%DX_l, p%DX_u, p%DC_l, p%DC_u )
   DEALLOCATE( X_orig, Y_orig, Z_orig, p%X, p%Y, p%Z, p%C )
   DEALLOCATE( p%X_l, p%X_u, p%C_l, p%C_u, p%A%ptr, p%L%ptr, STAT = i )

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
   ALLOCATE( p%A%ptr( n + 1 ), p%L%ptr( m + 1 ) )
   ALLOCATE( p%A%val( a_ne ), p%A%row( a_ne ), p%A%col( a_ne ) )
   ALLOCATE( p%L%val( l_ne ), p%L%row( l_ne ), p%L%col( l_ne ) )
   ALLOCATE( p%DB( o ), p%DX_l( n ), p%DX_u( n ), p%DC_l( m ), p%DC_u( m ) )
   IF ( ALLOCATED( p%A%type ) ) DEALLOCATE( p%A%type )
   CALL SMT_put( p%A%type, 'COORDINATE', smt_stat )
   IF ( ALLOCATED( p%L%type ) ) DEALLOCATE( p%L%type )
   CALL SMT_put( p%L%type, 'COORDINATE', smt_stat )
   p%n = n ; p%m = m ; p%o = o ; p%A%ne = a_ne ; p%L%ne = l_ne 
   p%B = (/ 0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp, 1.0_wp,    &
            0.0_wp, 2.0_wp, 0.0_wp, 0.0_wp, 2.0_wp, 0.0_wp, 2.0_wp /) 
   p%C_l = (/ 4.0_wp, 2.0_wp, 6.0_wp, - infinity, - infinity,                  &
              4.0_wp, 2.0_wp, 6.0_wp, - infinity, - infinity,                  &
              - 10.0_wp, - 10.0_wp, - 10.0_wp, - 10.0_wp,                      &
              - 10.0_wp, - 10.0_wp, - 10.0_wp /)
   p%C_u = (/ 4.0_wp, infinity, 10.0_wp, 2.0_wp, infinity,                     &
              4.0_wp, infinity, 10.0_wp, 2.0_wp, infinity,                     &
              10.0_wp, 10.0_wp, 10.0_wp, 10.0_wp,                              &
              10.0_wp, 10.0_wp, 10.0_wp /)
   p%X_l = (/ 1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, -infinity, -infinity, -infinity, &
              1.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, -infinity, -infinity, -infinity /)
   p%X_u = (/ 1.0_wp, infinity, infinity, 3.0_wp, 4.0_wp, 0.0_wp, infinity,    &
              1.0_wp, infinity, infinity, 3.0_wp, 4.0_wp, 0.0_wp, infinity /)
   p%DB = 1.0_wp
   p%DC_l = - 1.0_wp
   p%DC_u = 1.0_wp
   p%DX_l = - 1.0_wp
   p%DX_u = 1.0_wp
   p%A%val = (/ 1.0_wp, 1.0_wp, 2.0_wp, 2.0_wp, 3.0_wp, 3.0_wp, 4.0_wp,        &
                4.0_wp, 5.0_wp, 5.0_wp, 6.0_wp, 6.0_wp, 7.0_wp, 7.0_wp,        &
                8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp,        &
                8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp, 8.0_wp  /)
   p%A%row = (/ 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14,                 &
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15  /)
   p%A%col = (/ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,                      &
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 /)
   p%L%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                &
                1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp,                        &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp, 1.0_wp, - 1.0_wp,          &
                1.0_wp, - 1.0_wp /)
   p%L%row = (/ 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5,                &
                6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10,             &
                11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17 /)
   p%L%col = (/ 1, 3, 5, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 4, 6,                &
                8, 10, 12, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 9, 11, 13,      &
                1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13, 7, 14 /) 

   CALL LSP_initialize( map, control )
   control%infinity = infinity
   p%X = MAX( p%X_l, MIN( 0.0_wp, p%X_u ) ) ; p%Y = 0.0_wp ; p%Z = 0.0_wp
   CALL LX( p%m, p%n, p%L%type, p%L%ne, p%L%val, p%L%row, p%L%col, p%L%ptr,    &
            p%X, p%C )
   sname = 'reorder   '
   iname = iname + 1
   CALL LSP_reorder( map, control, info, d, p, .FALSE., .FALSE., .FALSE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'apply     '
   iname = iname + 1
   CALL LX( p%m, p%n, p%L%type, p%L%ne, p%L%val, p%L%row, p%L%col, p%L%ptr,    &
            p%X, p%C )
   CALL LSP_apply( map, info, p, get_all = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all = .TRUE. )

   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_apply( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'get_values'
   CALL LSP_get_values( map, info, p, X_orig, Y_orig, Z_orig )
   WRITE( 6, 10 ) st, 1, sname, info%status
   sname = 'restore   '
   CALL LSP_restore( map, info, p, get_all_parametric = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status

   sname = 'apply     '
   iname = iname + 1
   CALL LX( p%m, p%n, p%L%type, p%L%ne, p%L%val, p%L%row, p%L%col, p%L%ptr,    &
            p%X, p%C )
   CALL LSP_apply( map, info, p, get_A = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status 
   CALL LSP_apply( map, info, p, get_L = .TRUE. )
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
   CALL LSP_restore( map, info, p, get_L = .TRUE. )
   WRITE( 6, 10 ) st, 1, sname, info%status
   CALL LSP_restore( map, info, p, get_A = .TRUE. )
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
   DEALLOCATE( p%L%val, p%L%row, p%L%col, STAT = smt_stat )
   DEALLOCATE( p%A%val, p%A%row, p%A%col, STAT = smt_stat )
   WRITE( 6, "( /, ' tests completed' )" )

10 FORMAT( A2, I1, ': LSP_', A10, ' exit status = ', I6 )

   CONTAINS

   SUBROUTINE LX(  m, n, l_type, l_ne, L_val, L_row, L_col, L_ptr, X, C )
    
   INTEGER, INTENT( IN ) :: m, n, l_ne
   CHARACTER, INTENT( IN ), DIMENSION( : ) :: l_type
   INTEGER, INTENT( IN ), DIMENSION( : ) ::  L_row, L_col
   INTEGER, INTENT( IN ), DIMENSION( : ) :: L_ptr
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: L_val
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C
   INTEGER :: i, l

   C = 0.0_wp
   IF ( SMT_get( l_type ) == 'DENSE' .OR.                                      &
        SMT_get( l_type ) == 'DENSE_BY_ROWS' ) THEN
     l = 0
     DO i = 1, m
       C( i ) = DOT_PRODUCT( L_val( l + 1 : l + n ), X )
       l = l + n
     END DO
   ELSE IF ( SMT_get( l_type ) == 'DENSE_BY_COLUMNS' ) THEN
     l = 0 ; C = 0.0_wp
     DO i = 1, n
       C = C + L_val( l + 1 : l + m ) * X( i )
       l = l + m
     END DO
   ELSE IF ( SMT_get( l_type ) == 'SPARSE_BY_ROWS' ) THEN
     DO i = 1, m
       DO l = L_ptr( i ), L_ptr( i + 1 ) - 1
         C( i ) = C( i ) + L_val( l ) * X( L_col( l ) )
       END DO
     END DO
   ELSE IF ( SMT_get( l_type ) == 'SPARSE_BY_COLUMNS' ) THEN
     DO i = 1, n
       DO l = L_ptr( i ), L_ptr( i + 1 ) - 1
         C( L_row( l ) ) = C( L_row( l ) ) + L_val( l ) * X( i )
       END DO
     END DO
   ELSE  ! sparse co-ordinate
     DO l = 1, l_ne
       i = L_row( l )
       C( i ) = C( i ) + L_val( l ) * X( L_col( l ) )
     END DO 
   END IF

   RETURN
   END SUBROUTINE LX

   END PROGRAM GALAHAD_LSP_EXAMPLE

