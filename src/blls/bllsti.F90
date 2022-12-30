! THIS VERSION: GALAHAD 4.1 - 2022-12-30 AT 09:40 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_BLLS_interface_test
   USE GALAHAD_KINDS
   USE GALAHAD_BLLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( BLLS_control_type ) :: control
   TYPE ( BLLS_inform_type ) :: inform
   TYPE ( BLLS_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, m, A_ne, A_dense_ne, eval_status
   INTEGER ( KIND = ip_ ) :: i, j, l, nm, mask, data_storage_type, status
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Z, X_l, X_u, B, C, G
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_by_col_row
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_by_col_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_by_col_val
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_by_col_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat
   INTEGER ( KIND = ip_ ) :: nz_in_start, nz_in_end, nz_out_end
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: nz_in, nz_out
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V, P
   CHARACTER ( len = 3 ) :: st
   TYPE ( GALAHAD_userdata_type ) :: userdata

! set up problem data for min || A x - b || with
!   A = (  I  )  and b = (   e   )
!       ( e^T )          ( n + 1 )

   n = 10 ; m = n + 1 ; A_ne = 2 * n ; A_dense_ne = m * n
   ALLOCATE( X( n ), Z( n ), X_l( n ), X_u( n ), G( n ) )
   ALLOCATE( B( m ), C( m ), X_stat( n ) )
   X_l( 1 ) = - 1.0_rp_ ; X_l( 2 : n ) = - infinity ! variable lower bound
   X_u( 1 ) = 1.0_rp_ ; X_u( 2 ) = infinity ; X_u( 3 : n ) = 2.0_rp_ ! upper
   B( : m ) = 1.0_rp_ ! observations
   DO i = 1, n
     B( i ) = REAL( i, KIND = rp_ )
   END DO
   B( m ) = REAL( n + 1, KIND = rp_ )

!  set up A stored by rows

   ALLOCATE( A_val( A_ne ), A_row( A_ne ), A_col( A_ne ), A_ptr( m + 1 ) )
   l = 0
   DO i = 1, n
     l = l + 1 ; A_ptr( i ) = l
     A_row( l ) = i ; A_col( l ) = i ; A_val( l ) = 1.0_rp_ 
   END DO
   A_ptr( m ) = l + 1
   DO i = 1, n
     l = l + 1
     A_row( l ) = m ; A_col( l ) = i ; A_val( l ) = 1.0_rp_ 
   END DO
   A_ptr( m + 1 ) = l + 1
   l = 0
   ALLOCATE( A_dense( A_dense_ne ) )
   DO i = 1, n
     DO j = 1, n
       l = l + 1
       IF ( i == j ) THEN
         A_dense( l ) = 1.0_rp_
       ELSE
         A_dense( l ) = 0.0_rp_
       END IF
     END DO
   END DO
   A_dense( l + 1 : l + n ) = 1.0_rp_

!  set up A stored by columns

   ALLOCATE( A_by_col_val( A_ne ), A_by_col_row( A_ne ), A_by_col_ptr( n + 1 ) )
   l = 0
   DO i = 1, n
     l = l + 1 ; A_by_col_ptr( i ) = l
     A_by_col_row( l ) = i ; A_by_col_val( l ) = 1.0_rp_ 
     l = l + 1
     A_by_col_row( l ) = m ; A_by_col_val( l ) = 1.0_rp_ 
   END DO
   A_by_col_ptr( n + 1 ) = l + 1
   l = 0
   ALLOCATE( A_by_col_dense( A_dense_ne ) )
   DO i = 1, n
     DO j = 1, n
       l = l + 1
       IF ( i == j ) THEN
         A_by_col_dense( l ) = 1.0_rp_
       ELSE
         A_by_col_dense( l ) = 0.0_rp_
       END IF
     END DO
     l = l + 1
     A_by_col_dense( l ) = 1.0_rp_
   END DO

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of Jacobian storage formats', / )" )

   DO data_storage_type = 1, 5
     CALL BLLS_initialize( data, control, inform )
     X = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' CO'
       CALL BLLS_import( control, data, status, n, m, 'coordinate',            &
                         A_ne, A_row, A_col, null )
       CALL BLLS_solve_given_a( data, userdata, status, A_val, B,              &
                                X_l, X_u, X, Z, C, G, X_stat )
!      WRITE( 6, "( ' x = ', 5ES12.4, /, 5X, 5ES12.4 )" ) X
     CASE ( 2 ) ! sparse by rows
        st = ' SR'
        CALL BLLS_import( control, data, status, n, m, 'sparse_by_rows',       &
                          A_ne, null, A_col, A_ptr )
       CALL BLLS_solve_given_a( data, userdata, status, A_val, B,              &
                                X_l, X_u, X, Z, C, G, X_stat )
     CASE ( 3 ) ! dense_by_rows
       st = ' DR'
       CALL BLLS_import( control, data, status, n, m, 'dense_by_rows',         &
                                  A_ne, null, null, null )
       CALL BLLS_solve_given_a( data, userdata, status, A_dense, B,            &
                                X_l, X_u, X, Z, C, G, X_stat )
     CASE ( 4 ) ! sparse by cols
       st = ' SC'
       CALL BLLS_import( control, data, status, n, m, 'sparse_by_columns',     &
                                  A_ne, A_by_col_row, null, A_by_col_ptr )
       CALL BLLS_solve_given_a( data, userdata, status, A_by_col_val, B,       &
                                X_l, X_u, X, Z, C, G, X_stat )
     CASE ( 5 ) ! dense_by_cols
       st = ' DC'
       CALL BLLS_import( control, data, status, n, m, 'dense_by_columns',      &
                         A_ne, null, null, null )
       CALL BLLS_solve_given_a( data, userdata, status, A_by_col_dense, B,     &
                                X_l, X_u, X, Z, C, G, X_stat )
     END SELECT
     CALL BLLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A3, ':', I6, ' iterations. Optimal objective value = ',    &
     &    F6.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A3, ': BLLS_solve exit status = ', I0 ) " ) st,inform%status
     END IF
     CALL BLLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_dense )
   DEALLOCATE( A_by_col_val, A_by_col_row, A_by_col_ptr, A_by_col_dense )

   WRITE( 6, "( /, ' test of reverse-communication interface', / )" )

   nm = MAX( n, m )
   ALLOCATE( nz_in( nm ), nz_out( m ), V( nm ), P( nm ) )
   CALL BLLS_initialize( data, control, inform )
   X = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
   st = ' RC'
!  control%print_level = 1
!  control%maxit = 5
   CALL BLLS_import_without_a( control, data, status, n, m )
   status = 1
   DO 
     CALL BLLS_solve_reverse_a_prod( data, status, eval_status, B, X_l, X_u,   &
                                     X, Z, C, G, X_stat, V, P,                 &
                                     nz_in, nz_in_start, nz_in_end,            &
                                     nz_out, nz_out_end )
!    write(6, "( ' status = ', I0 )" ) status
     SELECT CASE( status )
     CASE ( : 0 )
       EXIT
     CASE ( 2 ) ! Av
       P( : n ) = V( : n )
       P( m ) = SUM( V( : n ) )
       eval_status = 0
     CASE ( 3 ) ! A^T v
       P( : n ) = V( : n ) + V( m )
       eval_status = 0
     CASE ( 4 ) ! A v using sparse v
       P( : m ) = 0.0_rp_
       DO l = nz_in_start, nz_in_end
         i = nz_in( l )
         P( i ) = V( i )
         P( m ) = P( m ) + V( i )
       END DO
       eval_status = 0
     CASE ( 5 ) ! sparse A v using sparse v
       nz_out_end = 0
       mask = 0
       DO l = nz_in_start, nz_in_end
         i = nz_in( l )
         nz_out_end = nz_out_end + 1
         nz_out( nz_out_end ) = i
         P( i ) = V( i )
         IF ( mask == 0 ) THEN
           mask = 1
           nz_out_end = nz_out_end + 1
           nz_out( nz_out_end ) = m
           P( m ) = V( i )
         ELSE
           P( m ) = P( m ) + V( i )
         END IF
       END DO
       eval_status = 0
     CASE ( 6 ) ! sparse A^T v
       DO l = nz_in_start, nz_in_end
         i = nz_in( l )
         P( i ) = V( i ) + V( m )
       END DO
       eval_status = 0
     END SELECT
   END DO
   CALL BLLS_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A3, ':', I6, ' iterations. Optimal objective value = ',      &
   &    F6.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A3, ': BLLS_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   CALL BLLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( B, X, Z, X_l, X_u, C, G, X_stat, NZ_in, NZ_out, V, P )
   WRITE( 6, "( /, ' tests completed' )" )
   END PROGRAM GALAHAD_BLLS_interface_test
