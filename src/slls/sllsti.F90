! THIS VERSION: GALAHAD 5.5 - 2026-02-14 AT 15:55 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SLLS_interface_test
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SLLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( SLLS_full_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: n, o, Ao_ne, Ao_dense_ne, eval_status
   INTEGER ( KIND = ip_ ) :: i, j, l, m, nm, data_storage_type, status
   REAL ( KIND = rp_ ) :: sigma  = 1.0_rp_
   INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: null_
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, Y, Z, B, R, G, W, X_s
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_row, Ao_col, Ao_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_val, Ao_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_by_col_row
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_by_col_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_by_col_val
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Ao_by_col_dense
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat, COHORT
   INTEGER ( KIND = ip_ ) :: lvl, lvu, lp, index
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IV, IP
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V, P

   CHARACTER ( len = 3 ) :: st
   TYPE ( USERDATA_type ) :: userdata

! set up problem data for min || A_o x - b || with e^T x = 1, x >= 0

   n = 10 ; o = n + 1 ; Ao_ne = 2 * n ; Ao_dense_ne = o * n ; m = 1
   ALLOCATE( X( n ), Y( m ), Z( n ), G( n ), B( o ), R( o ), X_stat( n ) )
   B( : o ) = 1.0_rp_ ! observations
   DO i = 1, n
     B( i ) = REAL( i, KIND = rp_ )
   END DO
   B( o ) = REAL( n + 1, KIND = rp_ )

!  set up A stored by rows

   ALLOCATE( Ao_val( Ao_ne ), Ao_row( Ao_ne ) )
   ALLOCATE( Ao_col( Ao_ne ), Ao_ptr( o + 1 ) )
   l = 0
   DO i = 1, n
     l = l + 1 ; Ao_ptr( i ) = l
     Ao_row( l ) = i ; Ao_col( l ) = i ; Ao_val( l ) = 1.0_rp_
   END DO
   Ao_ptr( o ) = l + 1
   DO i = 1, n
     l = l + 1
     Ao_row( l ) = o ; Ao_col( l ) = i ; Ao_val( l ) = 1.0_rp_
   END DO
   Ao_ptr( o + 1 ) = l + 1
   l = 0
   ALLOCATE( Ao_dense( Ao_dense_ne ) )
   DO i = 1, n
     DO j = 1, n
       l = l + 1
       IF ( i == j ) THEN
         Ao_dense( l ) = 1.0_rp_
       ELSE
         Ao_dense( l ) = 0.0_rp_
       END IF
     END DO
   END DO
   Ao_dense( l + 1 : l + n ) = 1.0_rp_

!  set up A stored by columns

   ALLOCATE( Ao_by_col_val( Ao_ne ), Ao_by_col_row( Ao_ne ) )
   ALLOCATE( Ao_by_col_ptr( n + 1 ) )
   l = 0
   DO i = 1, n
     l = l + 1 ; Ao_by_col_ptr( i ) = l
     Ao_by_col_row( l ) = i ; Ao_by_col_val( l ) = 1.0_rp_
     l = l + 1
     Ao_by_col_row( l ) = o ; Ao_by_col_val( l ) = 1.0_rp_
   END DO
   Ao_by_col_ptr( n + 1 ) = l + 1
   l = 0
   ALLOCATE( Ao_by_col_dense( Ao_dense_ne ) )
   DO i = 1, n
     DO j = 1, n
       l = l + 1
       IF ( i == j ) THEN
         Ao_by_col_dense( l ) = 1.0_rp_
       ELSE
         Ao_by_col_dense( l ) = 0.0_rp_
       END IF
     END DO
     l = l + 1
     Ao_by_col_dense( l ) = 1.0_rp_
   END DO

! problem data complete

!  =====================================
!  basic test of various storage formats
!  =====================================

   WRITE( 6, "( /, ' basic tests of Jacobian storage formats', / )" )

   DO data_storage_type = 1, 5
     CALL SLLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     X = 0.0_rp_ ! start from zero
     inform%status = 1
     SELECT CASE ( data_storage_type )
     CASE ( 1 ) ! sparse co-ordinate storage
       st = ' CO'
       CALL SLLS_import( control, data, status, n, o, m, 'coordinate',         &
                         Ao_ne, Ao_row, Ao_col, null_ )
       CALL SLLS_solve_given_a( data, userdata, status, Ao_val, B,             &
                                sigma, X, Y, Z, R, G, X_stat )
!      WRITE( 6, "( ' x = ', 5ES12.4, /, 5X, 5ES12.4 )" ) X
     CASE ( 2 ) ! sparse by rows
        st = ' SR'
        CALL SLLS_import( control, data, status, n, o, m, 'sparse_by_rows',    &
                          Ao_ne, null_, Ao_col, Ao_ptr )
       CALL SLLS_solve_given_a( data, userdata, status, Ao_val, B,             &
                                sigma, X, Y, Z, R, G, X_stat )
     CASE ( 3 ) ! dense_by_rows
       st = ' DR'
       CALL SLLS_import( control, data, status, n, o, m, 'dense_by_rows',      &
                                  Ao_ne, null_, null_, null_ )
       CALL SLLS_solve_given_a( data, userdata, status, Ao_dense, B,           &
                                sigma, X, Y, Z, R, G, X_stat )
     CASE ( 4 ) ! sparse by cols
       st = ' SC'
       CALL SLLS_import( control, data, status, n, o, m, 'sparse_by_columns',  &
                                  Ao_ne, Ao_by_col_row, null_, Ao_by_col_ptr )
       CALL SLLS_solve_given_a( data, userdata, status, Ao_by_col_val, B,      &
                                sigma, X, Y, Z, R, G, X_stat )
     CASE ( 5 ) ! dense_by_cols
       st = ' DC'
       CALL SLLS_import( control, data, status, n, o, m, 'dense_by_columns',   &
                         Ao_ne, null_, null_, null_ )
       CALL SLLS_solve_given_a( data, userdata, status, Ao_by_col_dense, B,    &
                                sigma, X, Y, Z, R, G, X_stat )
     END SELECT
     CALL SLLS_information( data, inform, status )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( A3, ':', I6, ' iteration(s). Optimal objective value = ',  &
     &    F6.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
     ELSE
       WRITE( 6, "( A3, ': SLLS_solve exit status = ', I0 ) " ) st,inform%status
     END IF
     CALL SLLS_terminate( data, control, inform )  ! delete internal workspace
   END DO
!  DEALLOCATE( Ao_val, Ao_row, Ao_col, Ao_ptr, Ao_dense )
   DEALLOCATE( Ao_by_col_val, Ao_by_col_row, Ao_by_col_ptr, Ao_by_col_dense )

   WRITE( 6, "( /, ' test of reverse-communication interface', / )" )

   nm = MAX( n, o )
   ALLOCATE( IV( nm ), IP( o ), V( nm ), P( nm ) )
   CALL SLLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   X = 0.0_rp_ ! start from zero
   st = ' RC'
!  control%print_level = 1
!  control%maxit = 5
   CALL SLLS_import_without_a( control, data, status, n, o, m )
   status = 1
   DO
     CALL SLLS_solve_reverse_a_prod( data, status, eval_status, B, sigma,      &
                                     X, Y, Z, R, G, X_stat, V, P,              &
                                     IV, lvl, lvu, index, IP, lp )
!    write(6, "( ' status = ', I0 )" ) status
     SELECT CASE( status )
     CASE ( : 0 )
       EXIT
     CASE ( 2 ) ! Av
       P( : n ) = V( : n )
       P( o ) = SUM( V( : n ) )
       eval_status = 0
     CASE ( 3 ) ! Ao^T v
       P( : n ) = V( : n ) + V( o )
       eval_status = 0
     CASE ( 4 ) ! sparse column of Ao
       lp = 1
       IP( lp ) = index
       P( lp ) = 1.0_rp_
       lp = lp + 1
       IP( lp ) = o
       P( lp ) = 1.0_rp_
       eval_status = 0
     CASE ( 5 ) ! Ao v using sparse v
       P( : o ) = 0.0_rp_
       DO l = lvl, lvu
         i = IV( l )
         P( i ) = V( i )
         P( o ) = P( o ) + V( i )
       END DO
       eval_status = 0
     CASE ( 6 ) ! sparse Ao^T v
       DO l = lvl, lvu
         i = IV( l )
         P( i ) = V( i ) + V( o )
       END DO
       eval_status = 0
     END SELECT
   END DO
   CALL SLLS_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A3, ':', I6, ' iteration(s). Optimal objective value = ',    &
   &    F6.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A3, ': SLLS_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   CALL SLLS_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( Y )

   WRITE( 6, "( /, ' test of explicit cohort + weights + shifts interface', /)")

   m = 1
   ALLOCATE( Y( m ), COHORT( n ), W( o ), X_s( n ) )
   COHORT = 1 ! all variables in a single cohort
   W = 1.0_rp_ ! weight of one
   X_s = 0.0_rp_ ! shifts of zero
   CALL SLLS_initialize( data, control, inform )
   CALL WHICH_sls( control )
   X = 0.0_rp_ ! start from zero
   st = ' CO'
   CALL SLLS_import( control, data, status, n, o, m, 'coordinate',             &
                     Ao_ne, Ao_row, Ao_col, null_, COHORT = COHORT )
   CALL SLLS_solve_given_a( data, userdata, status, Ao_val, B, sigma,          &
                            X, Y, Z, R, G, X_stat, W = W, X_s = X_s )
   CALL SLLS_information( data, inform, status )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( A3, ':', I6, ' iteration(s). Optimal objective value = ',    &
   &    F6.2, ' status = ', I0 )" ) st, inform%iter, inform%obj, inform%status
   ELSE
     WRITE( 6, "( A3, ': SLLS_solve exit status = ', I0 ) " ) st, inform%status
   END IF
   CALL SLLS_terminate( data, control, inform )  ! delete internal workspace
!write(6,"( ' x: ', 5ES10.2, /, ( 4X, 5ES10.2 ) )" ) X
!write(6,"( ' y: ', 5ES10.2 )" ) Y
!write(6,"( ' z: ', 5ES10.2, /, ( 4X, 5ES10.2 ) )" ) Z
!write(6,"( ' g: ', 5ES10.2, /, ( 4X, 5ES10.2 ) )" ) G
   DEALLOCATE( Ao_val, Ao_row, Ao_col, Ao_ptr, Ao_dense )
   DEALLOCATE( B, X, Y, Z, R, G, X_stat, W, X_s, IV, IP, V, P )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( SLLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls
   END PROGRAM GALAHAD_SLLS_interface_test
