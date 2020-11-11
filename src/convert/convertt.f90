! THIS VERSION: GALAHAD 3.3 - 29/10/2020 AT 08:30 GMT.
   PROGRAM GALAHAD_CONVERT_TEST
   USE GALAHAD_CONVERT_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: A, A_out
   TYPE ( CONVERT_control_type ) :: control
   TYPE ( CONVERT_inform_type ) :: inform
   INTEGER :: i, j, l, mode, s, status, type
   INTEGER, PARAMETER :: len_iw = 0, len_w = 0
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
   LOGICAL :: testdc, testdr, testsc, testsr, testco
! set problem data
   A%m = 4 ; A%n = 5 ; A%ne = 9

!  first try specific interfaces

!  GO TO 39
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
       ALLOCATE( IW( A%n ), W( A%n ) )
     ELSE
       ALLOCATE( IW( A%m ), W( A%m ) )
     END IF
     IW = 0
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,    &
                                             IW, W ) ! convert
       WRITE( 6, "( /, ' convert from ', A, ' to column format ')" )           &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         DO i = 1, A_out%n
           WRITE( 6, "( ' column ', I0, ', ( row value ): ',                   &
          & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%row( j ),          &
              A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%ptr, A_out%row, A_out%val, stat = i )
     END DO
     DEALLOCATE( IW, W, stat = i )
   END DO

 9 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
       ALLOCATE( IW( A%n ), W( A%n ) )
     ELSE
       ALLOCATE( IW( A%m ), W( A%m ) )
     END IF
     IW = 0

     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,           &
                                          IW, W ) ! convert
       WRITE( 6, "( /, ' convert from ', A, ' to row format ')" )              &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         DO i = 1, A_out%m
           WRITE( 6, "( ' row ', I0, ', ( column value ): ',                   &
          & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%col( j ),          &
              A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%ptr, A_out%col, A_out%val, stat = i )
     END DO
     DEALLOCATE( IW, W, stat = i )
   END DO

19 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_coordinate_format( A, A_out, control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to coordinate format ')" )       &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( '( row column value )' )" )
         WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                    &
             ( A_out%row( j ), A_out%col( j ), A_out%val( j ), j = 1, A_out%ne )
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%row, A_out%col, A_out%val, stat = i )
     END DO
   END DO

29 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_dense_row_format( A, A_out, control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to dense-row format ')" )        &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         l = 0
         DO i = 1, A_out%m
           WRITE( 6, "( ' row ', I0, ( 3( 10F5.1 ), : ) )" )                   &
             i, ( A_out%val( j ), j = l + 1, l + A_out%n )
           l = l + A_out%n
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

39 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_to_dense_column_format( A, A_out, control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to dense-column format ')" )     &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         l = 0
         DO j = 1, A_out%n
           WRITE( 6, "( ' column ', I0, ( 3( 10F5.1 ) : ) )" )                 &
             j, ( A_out%val( i ), i = l + 1, l + A_out%m )
           l = l + A_out%m
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

!  now try generic interface

40 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'SPARSE_BY_COLUMNS', A_out,     &
                                            control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to column format ')" )           &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         DO i = 1, A_out%n
           WRITE( 6, "( ' column ', I0, ', ( row value ): ',                   &
          & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%row( j ),          &
              A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%ptr, A_out%row, A_out%val, stat = i )
     END DO
   END DO

49 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'SPARSE_BY_ROWS', A_out,        &
                                            control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to row format ')" )              &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         DO i = 1, A_out%m
           WRITE( 6, "( ' row ', I0, ', ( column value ): ',                   &
          & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%col( j ),          &
              A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%ptr, A_out%col, A_out%val, stat = i )
     END DO
   END DO

59 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'COORDINATE', A_out,            &
                                            control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to coordinate format ')" )       &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( '( row column value )' )" )
         WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                    &
             ( A_out%row( j ), A_out%col( j ), A_out%val( j ), j = 1, A_out%ne )
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%row, A_out%col, A_out%val, stat = i )
     END DO
   END DO

69 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'DENSE_BY_ROWS', A_out,        &
                                            control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to dense-row format ')" )        &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         l = 0
         DO i = 1, A_out%m
           WRITE( 6, "( ' row ', I0, ( 3( 10F5.1 ), : ) )" )                   &
             i, ( A_out%val( j ), j = l + 1, l + A_out%n )
           l = l + A_out%n
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

79 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       SELECT CASE ( type )
       CASE ( 1 ) ! dense
         CALL SMT_put( A%type, 'DENSE', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate
         CALL SMT_put( A%type, 'COORDINATE', s )
         ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'DENSE_BY_COLUMNS', A_out,      &
                                             control, inform )
       WRITE( 6, "( /, ' convert from ', A, ' to dense-column format ')" )     &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         l = 0
         DO j = 1, A_out%n
           WRITE( 6, "( ' column ', I0, ( 3( 10F5.1 ) : ) )" )                 &
             j, ( A_out%val( i ), i = l + 1, l + A_out%m )
           l = l + A_out%m
         END DO
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       SELECT CASE ( type ) ! deallocate space
       CASE ( 1, 2 ) ! dense + dense by columns
         DEALLOCATE( A%val )
       CASE ( 3 ) ! sparse by rows
         DEALLOCATE( A%ptr, A%col, A%val )
       CASE ( 4 ) ! sparse by columns
         DEALLOCATE( A%ptr, A%row, A%val )
       CASE ( 5 ) ! sparse co-ordinate
         DEALLOCATE( A%row, A%col, A%val )
       END SELECT
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests, status should be -ve', / )" )
   CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
   ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
   A%m = 4 ; A%n = 5 ; A%ne = 9
   A%ptr = (/ 1, 4, 6, 8, 10 /)
   A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
   A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,                     &
              33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)

   testdc = .TRUE. ; testdr = .TRUE.
   testsc = .TRUE. ; testsr = .TRUE. ; testco = .TRUE.
   DO status = 1, 6
     IF ( status == 1 ) THEN
       A%m = 0
     ELSE IF ( status == 2 ) THEN
       A%m = 4 ; A%n = 0
     ELSE IF ( status == 3 ) THEN
       testdc = .FALSE. ; testdr = .FALSE. ; testco = .FALSE.
       A%n = 5
     ELSE IF ( status == 4 ) THEN
       ALLOCATE( IW( 0 ) )
     ELSE IF ( status == 5 ) THEN
       DEALLOCATE( IW )
       ALLOCATE( IW( 5 ) )
       IW = 0
     ELSE IF ( status == 6 ) THEN
       ALLOCATE( W( 0 ) )
     ELSE IF ( status == 7 ) THEN
       DEALLOCATE( W )
       ALLOCATE( W( 5 ) )
     END IF

     IF ( testdr ) THEN
       CALL CONVERT_to_dense_row_format( A, A_out, control, inform )
       WRITE( 6, "( ' dr status = ', I0 )" ) inform%status
     END IF
     IF ( testdc ) THEN
       CALL CONVERT_to_dense_column_format( A, A_out, control, inform )
       WRITE( 6, "( ' dc status = ', I0 )" ) inform%status
     END IF
     IF ( testsc ) THEN
       IF ( status <= 3 ) THEN
         CALL CONVERT_to_sparse_column_format( A, A_out, control, inform )
       ELSE IF ( status == 4 .OR. status == 5 ) THEN
         CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,      &
                                               IWORK = IW )
       ELSE
         CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,      &
                                               IWORK = IW, WORK = W )
       END IF
       WRITE( 6, "( ' sc status = ', I0 )" ) inform%status
     END IF
     IF ( testsr ) THEN
       IF ( status <= 3 ) THEN
         CALL CONVERT_to_sparse_row_format( A, A_out, control, inform )
       ELSE IF ( status == 4 .OR. status == 5 ) THEN
         CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,         &
                                               IWORK = IW )
       ELSE
         CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,         &
                                               IWORK = IW, WORK = W )
       END IF
       WRITE( 6, "( ' sr status = ', I0 )" ) inform%status
     END IF
     IF ( testco ) THEN
       CALL CONVERT_to_coordinate_format( A, A_out, control, inform )
       WRITE( 6, "( ' co status = ', I0 )" ) inform%status
     END IF
   END DO

   DEALLOCATE( A%ptr, A%col, A%val, IW, W, STAT = i )

   END PROGRAM GALAHAD_CONVERT_TEST
