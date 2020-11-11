! THIS VERSION: GALAHAD 3.3 - 29/10/2020 AT 08:30 GMT.
   PROGRAM GALAHAD_CONVERT_EXAMPLE
   USE GALAHAD_CONVERT_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: A, A_out
   TYPE ( CONVERT_control_type ) :: control
   TYPE ( CONVERT_inform_type ) :: inform
   INTEGER :: i, j, l, ne, mode, s, type
! set problem data
   A%m = 4 ; A%n = 5 ; ne = 9
   DO mode = 1, 2 ! write natrix (mode=1) or its transpose (mode=2)
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5 ! loop over storage types
       SELECT CASE ( type )
       CASE ( 1 ) ! dense input format
         CALL SMT_put( A%type, 'DENSE_BY_ROWS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 13.0_wp, 0.0_wp, 15.0_wp,                 &
                    0.0_wp, 22.0_wp, 0.0_wp, 24.0_wp, 0.0_wp,                  &
                    0.0_wp, 32.0_wp, 33.0_wp, 0.0_wp, 0.0_wp,                  &
                    0.0_wp, 0.0_wp, 0.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 2 ) ! dense by columns input format
         CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
         ALLOCATE( A%val( A%m * A%n ) )
         A%val = (/ 11.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                           &
                    0.0_wp, 22.0_wp, 32.0_wp, 0.0_wp,                          &
                    13.0_wp, 0.0_wp, 33.0_wp, 0.0_wp,                          &
                    0.0_wp, 24.0_wp, 0.0_wp, 44.0_wp,                          &
                    15.0_wp, 0.0_wp, 0.0_wp, 45.0_wp /)
       CASE ( 3 ) ! sparse by rows input format
         CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
         ALLOCATE( A%ptr( A%m + 1 ), A%col( ne ), A%val( ne ) )
         A%ptr = (/ 1, 4, 6, 8, 10 /)
         A%col = (/ 1, 5, 3, 2, 4, 3, 2, 4, 5 /)
         A%val = (/ 11.0_wp, 15.0_wp, 13.0_wp, 22.0_wp, 24.0_wp,               &
                    33.0_wp, 32.0_wp, 44.0_wp, 45.0_wp /)
       CASE ( 4 ) ! sparse by columns input format
         CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
         ALLOCATE( A%ptr( A%n + 1 ), A%row( ne ), A%val( ne ) )
         A%ptr = (/ 1, 2, 4, 6, 8, 10 /)
         A%row = (/ 1, 3, 2, 1, 3, 2, 4, 4, 1 /)
         A%val = (/ 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,               &
                    24.0_wp, 44.0_wp, 45.0_wp, 15.0_wp /)
       CASE ( 5 ) ! sparse co-ordinate input format
         CALL SMT_put( A%type, 'COORDINATE', s )
         A%ne = ne
         ALLOCATE( A%row( ne ), A%col( ne ), A%val( ne ) )
         A%row = (/ 4, 1, 3, 2, 1, 3, 4, 2, 1 /)
         A%col = (/ 5, 1, 2, 2, 3, 3, 4, 4, 5 /)
         A%val = (/ 45.0_wp, 11.0_wp, 32.0_wp, 22.0_wp, 13.0_wp, 33.0_wp,      &
                    44.0_wp, 24.0_wp, 15.0_wp /)
       END SELECT
       CALL CONVERT_between_matrix_formats( A, 'SPARSE_BY_ROWS', A_out,        &
                                            control, inform ) ! transform
       WRITE( 6, "( /, ' convert from ', A, ' to sparse-row format ')" )       &
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

   END PROGRAM GALAHAD_CONVERT_EXAMPLE
