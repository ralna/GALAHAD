! THIS VERSION: GALAHAD 5.6 - 2026-08-14 AT 15:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_CONVERT_TEST
   USE GALAHAD_KINDS_precision
   USE GALAHAD_CONVERT_precision         ! double precision version
   IMPLICIT NONE
   TYPE ( SMT_type ) :: A, A_out, H, H_out
   TYPE ( CONVERT_control_type ) :: control
   TYPE ( CONVERT_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: i, j, l, mode, s, status, type
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IW
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP
   LOGICAL :: testdc, testdr, testsc, testsr, testco

   WRITE( 6, "( /, ' tests for unsymmetric matices')" )

!  first try specific interfaces

!  GO TO 9
   DO mode = 1, 2
     control%order = .TRUE.
!    control%order = .FALSE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     ELSE
       control%transpose = .FALSE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       ALLOCATE( MAP( SIZE( A%val ) ) )
       CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,        &
                                             MAP = MAP )  ! convert
       WRITE( 6, "( /, ' convert from ', A, ' to sparse-column format ')" )    &
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
       WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
       A%val = 2.0_rp_ * A%val
       CALL CONVERT_map_values( A%ne, A%val, A_out%ne, A_out%val, MAP )
       A_out%val( : A_out%ne ) = 0.5_rp_ * A_out%val( : A_out%ne )
       WRITE( 6, "( /, ' new A' )" )
       DO i = 1, A_out%n
         WRITE( 6, "( ' column ', I0, ', ( row value ): ',                     &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%row( j ),            &
            A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
       END DO
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%ptr, A_out%row, A_out%val, MAP, stat = i )
     END DO
   END DO

!4 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     ELSE
       control%transpose = .FALSE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       ALLOCATE( MAP( SIZE( A%val ) ) )
       CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,           &
                                          MAP = MAP )  ! convert
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
       WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
       A%val = 2.0_rp_ * A%val
       CALL CONVERT_map_values( A%ne, A%val, A_out%ne, A_out%val, MAP )
       A_out%val( : A_out%ne ) = 0.5_rp_ * A_out%val( : A_out%ne )
       WRITE( 6, "( /, ' new A' )" )
       DO i = 1, A_out%m
         WRITE( 6, "( ' row ', I0, ', ( column value ): ',                     &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( A_out%col( j ),            &
            A_out%val( j ), j = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1 )
       END DO
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%ptr, A_out%col, A_out%val, MAP, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     ELSE
       control%transpose = .FALSE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       ALLOCATE( MAP( SIZE( A%val ) ) )
       CALL CONVERT_to_coordinate_format( A, A_out, control, inform,           &
                                          MAP = MAP )
       WRITE( 6, "( /, ' convert from ', A, ' to coordinate format ')" )       &
         SMT_get( A%type )
       IF ( inform%status == 0 ) THEN
         WRITE( 6, "( '( row column value )' )" )
         WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                    &
             ( A_out%row( j ), A_out%col( j ), A_out%val( j ), j = 1, A_out%ne )
       ELSE
         WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
       END IF
       WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
       A%val = 2.0_rp_ * A%val
       CALL CONVERT_map_values( A%ne, A%val, A_out%ne, A_out%val, MAP )
       A_out%val( : A_out%ne ) = 0.5_rp_ * A_out%val( : A_out%ne )
       WRITE( 6, "( /, ' new A' )" )
       WRITE( 6, "( '( row column value )' )" )
       WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                      &
           ( A_out%row( j ), A_out%col( j ), A_out%val( j ), j = 1, A_out%ne )
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%row, A_out%col, A_out%val, MAP, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     ELSE
       control%transpose = .FALSE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       ALLOCATE( MAP( SIZE( A%val ) ) )
!      CALL CONVERT_to_dense_row_format( A, A_out, control, inform )
       CALL CONVERT_to_dense_row_format( A, A_out, control, inform, MAP = MAP )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%val, MAP, stat = i )
     END DO
   END DO

!40 CONTINUE
   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     ELSE
       control%transpose = .FALSE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       ALLOCATE( MAP( SIZE( A%val ) ) )
       CALL CONVERT_to_dense_column_format( A, A_out, control, inform )
!      CALL CONVERT_to_dense_column_format( A, A_out, control, inform,         &
!                                           MAP = MAP )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%val, MAP, stat = i )
     END DO
   END DO

!  now try generic interface

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%ptr, A_out%row, A_out%val, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%ptr, A_out%col, A_out%val, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%row, A_out%col, A_out%val, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
       CALL CONVERT_between_matrix_formats( A, 'DENSE_BY_ROWS', A_out,         &
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

   DO mode = 1, 2
     control%order = .TRUE.
     IF ( mode == 2 ) THEN
       WRITE( 6, "( /, ' construct the transpose' )" )
       control%transpose = .TRUE.
     END IF
     DO type = 1, 5
       CALL SET_A( type, A )
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
       CALL DEALLOCATE_A( type, A )
       DEALLOCATE( A_out%val, stat = i )
     END DO
   END DO

!  ==================================================
   WRITE( 6, "( /, ' tests for symmetric matices')" )
!  ==================================================

!  first try specific interfaces

!  GO TO 40
   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     ALLOCATE( MAP( SIZE( H%val ) ) )
     CALL CONVERT_to_sparse_symmetric_column_format( H, H_out, control,        &
                                                     inform, MAP = MAP )
     WRITE( 6, "( /, ' convert from ', A, ' to column format ')" )             &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       DO i = 1, H_out%n
         WRITE( 6, "( ' column ', I0, ', ( row value ): ',                     &
          & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%row( j ),          &
            H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
     H%val = 2.0_rp_ * H%val
     CALL CONVERT_map_values( H%ne, H%val, H_out%ne, H_out%val, MAP )
     H_out%val( : H_out%ne ) = 0.5_rp_ * H_out%val( : H_out%ne )
     WRITE( 6, "( /, ' new H' )" )
     DO i = 1, H_out%n
       WRITE( 6, "( ' column ', I0, ', ( row value ): ',                       &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%row( j ),            &
          H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
     END DO
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%ptr, H_out%row, H_out%val, MAP, stat = i )
   END DO

!9 CONTINUE
   control%order = .TRUE.
!  control%order = .FALSE.

   DO type = 1, 5
     CALL SET_H( type, H )
     ALLOCATE( MAP( SIZE( H%val ) ) )
     CALL CONVERT_to_sparse_symmetric_row_format( H, H_out, control, inform,   &
                                                  MAP = MAP )
     WRITE( 6, "( /, ' convert from ', A, ' to row format ')" )                &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       DO i = 1, H_out%n
         WRITE( 6, "( ' row ', I0, ', ( column value ): ',                     &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%col( j ),            &
            H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
     H%val = 2.0_rp_ * H%val
     CALL CONVERT_map_values( H%ne, H%val, H_out%ne, H_out%val, MAP )
     H_out%val( : H_out%ne ) = 0.5_rp_ * H_out%val( : H_out%ne )
     WRITE( 6, "( /, ' new H' )" )
     DO i = 1, H_out%n
       WRITE( 6, "( ' row ', I0, ', ( column value ): ',                       &
      & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%col( j ),              &
          H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
     END DO
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%ptr, H_out%col, H_out%val, MAP, stat = i )
   END DO

!9 CONTINUE
   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     ALLOCATE( MAP( SIZE( H%val ) ) )
     CALL CONVERT_to_symmetric_coordinate_format( H, H_out, control, inform,   &
                                                  MAP = MAP )
     WRITE( 6, "( /, ' convert from ', A, ' to coordinate format ')" )         &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( '( row column value )' )" )
       WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                      &
           ( H_out%row( j ), H_out%col( j ), H_out%val( j ), j = 1, H_out%ne )
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     WRITE( 6, "( ' map:', 20( 1X, I0 ) )" ) MAP
     H%val = 2.0_rp_ * H%val
     CALL CONVERT_map_values( H%ne, H%val, H_out%ne, H_out%val, MAP )
     H_out%val( : H_out%ne ) = 0.5_rp_ * H_out%val( : H_out%ne )
     WRITE( 6, "( /, ' new H' )" )
     WRITE( 6, "( '( row column value )' )" )
     WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                        &
         ( H_out%row( j ), H_out%col( j ), H_out%val( j ), j = 1, H_out%ne )
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%row, H_out%col, H_out%val, MAP, stat = i )
   END DO

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_to_dense_symmetric_row_format( H, H_out, control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to dense-row format ')" )          &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       l = 0
       DO i = 1, H_out%n
         WRITE( 6, "( ' row ', I0, ( 3( 10F5.1 ), : ) )" )                     &
           i, ( H_out%val( j ), j = l + 1, l + i )
         l = l + i
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%val, stat = i )
   END DO

!40 CONTINUE
   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_to_dense_symmetric_column_format( H, H_out, control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to dense-column format ')" )       &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       l = 0
       DO j = 1, H_out%n
         WRITE( 6, "( ' column ', I0, ( 3( 10F5.1 ) : ) )" )                   &
           j, ( H_out%val( i ), i = l + 1, l + H_out%n - j + 1 )
         l = l + H_out%n - j + 1
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%val, stat = i )
   END DO

!  now try generic interface

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_between_symmetric_formats( H, 'SPARSE_BY_COLUMNS', H_out,    &
                                             control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to column format ')" )             &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       DO i = 1, H_out%n
         WRITE( 6, "( ' column ', I0, ', ( row value ): ',                     &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%row( j ),            &
            H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%ptr, H_out%row, H_out%val, stat = i )
   END DO

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_between_symmetric_formats( H, 'SPARSE_BY_ROWS', H_out,       &
                                             control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to row format ')" )                &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       DO i = 1, H_out%n
         WRITE( 6, "( ' row ', I0, ', ( column value ): ',                     &
        & ( 5( '(', I2, F5.1, ' )', : ) ) )" ) i, ( H_out%col( j ),            &
            H_out%val( j ), j = H_out%ptr( i ), H_out%ptr( i + 1 ) - 1 )
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%ptr, H_out%col, H_out%val, stat = i )
   END DO

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_between_symmetric_formats( H, 'COORDINATE', H_out,           &
                                             control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to coordinate format ')" )         &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       WRITE( 6, "( '( row column value )' )" )
       WRITE( 6, "( ( 5( ' (', 2I2, F5.1, ')', : ) ) )" )                      &
           ( H_out%row( j ), H_out%col( j ), H_out%val( j ), j = 1, H_out%ne )
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%row, H_out%col, H_out%val, stat = i )
   END DO

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_between_symmetric_formats( H, 'DENSE_BY_ROWS', H_out,        &
                                             control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to dense-row format ')" )          &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       l = 0
       DO i = 1, H_out%n
         WRITE( 6, "( ' row ', I0, ( 3( 10F5.1 ), : ) )" )                     &
           i, ( H_out%val( j ), j = l + 1, l + i )
         l = l + i
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%val, stat = i )
   END DO

   control%order = .TRUE.
   DO type = 1, 5
     CALL SET_H( type, H )
     CALL CONVERT_between_symmetric_formats( H, 'DENSE_BY_COLUMNS', H_out,     &
                                             control, inform )
     WRITE( 6, "( /, ' convert from ', A, ' to dense-column format ')" )       &
       SMT_get( H%type )
     IF ( inform%status == 0 ) THEN
       l = 0
       DO j = 1, H_out%n
         WRITE( 6, "( ' column ', I0, ( 3( 10F5.1 ) : ) )" )                   &
           j, ( H_out%val( i ), i = l + 1, l + H_out%n - j + 1 )
         l = l + H_out%n - j + 1
       END DO
     ELSE
       WRITE( 6, "( ' error return, status = ', I0 )" ) inform%status
     END IF
     CALL DEALLOCATE_H( type, H ) 
     DEALLOCATE( H_out%val, stat = i )
   END DO

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests, status should be -ve', / )" )
   CALL SET_A( 3, A ) ! sparse by rows

   testdc = .TRUE. ; testdr = .TRUE.
   testsc = .TRUE. ; testsr = .TRUE. ; testco = .TRUE.
   DO status = 1, 2
     IF ( status == 1 ) THEN
       A%m = - 1
     ELSE IF ( status == 2 ) THEN
       A%m = 4 ; A%n = 0
     ELSE IF ( status == 3 ) THEN
       testdc = .FALSE. ; testdr = .FALSE. ; testco = .FALSE.
       A%n = 5
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
       CALL CONVERT_to_sparse_column_format( A, A_out, control, inform )
       WRITE( 6, "( ' sc status = ', I0 )" ) inform%status
     END IF
     IF ( testsr ) THEN
       CALL CONVERT_to_sparse_row_format( A, A_out, control, inform )
       WRITE( 6, "( ' sr status = ', I0 )" ) inform%status
     END IF
     IF ( testco ) THEN
       CALL CONVERT_to_coordinate_format( A, A_out, control, inform )
       WRITE( 6, "( ' co status = ', I0 )" ) inform%status
     END IF
   END DO

   DEALLOCATE( A%ptr, A%col, A%val, STAT = i )

   WRITE( 6, "( /, ' error tests for symmetric matices', /)" )

   CALL SET_H( 4, H ) ! sparse by rows

   testdc = .TRUE. ; testdr = .TRUE.
   testsc = .TRUE. ; testsr = .TRUE. ; testco = .TRUE.
   DO status = 2, 2
     IF ( status == 1 ) THEN
       CYCLE
     ELSE IF ( status == 2 ) THEN
       H%n = 0
     ELSE IF ( status == 3 ) THEN
       testdc = .FALSE. ; testdr = .FALSE. ; testco = .FALSE.
       H%n = 4
     END IF

     IF ( testdr ) THEN
       CALL CONVERT_to_dense_symmetric_row_format( H, H_out, control, inform )
       WRITE( 6, "( ' dr status = ', I0 )" ) inform%status
     END IF
     IF ( testdc ) THEN
       CALL CONVERT_to_dense_symmetric_column_format( H, H_out, control,       &
                                                      inform )
       WRITE( 6, "( ' dc status = ', I0 )" ) inform%status
     END IF
     IF ( testsc ) THEN
       CALL CONVERT_to_sparse_symmetric_column_format( H, H_out, control,      &
                                                       inform )
       WRITE( 6, "( ' sc status = ', I0 )" ) inform%status
     END IF
     IF ( testsr ) THEN
       CALL CONVERT_to_sparse_symmetric_row_format( H, H_out, control,         &
                                                    inform )
       WRITE( 6, "( ' sr status = ', I0 )" ) inform%status
     END IF
     IF ( testco ) THEN
       CALL CONVERT_to_coordinate_format( H, H_out, control, inform )
       WRITE( 6, "( ' co status = ', I0 )" ) inform%status
     END IF
   END DO

   DEALLOCATE( H%ptr, H%col, H%val, STAT = i )

   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS

     SUBROUTINE SET_A( type, A )
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: type
     TYPE ( SMT_type ), INTENT( INOUT ) :: A
     A%m = 4 ; A%n = 5
     SELECT CASE ( type ) ! set up matrix A
     CASE ( 1 ) ! dense
       CALL SMT_put( A%type, 'DENSE', s )
       A%ne =  A%m * A%n
       ALLOCATE( A%val( A%ne ) )
       A%val = [ 11.0_rp_, 0.0_rp_, 13.0_rp_, 0.0_rp_, 15.0_rp_,               &
                  0.0_rp_, 22.0_rp_, 0.0_rp_, 24.0_rp_, 0.0_rp_,               &
                  0.0_rp_, 32.0_rp_, 33.0_rp_, 0.0_rp_, 0.0_rp_,               &
                  0.0_rp_, 0.0_rp_, 0.0_rp_, 44.0_rp_, 45.0_rp_ ]
     CASE ( 2 ) ! dense by columns
       CALL SMT_put( A%type, 'DENSE_BY_COLUMNS', s )
       A%ne =  A%m * A%n
       ALLOCATE( A%val( A%ne ) )
       A%val = [ 11.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,                          &
                 0.0_rp_, 22.0_rp_, 32.0_rp_, 0.0_rp_,                         &
                 13.0_rp_, 0.0_rp_, 33.0_rp_, 0.0_rp_,                         &
                 0.0_rp_, 24.0_rp_, 0.0_rp_, 44.0_rp_,                         &
                 15.0_rp_, 0.0_rp_, 0.0_rp_, 45.0_rp_ ]
     CASE ( 3 ) ! sparse by rows
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s )
       A%ne = 9
       ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) )
       A%ptr = [ 1, 4, 6, 8, 10 ]
       A%col = [ 1, 5, 3, 2, 4, 3, 2, 4, 5 ]
       A%val = [ 11.0_rp_, 15.0_rp_, 13.0_rp_, 22.0_rp_, 24.0_rp_,             &
                 33.0_rp_, 32.0_rp_, 44.0_rp_, 45.0_rp_ ]
     CASE ( 4 ) ! sparse by columns
       CALL SMT_put( A%type, 'SPARSE_BY_COLUMNS', s )
       A%ne = 9
       ALLOCATE( A%ptr( A%n + 1 ), A%row( A%ne ), A%val( A%ne ) )
       A%ptr = [ 1, 2, 4, 6, 8, 10 ]
       A%row = [ 1, 3, 2, 1, 3, 2, 4, 4, 1 ]
       A%val = [ 11.0_rp_, 32.0_rp_, 22.0_rp_, 13.0_rp_, 33.0_rp_,             &
                 24.0_rp_, 44.0_rp_, 45.0_rp_, 15.0_rp_ ]
     CASE ( 5 ) ! sparse co-ordinate
       CALL SMT_put( A%type, 'COORDINATE', s )
       A%ne = 9
       ALLOCATE( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ) )
       A%row = [ 4, 1, 3, 2, 1, 3, 4, 2, 1 ]
       A%col = [ 5, 1, 2, 2, 3, 3, 4, 4, 5 ]
       A%val = [ 45.0_rp_, 11.0_rp_, 32.0_rp_, 22.0_rp_, 13.0_rp_,             &
                 33.0_rp_, 44.0_rp_, 24.0_rp_, 15.0_rp_ ]
     END SELECT
     END SUBROUTINE SET_A

     SUBROUTINE DEALLOCATE_A( type, A )
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: type
     TYPE ( SMT_type ), INTENT( INOUT ) :: A
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
     END SUBROUTINE DEALLOCATE_A

     SUBROUTINE SET_H( type, H )
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: type
     TYPE ( SMT_type ), INTENT( INOUT ) :: H
     H%n = 4
     SELECT CASE ( type )
     CASE ( 1 ) ! dense
       CALL SMT_put( H%type, 'DENSE', s )
       H%ne = ( H%n * ( H%n + 1 ) ) / 2
       ALLOCATE( H%val( H%ne ) ) 
       H%val = [ 11.0_rp_, 0.0_rp_, 22.0_rp_, 0.0_rp_, 32.0_rp_,               &
                 33.0_rp_, 0.0_rp_, 42.0_rp_, 0.0_rp_, 44.0_rp_ ]
     CASE ( 2 ) ! dense by columns
       CALL SMT_put( H%type, 'DENSE_BY_COLUMNS', s )
       H%ne = ( H%n * ( H%n + 1 ) ) / 2
       ALLOCATE( H%val( H%ne ) )
       H%val = [ 11.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_, 22.0_rp_,                &
                 32.0_rp_, 42.0_rp_, 33.0_rp_, 0.0_rp_, 44.0_rp_ ]
     CASE ( 3 ) ! sparse by rows
       CALL SMT_put( H%type, 'SPARSE_BY_ROWS', s )
       H%ne = 6
       ALLOCATE( H%ptr( H%n + 1 ), H%col( H%ne ), H%val( H%ne ) )
       H%ptr = [ 1, 2, 3, 5, 7 ]
       H%col = [ 1, 2, 2, 3, 2, 4 ]
       H%val = [ 11.0_rp_, 22.0_rp_, 32.0_rp_, 33.0_rp_, 42.0_rp_, 44.0_rp_ ]
     CASE ( 4 ) ! sparse by columns
       CALL SMT_put( H%type, 'SPARSE_BY_COLUMNS', s )
       H%ne = 6
       ALLOCATE( H%ptr( H%n + 1 ), H%row( H%ne ), H%val( H%ne ) )
       H%ptr = [ 1, 2, 5, 6, 7 ]
       H%row = [ 1, 2, 3, 4, 3, 4 ]
       H%val = [ 11.0_rp_, 22.0_rp_, 32.0_rp_, 42.0_rp_, 33.0_rp_, 44.0_rp_ ]
     CASE ( 5 ) ! sparse co-ordinate
       CALL SMT_put( H%type, 'COORDINATE', s )
       H%ne = 6
       ALLOCATE( H%row( H%ne ), H%col( H%ne ), H%val( H%ne ) )
       H%row = [ 1, 2, 3, 3, 4, 4 ]
       H%col = [ 1, 2, 2, 3, 2, 4 ]
       H%val = [ 11.0_rp_, 22.0_rp_, 32.0_rp_, 33.0_rp_, 42.0_rp_, 44.0_rp_ ]
     END SELECT
     END SUBROUTINE SET_H

     SUBROUTINE DEALLOCATE_H( type, H ) 
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: type
     TYPE ( SMT_type ), INTENT( INOUT ) :: H
     SELECT CASE ( type ) ! deallocate space
     CASE ( 1, 2 ) ! dense + dense by columns
       DEALLOCATE( H%val )
     CASE ( 3 ) ! sparse by rows
       DEALLOCATE( H%ptr, H%col, H%val )
     CASE ( 4 ) ! sparse by columns
       DEALLOCATE( H%ptr, H%row, H%val )
     CASE ( 5 ) ! sparse co-ordinate
       DEALLOCATE( H%row, H%col, H%val )
     END SELECT
     END SUBROUTINE DEALLOCATE_H

   END PROGRAM GALAHAD_CONVERT_TEST
