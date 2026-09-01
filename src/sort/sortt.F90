! THIS VERSION: GALAHAD 5.6 - 2026-08-28 AT 08:40 GMT.
#include "galahad_modules.h"
 PROGRAM GALAHAD_SORT_HEAP_TEST_PROGRAM
 USE GALAHAD_KINDS_precision
 USE GALAHAD_SORT_precision
 USE GALAHAD_STRING, ONLY: STRING_ordinal
 IMPLICIT NONE
 INTEGER ( KIND = ip_ ), PARAMETER :: n = 20
 INTEGER ( KIND = ip_ ) :: i, l, m, inform
 LOGICAL :: largest
 INTEGER ( KIND = ip_ ) :: IA( n ), IX( n ), IA0( n ), IX0( n )
 REAL ( KIND = rp_ ) :: A( n ), RX( n ), A0( n ), RX0( n )
 CHARACTER ( LEN = 8 ) :: largest_string

 WRITE( 6, "( /, ' Test error returns ', / )" )

 A( 1 ) = 1.0
 CALL SORT_heapsort_build( 0_ip_, A, inform )
 WRITE( 6, "( ' inform = ', I0 )" ) inform
 CALL SORT_heapsort_build( 1_ip_, A, inform )
 CALL SORT_heapsort_smallest( 0_ip_, A, inform )
 WRITE( 6, "( ' inform = ', I0 )" ) inform

!  set initial values

 IA0 = (/ -5, -7, 2, 9, 0, -3, 3, 5, -2, -6,                                   &
           8, 7, -1, -8, 10, -4, 6, -9, 1, 4 /)
 A0 = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,                &
          8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /)
 IX0 = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                       &
          15, 16, 17, 18, 19, 20 /)
 RX0 = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,       &
          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 /) 

 DO l = 1, 2
   largest = l == 2
   IF ( largest ) THEN 
     largest_string = 'largest '
   ELSE
     largest_string = 'smallest'
   END IF

!  order real values

   A = A0
   WRITE( 6, "( /, ' Order real values ', / )" )
   CALL SORT_heapsort_build( n, A, inform, &
                             largest = largest ) ! Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, A, inform,                                &
                                  largest = largest ) ! Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value is ',                   &
    &       F5.1 ) " ) i, string_ordinal( i ), TRIM( largest_string ), A( m )
   END DO

   A = A0 ; IX = IX0
   WRITE( 6, "( /, ' Order real values ', / )" )
   CALL SORT_heapsort_build( n, A, inform, ix = IX,                            &
                             largest = largest ) ! Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, A, inform, ix = IX,                       &
                                  largest = largest ) ! Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', I2, ') is ',      &
    & F5.1 ) " ) i, string_ordinal( i ), TRIM( largest_string ), IX( m ), A( m )
   END DO

   A = A0 ; RX = RX0
   WRITE( 6, "( /, ' Order real values ', / )" )
   CALL SORT_heapsort_build( n, A, inform, rx = RX,                            &
                             largest = largest ) ! Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, A, inform, rx = RX,                       &
                             largest = largest ) ! Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', F4.1, ') is ',    &
    & F5.1 ) " ) i, string_ordinal( i ), TRIM( largest_string ), RX( m ), A( m )
   END DO

   A = A0 ; IX = IX0 ; RX = RX0
   WRITE( 6, "( /, ' Order real values ', / )" )
   CALL SORT_heapsort_build( n, A, inform, ix = IX, rx = RX,                   &
                             largest = largest ) !  Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, A, inform, ix = IX, rx = RX,              &
                             largest = largest ) ! Reorder
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', I2, ', ', F4.1,   &
    &  ') is ', F5.1 ) " ) i, string_ordinal( i ), TRIM( largest_string ),     &
       IX( m ), RX( m ), A( m )
   END DO

!  order integer values

   IA = IA0
   WRITE( 6, "( /, ' Order integer values ', / )" )
   CALL SORT_heapsort_build( n, IA, inform,                                    &
                             largest = largest ) !  Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, IA, inform,                               &
                                  largest = largest ) !  Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value is ',                   &
    &       I3 ) " ) i, string_ordinal( i ), TRIM( largest_string ), IA( m )
   END DO

   IA = IA0 ; IX = IX0
   WRITE( 6, "( /, ' Order integer values ', / )" )
   CALL SORT_heapsort_build( n, IA, inform, ix = IX,                           &
                             largest = largest ) !  Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, IA, inform, ix = IX,                      &
                                  largest = largest ) !Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', I2, ') is ',      &
    &  I3 ) " ) i, string_ordinal( i ), TRIM( largest_string ), IX( m ), IA( m )
   END DO

   IA = IA0 ; RX = RX0
   WRITE( 6, "( /, ' Order integer values ', / )" )
   CALL SORT_heapsort_build( n, IA, inform, rx = RX,                           &
                             largest = largest ) !  Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, IA, inform, rx = RX,                      &
                             largest = largest ) !Reorder the variables
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', F4.1, ') is ',    &
    &  I3 ) " ) i, string_ordinal( i ), TRIM( largest_string ), RX( m ), IA( m )
   END DO

   IA = IA0 ; IX = IX0 ; RX = RX0
   WRITE( 6, "( /, ' Order integer values ', / )" )
   CALL SORT_heapsort_build( n, IA, inform, ix = IX, rx = RX,                  &
                             largest = largest ) !  Build the heap
   DO i = 1, n
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, IA, inform, ix = IX, rx = RX,             &
                                 largest = largest ) ! Reorder
     WRITE( 6, "( ' The ', I2, '-', A2, 1X, A, ' value, a(', I2, ', ', F4.1,  &
    &  ') is ', I3 ) " ) i, string_ordinal( i ), TRIM( largest_string ),      &
         IX( m ), RX( m ), IA( m )
   END DO
 END DO

 WRITE( 6, "( /, ' tests completed' )" )

 STOP
 END PROGRAM GALAHAD_SORT_HEAP_TEST_PROGRAM
