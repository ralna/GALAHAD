! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
 PROGRAM GALAHAD_SORT_TESTDEC
 USE GALAHAD_SORT_double             ! double precision version
 IMPLICIT NONE
 INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
 INTEGER, PARAMETER :: n = 20
 INTEGER :: i, m, inform
 LOGICAL :: largest
 INTEGER :: INDA( n ), IA( n )
 REAL ( KIND = wp ) :: A( n )

 WRITE( 6, "( /, ' Test error returns ', / )" )

 A( 1 ) = 1.0
 CALL SORT_heapsort_build( 0, A, inform )
 WRITE( 6, "( ' inform =  ', I2 )" ) inform
 CALL SORT_heapsort_build( 1, A, inform )
 CALL SORT_heapsort_smallest( 0, A, inform )
 WRITE( 6, "( ' inform =  ', I2 )" ) inform

 A = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,                 &
         8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /) ! values

 WRITE( 6, "( /, ' Order real values ', / )" )
 CALL SORT_heapsort_build( n, A, inform ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   CALL SORT_heapsort_smallest( m, A, inform ) !  Reorder the variables
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value is ',                    &
  &       F5.1 ) " ) i, A( m )
 END DO

 A = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,                 &
         8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /) ! values
 INDA = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                      &
           15, 16, 17, 18, 19, 20 /) ! indices

 WRITE( 6, "( /, ' Order real values ', / )" )
 CALL SORT_heapsort_build( n, A, inform, ix = INDA ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   CALL SORT_heapsort_smallest( m, A, inform, ix = INDA ) !Reorder the variables
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value, a(', I2, ') is ',       &
  &       F5.1 ) " ) i, INDA( m ), A( m )
 END DO

 IA = (/ -5, -7, 2, 9, 0, -3, 3, 5, -2, -6,                                    &
         8, 7, -1, -8, 10, -4, 6, -9, 1, 4 /) ! values

 WRITE( 6, "( /, ' Order integer values ', / )" )
 CALL SORT_heapsort_build( n, IA, inform ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   CALL SORT_heapsort_smallest( m, IA, inform ) !  Reorder the variables
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value is ',                    &
  &       I3 ) " ) i, IA( m )
 END DO

 IA = (/ -5, -7, 2, 9, 0, -3, 3, 5, -2, -6,                                    &
         8, 7, -1, -8, 10, -4, 6, -9, 1, 4 /) ! values
 INDA = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                      &
           15, 16, 17, 18, 19, 20 /) ! indices

 WRITE( 6, "( /, ' Order integer values ', / )" )
 CALL SORT_heapsort_build( n, IA, inform, ix = INDA ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   CALL SORT_heapsort_smallest( m, IA, inform, ix = INDA )!Reorder the variables
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value, a(', I2, ') is ',       &
  &       I3 ) " ) i, INDA( m ), IA( m )
 END DO

! Sort in decreasing order

largest = .TRUE.

 A = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,                 &
         8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /) ! values

 WRITE( 6, "( /, ' Arrange real values in decreasing order', / )" )
 CALL SORT_heapsort_build( n, A, inform, largest = largest ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   !  Reorder the variables
   CALL SORT_heapsort_smallest( m, A, inform, largest = largest )
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value is ',                    &
  &       F5.1 ) " ) i, A( m )
 END DO

 A = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,                 &
         8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /) ! values
 INDA = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                      &
           15, 16, 17, 18, 19, 20 /) ! indices

 WRITE( 6, "( /, ' Arrange real values in decreasing order', / )" )
 !  Build the heap
 CALL SORT_heapsort_build( n, A, inform, ix = INDA, largest = largest )
 DO i = 1, n
   m = n - i + 1
  !  Reorder the variables
   CALL SORT_heapsort_smallest( m, A, inform, ix = INDA, largest = largest )
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value, a(', I2, ') is ',       &
  &       F5.1 ) " ) i, INDA( m ), A( m )
 END DO

 IA = (/ -5, -7, 2, 9, 0, -3, 3, 5, -2, -6,                                    &
         8, 7, -1, -8, 10, -4, 6, -9, 1, 4 /) ! values

 WRITE( 6, "( /, ' Arrange integer values in decreasing order', / )" )
 CALL SORT_heapsort_build( n, IA, inform, largest = largest ) !  Build the heap
 DO i = 1, n
   m = n - i + 1
   !  Reorder the variables
   CALL SORT_heapsort_smallest( m, IA, inform, largest = largest )
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value is ',                    &
  &       I3 ) " ) i, IA( m )
 END DO

 IA = (/ -5, -7, 2, 9, 0, -3, 3, 5, -2, -6,                                    &
         8, 7, -1, -8, 10, -4, 6, -9, 1, 4 /) ! values
 INDA = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                      &
           15, 16, 17, 18, 19, 20 /) ! indices

 WRITE( 6, "( /, ' Arrange integer values in decreasing order', / )" )
 !  Build the heap
 CALL SORT_heapsort_build( n, IA, inform, ix = INDA, largest = largest )
 DO i = 1, n
   m = n - i + 1
   !  Reorder the variables
   CALL SORT_heapsort_smallest( m, IA, inform, ix = INDA, largest = largest )
   WRITE( 6, "( ' The ', I2, '-th(-st) smallest value, a(', I2, ') is ',       &
  &       I3 ) " ) i, INDA( m ), IA( m )
 END DO

 STOP
 END PROGRAM GALAHAD_SORT_TESTDEC
