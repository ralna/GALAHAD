! THIS VERSION: GALAHAD 3.3 - 04/02/2020 AT 11:30 GMT.
   PROGRAM GALAHAD_SORT_EXAMPLE
   USE GALAHAD_SORT_double                   ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   INTEGER, PARAMETER :: n = 20
   INTEGER :: i, m, inform
   INTEGER :: p( n )
   REAL ( KIND = wp ) :: x( n )
   x = (/ -5.0, -7.0, 2.0, 9.0, 0.0, -3.0, 3.0, 5.0, -2.0, -6.0,              &
           8.0, 7.0, -1.0, -8.0, 10.0, -4.0, 6.0, -9.0, 1.0, 4.0 /)  ! values
   p = (/  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,                     &
          15, 16, 17, 18, 19, 20 /)                                  ! indices
! write the initial data
   WRITE( 6, "( /' The vector x is' / 2( 10 ( F5.1, 2X ) / ) ) " ) x( 1:n )
   WRITE( 6, "(  ' The permutation is' /  20 ( 2X, I2 ) / ) " ) p( 1:n )
! sort x and obtain the inverse permutation
   WRITE( 6, "( ' Sort x in ascending order' / )" )
   CALL SORT_quicksort( n, x, inform, p )
   WRITE( 6, "( ' The vector x is now' / 2( 10 ( F5.1, 2X ) / ) ) " ) x( 1:n )
   WRITE( 6, "( ' The permutation is now' /  20 ( 2X, I2 ) / ) " ) p( 1:n )
! apply the inverse permutation to x
   WRITE( 6, "( ' Apply the permutation to x' / )" )
   CALL SORT_inplace_permute( n, p, x )
   WRITE( 6, "( ' The vector x is now' / 2( 10 ( F5.1, 2X ) / ) ) " )  x( 1:n )
! restore the identity permutation
   WRITE( 6, "( ' Restore the identity permutation by sorting' / )" )
   CALL SORT_quicksort( n, p, inform )
   WRITE( 6, "( ' The permutation is now' /  20 ( 2X, I2 ) / ) " ) p( 1:n )
! get the 12 smallest components and the associated inverse permutation
   WRITE( 6, "( ' Get the 12 smallest components of x' / )" )
   CALL SORT_heapsort_build( n, x, inform, ix = p ) !  Build the heap
   DO i = 1, 12
     m = n - i + 1
     CALL SORT_heapsort_smallest( m, x, inform, ix = p ) ! Reorder the variables
     WRITE( 6, "( ' The ', I2, '-th(-st) smallest value, x(', I2, ') is ',     &
    &       F5.1 ) " ) i, p( m ), x( m )
   END DO
   WRITE( 6, "( / ' The permutation is now' /  20 ( 2X, I2 ) / ) " ) p( 1:n )
! compute the direct permutation in p
   WRITE( 6, "( ' Compute the inverse of this permutation' / )" )
   CALL SORT_inplace_invert( n, p )
   WRITE( 6, "( ' The permutation is now' /  20 ( 2X, I2 ) / ) " ) p( 1:n )
! apply inverse permutation
   WRITE( 6, "( ' Apply the resulting permutation to x' / )" )
   CALL SORT_inverse_permute( n, p, x )
   WRITE( 6, "( ' The final vector is' / 2( 10 ( F5.1, 2X ) / ) ) " ) x( 1:n )
   STOP
   END PROGRAM GALAHAD_SORT_EXAMPLE
