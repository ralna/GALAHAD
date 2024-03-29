! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

! COPYRIGHT (c) 2006 Council for the Central Laboratory
!                    of the Research Councils
! Original date 21 February 2006. Version 1.0.0.
! 6 March 2007 Version 1.1.0. Argument stat made non-optional

   MODULE hsl_zd11_precision

     USE GALAHAD_KINDS_precision

!  ==========================
!  Sparse matrix derived type
!  ==========================

     TYPE, PUBLIC :: ZD11_type
       INTEGER ( KIND = ip_ ) :: m, n, ne
       CHARACTER, ALLOCATABLE, DIMENSION( : ) :: id, type
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: row, col, ptr
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: val
     END TYPE

   CONTAINS

     SUBROUTINE ZD11_put( array, string, stat )
     CHARACTER, ALLOCATABLE :: array( : )
     CHARACTER ( * ), INTENT( IN ) ::  string
     INTEGER ( KIND = ip_ ), INTENT( OUT ) ::  stat

!  Copy a string into an array

     INTEGER ( KIND = ip_ ) :: i, l
     l = LEN_TRIM( string )
     IF ( ALLOCATED( array ) ) THEN
       DEALLOCATE( array, STAT = stat )
       IF ( stat /= 0 ) RETURN
     end if
     ALLOCATE( array( l ), STAT = stat )
     IF ( stat /= 0 ) RETURN
     DO i = 1, l
       array( i ) = string( i : i )
     END DO
     END SUBROUTINE ZD11_put

     FUNCTION ZD11_get( array )
     CHARACTER, INTENT( IN ) :: array( : )
     CHARACTER( SIZE( array ) ) :: ZD11_get

!  Give the value of array to string

     INTEGER ( KIND = ip_ ) :: i
     DO i = 1, SIZE( array )
       ZD11_get( i : i ) = array( i )
     END DO
     END FUNCTION ZD11_get

   END MODULE hsl_zd11_precision
