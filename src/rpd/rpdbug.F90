! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

      TYPE, PUBLIC :: problem_type
        CHARACTER, ALLOCATABLE, DIMENSION( : ) :: name
      END TYPE

   CONTAINS

     SUBROUTINE A( prob )
     TYPE ( problem_type ), INTENT( INOUT ) :: prob
     CHARACTER ( LEN = 10 ) :: pname
     pname = '1234567890'
     ALLOCATE( prob%name( 10 ) )
     prob%name = TRANSFER( pname, prob%name )
     END SUBROUTINE A

   END MODULE BUG
