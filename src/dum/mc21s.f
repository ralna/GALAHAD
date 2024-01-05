! THIS VERSION: GALAHAD 4.3 - 2024-01-04 AT 14:40 GMT.

#ifdef GALAHAD_64BIT_INTEGER
  INTEGER, PARAMETER :: ip_ = INT64
#else
  INTEGER, PARAMETER :: ip_ = INT32
#endif

! dummy routine

      SUBROUTINE MC21A( )
      END

      SUBROUTINE MC21B( n, ICN, licn, IP, LENR, IPERM, numnz, 
     *                  PR, ARP, CV, OUT )
      INTEGER ( KIND = ip_ ) :: licn, n, numnz
      INTEGER ( KIND = ip_ ) :: ARP( n ), CV( n )
      INTEGER ( KIND = ip_ ) :: ICN( licn ), IP( n ), IPERM( n )
      INTEGER ( KIND = ip_ ) :: LENR( n ), OUT( n ), PR( n )
      END
