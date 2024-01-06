! THIS VERSION: GALAHAD 4.3 - 2024-01-05 AT 14:40 GMT.

! dummy routine

      SUBROUTINE MC21A( )
      END

      SUBROUTINE MC21B( n, ICN, licn, IP, LENR, IPERM, numnz, 
     *                  PR, ARP, CV, OUT )
      USE GALAHAD_KINDS_single
      INTEGER ( KIND = ip_ ) :: licn, n, numnz
      INTEGER ( KIND = ip_ ) :: ARP( n ), CV( n )
      INTEGER ( KIND = ip_ ) :: ICN( licn ), IP( n ), IPERM( n )
      INTEGER ( KIND = ip_ ) :: LENR( n ), OUT( n ), PR( n )
      END
