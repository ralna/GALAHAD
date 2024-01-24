! THIS VERSION: GALAHAD 4.3 - 2024-01-24 AT 15:55 GMT.

#include "galahad_hsl.h"

! dummy routine

      SUBROUTINE MC21AD( )
      END

      SUBROUTINE MC21BD( n, ICN, licn, IP, LENR, IPERM, numnz,
     *                   PR, ARP, CV, OUT )
      USE GALAHAD_KINDS
      INTEGER ( KIND = ip_ ) :: licn, n, numnz
      INTEGER ( KIND = ip_ ) :: ARP( n ), CV( n )
      INTEGER ( KIND = ip_ ) :: ICN( licn ), IP( n ), IPERM( n )
      INTEGER ( KIND = ip_ ) :: LENR( n ), OUT( n ), PR( n )
      END
