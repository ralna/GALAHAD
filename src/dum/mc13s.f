! THIS VERSION: GALAHAD 4.3 - 2024-01-05 AT 14:30 GMT.

!     -*-*-*-*-*-*-  G A L A H A D  -  MC13  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 8th 2018

      SUBROUTINE MC13D( n, ICN, licn, IP, LENR, IOR, IB, num, IW )
      USE GALAHAD_KINDS_single
      INTEGER ( KIND = ip_ ) :: licn, n, num
      INTEGER ( KIND = ip_ ) :: IB( n ), ICN( licn ), IOR( n )
      INTEGER ( KIND = ip_ ) :: IP( n ), IW( n, 3 ), LENR( n )
      RETURN
      END SUBROUTINE MC13D


      SUBROUTINE MC13E( n, ICN, licn, IP, LENR, ARP, IB, num, LOWL,
     *                   NUMB, PREV )
      USE GALAHAD_KINDS_single
      INTEGER ( KIND = ip_ ) :: licn, n, num
      INTEGER ( KIND = ip_ ) :: ARP( n ), IB (n )
      INTEGER ( KIND = ip_ ) :: ICN( licn ), IP( n ), LENR( n )
      INTEGER ( KIND = ip_ ) :: LOWL( n ), NUMB( n ), PREV( n )
      RETURN
      END SUBROUTINE MC13E
