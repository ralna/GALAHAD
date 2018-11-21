! THIS VERSION: 13/11/2018 AT 08:37:00 GMT.
! Updated 13/11/2018: fully arguments supplied

!     -*-*-*-*-*-*-  G A L A H A D  -  MC13  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 8th 2018

      SUBROUTINE MC13D( n, ICN, licn, IP, LENR, IOR, IB, num, IW )
      INTEGER :: licn, n, num
      INTEGER :: IB( n ), ICN( licn ), IOR( n )
      INTEGER :: IP( n ), IW( n, 3 ), LENR( n )
      RETURN
      END SUBROUTINE MC13D


      SUBROUTINE MC13E( n, ICN, licn, IP, LENR, ARP, IB, num, LOWL,
     *                   NUMB, PREV )
      INTEGER :: licn, n, num
      INTEGER :: ARP( n ), IB (n ), ICN( licn ), IP( n ), LENR( n )
      INTEGER :: LOWL( n ), NUMB( n ), PREV( n )
      RETURN
      END SUBROUTINE MC13E
