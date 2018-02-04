! THIS VERSION: GALAHAD 2.6 - 03/07/2014 AT 12:00 GMT.

!-*-*-*-*-*-*-*-*  G A L A H A D _ R A N D   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and John Reid

!  History -
!   originally released pre GALAHAD Version 1.0. Sept 1st 1995
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      MODULE GALAHAD_RAND_double

!  Portable random number generator by Linus Schrange, TOMS 5, 1979, pp 132-138

         IMPLICIT NONE

         PRIVATE
         PUBLIC :: RAND_initialize, RAND_random_real, RAND_random_integer,     &
                   RAND_get_seed, RAND_set_seed

!  Define the working precision to be double

         INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

         INTEGER, PARAMETER :: a = 16807, b15 = 32768
         INTEGER, PARAMETER :: b16 = 65536, p = 2147483647
         INTEGER, PARAMETER :: b30 = 1073741824, q = 1073741823 

         TYPE, PUBLIC :: RAND_seed
           PRIVATE
           INTEGER :: ix =  b16 - 1
         END TYPE

         INTERFACE RAND_random_real
           MODULE PROCEDURE RAND_random_real_scalar,                           &
                            RAND_random_real_vector,                           &
                            RAND_random_real_matrix
         END INTERFACE

         INTERFACE RAND_random_integer
           MODULE PROCEDURE RAND_random_integer_scalar,                        &
                            RAND_random_integer_vector,                        &
                            RAND_random_integer_matrix
         END INTERFACE


      CONTAINS

!-*-*-*-*-*-  R A N D _ I N I T I A L I Z E  S U B R O U T I N E  *-*-*-*-*-*-

         SUBROUTINE RAND_initialize ( seed )

!  Set the word generator to its default value

         TYPE ( RAND_seed ), INTENT( OUT ) :: seed

         seed%ix = b16 - 1

         END SUBROUTINE RAND_initialize

!-*-  R A N D _ R A N D O M _ R E A L  _ S C A L A R  S U B R O U T I N E  -*-

         SUBROUTINE RAND_random_real_scalar( seed, positive, random ) 

!  Real random number in the range [0, 1] ( if positive is .TRUE. ) 
!  or [-1, 1] ( if positive is .FALSE. )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         LOGICAL, INTENT( IN ) :: positive
         REAL ( KIND = wp ), INTENT( OUT ) :: random

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: fhi, k, leftlo, xalo, xhi 
         REAL ( KIND = wp ) :: x 
         REAL ( KIND = wp ), PARAMETER ::                                      &
                one = 1.0_wp, two = 2.0_wp, rb16 = two ** 16,                  &
                big  = one / ( two ** 31 - one ), big2 = two * big

         xhi = seed%ix / b16 

! Get 16 lo bits of seed%ix and form lo product

         xalo = ( seed%ix - xhi * b16 ) * a 

!  Get 15 hi order bits of lo product

         leftlo = xalo / b16 

!  Form the 31 highest bits of full product

         fhi = xhi * a + leftlo 

!  Get overflopast 31st bit of full product

         k = fhi / b15 

!  Assemble all the parts and presubtract P. The parentheses are essential

         seed%ix = ( ( ( xalo - leftlo * b16 ) - p ) +                         &
                     ( fhi - k * b15 ) * b16 ) + k 

!  Add p back in if neccessary

         IF ( seed%ix < 0 ) seed%ix = seed%ix + p 

!  Multiply by 1/(2**31-1)

         xhi = seed%ix / b16 
         x = FLOAT( xhi ) * rb16 + FLOAT( seed%ix - xhi * b16 ) 
         IF ( positive ) THEN
            random = x * big
         ELSE
            random = x * big2 - one
         END IF

         END SUBROUTINE RAND_random_real_scalar

!-*-  R A N D _ R A N D O M _ R E A L  _ V E C T O R  S U B R O U T I N E  -*-

         SUBROUTINE RAND_random_real_vector( seed, positive, random ) 

!  Real random vector in the range [0, 1] ( if positive is .TRUE. ) 
!  or [-1, 1] ( if positive is .FALSE. )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         LOGICAL, INTENT( IN ) :: positive
         REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: i, dim1
         dim1 = SIZE( RANDOM, 1 )
         DO i = 1, dim1
           CALL RAND_random_real_scalar( seed, positive, RANDOM( i ) ) 
         END DO

         END SUBROUTINE RAND_random_real_vector

!-*-  R A N D _ R A N D O M _ R E A L  _ M A T R I X  S U B R O U T I N E  -*-

         SUBROUTINE RAND_random_real_matrix( seed, positive, random ) 

!  Real random matrix in the range [0, 1] ( if positive is .TRUE. ) 
!  or [-1, 1] ( if positive is .FALSE. )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         LOGICAL, INTENT( IN ) :: positive
         REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : , : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: i, j, dim1, dim2
         dim1 = SIZE( RANDOM, 1 ) ; dim2 = SIZE( RANDOM, 2 )
         DO j = 1, dim2
           DO i = 1, dim1
             CALL RAND_random_real_scalar( seed, positive, RANDOM( i, j ) ) 
           END DO
         END DO

         END SUBROUTINE RAND_random_real_matrix

!-  R A N D _ R A N D O M _ I N T E G E R _ S C A L A R   S U B R O U T I N E  -

         SUBROUTINE RAND_random_integer_scalar( seed, n, random ) 

!  Integer random number in the range [1,n] if n > 1.
!  Otherwise, the value n is returned  

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         INTEGER, INTENT( IN ) :: n
         INTEGER, INTENT( OUT ) :: random

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: be1, be2, c, d, f, fhi, g, k, leftlo
         INTEGER :: mhi, mlo, mu, nu, xalo, xhi, xlo 

         IF ( n > 1 ) THEN

           xhi = seed%ix / b16 

!  Get 16 lo bits of seed%ix and form lo product

           xalo = ( seed%ix - xhi * b16 ) * a 

!  Get 15 hi order bits of lo product

           leftlo = xalo / b16 

!  Form the 31 highest bits of full product

           fhi = xhi * a + leftlo 

!  Get overflopast 31st bit of full product

           k = fhi / b15 

!  Assemble all the parts and presubtract p. The parentheses are essential

           seed%ix = ( ( ( xalo - leftlo * b16 ) - p ) +                       &
                       ( fhi - k * b15 ) * b16 ) +  k
 
!  Add p back in if neccessary

           IF ( seed%ix < 0 ) seed%ix = seed%ix + p 

!  Multiply by n and divide by 2**31-1 in integer arithmetic.
!  Split seed%ix and n into hi and lo parts

           xhi = seed%ix / b15 ; xlo = seed%ix - b15 * xhi
           mhi = n / b15 ; mlo = n - b15 * mhi 

!  Calculate intermediate product and split into hi and lo parts.
!  Presubtract p

           f = ( xhi * mlo - p ) + xlo * mhi 

!  f is > 0 if intermediate product would have overflowed

           IF ( f <= 0 ) THEN 
             f = f + p ; be1 = f / b15 ; be2 = f - be1 * b15 
           ELSE 
             f = f - 1 ; be1 = f / b15 ; be2 = f - be1 * b15; be1 = be1 + b16
           ENDIF 

!  Form product of lo parts and add in lo part of intermediate product
!  to get lo part of complete product

           g = b15 * be2 + xlo * mlo 

!  Represent lo part of full product in base 2**30

           d = g / b30 ; c = xhi / 2 

!  Calculate full product divided by 2**30

           f = (( 2 * ( c * mhi - q ) - 1) + mhi * ( xhi - 2 * c )) + d + be1 

!  Get full product divided in base 2**31

           IF ( f <= 0 ) THEN 
             f = f + p ; nu = f / 2 ; mu = f - nu * 2 
           ELSE 
             f = f - 1 ; nu = f / 2 ; mu = f - 2 * nu ; nu = nu + b30 
           ENDIF 

!  Calculate remainder of product divided by 2**31

           f = ( b30 * mu - p ) + nu + ( g - b30 * d ) 
           random = nu + 1 

!  Add one if remainder is not < 2**31-1

           IF ( f >= 0 ) random = random + 1 
         ELSE

!  If n is less than or equal to 1, set random to n.

           random = n
         END IF

         END SUBROUTINE RAND_random_integer_scalar

!-  R A N D _ R A N D O M _ I N T E G E R _ V E C T O R   S U B R O U T I N E  -

         SUBROUTINE RAND_random_integer_vector( seed, n, RANDOM ) 

!  Integer random vector in the range [1,n] if n > 1.
!  Otherwise, the value n is returned  

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         INTEGER, INTENT( IN ) :: n
         INTEGER, INTENT( OUT ), DIMENSION( : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: i, dim1
         dim1 = SIZE( RANDOM, 1 )
         DO i = 1, dim1
           CALL RAND_random_integer_scalar( seed, n, RANDOM( i ) ) 
         END DO

         END SUBROUTINE RAND_random_integer_vector

!-  R A N D _ R A N D O M _ I N T E G E R _ M A T R I X   S U B R O U T I N E  -

         SUBROUTINE RAND_random_integer_matrix( seed, n, RANDOM ) 

!  Integer random matrix in the range [1,n] if n > 1.
!  Otherwise, the value n is returned  

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( INOUT ) :: seed
         INTEGER, INTENT( IN ) :: n
         INTEGER, INTENT( OUT ), DIMENSION( : , : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER :: i, j, dim1, dim2
         dim1 = SIZE( RANDOM, 1 ) ; dim2 = SIZE( RANDOM, 2 )
         DO j = 1, dim2
           DO i = 1, dim1
             CALL RAND_random_integer_scalar( seed, n, RANDOM( i, j ) ) 
           END DO
         END DO

         END SUBROUTINE RAND_random_integer_matrix

!-*-*-*-*-*-*-  R A N D _ G E T _ S E E D  S U B R O U T I N E  *-*-*-*-*-*-*-

         SUBROUTINE RAND_get_seed ( seed, value_seed ) 
 
!  Determine the current word generator.

         TYPE ( RAND_seed ), INTENT( IN ) :: seed
         INTEGER, INTENT( OUT ) :: value_seed

         value_seed = seed%ix 

         END SUBROUTINE RAND_get_seed  

!-*-*-*-*-*-*-  R A N D _ S E T _ S E E D   S U B R O U T I N E  *-*-*-*-*-*-

         SUBROUTINE RAND_set_seed ( seed, value_seed ) 

!  Reset the word generator to seed if seed lies in the
!  interval [1, 2**31 - 1]. More generally, seed is set 
!  to ( seed - 1 ) mod (2**31 -1) + 1

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE ( RAND_seed ), INTENT( OUT ) :: seed
         INTEGER, INTENT( IN ) :: value_seed

         seed%ix = MOD( value_seed - 1, p ) + 1

         END SUBROUTINE RAND_set_seed 

      END MODULE GALAHAD_RAND_double

