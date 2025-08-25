! THIS VERSION: GALAHAD 5.3 - 2025-08-23 AT 16:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*  G A L A H A D _ R A N D   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  elements *_lgc are from SPRAL,
!  COPYRIGHT (c) 2014 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  Principal authors: Nick Gould, John Reid and Jonathan Hogg
!  SPRAL components Forked and extended for GALAHAD, Nick Gould, 2016

!  History -
!   originally released pre GALAHAD Version 1.0. Sept 1st 1995
!   update released with GALAHAD Version 2.0. February 16th 2005
!   SPRAL components added GALAHAD 5.3, August 23, 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

      MODULE GALAHAD_RAND_precision

!  Implementations of both the portable random number generator by 
!  Linus Schrange, TOMS 5, 1979, pp 132-138 and the linear congruential 
!  pseudo-randomized number generator (LCG) by Lehmer, Thomson & Rotenberg
!  see https://en.wikipedia.org/wiki/Linear_congruential_generator

         USE GALAHAD_KINDS_precision

         IMPLICIT NONE

         PRIVATE
         PUBLIC :: RAND_initialize, RAND_random_real, RAND_random_integer,     &
                   RAND_get_seed, RAND_set_seed,                               &
                   RAND_random_real_lcg, RAND_random_integer_lcg,              &
                   RAND_random_logical_lcg,  RAND_get_seed_lcg,                &
                   RAND_set_seed_lcg

!  Define the working precision to be double

         INTEGER ( KIND = ip_ ), PARAMETER :: a = 16807, b15 = 32768
         INTEGER ( KIND = ip_ ), PARAMETER :: b16 = 65536, p = 2147483647
         INTEGER ( KIND = ip_ ), PARAMETER :: b30 = 1073741824, q = 1073741823

!  LCG data

         INTEGER( KIND = long_ ), PARAMETER :: a_lcg = 1103515245
         INTEGER( KIND = long_ ), PARAMETER :: c_lcg = 12345
         INTEGER( KIND = long_ ), PARAMETER :: m_lcg = 2**31_long_

         TYPE, PUBLIC :: RAND_seed
           PRIVATE
           INTEGER ( KIND = ip_ ) :: ix =  b16 - 1
         END TYPE RAND_seed

         TYPE, PUBLIC :: RAND_random_state_lcg
           PRIVATE
           INTEGER( KIND = ip_ ) :: x_lcg = 486502
         END TYPE RAND_random_state_lcg

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

         INTERFACE RAND_random_integer_lcg
           MODULE PROCEDURE RAND_random_integer_lcg32,                         &
                            RAND_random_integer_lcg64
         END INTERFACE RAND_random_integer_lcg

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
         REAL ( KIND = rp_ ), INTENT( OUT ) :: random

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: fhi, k, leftlo, xalo, xhi
         REAL ( KIND = rp_ ) :: x
         REAL ( KIND = rp_ ), PARAMETER ::                                     &
                one = 1.0_rp_, two = 2.0_rp_, rb16 = two ** 16,                &
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
         x = REAL( xhi, KIND = rp_ ) * rb16                                    &
               + REAL( seed%ix - xhi * b16, KIND = rp_ )
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
         REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: i, dim1
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
         REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( : , : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: i, j, dim1, dim2
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
         INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
         INTEGER ( KIND = ip_ ), INTENT( OUT ) :: random

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: be1, be2, c, d, f, fhi, g, k, leftlo
         INTEGER ( KIND = ip_ ) :: mhi, mlo, mu, nu, xalo, xhi, xlo

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
         INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
         INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: i, dim1
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
         INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
         INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : , : ) :: RANDOM

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER ( KIND = ip_ ) :: i, j, dim1, dim2
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
         INTEGER ( KIND = ip_ ), INTENT( OUT ) :: value_seed

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
         INTEGER ( KIND = ip_ ), INTENT( IN ) :: value_seed

         seed%ix = MOD( value_seed - 1, p ) + 1

         END SUBROUTINE RAND_set_seed

! =============================================================================
!                   contributions from SPRAL
! =============================================================================

!-*-*-*-*-  R A N D _ r a n d o m _ r e a l _ l c g  F U N C T I O N  -*-*-*-*-

         REAL( rp_ ) FUNCTION RAND_random_real_lcg( state, positive )

!  real LCG random number in the range
!   [ 0, 1] ( if positive is present and .TRUE. ); or
!   [-1, 1] ( otherwise )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE( RAND_random_state_lcg ), INTENT( INOUT ) :: state
         LOGICAL, OPTIONAL, INTENT( IN ) :: positive

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         LOGICAL :: pos

         pos = .FALSE.
         IF ( PRESENT( positive ) ) pos = positive

!  X_{n+1} = ( aX_n + c ) mod m

         state%x_lcg = INT( MOD( a_lcg * state%x_lcg + c_lcg, m_lcg ) )

!  convert to a RAND_random real

         IF ( pos ) THEN
           RAND_random_real_lcg = REAL( state%x_lcg, rp_ ) / REAL( m_lcg, rp_ )
         ELSE
           RAND_random_real_lcg                                                &
             = 1.0 - 2.0 * REAL( state%x_lcg, rp_ ) / REAL( m_lcg, rp_ )
         END IF

         END FUNCTION RAND_random_real_lcg

!-*-*-  R A N D _ r a n d o m _ i n t e g e r _ l c g 6 4   F U N C T I O N  -*-

         INTEGER( long_ ) FUNCTION RAND_random_integer_lcg64( state, n )

!  integer LCG random number in the range [1,n] if n > 1,
!  otherwise, the value n is returned

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE( RAND_random_state_lcg ), INTENT( INOUT ) :: state
         INTEGER( i8_ ), INTENT( IN ) :: n

         IF ( n <= 0_i8_ ) THEN
           RAND_random_integer_lcg64 = n
           RETURN
         END IF

!  X_{n+1} = ( aX_n + c ) mod m

         state%x_lcg = INT( MOD( a_lcg * state%x_lcg + c_lcg, m_lcg ) )

!  take modulo n for return value

         RAND_random_integer_lcg64 = INT( state%x_lcg * INT( REAL( n, rp_ )    &
            / REAL( m_lcg, rp_ ), ip_ ), long_ ) + 1

         END FUNCTION RAND_random_integer_lcg64

!-*-*-  R A N D _ r a n d o m _ i n t e g e r _ l c g 3 2   F U N C T I O N  -*-

         INTEGER function RAND_random_integer_lcg32( state, n )

!  integer LCG random number in the range [1,n] if n > 1,
!  otherwise, the value n is returned

         TYPE( RAND_random_state_lcg ), INTENT( INOUT ) :: state
         INTEGER( i4_ ), INTENT( IN ) :: n

!  just call 64-bit version with type casts

         RAND_random_integer_lcg32                                             &
           = INT( RAND_random_integer_lcg64( state, INT( n, long_ ) ) )

         END FUNCTION RAND_random_integer_lcg32

!-*-*-*-  R A N D _ r a n d o m _ l o g i c a l _ l c g  F U N C T I O N -*-*-*-

         LOGICAL FUNCTION RAND_random_logical_lcg( state )

!  generate an LCG random logical value

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE( RAND_random_state_lcg ), INTENT( INOUT ) :: state

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

         INTEGER( ip_ ) :: test

         test = RAND_random_integer_lcg( state, 2 )
         RAND_random_logical_lcg = test == 1

         END FUNCTION RAND_random_logical_lcg

!-*-*-*-*-*-*-  R A N D _ g e t _ s e e d _ l c g  F U N C T I O N  -*-*-*-*-*-

         INTEGER( ip_ ) FUNCTION RAND_get_seed_lcg( state )

!  get random seed for LCG

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE( RAND_random_state_lcg ), INTENT( IN ) :: state

         RAND_get_seed_lcg = state%x_lcg

         END FUNCTION RAND_get_seed_lcg

!-*-*-*-*-  R A N D _ s e t _ s e e d _ l c g  S U B R O U T I N E  -*-*-*-*-

         SUBROUTINE RAND_set_seed_lcg( state, seed )

!  set random seed for LCG

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

         TYPE( RAND_random_state_lcg ), INTENT( INOUT ) :: state
         INTEGER( ip_ ), INTENT( IN ) :: seed

         state%x_lcg = seed

         END SUBROUTINE RAND_set_seed_lcg

      END MODULE GALAHAD_RAND_precision

