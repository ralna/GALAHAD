! THIS VERSION: GALAHAD 2.6 - 31/03/2015 AT 14:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ S T R I N G   M O D U L E  *-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.0. September 14th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_STRING_double

!     ----------------------------
!    |  Set strings appropriate   |
!    !  for singular and plural   |
!    |  forms of words along with |
!    |  other useful strings      |
!     ----------------------------

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: STRING_pleural, STRING_verb_pleural, STRING_are, STRING_have,   &
               STRING_ies, STRING_their, STRING_sign, STRING_choice,           &
               STRING_lower, STRING_upper, STRING_lower_word,                  &
               STRING_upper_word, STRING_put, STRING_get, STRING_integer_6,    &
               STRING_real_7, STRING_real_12, STRING_trim_real_24,             &
               STRING_trim_integer_16, STRING_es, STRING_es12,                 &
               STRING_exponent, STRING_ordinal

!  Set precision

     INTEGER, PARAMETER :: sp = KIND( 1.0E+0 )
     INTEGER, PARAMETER :: dp = KIND( 1.0D+0 )

     REAL( KIND = sp ), PARAMETER :: teneps_s = 10.0_sp * EPSILON( 1.0_sp )
     REAL( KIND = dp ), PARAMETER :: teneps_d = 10.0_dp * EPSILON( 1.0_dp )
   
!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

     INTERFACE STRING_sign
       MODULE PROCEDURE STRING_sign_single, STRING_sign_double
     END INTERFACE

     INTERFACE STRING_real_7
       MODULE PROCEDURE STRING_real_7_single, STRING_real_7_double
      END INTERFACE

     INTERFACE STRING_real_12
       MODULE PROCEDURE STRING_real_12_single, STRING_real_12_double
     END INTERFACE

     INTERFACE STRING_trim_real_24
       MODULE PROCEDURE STRING_trim_real_24_single, STRING_trim_real_24_double
     END INTERFACE

     INTERFACE STRING_es
       MODULE PROCEDURE STRING_es_single, STRING_es_double
     END INTERFACE

     INTERFACE STRING_es12
       MODULE PROCEDURE STRING_es12_single, STRING_es12_double
     END INTERFACE

     INTERFACE STRING_exponent
       MODULE PROCEDURE STRING_exponent_single, STRING_exponent_double
     END INTERFACE

   CONTAINS

!-*-*-*-  G A L A H A D -  S T R I N G _ p l e u r a l   F U N C T I O N  -*-*-

     FUNCTION STRING_pleural( val )

!   Given an integer val, returns "s" if val /= 1, otherwise returns " "

     CHARACTER ( len = 1 ) :: STRING_pleural

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val /= 1 ) THEN
       STRING_pleural = "s"
     ELSE
       STRING_pleural = " "
     END IF

     RETURN

!  End of function STRING_pleural

      END FUNCTION STRING_pleural

!-  G A L A H A D -  S T R I N G _ v e r b _ p l e u r a l   F U N C T I O N  -

     FUNCTION STRING_verb_pleural( val )

!   Given an integer val, returns "s" if val = 1, otherwise returns " "

     CHARACTER ( len = 1 ) :: STRING_verb_pleural

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val == 1 ) THEN
       STRING_verb_pleural = "s"
     ELSE
       STRING_verb_pleural = " "
     END IF

     RETURN

!  End of function STRING_verb_pleural

      END FUNCTION STRING_verb_pleural

!-*-*-*-*-  G A L A H A D -  S T R I N G _ a r e   F U N C T I O N  -*-*-*-*-

     FUNCTION STRING_are( val )

!   Given an integer val, returns "are" if val /= 1, otherwise returns "is "

     CHARACTER ( len = 3 ) :: STRING_are

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val /= 1 ) THEN
       STRING_are = "are"
     ELSE
       STRING_are = "is "
     END IF

     RETURN

!  End of function STRING_are

      END FUNCTION STRING_are

!-*-*-*-*-  G A L A H A D -  S T R I N G _ h a v e   F U N C T I O N  -*-*-*-*-

     FUNCTION STRING_have( val )

!   Given an integer val, returns "have" if val /= 1, otherwise returns "has "

     CHARACTER ( len = 4 ) :: STRING_have

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val /= 1 ) THEN
       STRING_have = "have"
     ELSE
       STRING_have = "has "
     END IF

     RETURN

!  End of function STRING_have

      END FUNCTION STRING_have

!-*-*-*-*-  G A L A H A D -  S T R I N G _ i e s   F U N C T I O N  -*-*-*-*-

     FUNCTION STRING_ies( val )

!   Given an integer val, returns "ies" if val /= 1, otherwise returns "y  "

     CHARACTER ( len = 3 ) :: STRING_ies

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val /= 1 ) THEN
       STRING_ies = "ies"
     ELSE
       STRING_ies = "y  "
     END IF

     RETURN

!  End of function STRING_ies

      END FUNCTION STRING_ies

!-*-*-*-  G A L A H A D -  S T R I N G _ t h e i r    F U N C T I O N  -*-*-*-

     FUNCTION STRING_their( val )

!   Given an integer val, returns "their" if val /= 1, otherwise returns "its"

     CHARACTER ( len = 5 ) :: STRING_their

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val

     IF ( val /= 1 ) THEN
       STRING_their = "their"
     ELSE
       STRING_their = "its  "
     END IF

     RETURN

!  End of function STRING_their

      END FUNCTION STRING_their

!-*-*-*-  G A L A H A D -  S T R I N G _ c h o i c e    F U N C T I O N  -*-*-*-

     FUNCTION STRING_choice( val, string1, string2 )

!   Given an integer val, returns string1 if val /= 1, otherwise returns string2

     CHARACTER ( len = 120 ) :: STRING_choice
 
!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER, INTENT( IN ) :: val
     CHARACTER ( len = * ), INTENT( IN ) :: string1, string2

     IF ( val /= 1 ) THEN
       STRING_choice = string1
     ELSE
       STRING_choice = string2
     END IF

     RETURN

!  End of function STRING_choice

      END FUNCTION STRING_choice

!-*-  G A L A H A D -  S T R I N G _ s i g n _ s i n g l e  F U N C T I O N  -*-

     FUNCTION STRING_sign_single( val, show_plus )

!   Given a real number val, returns " " (or "+" if show_plus is true)
!   if val >= 0 and "-" if val < 0

     CHARACTER ( len = 1 ) :: STRING_sign_single

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = sp ), INTENT( IN ) :: val
     LOGICAL, INTENT( IN ) :: show_plus

     IF ( val < 0.0_sp ) THEN
       STRING_sign_single = "-"
     ELSE
       IF ( show_plus ) THEN
         STRING_sign_single = "+"
       ELSE
         STRING_sign_single = " "
       END IF
     END IF

     RETURN

!  End of function STRING_sign_single

     END FUNCTION STRING_sign_single

!-*-  G A L A H A D -  S T R I N G _ s i g n _ d o u b l e  F U N C T I O N  -*-

     FUNCTION STRING_sign_double( val, show_plus )

!   Given a real number val, returns " " (or "+" if show_plus is true)
!   if val >= 0 and "-" if val < 0

     CHARACTER ( len = 1 ) :: STRING_sign_double

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = dp ), INTENT( IN ) :: val
     LOGICAL, INTENT( IN ) :: show_plus

     IF ( val < 0.0_dp ) THEN
       STRING_sign_double = "-"
     ELSE
       IF ( show_plus ) THEN
         STRING_sign_double = "+"
       ELSE
         STRING_sign_double = " "
       END IF
     END IF

     RETURN

!  End of function STRING_sign_double

      END FUNCTION STRING_sign_double

!-*-*-*-*-*-*-*-   S T R I N G _ l o w e r   S U B R O U T I N E  -*-*-*-*-*-*-

     SUBROUTINE STRING_lower( string )

!  Convert a character variable from upper to lower case

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     CHARACTER, INTENT( INOUT ) :: string

!  Local variables

     INTEGER :: letter
     CHARACTER, DIMENSION( 26 ) :: LOWER, UPPER

     DATA LOWER / 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',          &
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',          &
                  'u', 'v', 'w', 'x', 'y', 'z' /
     DATA UPPER / 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',          &
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',          &
                  'U', 'V', 'W', 'X', 'Y', 'Z' /

!  See if the current letter is upper case. If so replace it by its
!  lower case counterpart

     DO letter = 1, 26
       IF ( string == UPPER( letter ) ) THEN
         string = LOWER( letter )
         EXIT
       END IF
     END DO

     RETURN

!  End of subroutine STRING_lower

     END SUBROUTINE STRING_lower

!-*-*-*-*-*-*-*-   S T R I N G _ u p p e r   S U B R O U T I N E  -*-*-*-*-*-*-

     SUBROUTINE STRING_upper( string )

!  Convert a character variable from lower to upper case

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     CHARACTER, INTENT( INOUT ) :: string

!  Local variables

     INTEGER :: letter
     CHARACTER, DIMENSION( 26 ) :: LOWER, UPPER

     DATA LOWER / 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',          &
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',          &
                  'u', 'v', 'w', 'x', 'y', 'z' /
     DATA UPPER / 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',          &
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',          &
                  'U', 'V', 'W', 'X', 'Y', 'Z' /

!  See if the current letter is lower case. If so replace it by its
!  upper case counterpart

     DO letter = 1, 26
       IF ( string == LOWER( letter ) ) THEN
         string = UPPER( letter )
         EXIT
       END IF
     END DO

     RETURN

!  End of subroutine STRING_upper

     END SUBROUTINE STRING_upper

!-*-*-*-*-   S T R I N G _ l o w e r  _ w o r d   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE STRING_lower_word( word )

!  Convert a word of character strings from upper to lower case

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     CHARACTER (LEN = * ), INTENT( INOUT ) :: word

!  Local variables

     INTEGER :: i

!  Change the word, letter by letter, to lower case

     DO i = 1,  LEN_TRIM( word )
       CALL STRING_lower( word( i : i ) )
     END DO

     RETURN

!  End of subroutine STRING_lower_word

     END SUBROUTINE STRING_lower_word

!-*-*-*-*-   S T R I N G _ u p p e r  _ w o r d   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE STRING_upper_word( word )

!  Convert a word of character strings from lower to upper case

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     CHARACTER (LEN = * ), INTENT( INOUT ) :: word

!  Local variables

     INTEGER :: i

!  Change the word, letter by letter, to upper case

     DO i = 1,  LEN_TRIM( word )
       CALL STRING_upper( word( i : i ) )
     END DO

     RETURN

!  End of subroutine STRING_upper_word

     END SUBROUTINE STRING_upper_word

!-*-*-*-  G A L A H A D -  S T R I N G _ p u t   S U B R O U T I N E  -*-*-*-

     SUBROUTINE STRING_put( array, string, stat )

!  allocate a character array and set its components from a character variable

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

!  array - character array which will be allocated to the length of the
!          input string and its components then filled with the string
!  string: input string
!  stat: output stat from allocate/deallocate: 0 = success

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array
     CHARACTER ( len = * ), INTENT( IN ) :: string
     INTEGER, INTENT( OUT ) ::  stat

!  Local variables

     INTEGER :: i, l

     l = LEN_TRIM( string )
     IF ( ALLOCATED( array ) ) THEN
       DEALLOCATE( array, stat = stat )
       IF ( stat /= 0 ) RETURN
     END IF
     ALLOCATE( array( l ), STAT = stat )
     IF ( stat == 0 ) THEN
       DO i = 1, l
         array( i ) = string( i : i )
       END DO
     END IF

     RETURN

!  End of STRING_put

     END SUBROUTINE STRING_put

!-*-*-*-*-  G A L A H A D -  S T R I N G _ g e t   F U N C T I O N  -*-*-*-*-

     FUNCTION STRING_get( array )

!  obtain the elements of a character array as a character variable

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

!  array - character array whose components hold the string
!  string_get - equivalent character string

     CHARACTER, INTENT( IN ), DIMENSION( : ) :: array
     CHARACTER( SIZE( array ) ) :: STRING_get

!  Local variable

     INTEGER :: i

     DO i = 1, SIZE( array )
        STRING_get( i : i ) = array( i )
     END DO

     RETURN

!  End of function STRING_get

     END FUNCTION STRING_get

!-*-  G A L A H A D -  S T R I N G _ i n t e g e r _ 6   F U N C T I O N  -*-*-

     FUNCTION STRING_integer_6( i )

     CHARACTER ( LEN = 6 ) :: STRING_integer_6

!  Obtain a 6 character representation of the integer i.
!  Extracted from LANCELOT module OTHERS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: i

!  Local variables

     INTEGER :: ik, im, ig
     CHARACTER :: ci * 6

     STRING_integer_6( 1 : 6 ) = '       '
     ik = i / 1000
     im = ik / 1000
     ig = im / 1000
     IF ( i <= 9 ) THEN
       WRITE( UNIT = ci, FMT = "( I1 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( i <= 99 ) THEN
       WRITE( UNIT = ci, FMT = "( I2 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( i <= 999 ) THEN
       WRITE( UNIT = ci, FMT = "( I3 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( i <= 9999 ) THEN
       WRITE( UNIT = ci, FMT = "( I4 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( i <= 99999 ) THEN
       WRITE( UNIT = ci, FMT = "( I5 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( i <= 999999 ) THEN
       WRITE( UNIT = ci, FMT = "( I6 )" ) i
       STRING_integer_6 = ci
     ELSE IF ( ik <= 99999 ) THEN
       WRITE( UNIT = ci( 1 : 5 ), FMT = "( I5 )" ) ik
       STRING_integer_6 = ci( 1 : 5 ) // 'k'
     ELSE IF ( im <= 99999 ) THEN
       WRITE( UNIT = ci( 1 : 5 ), FMT = "( I5 )" ) im
       STRING_integer_6 = ci( 1 : 5 ) // 'm'
     ELSE
       WRITE( UNIT = ci( 1 : 5 ), FMT = "( I5 )" ) ig
       STRING_integer_6 = ci( 1 : 5 ) // 'g'
     END IF

     RETURN

!  End of STRING_integer_6

     END FUNCTION STRING_integer_6

!-  G A L A H A D -  S T R I N G _ r e a l _ 7 _ s i n g l e  F U N C T I O N -

     FUNCTION STRING_real_7_single( re )

     CHARACTER ( LEN = 7 ) :: STRING_real_7_single

!  Obtain a 7 character representation of the single-precision real r
!  Extracted from LANCELOT module OTHERS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = sp ), INTENT( IN ) :: re

!  Local variables

     INTEGER :: ir
     REAL ( KIND = sp ) :: r, rm, rh, rd
     CHARACTER :: cr * 7

     STRING_real_7_single( 1 : 7 ) = '       '
     r = re
     rm = re / 60.0_sp
     rh = rm / 60.0_sp
     rd = rh / 24.0_sp
     IF ( r <= 9999.9_sp ) THEN
       r = re
       WRITE( UNIT = cr, FMT = "( 0P, F7.1 )" ) r
       STRING_real_7_single = cr
     ELSE IF ( r <= 99999.9_sp ) THEN
       r = re
       WRITE( UNIT = cr, FMT = "( 0P, F7.1 )" ) r
       STRING_real_7_single( 1 : 1 ) = ' '
       STRING_real_7_single( 2 : 7 ) = cr( 1 : 6 )
     ELSE IF ( r <= 999999.0_sp ) THEN
       ir = INT(re)
       WRITE( UNIT = cr, FMT = "( I7 )" ) ir
       STRING_real_7_single = cr
     ELSE IF ( rm <= 99999.9_sp ) THEN
       ir = INT( rm )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_single = cr( 1 : 6 ) // 'm'
     ELSE IF ( rh <= 99999.9_sp ) THEN
       ir = INT( rh )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_single = cr( 1 : 6 ) // 'h'
     ELSE IF ( rd <= 99999.9_sp ) THEN
       ir = INT( rd )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_single = cr( 1 : 6 ) // 'd'
     ELSE
       STRING_real_7_single = ' ******'
     END IF

     RETURN

!  End of STRING_real_7_single

     END FUNCTION STRING_real_7_single

!-  G A L A H A D -  S T R I N G _ r e a l _ 7 _ d o u b l e  F U N C T I O N -

     FUNCTION STRING_real_7_double( re )
     CHARACTER ( LEN = 7 ) :: STRING_real_7_double

!  Obtain a 7 character representation of the double-precision real r
!  Extracted from LANCELOT module OTHERS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = dp ), INTENT( IN ) :: re

!  Local variables

     INTEGER :: ir
     REAL ( KIND = dp ) :: r, rm, rh, rd
     CHARACTER :: cr * 7

     STRING_real_7_double( 1 : 7 ) = '       '
     r = re
     rm = re / 60.0_dp
     rh = rm / 60.0_dp
     rd = rh / 24.0_dp
     IF ( r <= 9999.9_dp ) THEN
       r = re
       WRITE( UNIT = cr, FMT = "( 0P, F7.1 )" ) r
       STRING_real_7_double = cr
     ELSE IF ( r <= 99999.9_dp ) THEN
       r = re
       WRITE( UNIT = cr, FMT = "( 0P, F7.1 )" ) r
       STRING_real_7_double( 1 : 1 ) = ' '
       STRING_real_7_double( 2 : 7 ) = cr( 1 : 6 )
     ELSE IF ( r <= 999999.0_dp ) THEN
       ir = INT(re)
       WRITE( UNIT = cr, FMT = "( I7 )" ) ir
       STRING_real_7_double = cr
     ELSE IF ( rm <= 99999.9_dp ) THEN
       ir = INT( rm )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_double = cr( 1 : 6 ) // 'm'
     ELSE IF ( rh <= 99999.9_dp ) THEN
       ir = INT( rh )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_double = cr( 1 : 6 ) // 'h'
     ELSE IF ( rd <= 99999.9_dp ) THEN
       ir = INT( rd )
       WRITE( UNIT = cr( 1 : 6 ), FMT = "( I6 )" ) ir
       STRING_real_7_double = cr( 1 : 6 ) // 'd'
     ELSE
       STRING_real_7_double = ' ******'
     END IF

     RETURN


!  End of STRING_real_7_double

     END FUNCTION STRING_real_7_double

!-  G A L A H A D -  S T R I N G _ r e a l _ 1 2 _ s i n g l e  F U N C T I O N 

     FUNCTION STRING_real_12_single( r )

!  writes the real r as a 12 character string with as much precision as possible

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  r - a floating-point number
!  string_real - equivalent 12 character string

     REAL( KIND = sp ), INTENT( IN )  :: r
     CHARACTER ( LEN = 12 ) :: STRING_real_12_single

!  Local variables

     INTEGER :: i, i1, i2, j
     REAL( KIND = sp ) :: s
     CHARACTER ( LEN = 11 ) :: field11
     CHARACTER ( LEN = 12 ) :: field
     CHARACTER ( LEN = 13 ) :: field13

!  cram r into 12 characters

     IF ( r == 0.0_sp ) THEN
       field = "0.0         "
     ELSE IF ( SIGN( 1.0_sp, r ) > 0.0_sp ) THEN
!      IF ( r >= ( 10.0_sp ) ** 100 ) THEN
!        WRITE( field13, "( ES13.6E3 )" ) r
!        field = field13( 1 : 9 ) // field13( 11 : 13 )
       IF ( r >= ( 10.0_sp ) ** 11 ) THEN
         WRITE( field13, "( ES13.7 )" ) r
         field = field13( 1 : 10 ) // field13( 12 : 13 )
       ELSE IF ( r >= ( 10.0_sp ) ** 10 ) THEN
         WRITE( field, "( F12.0 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 9 ) THEN
         WRITE( field, "( F12.1 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 8 ) THEN
         WRITE( field, "( F12.2 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 7 ) THEN
         WRITE( field, "( F12.3 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 6 ) THEN
         WRITE( field, "( F12.4 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 5 ) THEN
         WRITE( field, "( F12.5 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 4 ) THEN
         WRITE( field, "( F12.6 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 3 ) THEN
         WRITE( field, "( F12.7 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 2 ) THEN
         WRITE( field, "( F12.8 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 1 ) THEN
         WRITE( field, "( F12.9 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** 0 ) THEN
         WRITE( field, "( F12.10 )" ) r
       ELSE IF ( r >= ( 10.0_sp ) ** ( - 4 ) ) THEN
         WRITE( field13, "( F13.11 )" ) r
         field = field13( 2 : 13 )
       ELSE IF ( r >= ( 10.0_sp ) ** ( - 9 ) ) THEN
         WRITE( field13, "( ES13.7 )" ) r
         field = field13( 1 : 11 ) // field13( 13 : 13 )
!      ELSE IF ( r >= ( 10.0_sp ) ** ( - 99 ) ) THEN
!        WRITE( field, "( ES12.6 )" ) r
!      ELSE IF ( r >= ( 10.0_sp ) ** ( - 999 ) ) THEN
!        WRITE( field, "( ES12.5E3 )" ) r
       ELSE
         WRITE( field, "( ES12.5E4 )" ) r
       END IF
     ELSE
       s = - r
       IF ( ABS( s - 1.0_sp ) <= teneps_s ) s = 1.0_sp
!      IF ( s >= ( 10.0_sp ) ** 100 ) THEN
!        WRITE( field, "( ES12.5E3 )" ) s
!        field11 = field( 1 : 8 ) // field( 10 : 12 )
       IF ( s >= ( 10.0_sp ) ** 10 ) THEN
         WRITE( field, "( ES12.6 )" ) s
         field11 = field( 1 : 9 ) // field( 11 : 12 )
       ELSE IF ( s >= ( 10.0_sp ) ** 9 ) THEN
         WRITE( field11, "( F11.0 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 8 ) THEN
         WRITE( field11, "( F11.1 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 7 ) THEN
         WRITE( field11, "( F11.2 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 6 ) THEN
         WRITE( field11, "( F11.3 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 5 ) THEN
         WRITE( field11, "( F11.4 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 4 ) THEN
         WRITE( field11, "( F11.5 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 3 ) THEN
         WRITE( field11, "( F11.6 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 2 ) THEN
         WRITE( field11, "( F11.7 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 1 ) THEN
         WRITE( field11, "( F11.8 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** 0 ) THEN
         WRITE( field11, "( F11.9 )" ) s
       ELSE IF ( s >= ( 10.0_sp ) ** ( - 4 ) ) THEN
         WRITE( field, "( F12.10 )" ) s
         field11 = field( 2 : 12 )
       ELSE IF ( s >= ( 10.0_sp ) ** ( - 9 ) ) THEN
         WRITE( field, "( ES12.6 )" ) s
         field11 = field( 1 : 10 ) // field( 12 : 12 )
!      ELSE IF ( s > ( 10.0_sp ) ** ( - 99 ) ) THEN
!        WRITE( field11, "( ES11.5 )" ) s
!      ELSE IF ( s > ( 10.0_sp ) ** ( - 999 ) ) THEN
!        WRITE( field11, "( ES11.4E3 )" ) s
       ELSE
         WRITE( field11, "( ES11.5E4 )" ) s
       END IF
       field = "-" // field11
     END IF

!  remove any superflous 0s

     i2 = 13
     DO i = 1, 12
       IF ( field( i : i ) == '.' ) i1 = i
       IF ( field( i : i ) == 'E' ) i2 = i
     END DO

     IF ( i2 - 1 >= i1 + 1 ) THEN
       IF ( i2 == 13 ) THEN
         DO i = 12, i1 + 1, - 1
           IF ( i <= 11 ) field( i + 1 : i + 1 ) = ' '
           IF ( field( i : i ) /= '0' ) EXIT
         END DO
       ELSE
         DO i = i2 - 1, i1 + 1, - 1
           IF ( i < i2 - 1 ) field( i + 1 : i + 1 ) = ' '
           IF ( field( i : i ) /= '0' .OR. i == i1 + 1 ) THEN
             DO j = 1, 12 - i2 + 1
               field( i + j : i + j ) = field( i2 - 1 + j : i2 - 1 + j )
             END DO
             DO j = i + 12 - i2 + 1 + 1, 12
               field( j : j ) = ' '
             END DO
             EXIT
           END IF
         END DO
       END IF
     END IF

     STRING_real_12_single = field

     RETURN

!  End of function STRING_real_12_single

     END FUNCTION STRING_real_12_single

!-  G A L A H A D -  S T R I N G _ r e a l _ 1 2 _ d o u b l e  F U N C T I O N 

     FUNCTION STRING_real_12_double( r )

!  writes the real r as a 12 character string with as much precision as possible

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  r - a floating-point number
!  string_real - equivalent 12 character string

     REAL( KIND = dp ), INTENT( IN )  :: r
     CHARACTER ( LEN = 12 ) :: STRING_real_12_double

!  Local variables

     INTEGER :: i, i1, i2, j
     REAL( KIND = dp ) :: s
     CHARACTER ( LEN = 11 ) :: field11
     CHARACTER ( LEN = 12 ) :: field
     CHARACTER ( LEN = 13 ) :: field13

!  cram r into 12 characters

     IF ( r == 0.0_dp ) THEN
       field = "0.0         "
     ELSE IF ( SIGN( 1.0_dp, r ) > 0.0_dp ) THEN
       IF ( r >= ( 10.0_dp ) ** 100 ) THEN
         WRITE( field13, "( ES13.6E3 )" ) r
         field = field13( 1 : 9 ) // field13( 11 : 13 )
       ELSE IF ( r >= ( 10.0_dp ) ** 11 ) THEN
         WRITE( field13, "( ES13.7 )" ) r
         field = field13( 1 : 10 ) // field13( 12 : 13 )
       ELSE IF ( r >= ( 10.0_dp ) ** 10 ) THEN
         WRITE( field, "( F12.0 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 9 ) THEN
         WRITE( field, "( F12.1 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 8 ) THEN
         WRITE( field, "( F12.2 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 7 ) THEN
         WRITE( field, "( F12.3 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 6 ) THEN
         WRITE( field, "( F12.4 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 5 ) THEN
         WRITE( field, "( F12.5 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 4 ) THEN
         WRITE( field, "( F12.6 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 3 ) THEN
         WRITE( field, "( F12.7 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 2 ) THEN
         WRITE( field, "( F12.8 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 1 ) THEN
         WRITE( field, "( F12.9 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** 0 ) THEN
         WRITE( field, "( F12.10 )" ) r
       ELSE IF ( r >= ( 10.0_dp ) ** ( - 4 ) ) THEN
         WRITE( field13, "( F13.11 )" ) r
         field = field13( 2 : 13 )
       ELSE IF ( r >= ( 10.0_dp ) ** ( - 9 ) ) THEN
         WRITE( field13, "( ES13.7 )" ) r
         field = field13( 1 : 11 ) // field13( 13 : 13 )
       ELSE IF ( r >= ( 10.0_dp ) ** ( - 99 ) ) THEN
         WRITE( field, "( ES12.6 )" ) r
!      ELSE IF ( r >= ( 10.0_dp ) ** ( - 999 ) ) THEN
!        WRITE( field, "( ES12.5E3 )" ) r
       ELSE
         WRITE( field, "( ES12.5E4 )" ) r
       END IF
     ELSE
       s = - r
       IF ( ABS( s - 1.0_dp ) <= teneps_d ) s = 1.0_dp
       IF ( s >= ( 10.0_dp ) ** 100 ) THEN
         WRITE( field, "( ES12.5E3 )" ) s
         field11 = field( 1 : 8 ) // field( 10 : 12 )
       ELSE IF ( s >= ( 10.0_dp ) ** 10 ) THEN
         WRITE( field, "( ES12.6 )" ) s
         field11 = field( 1 : 9 ) // field( 11 : 12 )
       ELSE IF ( s >= ( 10.0_dp ) ** 9 ) THEN
         WRITE( field11, "( F11.0 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 8 ) THEN
         WRITE( field11, "( F11.1 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 7 ) THEN
         WRITE( field11, "( F11.2 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 6 ) THEN
         WRITE( field11, "( F11.3 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 5 ) THEN
         WRITE( field11, "( F11.4 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 4 ) THEN
         WRITE( field11, "( F11.5 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 3 ) THEN
         WRITE( field11, "( F11.6 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 2 ) THEN
         WRITE( field11, "( F11.7 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 1 ) THEN
         WRITE( field11, "( F11.8 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** 0 ) THEN
         WRITE( field11, "( F11.9 )" ) s
       ELSE IF ( s >= ( 10.0_dp ) ** ( - 4 ) ) THEN
         WRITE( field, "( F12.10 )" ) s
         field11 = field( 2 : 12 )
       ELSE IF ( s >= ( 10.0_dp ) ** ( - 9 ) ) THEN
         WRITE( field, "( ES12.6 )" ) s
         field11 = field( 1 : 10 ) // field( 12 : 12 )
       ELSE IF ( s > ( 10.0_dp ) ** ( - 99 ) ) THEN
         WRITE( field11, "( ES11.5 )" ) s
!      ELSE IF ( s > ( 10.0_dp ) ** ( - 999 ) ) THEN
!        WRITE( field11, "( ES11.4E3 )" ) s
       ELSE
         WRITE( field11, "( ES11.5E4 )" ) s
       END IF
       field = "-" // field11
     END IF

!  remove any superflous 0s

     i2 = 13
     DO i = 1, 12
       IF ( field( i : i ) == '.' ) i1 = i
       IF ( field( i : i ) == 'E' ) i2 = i
     END DO

     IF ( i2 - 1 >= i1 + 1 ) THEN
       IF ( i2 == 13 ) THEN
         DO i = 12, i1 + 1, - 1
           IF ( i <= 11 ) field( i + 1 : i + 1 ) = ' '
           IF ( field( i : i ) /= '0' ) EXIT
         END DO
       ELSE
         DO i = i2 - 1, i1 + 1, - 1
           IF ( i < i2 - 1 ) field( i + 1 : i + 1 ) = ' '
           IF ( field( i : i ) /= '0' .OR. i == i1 + 1 ) THEN
             DO j = 1, 12 - i2 + 1
               field( i + j : i + j ) = field( i2 - 1 + j : i2 - 1 + j )
             END DO
             DO j = i + 12 - i2 + 1 + 1, 12
               field( j : j ) = ' '
             END DO
             EXIT
           END IF
         END DO
       END IF
     END IF

     STRING_real_12_double = field

     RETURN

!  End of function STRING_real_12_double

     END FUNCTION STRING_real_12_double

!  -- S T R I N G _ T R I M _ R E A L _ 2 4 _ S I N G L E    F U N C T I O N --

     FUNCTION STRING_trim_real_24_single( value )
     CHARACTER ( LEN = 24 ) :: STRING_trim_real_24_single
     REAL ( KIND = sp ) :: value

!  write a real value into 24 characters trimming as much as possible 
!  without losing precision

     INTEGER :: i, i_start, i_point, i_end, j, k, l, zs
     REAL ( KIND = sp ) :: minus_value
     LOGICAL :: zeros
     CHARACTER ( LEN = 22 ) :: field22
     CHARACTER ( LEN = 23 ) :: field
     CHARACTER ( LEN = 24 ) :: field24

!  cram value into 23 characters

!write(6,*) value
     IF ( value == 0.0_sp ) THEN
       field = "0.0         "
     ELSE IF ( SIGN( 1.0_sp, value ) > 0.0_sp ) THEN
!      IF ( value >= ( 10.0_sp ) ** 100 ) THEN
!        WRITE( field24, "( ES24.15E3 )" ) value
!        field = field24( 1 : 20 ) // field24( 22 : 24 )
       IF ( value >= ( 10.0_sp ) ** 16 ) THEN
         WRITE( field24, "( ES24.15 )" ) value
         field = field24( 1 : 21 ) // field24( 23 : 24 )
       ELSE IF ( value >= ( 10.0_sp ) ** 15 ) THEN
         WRITE( field, "( F23.0 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 14 ) THEN
         WRITE( field, "( F23.1 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 13 ) THEN
         WRITE( field, "( F23.2 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 12 ) THEN
         WRITE( field, "( F23.3 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 11 ) THEN
         WRITE( field, "( F23.4 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 10 ) THEN
         WRITE( field, "( F23.5 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 9 ) THEN
         WRITE( field, "( F23.6 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 8 ) THEN
         WRITE( field, "( F23.7 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 7 ) THEN
         WRITE( field, "( F23.8 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 6 ) THEN
         WRITE( field, "( F23.9 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 5 ) THEN
         WRITE( field, "( F23.10 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 4 ) THEN
         WRITE( field, "( F23.11 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 3 ) THEN
         WRITE( field, "( F23.12 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 2 ) THEN
         WRITE( field, "( F23.13 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 1 ) THEN
         WRITE( field, "( F23.14 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** 0 ) THEN
         WRITE( field, "( F23.15 )" ) value
       ELSE IF ( value >= ( 10.0_sp ) ** ( - 1 ) ) THEN
         WRITE( field24, "( F24.16 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_sp ) ** ( - 2 ) ) THEN
         WRITE( field24, "( F24.17 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_sp ) ** ( - 3 ) ) THEN
         WRITE( field24, "( F24.18 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_sp ) ** ( - 4 ) ) THEN
         WRITE( field24, "( F24.16 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_sp ) ** ( - 9 ) ) THEN
         WRITE( field24, "( ES24.15 )" ) value
         field = field24( 1 : 22 ) // field24( 24 : 24 )
!      ELSE IF ( value >= ( 10.0_sp ) ** ( - 99 ) ) THEN
!        WRITE( field, "( ES23.15 )" ) value
!      ELSE IF ( value >= ( 10.0_sp ) ** ( - 999 ) ) THEN
!        WRITE( field, "( ES23.15E3 )" ) value
       ELSE
         WRITE( field, "( ES23.15E4 )" ) value
       END IF
     ELSE
       minus_value = - value
       IF ( ABS( minus_value - 1.0_sp ) <= teneps_d ) minus_value = 1.0_sp
!      IF ( minus_value >= ( 10.0_sp ) ** 100 ) THEN
!        WRITE( field, "( ES23.15E3 )" ) minus_value
!        field22 = field( 1 : 19 ) // field( 21 : 23 )
       IF ( minus_value >= ( 10.0_sp ) ** 16 ) THEN
         WRITE( field, "( ES23.15 )" ) minus_value
         field22 = field( 1 : 20 ) // field( 22 : 23 )
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 15 ) THEN
         WRITE( field22, "( F22.0 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 14 ) THEN
         WRITE( field22, "( F22.1 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 13 ) THEN
         WRITE( field22, "( F22.2 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 12 ) THEN
         WRITE( field22, "( F22.3 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 11 ) THEN
         WRITE( field22, "( F22.4 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 10 ) THEN
         WRITE( field22, "( F22.5 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 9 ) THEN
         WRITE( field22, "( F22.6 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 8 ) THEN
         WRITE( field22, "( F22.7 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 7 ) THEN
         WRITE( field22, "( F22.8 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 6 ) THEN
         WRITE( field22, "( F22.9 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 5 ) THEN
         WRITE( field22, "( F22.10 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 4 ) THEN
         WRITE( field22, "( F22.11 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 3 ) THEN
         WRITE( field22, "( F22.12 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 2 ) THEN
         WRITE( field22, "( F22.13 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 1 ) THEN
         WRITE( field22, "( F22.14 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** 0 ) THEN
         WRITE( field22, "( F22.15 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_sp ) ** ( - 1 ) ) THEN
         WRITE( field, "( F23.16 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_sp ) ** ( - 2 ) ) THEN
         WRITE( field, "( F23.17 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_sp ) ** ( - 3 ) ) THEN
         WRITE( field, "( F23.18 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_sp ) ** ( - 4 ) ) THEN
         WRITE( field, "( F23.15 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_sp ) ** ( - 9 ) ) THEN
         WRITE( field, "( ES23.15 )" ) minus_value
         field22 = field( 1 : 21 ) // field( 23 : 23 )
!      ELSE IF ( minus_value > ( 10.0_sp ) ** ( - 99 ) ) THEN
!        WRITE( field22, "( ES22.15 )" ) minus_value
!      ELSE IF ( minus_value > ( 10.0_sp ) ** ( - 999 ) ) THEN
!        WRITE( field22, "( ES22.15E3 )" ) minus_value
       ELSE
         WRITE( field22, "( ES22.15E4 )" ) minus_value
       END IF
       field = "-" //  ADJUSTL( field22 )
     END IF

!  shift the value left

     field24 = ADJUSTL( field ) // ' '

!  find the positions of the first digit in the mantissa

     IF ( field24( 1 : 1 ) == '-' ) THEN
       i_start = 2
     ELSE
       i_start = 1
     END IF

!  find the positions of the decimal point and last digit in the mantissa

     DO i = 1, 23
       IF ( field24( i : i ) == '.' ) i_point = i
       IF ( field24( i + 1 : i + 1 ) == ' ' .OR.                               &
            field24( i + 1 : i + 1 ) == 'e' .OR.                               &
            field24( i + 1 : i + 1 ) == 'E' .OR.                               &
            field24( i + 1 : i + 1 ) == 'd' .OR.                               &
            field24( i + 1 : i + 1 ) == 'D' ) THEN
         i_end = i
         EXIT
       END IF
     END DO

!    IF ( i_end - i_point >= 15 ) THEN
     IF ( i_end - i_start >= 15 ) THEN

!  round down any *01 to *00

       IF ( field24( i_end - 1 : i_end ) == '01' )                             &
         field24( i_end - 1 : i_end ) = '00'

!  round any *9r to **0r where ** = *+1

       IF ( field24( i_end - 1 : i_end ) == '99' ) THEN
         DO i = i_end, i_point + 1, - 1
           IF ( field24( i : i ) == '9' ) THEN
             field24( i : i ) = '0'
           ELSE
             READ( field24( i : i ), "( I1 )" ) l
             WRITE( field24( i : i ), "( I1 )" ) l + 1
             EXIT
           END IF
           IF ( i == i_point + 1 ) THEN
             DO j = i_point - 1, i_start, - 1
               IF ( field24( j : j ) == '9' ) THEN
                 field24( j : j ) = '0'
               ELSE
                 READ( field24( j : j ), "( I1 )" ) l
                 WRITE( field24( j : j ), "( I1 )" ) l + 1
                 EXIT
               END IF
               IF ( j == i_start ) THEN
                 DO l = i_end - 1, i_start, - 1
                   field24( l + 1 : l + 1 ) = field24( l : l ) 
                 END DO
                 field24( i_start : i_start ) = '1'
               END IF
             END DO
           END IF
         END DO
       END IF
     END IF

!    field24 = REPEAT( ' ', 24 )
!    IF ( value > - 10.0_wp .AND. value < 10.0_wp ) THEN
!      WRITE( field24, "( F19.16 )" ) value
!    ELSE
!      WRITE( field24, "( ES23.16 )" ) value
!    END IF

     STRING_trim_real_24_single = field24

!  remove any leading space

!    IF ( STRING_trim_real_24_single( 1 : 1 ) == ' ' ) THEN
!      DO i = 2, 24
!        STRING_trim_real_24_single( i - 1 : i - 1 )                           &
!          = STRING_trim_real_24_single( i : i )
!      END DO
!    END IF 

     zeros = .FALSE.
     DO i = 1, 24
       IF ( STRING_trim_real_24_single( i : i ) == '0' ) THEN
         IF ( .NOT. zeros ) THEN
           zs = i
           zeros = .TRUE.
         END IF
       ELSE IF ( STRING_trim_real_24_single( i : i ) == 'E' .OR.               &
                 STRING_trim_real_24_single( i : i ) == 'e' .OR.               &
                 STRING_trim_real_24_single( i : i ) == 'D' .OR.               &
                 STRING_trim_real_24_single( i : i ) == 'd' ) THEN
         IF ( zeros ) THEN
           DO j = zs + 1, zs + 25 - i
             k = i + ( j - zs - 1 )
             STRING_trim_real_24_single( j : j )                               &
               = STRING_trim_real_24_single( k : k  )
           END DO
           DO j = zs + 26 - i, 24
             STRING_trim_real_24_single( j : j ) = ' '
           END DO
         END IF
         zeros = .FALSE.
         EXIT
       ELSE IF ( STRING_trim_real_24_single( i : i ) == ' ' ) THEN
         IF ( zeros ) THEN
           DO j = zs + 1, i
             STRING_trim_real_24_single( j : j ) = ' '
           END DO
         END IF
         zeros = .FALSE.
         EXIT
       ELSE
         zeros = .FALSE.
       END IF
     END DO
     IF ( zeros ) THEN
       DO j = zs + 1, i
         STRING_trim_real_24_single( j : j ) = ' '
       END DO
     END IF

!  remove superflous 0 from the exponent

     DO i = 1, 24
       IF ( STRING_trim_real_24_single( i : i ) == 'E' .OR.                    &
            STRING_trim_real_24_single( i : i ) == 'e' .OR.                    &
            STRING_trim_real_24_single( i : i ) == 'D' .OR.                    &
            STRING_trim_real_24_single( i : i ) == 'd' ) THEN
         IF ( STRING_trim_real_24_single( i + 1 : i + 1 ) == '+' .OR.          &
              STRING_trim_real_24_single( i + 1 : i + 1 ) == '-' ) THEN
           IF ( STRING_trim_real_24_single( i + 2 : i + 2 ) == '0' ) THEN
             IF ( STRING_trim_real_24_single( i + 3 : i + 3 ) == '0' ) THEN
               IF ( STRING_trim_real_24_single( i + 4 : i + 4 ) == ' ' ) THEN
                 STRING_trim_real_24_single( i + 3 : i + 3 ) = '0'
               ELSE
                 STRING_trim_real_24_single( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_single( i + 4 : i + 4 )
                 STRING_trim_real_24_single( i + 3 : i + 4 ) = '  '
               END IF
             ELSE
               IF ( STRING_trim_real_24_single( i + 4 : i + 4 ) == ' ' ) THEN
               ELSE
                 STRING_trim_real_24_single( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_single( i + 3 : i + 3 )
                 STRING_trim_real_24_single( i + 3 : i + 3 )                   &
                   = STRING_trim_real_24_single( i + 4 : i + 4 )
                 STRING_trim_real_24_single( i + 4 : i + 4 ) = ' '
               END IF
             END IF
           END IF
         ELSE
           IF ( STRING_trim_real_24_single( i + 1 : i + 1 ) == '0' ) THEN
             IF ( STRING_trim_real_24_single( i + 2 : i + 2 ) == '0' ) THEN
               IF ( STRING_trim_real_24_single( i + 3 : i + 3 ) == ' ' ) THEN
                 STRING_trim_real_24_single( i + 2 : i + 2 ) = '0'
               ELSE
                 STRING_trim_real_24_single( i + 1 : i + 1 )                   &
                   = STRING_trim_real_24_single( i + 3 : i + 3 )
                 STRING_trim_real_24_single( i + 2 : i + 3 ) = '  '
               END IF
             ELSE
               IF ( STRING_trim_real_24_single( i + 3 : i + 3 ) == ' ' ) THEN
               ELSE
                 STRING_trim_real_24_single( i + 1 : i + 1 )                   &
                   = STRING_trim_real_24_single( i + 2 : i + 2 )
                 STRING_trim_real_24_single( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_single( i + 3 : i + 3 )
                 STRING_trim_real_24_single( i + 3 : i + 3 ) = ' '
               END IF
             END IF
           END IF
         END IF
         EXIT
       END IF

!  remove trailing 0 unless it is preceeded by a .

       IF ( STRING_trim_real_24_single( i : i ) == ' ' ) THEN
         IF ( i < 3 ) EXIT
         IF ( STRING_trim_real_24_single( i - 1 : i - 1 ) == '0' .AND.         &
              STRING_trim_real_24_single( i - 2 : i - 2 ) /= '.' ) THEN
              STRING_trim_real_24_single( i - 1 : i - 1 ) = ' '
         END IF
         EXIT
       END IF

     END DO

!  if the string starts with a ., add a 0 at the front

     IF ( STRING_trim_real_24_single( 1 : 1 ) == '.' ) THEN
       DO i = 24, 2, -1
         STRING_trim_real_24_single( i : i )                                   &
           = STRING_trim_real_24_single( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_single( 1 : 1 ) = '0'
     END IF

!  if the string starts with a ., add a 0 at the front

     IF ( STRING_trim_real_24_single( 1 : 1 ) == '.' ) THEN
       DO i = 24, 2, -1
         STRING_trim_real_24_single( i : i )                                   &
           = STRING_trim_real_24_single( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_single( 1 : 1 ) = '0'
     END IF

!  if the string starts with a -., replace by -0. at the front

     IF ( STRING_trim_real_24_single( 1 : 2 ) == '-.' ) THEN
       DO i = 24, 3, -1
         STRING_trim_real_24_single( i : i )                                   &
           = STRING_trim_real_24_single( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_single( 2 : 2 ) = '0'
     END IF
     RETURN

!  end of function STRING_trim_real_24_single

     END FUNCTION STRING_trim_real_24_single

!  -- S T R I N G _ T R I M _ R E A L _ 2 4 _ D O U B L E    F U N C T I O N --

     FUNCTION STRING_trim_real_24_double( value )
     CHARACTER ( LEN = 24 ) :: STRING_trim_real_24_double
     REAL ( KIND = dp ) :: value

!  write a real value into 24 characters trimming as much as possible 
!  without losing precision

     INTEGER :: i, i_start, i_point, i_end, j, k, l, zs
     REAL ( KIND = dp ) :: minus_value
     LOGICAL :: zeros
     CHARACTER ( LEN = 22 ) :: field22
     CHARACTER ( LEN = 23 ) :: field
     CHARACTER ( LEN = 24 ) :: field24

!  cram value into 23 characters

!write(6,*) value
     IF ( value == 0.0_dp ) THEN
       field = "0.0         "
     ELSE IF ( SIGN( 1.0_dp, value ) > 0.0_dp ) THEN
       IF ( value >= ( 10.0_dp ) ** 100 ) THEN
         WRITE( field24, "( ES24.15E3 )" ) value
         field = field24( 1 : 20 ) // field24( 22 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** 16 ) THEN
         WRITE( field24, "( ES24.15 )" ) value
         field = field24( 1 : 21 ) // field24( 23 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** 15 ) THEN
         WRITE( field, "( F23.0 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 14 ) THEN
         WRITE( field, "( F23.1 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 13 ) THEN
         WRITE( field, "( F23.2 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 12 ) THEN
         WRITE( field, "( F23.3 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 11 ) THEN
         WRITE( field, "( F23.4 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 10 ) THEN
         WRITE( field, "( F23.5 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 9 ) THEN
         WRITE( field, "( F23.6 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 8 ) THEN
         WRITE( field, "( F23.7 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 7 ) THEN
         WRITE( field, "( F23.8 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 6 ) THEN
         WRITE( field, "( F23.9 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 5 ) THEN
         WRITE( field, "( F23.10 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 4 ) THEN
         WRITE( field, "( F23.11 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 3 ) THEN
         WRITE( field, "( F23.12 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 2 ) THEN
         WRITE( field, "( F23.13 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 1 ) THEN
         WRITE( field, "( F23.14 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** 0 ) THEN
         WRITE( field, "( F23.15 )" ) value
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 1 ) ) THEN
         WRITE( field24, "( F24.16 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 2 ) ) THEN
         WRITE( field24, "( F24.17 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 3 ) ) THEN
         WRITE( field24, "( F24.18 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 4 ) ) THEN
         WRITE( field24, "( F24.16 )" ) value
         field = field24( 2 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 9 ) ) THEN
         WRITE( field24, "( ES24.15 )" ) value
         field = field24( 1 : 22 ) // field24( 24 : 24 )
       ELSE IF ( value >= ( 10.0_dp ) ** ( - 99 ) ) THEN
         WRITE( field, "( ES23.15 )" ) value
!      ELSE IF ( value >= ( 10.0_dp ) ** ( - 999 ) ) THEN
!        WRITE( field, "( ES23.15E3 )" ) value
       ELSE
         WRITE( field, "( ES23.15E4 )" ) value
       END IF
     ELSE
       minus_value = - value
       IF ( ABS( minus_value - 1.0_dp ) <= teneps_d ) minus_value = 1.0_dp
       IF ( minus_value >= ( 10.0_dp ) ** 100 ) THEN
         WRITE( field, "( ES23.15E3 )" ) minus_value
         field22 = field( 1 : 19 ) // field( 21 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 16 ) THEN
         WRITE( field, "( ES23.15 )" ) minus_value
         field22 = field( 1 : 20 ) // field( 22 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 15 ) THEN
         WRITE( field22, "( F22.0 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 14 ) THEN
         WRITE( field22, "( F22.1 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 13 ) THEN
         WRITE( field22, "( F22.2 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 12 ) THEN
         WRITE( field22, "( F22.3 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 11 ) THEN
         WRITE( field22, "( F22.4 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 10 ) THEN
         WRITE( field22, "( F22.5 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 9 ) THEN
         WRITE( field22, "( F22.6 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 8 ) THEN
         WRITE( field22, "( F22.7 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 7 ) THEN
         WRITE( field22, "( F22.8 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 6 ) THEN
         WRITE( field22, "( F22.9 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 5 ) THEN
         WRITE( field22, "( F22.10 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 4 ) THEN
         WRITE( field22, "( F22.11 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 3 ) THEN
         WRITE( field22, "( F22.12 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 2 ) THEN
         WRITE( field22, "( F22.13 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 1 ) THEN
         WRITE( field22, "( F22.14 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** 0 ) THEN
         WRITE( field22, "( F22.15 )" ) minus_value
       ELSE IF ( minus_value >= ( 10.0_dp ) ** ( - 1 ) ) THEN
         WRITE( field, "( F23.16 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** ( - 2 ) ) THEN
         WRITE( field, "( F23.17 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** ( - 3 ) ) THEN
         WRITE( field, "( F23.18 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** ( - 4 ) ) THEN
         WRITE( field, "( F23.15 )" ) minus_value
         field22 = field( 2 : 23 )
       ELSE IF ( minus_value >= ( 10.0_dp ) ** ( - 9 ) ) THEN
         WRITE( field, "( ES23.15 )" ) minus_value
         field22 = field( 1 : 21 ) // field( 23 : 23 )
       ELSE IF ( minus_value > ( 10.0_dp ) ** ( - 99 ) ) THEN
         WRITE( field22, "( ES22.15 )" ) minus_value
!      ELSE IF ( minus_value > ( 10.0_dp ) ** ( - 999 ) ) THEN
!        WRITE( field22, "( ES22.15E3 )" ) minus_value
       ELSE
         WRITE( field22, "( ES22.15E4 )" ) minus_value
       END IF
       field = "-" //  ADJUSTL( field22 )
     END IF

!  shift the value left

     field24 = ADJUSTL( field ) // ' '

!  find the positions of the first digit in the mantissa

     IF ( field24( 1 : 1 ) == '-' ) THEN
       i_start = 2
     ELSE
       i_start = 1
     END IF

!  find the positions of the decimal point and last digit in the mantissa

     DO i = 1, 23
       IF ( field24( i : i ) == '.' ) i_point = i
       IF ( field24( i + 1 : i + 1 ) == ' ' .OR.                               &
            field24( i + 1 : i + 1 ) == 'e' .OR.                               &
            field24( i + 1 : i + 1 ) == 'E' .OR.                               &
            field24( i + 1 : i + 1 ) == 'd' .OR.                               &
            field24( i + 1 : i + 1 ) == 'D' ) THEN
         i_end = i
         EXIT
       END IF
     END DO

!    IF ( i_end - i_point >= 15 ) THEN
     IF ( i_end - i_start >= 15 ) THEN

!  round down any *01 to *00

       IF ( field24( i_end - 1 : i_end ) == '01' )                             &
         field24( i_end - 1 : i_end ) = '00'

!  round any *9r to **0r where ** = *+1

       IF ( field24( i_end - 1 : i_end ) == '99' ) THEN
         DO i = i_end, i_point + 1, - 1
           IF ( field24( i : i ) == '9' ) THEN
             field24( i : i ) = '0'
           ELSE
             READ( field24( i : i ), "( I1 )" ) l
             WRITE( field24( i : i ), "( I1 )" ) l + 1
             EXIT
           END IF
           IF ( i == i_point + 1 ) THEN
             DO j = i_point - 1, i_start, - 1
               IF ( field24( j : j ) == '9' ) THEN
                 field24( j : j ) = '0'
               ELSE
                 READ( field24( j : j ), "( I1 )" ) l
                 WRITE( field24( j : j ), "( I1 )" ) l + 1
                 EXIT
               END IF
               IF ( j == i_start ) THEN
                 DO l = i_end - 1, i_start, - 1
                   field24( l + 1 : l + 1 ) = field24( l : l ) 
                 END DO
                 field24( i_start : i_start ) = '1'
               END IF
             END DO
           END IF
         END DO
       END IF
     END IF

!    field24 = REPEAT( ' ', 24 )
!    IF ( value > - 10.0_wp .AND. value < 10.0_wp ) THEN
!      WRITE( field24, "( F19.16 )" ) value
!    ELSE
!      WRITE( field24, "( ES23.16 )" ) value
!    END IF

     STRING_trim_real_24_double = field24

!  remove any leading space

!    IF ( STRING_trim_real_24_double( 1 : 1 ) == ' ' ) THEN
!      DO i = 2, 24
!        STRING_trim_real_24_double( i - 1 : i - 1 )                           &
!          = STRING_trim_real_24_double( i : i )
!      END DO
!    END IF 

     zeros = .FALSE.
     DO i = 1, 24
       IF ( STRING_trim_real_24_double( i : i ) == '0' ) THEN
         IF ( .NOT. zeros ) THEN
           zs = i
           zeros = .TRUE.
         END IF
       ELSE IF ( STRING_trim_real_24_double( i : i ) == 'E' .OR.               &
                 STRING_trim_real_24_double( i : i ) == 'e' .OR.               &
                 STRING_trim_real_24_double( i : i ) == 'D' .OR.               &
                 STRING_trim_real_24_double( i : i ) == 'd' ) THEN
         IF ( zeros ) THEN
           DO j = zs + 1, zs + 25 - i
             k = i + ( j - zs - 1 )
             STRING_trim_real_24_double( j : j )                               &
               = STRING_trim_real_24_double( k : k  )
           END DO
           DO j = zs + 26 - i, 24
             STRING_trim_real_24_double( j : j ) = ' '
           END DO
         END IF
         zeros = .FALSE.
         EXIT
       ELSE IF ( STRING_trim_real_24_double( i : i ) == ' ' ) THEN
         IF ( zeros ) THEN
           DO j = zs + 1, i
             STRING_trim_real_24_double( j : j ) = ' '
           END DO
         END IF
         zeros = .FALSE.
         EXIT
       ELSE
         zeros = .FALSE.
       END IF
     END DO
     IF ( zeros ) THEN
       DO j = zs + 1, i
         STRING_trim_real_24_double( j : j ) = ' '
       END DO
     END IF

!  remove superflous 0 from the exponent

     DO i = 1, 24
       IF ( STRING_trim_real_24_double( i : i ) == 'E' .OR.                    &
            STRING_trim_real_24_double( i : i ) == 'e' .OR.                    &
            STRING_trim_real_24_double( i : i ) == 'D' .OR.                    &
            STRING_trim_real_24_double( i : i ) == 'd' ) THEN
         IF ( STRING_trim_real_24_double( i + 1 : i + 1 ) == '+' .OR.          &
              STRING_trim_real_24_double( i + 1 : i + 1 ) == '-' ) THEN
           IF ( STRING_trim_real_24_double( i + 2 : i + 2 ) == '0' ) THEN
             IF ( STRING_trim_real_24_double( i + 3 : i + 3 ) == '0' ) THEN
               IF ( STRING_trim_real_24_double( i + 4 : i + 4 ) == ' ' ) THEN
                 STRING_trim_real_24_double( i + 3 : i + 3 ) = '0'
               ELSE
                 STRING_trim_real_24_double( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_double( i + 4 : i + 4 )
                 STRING_trim_real_24_double( i + 3 : i + 4 ) = '  '
               END IF
             ELSE
               IF ( STRING_trim_real_24_double( i + 4 : i + 4 ) == ' ' ) THEN
               ELSE
                 STRING_trim_real_24_double( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_double( i + 3 : i + 3 )
                 STRING_trim_real_24_double( i + 3 : i + 3 )                   &
                   = STRING_trim_real_24_double( i + 4 : i + 4 )
                 STRING_trim_real_24_double( i + 4 : i + 4 ) = ' '
               END IF
             END IF
           END IF
         ELSE
           IF ( STRING_trim_real_24_double( i + 1 : i + 1 ) == '0' ) THEN
             IF ( STRING_trim_real_24_double( i + 2 : i + 2 ) == '0' ) THEN
               IF ( STRING_trim_real_24_double( i + 3 : i + 3 ) == ' ' ) THEN
                 STRING_trim_real_24_double( i + 2 : i + 2 ) = '0'
               ELSE
                 STRING_trim_real_24_double( i + 1 : i + 1 )                   &
                   = STRING_trim_real_24_double( i + 3 : i + 3 )
                 STRING_trim_real_24_double( i + 2 : i + 3 ) = '  '
               END IF
             ELSE
               IF ( STRING_trim_real_24_double( i + 3 : i + 3 ) == ' ' ) THEN
               ELSE
                 STRING_trim_real_24_double( i + 1 : i + 1 )                   &
                   = STRING_trim_real_24_double( i + 2 : i + 2 )
                 STRING_trim_real_24_double( i + 2 : i + 2 )                   &
                   = STRING_trim_real_24_double( i + 3 : i + 3 )
                 STRING_trim_real_24_double( i + 3 : i + 3 ) = ' '
               END IF
             END IF
           END IF
         END IF
         EXIT
       END IF

!  remove trailing 0 unless it is preceeded by a .

       IF ( STRING_trim_real_24_double( i : i ) == ' ' ) THEN
         IF ( i < 3 ) EXIT
         IF ( STRING_trim_real_24_double( i - 1 : i - 1 ) == '0' .AND.         &
              STRING_trim_real_24_double( i - 2 : i - 2 ) /= '.' ) THEN
              STRING_trim_real_24_double( i - 1 : i - 1 ) = ' '
         END IF
         EXIT
       END IF

     END DO

!  if the string starts with a ., add a 0 at the front

     IF ( STRING_trim_real_24_double( 1 : 1 ) == '.' ) THEN
       DO i = 24, 2, -1
         STRING_trim_real_24_double( i : i )                                   &
           = STRING_trim_real_24_double( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_double( 1 : 1 ) = '0'
     END IF

!  if the string starts with a ., add a 0 at the front

     IF ( STRING_trim_real_24_double( 1 : 1 ) == '.' ) THEN
       DO i = 24, 2, -1
         STRING_trim_real_24_double( i : i )                                   &
           = STRING_trim_real_24_double( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_double( 1 : 1 ) = '0'
     END IF

!  if the string starts with a -., replace by -0. at the front

     IF ( STRING_trim_real_24_double( 1 : 2 ) == '-.' ) THEN
       DO i = 24, 3, -1
         STRING_trim_real_24_double( i : i )                                   &
           = STRING_trim_real_24_double( i - 1 : i - 1 )
       END DO
       STRING_trim_real_24_double( 2 : 2 ) = '0'
     END IF
     RETURN

!  end of function STRING_trim_real_24_double

     END FUNCTION STRING_trim_real_24_double

!  ------- S T R I N G _ T R I M _ I N T E G E R _ 1 6    F U N C T I O N -----

     FUNCTION STRING_trim_integer_16( i )
     CHARACTER ( LEN = 16 ) :: STRING_trim_integer_16
     INTEGER :: i

!  write integer as a left shifted length 16 character

     STRING_trim_integer_16 = REPEAT( ' ', 16 )
     WRITE( STRING_trim_integer_16, "( I0 )" ) i
     RETURN

!  end of function STRING_trim_integer_16

     END FUNCTION STRING_trim_integer_16

! -*-*- G A L A H A D - S T R I N G _ e s _ s i n g l e  F U N C T I O N -*-*-

     FUNCTION STRING_es_single( d, val )

!  obtain a left-shited character representation of a real variable with
!  d digits of precision

     INTEGER, INTENT( IN ) :: d
     REAL ( KIND = sp ) :: val
     CHARACTER ( len = d + 6 + COUNT( SPREAD( val, 1, 1 ) < 0.0 ) ) ::         &
       STRING_es_single

!  local variables

     INTEGER :: l
     CHARACTER ( len = 80 ) :: cval
     CHARACTER ( len = 40 ) :: cl, cd

     l = d + 6 + COUNT( SPREAD( val, 1, 1 ) < 0.0 )
     cval = REPEAT( ' ', 80 )
     WRITE( unit = cl, fmt = "( I40 )" ) l
     WRITE( unit = cd, fmt = "( I40 )" ) d
     WRITE( unit = cval, fmt = '(ES' // TRIM( ADJUSTL( cl ) ) //               &
                                 '.' // TRIM( ADJUSTL( cd ) ) // ')' ) val
     STRING_es_single = ADJUSTL( cval( 1 : l ) )

     RETURN

!  End of function STRING_es_single

     END FUNCTION STRING_es_single

! -*-*- G A L A H A D - S T R I N G _ e s _ d o u b l e  F U N C T I O N -*-*-

     FUNCTION STRING_es_double( d, val )

!  obtain a left-shited character representation of a real variable with
!  d digits of precision

     INTEGER, INTENT( IN ) :: d
     REAL ( KIND = dp ) :: val
     CHARACTER ( len = d + 6 + COUNT( SPREAD( val, 1, 1 ) < 0.0 ) ) ::         &
       STRING_es_double

!  local variables

     INTEGER :: l
     CHARACTER ( len = 80 ) :: cval
     CHARACTER ( len = 40 ) :: cl, cd

     l = d + 6 + COUNT( SPREAD( val, 1, 1 ) < 0.0 )
     cval = REPEAT( ' ', 80 )
     WRITE( unit = cl, fmt = "( I40 )" ) l
     WRITE( unit = cd, fmt = "( I40 )" ) d
     WRITE( unit = cval, fmt = '(ES' // TRIM( ADJUSTL( cl ) ) //               &
                                 '.' // TRIM( ADJUSTL( cd ) ) // ')' ) val
     STRING_es_double = ADJUSTL( cval( 1 : l ) )

     RETURN

!  End of function STRING_es_double

     END FUNCTION STRING_es_double

! -*- G A L A H A D - S T R I N G _ e s 1 2 _ s i n g l e  F U N C T I O N -*-

     FUNCTION STRING_es12_single( val )

!  obtain a left-shited ES12.5 character representation of a real variable

     CHARACTER ( len = 12 ) :: STRING_es12_single
     REAL ( KIND = sp ), INTENT( IN ) :: val

!  local variables

     CHARACTER ( LEN = 12 ) :: cval
     WRITE( cval, "( ES12.5 )" ) val
     STRING_es12_single = ADJUSTL( cval )

     RETURN

!  End of function STRING_es12_single

     END FUNCTION STRING_es12_single

! -*- G A L A H A D - S T R I N G _ e s 1 2 _ d o u b l e  F U N C T I O N -*-

     FUNCTION STRING_es12_double( val )

!  obtain a left-shited ES12.5 character representation of a real variable

     CHARACTER ( len = 12 ) :: STRING_es12_double
     REAL ( KIND = dp ), INTENT( IN ) :: val

!  local variables

     CHARACTER ( LEN = 12 ) :: cval
     WRITE( cval, "( ES12.5 )" ) val
     STRING_es12_double = ADJUSTL( cval )

     RETURN

!  End of function STRING_es12_double

     END FUNCTION STRING_es12_double

! - G A L A H A D - S T R I N G _ e x p o n e n t _ s i n g l e  F U N C T I O N

     FUNCTION STRING_exponent_single( r )

!  Obtain a 3 character representation of the exponent of the real r.
!  Exponents outside [-99,+99] will be represented as +in or -in as relevant

     CHARACTER ( LEN = 3 ) :: STRING_exponent_single

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     REAL ( KIND = sp ) :: r

!  Local variable

     INTEGER :: i

     IF ( r == 0.0_sp ) THEN
       WRITE( STRING_exponent_single, "( '-in' )" )
     ELSE
       i = FLOOR( LOG10( r ) )
       IF ( i <= 99 .AND. i >= - 99 ) THEN
         WRITE( STRING_exponent_single, "( SP, I3 )" ) i
       ELSE IF ( i < - 99 ) THEN
         WRITE( STRING_exponent_single, "( '-in' )" )
       ELSE
         WRITE( STRING_exponent_single, "( '+in' )" )
       END IF
     END IF
     RETURN

!  End of STRING_exponent_single

     END FUNCTION STRING_exponent_single

! - G A L A H A D - S T R I N G _ e x p o n e n t _ d o u b l e  F U N C T I O N

     FUNCTION STRING_exponent_double( r )

!  Obtain a 3 character representation of the exponent of the real r.
!  Exponents outside [-99,+99] will be represented as +in or -in as relevant

     CHARACTER ( LEN = 3 ) :: STRING_exponent_double

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     REAL ( KIND = dp ) :: r

!  Local variable

     INTEGER :: i

     IF ( r == 0.0_dp ) THEN
       WRITE( STRING_exponent_double, "( '-in' )" )
     ELSE
       i = FLOOR( LOG10( r ) )
       IF ( i <= 99 .AND. i >= - 99 ) THEN
         WRITE( STRING_exponent_double, "( SP, I3 )" ) i
       ELSE IF ( i < - 99 ) THEN
         WRITE( STRING_exponent_double, "( '-in' )" )
       ELSE
         WRITE( STRING_exponent_double, "( '+in' )" )
       END IF
     END IF
     RETURN

!  End of STRING_exponent_double

     END FUNCTION STRING_exponent_double

!-*-*-  G A L A H A D -  S T R I N G _ o r d i n a l   F U N C T I O N  -*-*-

     FUNCTION STRING_ordinal( num )

     CHARACTER ( len = 2 ) :: STRING_ordinal

!  Obtain the ordinal suffix (i.e., "st", "nd", "rd", "th") for the integer num.
!  Extracted from GALAHAD module TOOLS

!--------------------------------
!   D u m m y   A r g u m e n t
!--------------------------------

     INTEGER :: num

     SELECT CASE( MOD( num, 100 ) )
     CASE( 1, 21, 31, 41, 51, 61, 71, 81, 91 )
       STRING_ordinal = 'st'
     CASE( 2, 22, 32, 42, 52, 62, 72, 82, 92 )
       STRING_ordinal = 'nd'
     CASE( 3, 23, 33, 43, 53, 63, 73, 83, 93 )
       STRING_ordinal = 'rd'
     CASE DEFAULT
       STRING_ordinal = 'th'
     END SELECT
     RETURN

!   End of STRING_ordinal

     END FUNCTION STRING_ordinal

!  End of module GALAHAD_STRING

   END MODULE GALAHAD_STRING_double
