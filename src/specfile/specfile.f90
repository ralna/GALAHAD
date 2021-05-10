! THIS VERSION: GALAHAD 2.4 - 12/08/2009 AT 09:30 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*    GALAHAD SPECFILE  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Philippe Toint

!  History -
!   originally released pre GALAHAD Version 1.0. July 16th 2002
!   update released with GALAHAD Version 2.0. August 25th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!
!             +-----------------------------------------------+
!             |  Read the algorithmic specifications for      |
!             |  GALAHAD programs or routines                 |
!             |                                               |
!             |  Philippe Toint                               |
!             |  Spring 2002                                  |
!             +-----------------------------------------------+
!
!  Several programs or routines of the GALAHAD library feature a number of
!  "control parameters", that is parameters that condition the various
!  algorithmic, printing and other options of the program/routine. All these
!  parameters have default values, but it is often useful to overide these
!  default values for specific purposes.  Two mechanisms are possible:
!  1) write a suitable assignment in the calling program (after the default
!     values have been set in an initialization phase and before the main
!     work of the routine is performed);
!  2) read the values of the control parameters that must overide the
!     corresponding defaults from a "specification file" (specfile).
!  (Note that the first mechanism only applies to routines and requires
!  recompilation of the code for each change in the control parameters.)
!  The purpose of this module is to facilitate the use of the second of
!  these mechanisms.
!
!  The syntax of a specfile
!  ------------------------
!
!  A specification file, or specfile, is assumed to consist of a number of
!  "specification commands",  each of these being decomposed into
!  - a "keyword", which is a string (in a close-to-natural language) that will
!    be used to identify a control parameter in the specfile, and
!  - an (optional) "value", which is the value to be attributed to the
!    said control parameter (overiding its default).
!  A specific algorithmic control parameter is associated to each such
!  keyword, and the effect of interpreting the specification file is to assign
!  the value associated with the keyword (in each specification command) to
!  the corresponding algorithmic parameter.  A specification file must start
!  with a "BEGIN (program/routine name)" command and ends with an
!  "END" command.  The syntax of the specfile is thus defined as follows:
!
!      BEGIN (program/routine name)
!         keyword            value
!         .......            .....
!         keyword            value
!      END
!
!  where keyword and value are two strings separated by (at least) one blank.
!  The BEGIN and END delimiter commands may be completed after a blank, so
!  that lines such as
!
!      BEGIN myroutine SPECIFICATIONS
!
!  and
!
!      END myroutine SPECIFICATIONS
!
!  are acceptable. Furthermore, the specification commands (between the
!  BEGIN and END delimiters) may be specified in any order.  Blank lines and
!  lines whose first non-blank character is ! or * are ignored. The content
!  of a line after a ! or * character is also ignored (as is the ! or *
!  character itself). This provides an easy manner to "comment off" some
!  specification commands or to add comments associated to specific values
!  of certain control parameters.  The specification file must be open for
!  input when SPECFILE_read is called, and the associated device number
!  passed to the routine in device (see below). Note that the corresponding
!  file is REWINDed, which make it possible to combine the specifications
!  for more than one program/routine.  For the same reason, the file is not
!  closed by SPECFILE_read.
!
!  The value of a control parameters may be of five different types, namely
!  - integer,
!  - logical,
!  - real,
!  - character string,
!  - symbolic.
!  A symbolic value is a special, predefined (in the SYMBOLS module) string,
!  which may help to express an (integer) control parameter for an algorithm
!  in a "language" that is close to natural.  The possible values for logical
!  parameters are "ON", "TRUE", ".TRUE.", "T", "YES", "Y", or "OFF", "NO",
!  "N", "FALSE", ".FALSE." and "F". Empty values are also addmitted for
!  logical control parameters, and interpreted as "TRUE".  Note that the
!  keywords, as well as symbol and logical values in specification commands
!  are case insensitive. The keywords must not contain more than 50
!  characters, and the value not more than 30 characters.  Furthermore, each
!  line of the specfile is limited to 80 characters, including the blanks
!  separating keyword and value.
!
!
!  How to use this module
!  ----------------------
!
!  This module provides tools to read the specfile and to interpret its
!  specfication commands. A typical use consists of three successive steps:
!
!              +--------------------------------------------------+
!              |            define the content and type           |
!              |          of the specifications commands          |
!              |                                                  |
!              |    (by initializing a specific datastructure)    |
!              +--------------------------------------------------+
!                                      |
!                                      V
!              +--------------------------------------------------+
!              |               parse the specfile                 |
!              |                                                  |
!              | (using the SPECFILE_read routine of this module) |
!              +--------------------------------------------------+
!                                      |
!                                      V
!              +--------------------------------------------------+
!              |       interpret the specifications commands      |
!              |                                                  |
!              |    (using the SPECFILE_assign_* functions of     |
!              |                  this module)                    |
!              +--------------------------------------------------+
!
!  The data structure to initialize at step one is an array of
!  TYPE( SPECFILE_item_type ) (see TYPE definition below), whose length
!  is equal to the number of possible different specifications commands
!  in the considered specfile.
!
!  The details are best shown by an example. Assume that one has the routine
!  called "mystuff" whose associated specifications commands are used to specify
!  the print level (either SILENT or VERBOSE), an output device number (an
!  integer), an output file name (a string) and a final accuracy (a real).  The
!  code in charge of reading the specfile must thus first use the SPECFILE
!  module:
!
!     USE GALAHAD_SPECFILE_double
!
!  then declare the used variables, amongst which an array of SPECFILE_item_type
!  of length 4:
!
!     INTEGER               :: spec_device, errout, print_level, out_device
!     CHARACTER( LEN = 16 ) :: algo_name
!     CHARACTER( LEN = 30 ) :: out_filename
!     REAL( KIND = wp )     :: accuracy
!     TYPE( SPECFILE_item_type ), DIMENSION( 4 ) :: specs
!
!  then initialize it:
!
!     specs( 1 )%keyword = 'print-level'
!     specs( 2 )%keyword = 'output-file-name'
!     specs( 3 )%keyword = 'output-file-device'
!     specs( 4 )%keyword = 'final-accuracy'
!
!  then open the specfile:
!
!     OPEN( UNIT = spec_device, FILE = 'MYSTUFF.SPC' )
!
!  then call the SPECFILE_read routine to read the specfile MYSTUFF.SPC:
!
!     algo_name = 'mystuff'
!     CALL SPECFILE_read( spec_device, algo_name, specs, 4, errout )
!
!  and finally translate the obtained values (all contained in strings so far)
!  into their proper types and assign the resulting (translated) values to the
!  corresponding control parameters:
!
!     CALL SPECFILE_assign_symbol ( specs( 1 ), print_level , errout )
!     CALL SPECFILE_assign_string ( specs( 2 ), out_filename, errout )
!     CALL SPECFILE_assign_integer( specs( 3 ), out_device  , errout )
!     CALL SPECFILE_assign_real   ( specs( 4 ), accuracy    , errout )
!
!  (In these calls, errout is the device number on which error messages should
!  be output.) Note that no assignment is performed for a control parameter
!  if no value is specified for this control parameter in the specfile. Thus,
!  if the file named MYSTUFF.SPC contains
!
!     BEGIN mystuff SPECIFICATIONS
!         print-level          VERBOSE
!         output-file-name     myoutput.d
!         output-file-device   57
!         final-accuracy       1.0D-5
!     END mystuff SPECIFICATIONS
!
!  then the following assignments are made:
!
!     print_level  = (integer value corresponding to the VERBOSE symbol)
!     out_filename = 'myoutput.d'
!     out_device   = 57
!     accuracy     = 1.0D-5
!
!  Note: if the specfile is used to reassign the error output device itself,
!        it is better to define the corresponding specification command to be
!        the first one (specs(1) in our example), for the error messages
!        potentially occurring when interpreting further specification
!        commands to be output on the desired (updated) device.
!
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   MODULE GALAHAD_SPECFILE_double

      USE GALAHAD_SYMBOLS  ! to make the symbols known for translation

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Make the SPECFILE tools calls public

      PUBLIC :: SPECFILE_read         , SPECFILE_assign_integer,               &
                SPECFILE_assign_real  , SPECFILE_assign_logical,               &
                SPECFILE_assign_symbol, SPECFILE_assign_string,                &
                SPECFILE_assign_value , SPECFILE_assign_long

      INTERFACE SPECFILE_assign_value
        MODULE PROCEDURE SPECFILE_assign_integer, SPECFILE_assign_real,        &
                         SPECFILE_assign_logical, SPECFILE_assign_string
      END INTERFACE

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0E+0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp
      INTEGER, PRIVATE, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!                      PUBLIC TYPES AND DEFINITIONS
!
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!-------------------------------------------------------------------------------
!       The structure that controls the PRESOLVE execution by defining
!              the content of a (single) specification command
!-------------------------------------------------------------------------------

!     The following data type is intended to contain the information related
!     to a single control parameter assignment.  The specfile itself consists
!     of a series of such assignments, preceded by a BEGIN statement and
!     closed by a CLOSE statement. Reading a datafile thus requires specifying
!     an array of SPECFILE_item_type whose length is equal to the number of
!     parameters that can be assigned in the specfile (see above).

      TYPE, PUBLIC :: SPECFILE_item_type

         CHARACTER( LEN = 50 ) :: keyword

!              the string that specifies the control parameter whose value
!              is to be assigned.

         CHARACTER( LEN = 30 ) :: value

!              the value assigned to the control parameter, expressed as
!              a string.

         INTEGER :: line

!              0   : if no specfile command has been read so far that assigns a
!                    value to the considered control parameter
!                    (in which case no assignment is made for the control
!                    parameter in the SPECFILE module),
!              > 0 : the line number of the specfile at which the value of the
!                    control parameter has been assigned.

      END TYPE

!==============================================================================
!==============================================================================

   CONTAINS

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_squeeze( string, new_length, break )

!     Removes repeated, leading and trailing blanks, and computes the position
!     of the command value within the (cleaned up) command line.

!     Argument

      INTEGER              , INTENT( OUT )   :: new_length, break
      CHARACTER( LEN = 80 ), INTENT( INOUT ) :: string

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER              :: i, j, nl, nb, last
      CHARACTER( LEN = 1 ) :: c

!     Remove the leading and trailing blanks

      string = ADJUSTL( TRIM( string ) )

!     Compute the length of the resulting string

      new_length = LEN_TRIM( string )

!     Return if nothing is left

      IF ( new_length == 0 ) RETURN

!     Initialize the position of the second "word"

      break = 0

!     Initialize the position of the last non-blank seen (character 1
!     must be non-blank because the leading blanks have been removed)

      last  = 1

!     Loop of the string characters, starting from character 2

      i     = 2
      DO j  = 2, new_length

!        Get the next character

         c = string( i:i )

!        Action if it is not blank

         IF ( c /= ' ' ) THEN

!           If the rest of the line is a comment, skip it

            IF ( c == '!' .OR. c == '*' ) EXIT

!           Compute the number of blanks since the last non-blank character

            nb = i - last - 1

!           If this is the first internal blank, remember the position
!           that follows it

            IF ( nb >= 1 .AND. break == 0 ) break = last + 2

!           If there is more than one blank, remove the redundant ones

            IF ( nb > 1 ) THEN
               nl = new_length - nb + 1
               string( last+2:nl ) = string( i:new_length )
               new_length = nl
               i = i - nb + 1
            END IF

!           Remember that this character is not a blank

            last = i

         END IF

!        Consider the next character

         i = i + 1

!        Exit if we have seen all non-blank characters

         IF ( i > new_length ) EXIT

      END DO

      new_length = last
      IF ( break == 0 ) break = new_length + 1

      RETURN

      END SUBROUTINE SPECFILE_squeeze

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_integer( specitem, iparam, errout )

!     Translate the string value of an integer control parameter into
!     its proper value

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      INTEGER, INTENT( INOUT ) :: iparam

!        The control parameter to be assigned.

      INTEGER, INTENT ( INOUT ) :: errout

!        The error output device number. Note that it has INOUT for INTENT as
!        the assignment may be used to reassign errout itself.

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER               :: lvalue, ios, itmp
      CHARACTER( LEN = 8 )  :: fmt

      IF ( specitem%line > 0 ) THEN
         lvalue = LEN_TRIM( specitem%value )
         IF ( lvalue == 0 ) THEN
            WRITE( errout, * ) ' *** SPECFILE WARNING: integer value ',        &
                    TRIM( specitem%value )
            WRITE( errout, * ) '      not recognized in line',                 &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
            RETURN
         ELSE IF ( lvalue < 10 ) THEN
             WRITE( fmt, '( ''(I'',I1, '')'' )' ) lvalue
         ELSE
             WRITE( fmt, '( ''(I'',I2, '')'' )' ) lvalue
         END IF
         READ( specitem%value, fmt, IOSTAT = ios ) itmp
         IF ( ios == 0 ) THEN
            iparam = itmp
         ELSE
            WRITE( errout, * ) ' *** SPECFILE WARNING: integer value ',        &
                    TRIM( specitem%value )
            WRITE( errout, * ) '     not recognized in line',                  &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
         END IF
      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_integer

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_long( specitem, iparam, errout )

!     Translate the string value of an integer control parameter into
!     its proper value

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      INTEGER ( kind = long ), INTENT( INOUT ) :: iparam

!        The control parameter to be assigned.

      INTEGER, INTENT ( INOUT ) :: errout

!        The error output device number. Note that it has INOUT for INTENT as
!        the assignment may be used to reassign errout itself.

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER               :: lvalue, ios, itmp
      CHARACTER( LEN = 8 )  :: fmt

      IF ( specitem%line > 0 ) THEN
         lvalue = LEN_TRIM( specitem%value )
         IF ( lvalue == 0 ) THEN
            WRITE( errout, * ) ' *** SPECFILE WARNING: integer value ',        &
                    TRIM( specitem%value )
            WRITE( errout, * ) '      not recognized in line',                 &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
            RETURN
         ELSE IF ( lvalue < 10 ) THEN
             WRITE( fmt, '( ''(I'',I1, '')'' )' ) lvalue
         ELSE
             WRITE( fmt, '( ''(I'',I2, '')'' )' ) lvalue
         END IF
         READ( specitem%value, fmt, IOSTAT = ios ) itmp
         IF ( ios == 0 ) THEN
            iparam = itmp
         ELSE
            WRITE( errout, * ) ' *** SPECFILE WARNING: integer value ',        &
                    TRIM( specitem%value )
            WRITE( errout, * ) '     not recognized in line',                  &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
         END IF
      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_long

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_string( specitem, sparam, errout )

!     Translate the string value of a character control parameter into
!     its proper value.

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      CHARACTER( LEN = 30 ), INTENT( INOUT ) :: sparam

!        The control parameter to be assigned.

      INTEGER, INTENT ( IN ) :: errout

!        The error output device number

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER               :: lvalue

      IF ( specitem%line > 0 ) THEN
         lvalue = LEN_TRIM( specitem%value )
         IF ( lvalue > 0 ) THEN
            sparam = TRIM( specitem%value )
         ELSE
            WRITE( errout, * ) ' *** SPECFILE WARNING: string value ',         &
                    TRIM( specitem%value )
            WRITE( errout, * ) '    not recognized in line',                   &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '    Corresponding assignment skipped.'
         END IF
      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_string

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_real( specitem, rparam, errout )

!     Translate the string value of an real control parameter into
!     its proper value

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      REAL( KIND = wp ), INTENT( INOUT ) :: rparam

!        The control parameter to be assigned.

      INTEGER, INTENT ( IN ) :: errout

!        The error output device number

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER               :: lvalue, ios
      REAL( KIND = wp )     :: rtmp
      CHARACTER( LEN = 8 )  :: fmt

      IF ( specitem%line > 0 ) THEN
         lvalue = LEN_TRIM( specitem%value )
         IF ( lvalue == 0 ) THEN
            WRITE( errout, * )                                                 &
                    ' *** SPECFILE WARNING: keyword ', TRIM( specitem%keyword )
            WRITE( errout, * ) '     not recognized in line',                  &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
            RETURN
         ELSE IF ( lvalue < 10 ) THEN
            WRITE( fmt, '( ''(E'',I1, ''.1)'' )' ) lvalue
         ELSE
            WRITE( fmt, '( ''(E'',I2, ''.1)'' )' ) lvalue
         END IF
         READ( specitem%value, fmt, IOSTAT = ios ) rtmp
         IF ( ios == 0 ) THEN
            rparam = rtmp
         ELSE
            WRITE( errout, * )                                                 &
                    ' *** SPECFILE WARNING: real value ', TRIM( specitem%value )
            WRITE( errout, * ) '     not recognized in line',                  &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
         END IF

      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_real

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_logical( specitem, lparam, errout )

!     Translate the string value of an real control parameter into
!     its proper value

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      LOGICAL, INTENT( INOUT ) :: lparam

!        The control parameter to be assigned.

      INTEGER, INTENT ( IN ) :: errout

!        The error output device number

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      CHARACTER( LEN = 30 ) :: tmp
      INTEGER :: len_trim_tmp

      IF ( specitem%line > 0 ) THEN
         tmp = TRIM( specitem%value )
         len_trim_tmp = LEN_TRIM( tmp )
         CALL SPECFILE_upper_case( tmp, len_trim_tmp  )
         IF (      tmp == 'ON'     .OR. tmp == 'T'       .OR. &
                   tmp == 'YES'    .OR. tmp == 'Y'       .OR. &
                   tmp == '.TRUE.' .OR. tmp == 'TRUE'    .OR. &
                   tmp == ''                                  ) THEN
            lparam = .TRUE.
         ELSE IF ( tmp == 'OFF'    .OR. tmp == 'F'       .OR. &
                   tmp == 'NO'     .OR. tmp == 'N'       .OR. &
                   tmp == 'FALSE'  .OR. tmp == '.FALSE.'      ) THEN
            lparam = .FALSE.
         ELSE
            WRITE( errout, * ) ' *** SPECFILE WARNING: logical value ',        &
                    TRIM( specitem%value )
            WRITE( errout, * ) '     not recognized in line',                  &
                    specitem%line, 'of the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
            RETURN
         END IF

      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_logical

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_assign_symbol( specitem, param, errout )

!     Translate the symbol value of a control parameter into
!     its proper (integer) value

      TYPE( SPECFILE_item_type ), INTENT( IN ):: specitem

!        The specification item

      INTEGER, INTENT( INOUT ) :: param

!        The control parameter to be assigned.

      INTEGER, INTENT ( IN ) :: errout

!        The error output device number

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      CHARACTER( LEN = 30 ) :: upper_value
      INTEGER :: len_trim_upper_value

      IF ( specitem%line > 0 ) THEN

!        Convert the value to upper case before interpretation

         upper_value = TRIM( specitem%value )
         len_trim_upper_value = LEN_TRIM( upper_value )
         CALL SPECFILE_upper_case( upper_value, len_trim_upper_value )

!        Interpret the value

         IF ( TRIM( upper_value )      == 'SILENT'                ) THEN
            param = GALAHAD_SILENT
         ELSE IF ( TRIM( upper_value ) == 'TRACE'                 ) THEN
            param = GALAHAD_TRACE
         ELSE IF ( TRIM( upper_value ) == 'ACTION'                ) THEN
            param = GALAHAD_ACTION
         ELSE IF ( TRIM( upper_value ) == 'DETAILS'               ) THEN
            param = GALAHAD_DETAILS
         ELSE IF ( TRIM( upper_value ) == 'DEBUG'                 ) THEN
            param = GALAHAD_DEBUG
         ELSE IF ( TRIM( upper_value ) == 'CRAZY'                 ) THEN
            param = GALAHAD_CRAZY
         ELSE IF ( TRIM( upper_value ) == 'KEEP'                  ) THEN
           param = GALAHAD_KEEP
         ELSE IF ( TRIM( upper_value ) == 'DELETE'                ) THEN
            param = GALAHAD_DELETE
         ELSE IF ( TRIM( upper_value ) == 'NONE'                  ) THEN
            param = GALAHAD_NONE
         ELSE IF ( TRIM( upper_value ) == 'BASIC'                 ) THEN
            param = GALAHAD_BASIC
         ELSE IF ( TRIM( upper_value ) == 'SEVERE'                ) THEN
            param = GALAHAD_SEVERE
         ELSE IF ( TRIM( upper_value ) == 'REDUCED_SIZE'          ) THEN
            param = GALAHAD_REDUCED_SIZE
         ELSE IF ( TRIM( upper_value ) == 'FULL_PRESOLVE'         ) THEN
            param = GALAHAD_FULL_PRESOLVE
         ELSE IF ( TRIM( upper_value ) == 'ALL_ZEROS'             ) THEN
            param = GALAHAD_ALL_ZEROS
         ELSE IF ( TRIM( upper_value ) == 'ALL_ONES'              ) THEN
            param = GALAHAD_ALL_ONES
         ELSE IF ( TRIM( upper_value ) == 'GENERAL'               ) THEN
            param = GALAHAD_GENERAL
         ELSE IF ( TRIM( upper_value ) == 'POSITIVE'              ) THEN
            param = GALAHAD_POSITIVE
         ELSE IF ( TRIM( upper_value ) == 'NEGATIVE'              ) THEN
            param = GALAHAD_NEGATIVE
         ELSE IF ( TRIM( upper_value ) == 'LEAVE_AS_IS'           ) THEN
            param = GALAHAD_LEAVE_AS_IS
         ELSE IF ( TRIM( upper_value ) == 'FORCE_TO_ZERO'         ) THEN
            param = GALAHAD_FORCE_TO_ZERO
         ELSE IF ( TRIM( upper_value ) == 'TIGHTEST'              ) THEN
            param = GALAHAD_TIGHTEST
         ELSE IF ( TRIM( upper_value ) == 'NON_DEGENERATE'        ) THEN
            param = GALAHAD_NON_DEGENERATE
         ELSE IF ( TRIM( upper_value ) == 'LOOSEST'               ) THEN
            param = GALAHAD_LOOSEST
         ELSE IF ( TRIM( upper_value ) == 'DENSE'                 ) THEN
            param = GALAHAD_DENSE
         ELSE IF ( TRIM( upper_value ) == 'SPARSE'                ) THEN
            param = GALAHAD_SPARSE_BY_ROWS
         ELSE IF ( TRIM( upper_value ) == 'COORDINATE'            ) THEN
            param = GALAHAD_COORDINATE
         ELSE IF ( TRIM( upper_value ) == 'ELEMENTAL'             ) THEN
            param = GALAHAD_ELEMENTAL
         ELSE IF ( TRIM( upper_value ) == 'UNCONSTRAINED'         ) THEN
            param = GALAHAD_UNCONSTRAINED
         ELSE IF ( TRIM( upper_value ) == 'CONSTRAINED'           ) THEN
            param = GALAHAD_CONSTRAINED
         ELSE IF ( TRIM( upper_value ) == 'INACTIVE'              ) THEN
            param = GALAHAD_INACTIVE
         ELSE IF ( TRIM( upper_value ) == 'ELIMINATED'            ) THEN
            param = GALAHAD_ELIMINATED
         ELSE IF ( TRIM( upper_value ) == 'ACTIVE'                ) THEN
            param = GALAHAD_ACTIVE
         ELSE IF ( TRIM( upper_value ) == 'FIXED'                 ) THEN
            param = GALAHAD_FIXED
         ELSE IF ( TRIM( upper_value ) == 'RANGE'                 ) THEN
            param = GALAHAD_RANGE
         ELSE IF ( TRIM( upper_value ) == 'UPPER'                 ) THEN
            param = GALAHAD_UPPER
         ELSE IF ( TRIM( upper_value ) == 'LOWER'                 ) THEN
            param = GALAHAD_LOWER
         ELSE IF ( TRIM( upper_value ) == 'FREE'                  ) THEN
            param = GALAHAD_FREE
         ELSE IF ( TRIM( upper_value ) == 'POSITIVE'              ) THEN
            param = GALAHAD_POSITIVE
         ELSE IF ( TRIM( upper_value ) == 'NEGATIVE'              ) THEN
            param = GALAHAD_NEGATIVE
         ELSE IF ( TRIM( upper_value ) == 'SUCCESS'               ) THEN
            param = GALAHAD_SUCCESS
         ELSE IF ( TRIM( upper_value ) == 'MEMORY_FULL'           ) THEN
            param = GALAHAD_MEMORY_FULL
         ELSE IF ( TRIM( upper_value ) == 'COULD_NOT_WRITE'       ) THEN
            param = GALAHAD_COULD_NOT_WRITE
         ELSE IF ( TRIM( upper_value ) == 'FILE_NOT_OPENED'       ) THEN
            param = GALAHAD_FILE_NOT_OPENED
         ELSE IF ( TRIM( upper_value ) == 'EXACT'                 ) THEN
            param = GALAHAD_EXACT
         ELSE IF ( TRIM( upper_value ) == 'FORWARD'               ) THEN
            param = GALAHAD_FORWARD
         ELSE IF ( TRIM( upper_value ) == 'CENTRAL'               ) THEN
            param = GALAHAD_CENTRAL
         ELSE IF ( TRIM( upper_value ) == 'BFGS'                  ) THEN
            param = GALAHAD_BFGS
         ELSE IF ( TRIM( upper_value ) == 'DFP'                   ) THEN
            param = GALAHAD_DFP
         ELSE IF ( TRIM( upper_value ) == 'PSB'                   ) THEN
            param = GALAHAD_PSB
         ELSE IF ( TRIM( upper_value ) == 'SR1'                   ) THEN
            param = GALAHAD_SR1
         ELSE IF ( TRIM( upper_value ) == 'CG'                    ) THEN
            param = GALAHAD_CG
         ELSE IF ( TRIM( upper_value ) == 'DIAGONAL_CG'           ) THEN
            param = GALAHAD_DIAGONAL_CG
         ELSE IF ( TRIM( upper_value ) == 'USERS_CG'              ) THEN
            param = GALAHAD_USERS_CG
         ELSE IF ( TRIM( upper_value ) == 'EXPANDING_BAND_CG'     ) THEN
            param = GALAHAD_EXPANDING_BAND_CG
         ELSE IF ( TRIM( upper_value ) == 'MUNKSGAARD_CG'         ) THEN
            param = GALAHAD_MUNKSGAARD_CG
         ELSE IF ( TRIM( upper_value ) == 'SCHNABEL_ESKOW_CG'     ) THEN
            param = GALAHAD_SCHNABEL_ESKOW_CG
         ELSE IF ( TRIM( upper_value ) == 'GMPS_CG'               ) THEN
            param = GALAHAD_GMPS_CG
         ELSE IF ( TRIM( upper_value ) == 'BAND_CG'               ) THEN
            param = GALAHAD_BAND_CG
         ELSE IF ( TRIM( upper_value ) == 'LIN_MORE_CG'           ) THEN
            param = GALAHAD_LIN_MORE_CG
         ELSE IF ( TRIM( upper_value ) == 'MULTIFRONTAL'          ) THEN
            param = GALAHAD_MULTIFRONTAL
         ELSE IF ( TRIM( upper_value ) == 'MODIFIED_MULTIFRONTAL' ) THEN
            param = GALAHAD_MODIFIED_MULTIFRONTAL
         ELSE IF ( TRIM( upper_value ) == 'BANDED'                ) THEN
            param = GALAHAD_BANDED
         ELSE IF ( TRIM( upper_value ) == 'AUTOMATIC'             ) THEN
            param = GALAHAD_AUTOMATIC
         ELSE IF ( TRIM( upper_value ) == 'USER_DEFINED'          ) THEN
            param = GALAHAD_USER_DEFINED
         ELSE IF ( TRIM( upper_value ) == 'NEVER'                 ) THEN
            param = GALAHAD_NEVER
         ELSE IF ( TRIM( upper_value ) == 'INITIAL'               ) THEN
            param = GALAHAD_INITIAL
         ELSE IF ( TRIM( upper_value ) == 'ALWAYS'                ) THEN
            param = GALAHAD_ALWAYS
         ELSE IF ( TRIM( upper_value ) == 'GAUSS_NEWTON'          ) THEN
            param = GALAHAD_GAUSS_NEWTON
         ELSE IF ( TRIM( upper_value ) == 'NEWTON'                ) THEN
            param = GALAHAD_NEWTON
         ELSE IF ( TRIM( upper_value ) == 'ADAPTIVE'              ) THEN
            param = GALAHAD_ADAPTIVE
         ELSE IF ( TRIM( upper_value ) == 'FULL'                  ) THEN
            param = GALAHAD_FULL
         ELSE IF ( TRIM( upper_value ) == 'CURRENT'               ) THEN
            param = GALAHAD_CURRENT
         ELSE IF ( TRIM( upper_value ) == 'SMALLEST'              ) THEN
            param = GALAHAD_SMALLEST
         ELSE IF ( TRIM( upper_value ) == 'BEST_FIT'              ) THEN
            param = GALAHAD_BEST_FIT
         ELSE IF ( TRIM( upper_value ) == 'BEST_REDUCTION'        ) THEN
            param = GALAHAD_BEST_REDUCTION
         ELSE IF ( TRIM( upper_value ) == '1' ) THEN
            param = GALAHAD_1
         ELSE IF ( TRIM( upper_value ) == '2' ) THEN
            param = GALAHAD_2
         ELSE IF ( TRIM( upper_value ) == '3' ) THEN
            param = GALAHAD_3
         ELSE IF ( TRIM( upper_value ) == '4' ) THEN
            param = GALAHAD_4
         ELSE IF ( TRIM( upper_value ) == '5' ) THEN
            param = GALAHAD_5
         ELSE IF ( TRIM( upper_value ) == '6' ) THEN
            param = GALAHAD_6
         ELSE IF ( TRIM( upper_value ) == '7' ) THEN
            param = GALAHAD_7
         ELSE IF ( TRIM( upper_value ) == '8' ) THEN
            param = GALAHAD_8
         ELSE IF ( TRIM( upper_value ) == '9' ) THEN
            param = GALAHAD_9
         ELSE IF ( TRIM( upper_value ) == '10' ) THEN
            param = GALAHAD_10
         ELSE IF ( TRIM( upper_value ) == '11' ) THEN
            param = GALAHAD_11
         ELSE IF ( TRIM( upper_value ) == '12' ) THEN
            param = GALAHAD_12
         ELSE
            WRITE( errout, * )                                                 &
                 ' *** SPECFILE WARNING: symbol value ', TRIM( upper_value )
            WRITE( errout, * ) '     not recognized at line ', specitem%line,  &
                'in the specification file.'
            WRITE( errout, * ) '     Corresponding assignment skipped.'
         END IF
      END IF

      RETURN

      END SUBROUTINE SPECFILE_assign_symbol

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_upper_case( string, length )

!     Convert any lower case characters in the character array string to
!     upper case. This is not very efficient and maybe should be replaced
!     by a hashing routine.

!     Arguments:

      INTEGER, INTENT( IN ) :: length

!            the length of the string to convert

      CHARACTER ( LEN = length ), INTENT( INOUT ) :: string

!            the string to convert

!     Programming: N. Gould, 1991, with modifs by Ph. Toint, January 2002.
!
!===============================================================================

!     Local variables

      INTEGER :: i, letter
      CHARACTER, DIMENSION( 26 ) :: LOWER, UPPER

      DATA LOWER / 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',         &
                   'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',         &
                   'u', 'v', 'w', 'x', 'y', 'z' /
      DATA UPPER / 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',         &
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',         &
                   'U', 'V', 'W', 'X', 'Y', 'Z' /

!     Loop over each character in the string

      DO i = 1, length

!     See if the current letter is lower case. If so replace it by its
!     upper case counterpart

         DO letter = 1, 26
            IF ( string( i:i ) == LOWER( letter ) ) THEN
               string( i:i )   =  UPPER( letter )
               EXIT
            END IF
         END DO

      END DO

      RETURN

      END SUBROUTINE SPECFILE_upper_case

!==============================================================================
!==============================================================================

      SUBROUTINE SPECFILE_read( device, algo_name, spec, lspec, errout )

!     Reads the content of a specification files and performs the assignment
!     of values associated with given keywords to the corresponding control
!     parameters.

!     Arguments

      INTEGER, INTENT( IN ) :: device

!            The device number associated with the specification file. Note
!            that the file must be open for input.  The file is REWINDed
!            before use.

      CHARACTER( LEN = * ), INTENT ( IN ) :: algo_name

      INTEGER, INTENT( IN ) :: lspec

      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ), INTENT( INOUT ) :: spec

      INTEGER, INTENT( IN ) :: errout

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER               :: ios, break, endvalue, lenline, nline, ikey
      INTEGER               :: len_trim_spec
      LOGICAL               :: applicable, found, opened
      CHARACTER( LEN = 30 ) :: value
      CHARACTER( LEN = 50 ) :: keyword
      CHARACTER( LEN = 80 ) :: line

!     Unset all items in the spec list

      DO ikey = 1, lspec
         spec( ikey )%line = 0
      END DO

!     Check that the specfile is open

      INQUIRE( UNIT = device, OPENED = opened )
      IF ( .NOT. opened ) THEN
         WRITE( errout, * )                                                    &
              ' *** SPECFILE WARNING: specfile not opened on device', device,'.'
         WRITE( errout, * ) '     Corresponding specifications skipped.'
         RETURN
      END IF

!     Assume a priori that the line is not applicable to PRESOLVE

      applicable = .FALSE.

!     Start at the beginning of the file

      REWIND device

!     Make sure all keywords are in upper case
!     (in order to make keyword matching case insensitive)

      DO ikey = 1, lspec
         len_trim_spec = LEN_TRIM( spec( ikey )%keyword )
         CALL SPECFILE_upper_case( spec( ikey )%keyword, len_trim_spec )

      END DO

!     Loop over the file lines

      nline = 0
      DO

!        Read the next line

         READ( device, '(A80)', IOSTAT = ios ) line
         IF ( ios /= 0 ) RETURN

!        Increment the line number

         nline = nline + 1

!        Remove the unnecessary blanks and compute the position of the
!        second keyword

         CALL SPECFILE_squeeze( line, lenline, break )
!        If the line is commented out or blank, ignore it

         IF ( line( 1:1 ) == '!' .OR. line( 1:1 ) == '*' .OR. &
              lenline == 0                                    ) CYCLE

!        Consider first the case where the line is applicable

         IF ( applicable ) THEN

!           If this is the end of the specifications, ignore what follows

            IF ( line( 1:3 ) == 'END' ) RETURN

!           Otherwise, isolate the keyword ...

            keyword = line( 1:break-1 )

!           ... convert it to upper case...

            CALL SPECFILE_upper_case( keyword, break-1 )

!           ... and isolate the associated string value

            IF ( break > lenline ) THEN
               value = ''
            ELSE
               IF ( line( break:break ) == "'" ) THEN
                 endvalue = INDEX( line( break+1:lenline ), "'" ) + 1
               ELSE IF ( line( break:break ) == '"' ) THEN
                 endvalue = INDEX( line( break+1:lenline ), '"' ) + 1
               ELSE
                 endvalue = INDEX( line( break:lenline ), ' ' )
               END IF
               IF ( endvalue == 0 ) THEN
                  value = line( break:lenline )
               ELSE
                  value = line( break:break+endvalue-1 )
               END IF
            END IF

!           Perform the implied assignment

            found = .FALSE.
            DO ikey = 1, lspec
               IF ( keyword == spec( ikey )%keyword ) THEN
                  spec( ikey )%line  = nline
                  spec( ikey )%value = value
                  found = .TRUE.
                  EXIT
               END IF
            END DO

            IF ( .NOT. found ) THEN
               WRITE( errout, * )                                              &
                    ' *** SPECFILE WARNING: keyword ', TRIM( keyword )
               WRITE( errout, * ) '     not recognized in line',               &
                    nline, 'of the specification file.'
               WRITE( errout, * ) '     Corresponding assignment skipped.'
            END IF

!        Check for the beginning of the specifications

         ELSE IF ( line( 1:5 ) ==  'BEGIN' ) THEN
            IF ( INDEX( line, TRIM( algo_name ) ) == break ) applicable = .TRUE.
         END IF

      END DO

      RETURN

      END SUBROUTINE SPECFILE_read

!===============================================================================
!===============================================================================

   END MODULE GALAHAD_SPECFILE_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*   END GALAHAD SPECFILE  M O D U L E   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
