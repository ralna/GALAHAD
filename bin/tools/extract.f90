  PROGRAM extract_types

!  create C interface files package_ciface.f90 and package.h from
!  date in the package file package.f90

  INTEGER, DIMENSION( 100 ) :: control_clen
  INTEGER, DIMENSION( 100 ) :: inform_clen
  CHARACTER ( LEN = 1 ) :: st1, c
  CHARACTER ( LEN = 2 ) :: day, month, hour, minute
  CHARACTER ( LEN = 4 ) :: year
  CHARACTER ( LEN = 8 ) :: date
  CHARACTER ( LEN = 10 ) :: time
  CHARACTER ( LEN = 10 ) :: package = REPEAT( ' ', 10 )
  CHARACTER ( LEN = 10 ) :: package_lower = REPEAT( ' ', 10 )
  CHARACTER ( LEN = 10 ) :: package_upper = REPEAT( ' ', 10 )
  CHARACTER ( LEN = 14 ) :: package_f90 = REPEAT( ' ', 14 )
  CHARACTER ( LEN = 12 ) :: package_h = REPEAT( ' ', 12 )
  CHARACTER ( LEN = 19 ) :: package_ciface = REPEAT( ' ', 19 )
  CHARACTER ( LEN = 80 ) :: line = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 80 ) :: first_line = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 80 ) :: lineb = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 7 ) :: dtype
  CHARACTER ( LEN = 31 ) :: depends_lower, depends_upper
  CHARACTER ( LEN = 31 ) :: line_lower, line_upper
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: depends = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_i = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_r = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_s = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_l = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_c = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: control_t = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_i = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_r = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_s = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_l = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_c = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: inform_t = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: time_r = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 31 ), DIMENSION( 100 ) :: time_s = REPEAT( ' ', 80 )
  CHARACTER ( LEN = 9 ), DIMENSION( 12 ) :: months = (/                        &
   'January  ', 'February ', 'March    ', 'April    ',                         &
   'May      ', 'June     ', 'July     ', 'August   ',                         &
   'September', 'October  ', 'November ', 'December ' /)
  CHARACTER ( LEN = 2 ), DIMENSION( 31 ) :: days = (/                          &
   'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th',                 &
   'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th',                 &
   'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'st' /)
  INTEGER :: control_ni = 0
  INTEGER :: control_nr = 0
  INTEGER :: control_ns = 0
  INTEGER :: control_nl = 0
  INTEGER :: control_nc = 0
  INTEGER :: control_nt = 0
  INTEGER :: inform_ni = 0
  INTEGER :: inform_nr = 0
  INTEGER :: inform_ns = 0
  INTEGER :: inform_nl = 0
  INTEGER :: inform_nc = 0
  INTEGER :: inform_nt = 0
  INTEGER :: time_nr = 0
  INTEGER :: time_ns = 0
  INTEGER :: i, ios, iday, imonth, l, len, len_name, len_dname, len_tname
  INTEGER :: ni, nr, ns, nl, nc, nt, ndeps, dim, past
  INTEGER :: n_control = 0
  INTEGER :: n_inform = 0
  INTEGER :: n_type = 0
  INTEGER, PARAMETER :: out = 6
  INTEGER, PARAMETER :: package_unit = 20
  INTEGER, PARAMETER :: ciface_unit = 21
  INTEGER, PARAMETER :: h_unit = 22
  INTEGER, PARAMETER :: ciface_scratch = 31
  INTEGER, PARAMETER :: h_scratch = 32
  LOGICAL :: blank, file_exists, skip_next, comment_start, first_component
  LOGICAL :: control_type = .FALSE.
  LOGICAL :: time_type = .FALSE.
  LOGICAL :: inform_type = .FALSE.

!  read package name

  WRITE( out, "( ' please state package name:' )" )
  READ( 5, * ) package
  WRITE( out, "( ' package ', A, ' selected ' )" ) TRIM( package )
  len_name = LEN_TRIM( package )
  DO i = 1, len_name
    st1 = package( i : i ) 
    CALL STRING_lower_scalar( st1 )
    package_lower( i : i ) = st1
    CALL STRING_upper_scalar( st1 )
    package_upper( i : i ) = st1
  END DO
! WRITE( out, "( ' lower and upper case package names are ', A, ' & ', A )" )  &
!    TRIM( package_lower ), TRIM( package_upper )  

!  check that package.f90 exists

  package_f90 = TRIM( package_lower ) // '.f90'
  INQUIRE( FILE = package_f90, EXIST = file_exists )
  IF ( file_exists ) THEN
!   WRITE( out, "( ' package file ', A, ' exists' )" ) TRIM( package_f90 )
    OPEN( UNIT = package_unit, FILE = package_f90, IOSTAT = ios )
  ELSE
    WRITE( out, "( ' package file ', A, ' does not exist. Stopping' )" )       &
      TRIM( package_f90 )
    STOP
  END IF 

  CALL DATE_AND_TIME( date, time )
  day = date( 7 : 8 ) ;  month = date( 5 : 6 ) ;  year = date( 1 : 4 )
  hour = time( 1 : 2 ) ; minute = time( 3 : 4 )
  READ( month, "( I2 )" ) imonth
  READ( day, "( I2 )" ) iday

!  set up and initialize the C header file, along with an associated 
!  scratch file

  package_h = TRIM( package_lower ) // '.h'
  INQUIRE( FILE = package_h, EXIST = file_exists )
  IF ( file_exists ) THEN
    OPEN( UNIT = h_unit, FILE = package_h, FORM = 'FORMATTED',                 &
          STATUS = 'OLD', IOSTAT = ios )
  ELSE
    OPEN( UNIT = h_unit, FILE = package_h, FORM = 'FORMATTED',                 &
          STATUS = 'NEW', IOSTAT = ios )
  END IF
  IF ( ios /= 0 ) THEN
    WRITE( out, "( ' IOSTAT = ', I0, ' when opening file ', A,                 &
   &   '. Stopping')" ) ios, TRIM( package_h )
    STOP
  END IF

!  open the scratch file

  OPEN( UNIT = h_scratch, FORM = 'FORMATTED',                                  &
        STATUS = 'SCRATCH', IOSTAT = ios )
  IF ( ios /= 0 ) THEN
    WRITE( out, "( ' IOSTAT = ', I0, ' when opening h scratch file ',          &
   &   '. Stopping')" ) ios
    STOP
  END IF

!  insert leading data

  WRITE( h_unit, "( '//* \file ', A, '.h */', / )" ) TRIM( package_lower )

  WRITE( h_unit, "( '/*', /, &
 &  ' * THIS VERSION: GALAHAD 3.3 - ', A2, '/', A2, '/', A4, ' AT ',           &
 &  A2, ':', A2, ' GMT.' )" ) day, month, year, hour, minute
  WRITE( h_unit, "( ' *', /, &
 &  ' *-*-*-*-*-*-*-*-*-  GALAHAD_', A, &
 &     ' C INTERFACE  *-*-*-*-*-*-*-*-*-*-', /, &
 &  ' *', /, &
 &  ' *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions', /, &
 &  ' *  Principal author: Jaroslav Fowkes & Nick Gould', /, &
 &  ' *', /, &
 &  ' *  History -', /, &
 &  ' *   originally released GALAHAD Version 3.3. ', &
 &  A, 1X, I0, A2, 1X, A4 )" ) &
     TRIM( package_upper ), TRIM( months( imonth) ), iday, days( iday ), year
  WRITE( h_unit, "( ' *', /, &
 &  ' *  For full documentation, see', /, &
 &  ' *   http://galahad.rl.ac.uk/galahad-www/specs.html', /, &
 &  ' */', /, &
 &  '', /, &
 &  '/*! \mainpage GALAHAD C package ', A, /, &
 &  ' ', /, &
 &  '  \section ', A, '_intro Introduction', /, &
 &  '', /, &
 &  '  \subsection ', A, '_purpose Purpose', /, &
 &  '', /, &
 &  '  \subsection ', A, '_authors Authors', /, &
 &  '  N. I. M. Gould, STFC-Rutherford Appleton Laboratory, England.', /, &
 &  '', /, &
 &  '  C interface, additionally J. Fowkes,', &
 &  ' STFC-Rutherford Appleton Laboratory.', /, &
 &  '', /, &
 &  '  \subsection ', A, '_date Originally released', /, &
 &  '', /, &
 &  '  \subsection ', A, '_terminology Terminology', /, &
 &  '', /, &
 &  '  \subsection ', A, '_method Method', /, &
 &  '', /, &
 &  '  \subsection ', A, '_references Reference', /, &
 &  '', /, &
 &  '  \subsection ', A, '_call_order Call order', /, &
 &  ' */', /, &
 &  '', /, &
 &  '#ifdef __cplusplus', /, &
 &  'extern ""C"" {', /, &
 &  '#else', /, &
 &  '#include <stdbool.h>', /, &
 &  '#endif', /, &
 &  '', /, &
 &  '// include guard', /, &
 &  '#ifndef GALAHAD_', A, '_H ', /, &
 &  '#define GALAHAD_', A, '_H', /, &
 &  '', /, &
 &  '// precision', /, &
 &  '#include ""galahad_precision.h""' &
 &   )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ), &
          TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ), &
          TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ), &
          TRIM( package_upper ), TRIM( package_upper )

!  set up and initialize the C interface file, along with an associated 
!  scratch file

  package_ciface = TRIM( package_lower ) // '_ciface.f90'
  INQUIRE( FILE = package_ciface, EXIST = file_exists )
  IF ( file_exists ) THEN
    OPEN( UNIT = ciface_unit, FILE = package_ciface, FORM = 'FORMATTED',       &
          STATUS = 'OLD', IOSTAT = ios )
  ELSE
    OPEN( UNIT = ciface_unit, FILE = package_ciface, FORM = 'FORMATTED',       &
          STATUS = 'NEW', IOSTAT = ios )
  END IF
  IF ( ios /= 0 ) THEN
    WRITE( out, "( ' IOSTAT = ', I0, ' when opening file ', A,                 &
   &   '. Stopping')" ) ios, TRIM( package_ciface )
    STOP
  END IF

!  open the scratch file

  OPEN( UNIT = ciface_scratch, FORM = 'FORMATTED',                             &
        STATUS = 'SCRATCH', IOSTAT = ios )
  IF ( ios /= 0 ) THEN
    WRITE( out, "( ' IOSTAT = ', I0, ' when opening ciface scratch file ',     &
   &   '. Stopping')" ) ios
    STOP
  END IF

!  insert leading data

  WRITE( ciface_unit, "(                                                       &
 &  '! THIS VERSION: GALAHAD 3.3 - ', A2, '/', A2, '/', A4, ' AT ',            &
 &  A2, ':', A2, ' GMT.' )" ) day, month, year, hour, minute
  WRITE( ciface_unit, "(                                                       &
 &  '', /, &
 &  '!-*-*-*-*-*-*-*-  G A L A H A D _ ', 10( 1X, A1 ) )", ADVANCE = 'NO' )    &
   ( package_upper( i : i ), i = 1, len_name )   
  WRITE( ciface_unit, "(                                                       &
 &  '   C   I N T E R F A C E  -*-*-*-*-*-', /, &
 &  '', /, &
 &  '!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions', /, &
 &  '!  Principal authors: Jaroslav Fowkes & Nick Gould', /, &
 &  '', /, &
 &  '!  History -', /, &
 &  '!    originally released GALAHAD Version 3.3. ', &
 &  A, 1X, I0, A2, 1X, A4 )" ) &
     TRIM( months( imonth) ), iday, days( iday ), year
  WRITE( ciface_unit, "(                                                       &
 &  '', /, &
 &  '!  For full documentation, see', /, &
 &  '!   http://galahad.rl.ac.uk/galahad-www/specs.html', /, &
 &  '', /, &
 &  '  MODULE GALAHAD_', A, '_double_ciface', /, &
 &  '    USE iso_c_binding', /, &
 &  '    USE GALAHAD_common_ciface' &
 &   )" ) TRIM( package_upper )

!  search the package file for derived type definitions

  ndeps = 0
  DO
    READ( package_unit, "( A80 )", iostat = ios ) line
    IF ( ios /= 0 ) EXIT
    line = ADJUSTL( line )
    IF ( line( 1 : 16 ) /= 'TYPE, PUBLIC :: ' ) CYCLE
    DO i = 1, 16 + len_name + 1
      line( i : i ) = ' '
    END DO
    line = ADJUSTL( line )
!   WRITE( out, * ) TRIM( line )

!  control section

    IF ( line( 1 : 7 ) == "control" .OR.                                       &
         line( 1 : 18 ) == "subproblem_control" ) THEN
      dtype = 'control'
      control_type = .TRUE.

!  time section

    ELSE IF ( line( 1 : 4 ) == "time" ) THEN
      dtype = 'time   '
      time_type = .TRUE.

!  inform section

    ELSE IF ( line( 1 : 6 ) == "inform" .OR.                                   &
              line( 1 : 17 ) == "subproblem_inform" ) THEN
      dtype = 'inform '
      inform_type = .TRUE.
    ELSE
      CYCLE
    END IF
    WRITE( h_scratch, "(                                                       &
   &  '', /,  &
   & '/**', /, &
   & ' * ', A, ' derived type as a C struct', /, &
   & ' */', /, &
   & 'struct ', A, '_', A, '_type { ' &
   &   )" ) TRIM( dtype ), TRIM( package_lower ), TRIM( dtype )
    WRITE( ciface_scratch, "( '', /,                                           &
   &  '    TYPE, BIND( C ) :: ', A, '_', A, '_type' )" )                       &
      TRIM( package_lower ), TRIM( dtype )

    IF ( dtype == 'control' ) THEN
      WRITE( h_scratch, "(                                                     &
     &  '', /, &
     &  '    /// \brief', /, &
     &  '    /// use C or Fortran sparse matrix indexing', /, &
     &  '    bool f_indexing;' &
   & )" )  
      WRITE( ciface_scratch,                                                   &
        "( '      LOGICAL ( KIND = C_BOOL ) :: f_indexing' )" )
    END IF

!  obtain the guts of the relevant type definition

    ni = 0 ; nr = 0; ns = 0; nl = 0; nc = 0; nt = 0
    blank = .TRUE.
    skip_next = .FALSE.
    comment_start = .TRUE.
    first_component = .TRUE.
    DO
      READ( package_unit, "( A80 )", iostat = ios ) line
      IF ( ios /= 0 ) EXIT
      IF ( skip_next ) THEN
        skip_next = .FALSE.
        CYCLE
      END IF
      line = ADJUSTL( line )
      IF ( line( 1 : 8 ) == 'END TYPE' ) THEN
        WRITE(  h_scratch, "( '};' )" )
        WRITE(  ciface_scratch, "( '    END TYPE ', A, '_', A, '_type' )" )    &
          TRIM( package_lower ), TRIM( dtype )
        SELECT CASE( TRIM( dtype ) )
        CASE ( 'control' )
          control_ni = ni ; control_nr = nr ; control_ns = ns
          control_nl = nl ; control_nc = nc ; control_nt = nt
        CASE ( 'time' )
          time_nr = nr ; time_ns = ns
        CASE ( 'inform' )
          inform_ni = ni ; inform_nr = nr ; inform_ns = ns
          inform_nl = nl ; inform_nc = nc ; inform_nt = nt
        END SELECT
        EXIT
      END IF

! this is a blank line

      IF ( LEN_TRIM( line ) == 0 ) THEN
        first_component = .TRUE.

! this is a comment line

      ELSE IF ( line( 1 : 1 ) == '!' ) THEN
        first_component = .TRUE.
        IF ( blank ) THEN
          WRITE( h_scratch, "( '' )" )
          blank = .FALSE.
        END IF
        line( 1 : 1 ) = ' '
        line = ADJUSTL( line )
        IF ( comment_start ) THEN
          WRITE( h_scratch, "( '    /// \brief' )" )
          comment_start = .FALSE.
        END IF
        DO i = 1, LEN_TRIM( line ) ! convert type deliminter % to struct .
          IF ( line( i : i ) == '%' ) line( i : i ) = '.'
        END DO
        WRITE( h_scratch, "( '    /// ', A )" ) TRIM( line )

! this is a component line

      ELSE
        comment_start = .TRUE.

! check to see if the component is an array

        CALL contains_dimension( line, past ) 
        IF ( past > 0 ) THEN ! if so, find its dimension
          READ( line( past : 80 ), * ) dim
        END IF

! if the line is continued, skip the next one

        blank = .TRUE.
        l = SCAN( line, "&", BACK = .TRUE. )
        IF ( l > 0 ) THEN
          line( l : l ) = ' '
          skip_next = .TRUE.
        END IF

!  determine the kind of component

        IF ( line( 1 : 7 ) == 'INTEGER' ) THEN
          c = 'i'
        ELSE IF ( line( 1 : 18 ) == 'REAL ( KIND = wp )' ) THEN
          c = 'r'
        ELSE IF ( line( 1 : 18 ) == 'REAL ( KIND = dp )' ) THEN
          c = 'r'
        ELSE IF ( line( 1 : 18 ) == 'REAL ( KIND = sp )' ) THEN
          c = 's'
        ELSE IF ( line( 1 : 4 ) == 'REAL' ) THEN
          c = 's'
        ELSE IF ( line( 1 : 7 ) == 'LOGICAL' ) THEN
          c = 'l'
        ELSE IF ( line( 1 : 9 ) == 'CHARACTER' ) THEN
          c = 'c'
          lineb = line
          l = SCAN( lineb, "=" ) ! find the length of the string
          IF ( l > 0 ) THEN
            DO i = 1, l
              lineb( i : i ) = ' '
            END DO
            lineb = ADJUSTL( lineb )
            READ( lineb, * ) len
          ELSE
            len = 1
          END IF
        ELSE IF ( line( 1 : 4 ) == 'TYPE' ) THEN
          c = 't'
        ELSE
          WRITE( out, "( ' no idea what kind of variable this is! ')" )
          WRITE( out, * ) TRIM( line )
          STOP
        END IF
        l = SCAN( line, ":", BACK = .TRUE. )
!       write( out,"( ' position ', I0 )" ) l
        DO i = 1, l
          line( i : i ) = ' '
        END DO
        line = ADJUSTL( line )
        l = SCAN( line, " " )
        DO i = l, 80
          line( i : i ) = ' '
        END DO
!       WRITE( out, * ) TRIM( line )

!  if there are multiple definitions before the next comment, record the first

        IF ( first_component ) THEN
          first_component = .FALSE.
          first_line = line
        ELSE 
          WRITE( h_scratch, "( '    /// see ', A )" ) TRIM( first_line )
        END IF 

        SELECT CASE( c )
        CASE ( 'i' ) ! integer
          IF ( past > 0 ) THEN
            WRITE( h_scratch, "( '    int ', A, '[', I0, '];' )" )             &
              TRIM( line ), dim
          ELSE
            WRITE( h_scratch, "( '    int ', A, ';' )" ) TRIM( line )
          END IF
          WRITE( ciface_scratch,                                               &
        "( '      INTEGER ( KIND = C_INT ) :: ', A )" ) TRIM( line )
          ni = ni + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_i( ni ) = line
          CASE ( 'inform' )
            inform_i( ni ) = line
          END SELECT
        CASE ( 'r' )
          IF ( past > 0 ) THEN
            WRITE( h_scratch, "( '    real_wp_ ', A, '[', I0, '];' )" )        &
               TRIM( line ), dim
          ELSE
            WRITE( h_scratch, "( '    real_wp_ ', A, ';' )" ) TRIM( line )
          END IF
          WRITE( ciface_scratch,                                               &
        "( '      REAL ( KIND = wp ) :: ', A )" ) TRIM( line )
          nr = nr + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_r( nr ) = line
          CASE ( 'time' )
            time_r( nr ) = line
          CASE ( 'inform' )
            inform_r( nr ) = line
          END SELECT
        CASE ( 's' )
          IF ( past > 0 ) THEN
            WRITE( h_scratch, "( '    real_sp_ ', A, '[', I0, '];' )" )        &
              TRIM( line ), dim
          ELSE
            WRITE( h_scratch, "( '    real_sp_ ', A, ';' )" ) TRIM( line )
          END IF
          WRITE( ciface_scratch,                                               &
        "( '      REAL ( KIND = sp ) :: ', A )" ) TRIM( line )
          ns = ns + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_s( ns ) = line
          CASE ( 'time' )
            time_s( ns ) = line
          CASE ( 'inform' )
            inform_s( ns ) = line
          END SELECT
        CASE ( 'l' )
          IF ( past > 0 ) THEN
            WRITE( h_scratch, "( '    bool ', A, '[', I0, '];' )" )            &
              TRIM( line ), dim
          ELSE
            WRITE( h_scratch, "( '    bool ', A, ';' )" ) TRIM( line )
          END IF
          WRITE( ciface_scratch,                                               &
        "( '      LOGICAL ( KIND = C_BOOL ) :: ', A )" ) TRIM( line )
          nl = nl + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_l( nl ) = line
          CASE ( 'inform' )
            inform_l( nl ) = line
          END SELECT
        CASE ( 'c' )
          WRITE( h_scratch, "( '    char ', A, '[', I0, '];' )" )              &
            TRIM( line ), len + 1
          WRITE( ciface_scratch, "( '      CHARACTER ( KIND = C_CHAR ), ',     &
        &  'DIMENSION( ', I0, ' ) :: ', A )" ) len + 1, TRIM( line )
          nc = nc + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_c( nc ) = line
            control_clen( nc ) = len
          CASE ( 'inform' )
            inform_c( nc ) = line
            inform_clen( nc ) = len
          END SELECT
        CASE ( 't' )
          l = SCAN( line, "_" )
          IF ( l > 0 ) THEN
            DO i = l, 80
              line( i : i ) = ' '
            END DO
            IF ( ndeps == 0 ) THEN
              ndeps = 1
              depends( ndeps ) = line
            ELSE
              i = COUNT( depends( : ndeps ) == line )
              IF ( i == 0 ) THEN
                ndeps = ndeps + 1
                depends( ndeps ) = line
              END IF
            END IF
          END IF
          len_tname = LEN_TRIM( line )
          DO i = 1, len_tname
            st1 = line( i : i ) 
            CALL STRING_lower_scalar( st1 )
            line_lower( i : i ) = st1
          END DO
          IF ( line_lower( 1 : len_tname ) == 'time' ) THEN
            WRITE( h_scratch, "( '    struct ', A, '_time_type time;' )" )     &
              TRIM( package_lower )
            WRITE( ciface_scratch,                                             &
              "( '      TYPE ( ', A, '_', A, '_type ) :: ', A )" )             &
              TRIM( package_lower ), line_lower( 1 : len_tname ),              &
              line_lower( 1 : len_tname )
          ELSE
            WRITE( h_scratch, "( '    struct ', A, '_', A, '_type ',           &
           &  A, '_', A, ';' )" )                                              &
              line_lower( 1 : len_tname ), TRIM( dtype ),                      &
              line_lower( 1 : len_tname ), TRIM( dtype )
            WRITE( ciface_scratch,                                             &
              "( '      TYPE ( ', A, '_', A, '_type ) :: ', A, '_', A )" )     &
                line_lower( 1 : len_tname ), TRIM( dtype ),                    &
                line_lower( 1 : len_tname ), TRIM( dtype )
          END IF
          nt = nt + 1
          SELECT CASE( TRIM( dtype ) )
          CASE ( 'control' )
            control_t( nt ) = line
          CASE ( 'inform' )
            inform_t( nt ) = line
          END SELECT
        END SELECT
      END IF
    END DO

  END DO

!  add dependency information to output files

  IF ( ndeps > 0 ) THEN
    WRITE( h_unit, "( '', /, '// required packages' )" )
    DO l = 1, ndeps
      len_dname = LEN_TRIM( depends( l ) )
      DO i = 1, len_dname
        st1 = depends( l )( i : i ) 
        CALL STRING_lower_scalar( st1 )
        depends_lower( i : i ) = st1
      END DO
      WRITE( h_unit, "( '#include ""', A, '.h""' )" )                          &
         depends_lower( 1 : len_dname )
    END DO
  END IF

  WRITE( ciface_unit, "( '    USE GALAHAD_', A, '_double, ONLY: &' )" )        &
    TRIM( package_upper )
  IF ( control_type )                                                          &
    WRITE( ciface_unit, "( '        f_', A, '_control_type => ', A,            &
   & '_control_type, &' )" ) TRIM( package_lower ), TRIM( package_upper )
  IF ( time_type )                                                             &
    WRITE( ciface_unit, "( '        f_', A, '_time_type => ', A,               &
   & '_time_type, &' )" ) TRIM( package_lower ), TRIM( package_upper )
  IF ( inform_type )                                                           &
    WRITE( ciface_unit, "( '        f_', A, '_inform_type => ', A,             &
   & '_inform_type, &' )" ) TRIM( package_lower ), TRIM( package_upper )
  WRITE( ciface_unit, "( &
   & '        f_', A, '_full_data_type => ', A, '_full_data_type, &', /, &
   & '        f_', A, '_initialize => ', A, '_initialize, &', /, &
   & '        f_', A, '_read_specfile => ', A, '_read_specfile, &', /, &
   & '        f_', A, '_import => ', A, '_import, &', /, &
   & '        f_', A, '_reset_control => ', A, '_reset_control, &', /, &
   & '        f_', A, '_information => ', A, '_information, &', /, &
   & '        f_', A, '_terminate => ', A, '_terminate' )" )                   &
    TRIM( package_lower ), TRIM( package_upper ),                              &
    TRIM( package_lower ), TRIM( package_upper ),                              &
    TRIM( package_lower ), TRIM( package_upper ),                              &
    TRIM( package_lower ), TRIM( package_upper ),                              &
    TRIM( package_lower ), TRIM( package_upper ),                              &
    TRIM( package_lower ), TRIM( package_upper )

  DO l = 1, ndeps
    len_dname = LEN_TRIM( depends( l ) )
    DO i = 1, len_dname
      st1 = depends( l )( i : i ) 
      CALL STRING_lower_scalar( st1 )
      depends_lower( i : i ) = st1
      CALL STRING_upper_scalar( st1 )
      depends_upper( i : i ) = st1
    END DO
    WRITE( ciface_unit, "( '', /,                                              &
   &  '    USE GALAHAD_', A, '_double_ciface, ONLY: &', /, &
   &  '        ', A, '_inform_type, &', /, &
   &  '        ', A, '_control_type, &', /, &
   &  '        copy_', A, '_inform_in => copy_inform_in, &' , /, &
   &  '        copy_', A, '_inform_out => copy_inform_out, &', /, &
   &  '        copy_', A, '_control_in => copy_control_in, &', /, &
   &  '        copy_', A, '_control_out => copy_control_out' &
   & ) " ) depends_upper( 1 : len_dname ), &
    depends_lower( 1 : len_dname ), depends_lower( 1 : len_dname ), &
    depends_lower( 1 : len_dname ), depends_lower( 1 : len_dname ), &
    depends_lower( 1 : len_dname ), depends_lower( 1 : len_dname )
  END DO

  WRITE( ciface_unit, "( '', /,                                                &
 &  '    IMPLICIT NONE', /, &
 &  '', /, &
 &  '!--------------------', /, &
 &  '!   P r e c i s i o n', /, &
 &  '!--------------------', /, &
 &  '', /, &
 &  '    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision', /, &
 &  '    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision', /, &
 &  '', /, &
 &  '!-------------------------------------------------', /, &
 &  '!  D e r i v e d   t y p e   d e f i n i t i o n s', /, &
 &  '!-------------------------------------------------'  )" )

!  copy scratch file data into the true output files

  REWIND( UNIT = h_scratch )
  DO
    READ( h_scratch, "( A80 )", iostat = ios ) line
    IF ( ios /= 0 ) EXIT
    WRITE( h_unit, "( A )" ) TRIM( line )
  END DO
  CLOSE( UNIT = h_scratch )

  REWIND( UNIT = ciface_scratch )
  DO
    READ( ciface_scratch, "( A80 )", iostat = ios ) line
    IF ( ios /= 0 ) EXIT
    WRITE( ciface_unit, "( A )" ) TRIM( line )
  END DO
  CLOSE( UNIT = ciface_scratch )

!  finish writing the output files. Start with the h file

  WRITE( h_unit, "(                                                            &
 & '', /,  &
 &  '// *-*-*-*-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )            &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ I N I T I A L I Z E    -*-*-*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_initialize( void **data, ', /, &
 & '                     struct ', A, '_control_type *control,', /, &
 & '                     struct ', A, '_inform_type *inform );', /, &
 & '', /, &
 &  '/*!<', /, &
 &  ' Set default control values and initialize private data', /, &
 &  '', /, &
 &  '  @param[in,out] data  holds private internal data', /, &
 &  '  @param[out] control  is a struct containing control information ', /, &
 &  '              (see ', A, '_control_type)', /, &
 &  '  @param[out] inform   is a struct containing output information', /, &
 &  '              (see ', A, '_inform_type) ', /, &
 &  '*/' &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),   &
        TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 &  '// *-*-*-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )              &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ R E A D _ S P E C F I L E   -*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_read_specfile( struct ', A, '_control_type *control, ', /, &
 & '                        const char specfile[] );', /, &
 & '', /, &
 &  '/*!<', /, &
 &  '  Read the content of a specification file, and assign values', &   
 &  ' associated ', /, &
 &  '  with given keywords to the corresponding control parameters', /, &
 &  '', /, &
 &  '  @param[in,out]  control  is a struct containing control', & 
 &  ' information ', /, &
 &  '              (see ', A, '_control_type)', /, &
 &  '  @param[in]  specfile  is a character string containing the name of', /, &
 &  '              the specification file', /, &
 &  '*/'  &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 &  '// *-*-*-*-*-*-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )        &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ I M P O R T   -*-*-*-*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_import( struct ', A, '_control_type *control,', /, &
 & '                 void **data,', /, &
 & '                 int *status );', /, &
 & '', /, &
 &  '/*!<', /, &
 &  ' Import problem data into internal storage prior to solution. ', /, &
 &  '', /, &
 &  ' @param[in] control is a struct whose members provide control', /, &
 &  '  paramters for the remaining prcedures (see ', A, '_control_type)', /, &
 &  '', /, &
 &  ' @param[in,out] data holds private internal data', /, &
 &  '', /, &
 &  ' @param[in,out] status is a scalar variable of type int, that gives', /, &
 &  '    the exit status from the package. Possible values are:', /, &
 &  '  \li  1. The import was succesful, and the package is ready for', &    
 &  ' the solve phase', /, &
 &  '  \li -1. An allocation error occurred. A message indicating the ', /, &
 &  '       offending array is written on unit control.error, and the ', /, &
 &  '       returned allocation status and a string containing the name ', /, &
 &  '       of the offending array are held in inform.alloc_status and ', /, &
 &  '       inform.bad_alloc respectively.', /, &
 &  '  \li -2. A deallocation error occurred.  A message indicating the ', /, &
 &  '       offending array is written on unit control.error and the ', /, &
 &  '       returned allocation status and a string containing the', /, &
 &  '       name of the offending array are held in ', /, &
 &  '       inform.alloc_status and inform.bad_alloc respectively.', /, &
 &  '  \li -3. The restriction n > 0 or requirement that type contains', /, &
 &  '       its relevant string ''dense'', ''coordinate'', ', &     
 &  '''sparse_by_rows'',', /, &
 &  '       ''diagonal'' or ''absent'' has been violated.', /, &
 &  '*/'  &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 & '// *-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )                   &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ R E S E T _ C O N T R O L   -*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_reset_control( struct ', A, '_control_type *control,', /, &
 & '                 void **data,', /, &
 & '                 int *status );', /, &
 & '', /, &
 &  '/*!<', /, &
 &  ' Reset control parameters after import if required. ', /, &
 &  '', /, &
 &  ' @param[in] control is a struct whose members provide control', /, &
 &  '  paramters for the remaining prcedures (see ', A, '_control_type)', /, &
 &  '', /, &
 &  ' @param[in,out] data holds private internal data', /, &
 &  '', /, &
 &  ' @param[in,out] status is a scalar variable of type int, that gives', /, &
 &  '    the exit status from the package. Possible values are:', /, &
 &  '  \li  1. The import was succesful, and the package is ready for', &    
 &  ' the solve phase', /, &
 &  '*/'  &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 & '// *-*-*-*-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )             &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ I N F O R M A T I O N   -*-*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_information( void **data,', /, &
 & '                      struct ', A, '_inform_type *inform,', /, &
 & '                      int *status );', /, &
 & '', /, &
 &  '/*!<', /, &
 &  '  Provides output information', /, &
 &  '', /, &
 &  '  @param[in,out] data  holds private internal data', /, &
 &  '', /, &
 &  '  @param[out] inform   is a struct containing output information', /, &
 &  '              (see ', A, '_inform_type) ', /, &
 &  '', /, &
 &  '  @param[out] status is a scalar variable of type int, that gives', /, &
 &  '              the exit status from the package.', /, & 
 &  '              Possible values are (currently):', /, &
 &  '  \li  0. The values were recorded succesfully', /, &
 &  '*/' &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 &  '// *-*-*-*-*-*-*-*-*-*-   ', 10( 1X, A1 ) )", ADVANCE = 'NO' )            &
   ( package_upper( i : i ), i = 1, len_name )   

  WRITE( h_unit, "(                                                            &
 & ' _ T E R M I N A T E   -*-*-*-*-*-*-*-*-*-*', /, &
 & '', /, &
 & 'void ', A, '_terminate( void **data, ', /, &
 & '                    struct ', A, '_control_type *control, ', /, &
 & '                    struct ', A, '_inform_type *inform );', /, &
 & '', /, &
 & '/*!<', /, &
 & '  Deallocate all internal private storage', /, &
 & '', /, &
 & '  @param[in,out] data  holds private internal data', /, &
 & '', /, &
 & '  @param[out] control  is a struct containing control information ', /, &
 & '              (see ', A, '_control_type)', /, &
 & '', /, &
 & '  @param[out] inform   is a struct containing output information', /, &
 & '              (see ', A, '_inform_type)', /, &
 & ' */' &
 & )" ) TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),   &
        TRIM( package_lower ), TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 & '', /, &
 &  '/** \example ', A, 't.c', /, &
 &  '   This is an example of how to use the package.\n', /, &
 & ' */' &
 & )" ) TRIM( package_lower )

  WRITE( h_unit, "(                                                            &
 &  '', /, &
 &  '// end include guard', /, &
 &  '#endif', /, &
 &  '', /, &
 &  '#ifdef __cplusplus', /, &
 &  '} /* extern ""C"" */', /, &
 &  '#endif' )" )
  CLOSE( UNIT = h_unit )
  WRITE( out, "( 1X, A, '.h constructed' )" ) TRIM( package_lower )

!  now finish writing the ciface file

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!----------------------', /, &
 & '!   P r o c e d u r e s', /, &
 & '!----------------------', /, &
 & '', /, &
 & '  CONTAINS', /, &
 & '' &
 )" )

  IF ( control_type ) THEN
    WRITE( ciface_unit, "(                                                     &
 & '!  copy C control parameters to fortran', /, &
 & '', /, &
 & '    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) ', /, &
 & '    TYPE ( ', A, '_control_type ), INTENT( IN ) :: ccontrol', /, &
 & '    TYPE ( f_', A, '_control_type ), INTENT( OUT ) :: fcontrol', /, &
 & '    LOGICAL, optional, INTENT( OUT ) :: f_indexing', /, &
 & '    INTEGER :: i', /, &
 & '    ', /, &
 & '    ! C or Fortran sparse matrix indexing', /, &
 & '    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( control_ni > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Integers' )" )
      DO i = 1, control_ni
        WRITE( ciface_unit, "('    fcontrol%', A, ' = ccontrol%', A )" )       &
          TRIM( control_i( i ) ), TRIM( control_i( i ) )
      END DO
    END IF  
    IF ( control_nr + control_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, control_nr
        WRITE( ciface_unit, "('    fcontrol%', A, ' = ccontrol%', A )" )       &
          TRIM( control_r( i ) ), TRIM( control_r( i ) )
      END DO
      DO i = 1, control_ns
        WRITE( ciface_unit, "('    fcontrol%', A, ' = ccontrol%', A )" )       &
          TRIM( control_s( i ) ), TRIM( control_s( i ) )
      END DO
    END IF  
    IF ( control_nl > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Logicals' )" )
      DO i = 1, control_nl
        WRITE( ciface_unit, "('    fcontrol%', A, ' = ccontrol%', A )" )       &
          TRIM( control_l( i ) ), TRIM( control_l( i ) )
      END DO
    END IF  
    IF ( control_nt > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Derived types' )" )
      DO i = 1, control_nt
        len_name = LEN_TRIM( control_t( i ) ) 
        DO l = 1, len_name
          st1 = control_t( i )( l : l ) 
          CALL STRING_lower_scalar( st1 )
          depends_lower( l : l ) = st1
        END DO
        WRITE( ciface_unit, "('    CALL copy_', A, '_control_in( ccontrol%',   &
       &  A, '_control, fcontrol%', A, '_control )' )" )                       &
          depends_lower( 1 : len_name ), depends_lower( 1 : len_name ),        &
          depends_lower( 1 : len_name )
      END DO
    END IF  
    IF ( control_nc > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Strings' )" )
      DO i = 1, control_nc
        WRITE( ciface_unit, "( &
      & '    DO i = 1, ', I0, /, &
      & '      IF ( ccontrol%', A, '( i ) == C_NULL_CHAR ) EXIT', /, &
      & '      fcontrol%', A, '( i : i ) = ccontrol%', A, '( i )', /, &
      & '    END DO' &
      & )" ) control_clen( i ) + 1, TRIM( control_c( i ) ),                    &
         TRIM( control_c( i ) ), TRIM( control_c( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_control_in', / '' &
   )" )

    WRITE( ciface_unit, "(                                                     &
 & '!  copy fortran control parameters to C', /, &
 & '', /, &
 & '    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) ', /, &
 & '    TYPE ( f_', A, '_control_type ), INTENT( IN ) :: fcontrol', /, &
 & '    TYPE ( ', A, '_control_type ), INTENT( OUT ) :: ccontrol', /, &
 & '    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing', /, &
 & '    INTEGER :: i', /, &
 & '    ', /, &
 & '    ! C or Fortran sparse matrix indexing', /, &
 & '    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( control_ni > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Integers' )" )
      DO i = 1, control_ni
        WRITE( ciface_unit, "('    ccontrol%', A, ' = fcontrol%', A )" )       &
          TRIM( control_i( i ) ), TRIM( control_i( i ) )
      END DO
    END IF  
    IF ( control_nr + control_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, control_nr
        WRITE( ciface_unit, "('    ccontrol%', A, ' = fcontrol%', A )" )       &
          TRIM( control_r( i ) ), TRIM( control_r( i ) )
      END DO
      DO i = 1, control_ns
        WRITE( ciface_unit, "('    ccontrol%', A, ' = fcontrol%', A )" )       &
          TRIM( control_s( i ) ), TRIM( control_s( i ) )
      END DO
    END IF  
    IF ( control_nl > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Logicals' )" )
      DO i = 1, control_nl
        WRITE( ciface_unit, "('    ccontrol%', A, ' = fcontrol%', A )" )       &
          TRIM( control_l( i ) ), TRIM( control_l( i ) )
      END DO
    END IF  
    IF ( control_nt > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Derived types' )" )
      DO i = 1, control_nt
        len_name = LEN_TRIM( control_t( i ) ) 
        DO l = 1, len_name
          st1 = control_t( i )( l : l ) 
          CALL STRING_lower_scalar( st1 )
          depends_lower( l : l ) = st1
        END DO
        WRITE( ciface_unit, "('    CALL copy_', A, '_control_out( fcontrol%',  &
       &  A, '_control, ccontrol%', A, '_control )' )" )                       &
          depends_lower( 1 : len_name ), depends_lower( 1 : len_name ),        &
          depends_lower( 1 : len_name )
      END DO
    END IF  
    IF ( control_nc > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Strings' )" )
      DO i = 1, control_nc
        WRITE( ciface_unit, "( &
      & '    DO i = 1, LEN( fcontrol%', A, ' )', /, &
      & '      ccontrol%', A, '( i ) = fcontrol%', A, '( i : i )', /, &
      & '    END DO', /, &
      & '    ccontrol%', A, '( LEN( fcontrol%', A, ' ) + 1 ) = C_NULL_CHAR' &
      & )" ) TRIM( control_c( i ) ), TRIM( control_c( i ) ),                   &
         TRIM( control_c( i ) ), TRIM( control_c( i ) ), TRIM( control_c( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_control_out', /, '' &
   )" )
  END IF

  IF ( time_type ) THEN
    WRITE( ciface_unit, "(                                                     &
 & '!  copy C time parameters to fortran', /, &
 & '', /, &
 & '    SUBROUTINE copy_time_in( ctime, ftime ) ', /, &
 & '    TYPE ( ', A, '_time_type ), INTENT( IN ) :: ctime', /, &
 & '    TYPE ( f_', A, '_time_type ), INTENT( OUT ) :: ftime' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( time_nr + time_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, time_nr
        WRITE( ciface_unit, "('    ftime%', A, ' = ctime%', A )" )       &
          TRIM( time_r( i ) ), TRIM( time_r( i ) )
      END DO
      DO i = 1, time_ns
        WRITE( ciface_unit, "('    ftime%', A, ' = ctime%', A )" )       &
          TRIM( time_s( i ) ), TRIM( time_s( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_time_in', / '' &
   )" )

    WRITE( ciface_unit, "(                                                     &
 & '!  copy fortran time parameters to C', /, &
 & '', /, &
 & '    SUBROUTINE copy_time_out( ftime, ctime ) ', /, &
 & '    TYPE ( f_', A, '_time_type ), INTENT( IN ) :: ftime', /, &
 & '    TYPE ( ', A, '_time_type ), INTENT( OUT ) :: ctime' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( time_nr + time_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, time_nr
        WRITE( ciface_unit, "('    ctime%', A, ' = ftime%', A )" )       &
          TRIM( time_r( i ) ), TRIM( time_r( i ) )
      END DO
      DO i = 1, time_ns
        WRITE( ciface_unit, "('    ctime%', A, ' = ftime%', A )" )       &
          TRIM( time_s( i ) ), TRIM( time_s( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_time_out', /, '' &
   )" )
  END IF

  IF ( inform_type ) THEN
    WRITE( ciface_unit, "(                                                     &
 & '!  copy C inform parameters to fortran', /, &
 & '', /, &
 & '    SUBROUTINE copy_inform_in( cinform, finform ) ', /, &
 & '    TYPE ( ', A, '_inform_type ), INTENT( IN ) :: cinform', /, &
 & '    TYPE ( f_', A, '_inform_type ), INTENT( OUT ) :: finform', /, &
 & '    INTEGER :: i' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( inform_ni > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Integers' )" )
      DO i = 1, inform_ni
        WRITE( ciface_unit, "('    finform%', A, ' = cinform%', A )" )       &
          TRIM( inform_i( i ) ), TRIM( inform_i( i ) )
      END DO
    END IF  
    IF ( inform_nr + inform_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, inform_nr
        WRITE( ciface_unit, "('    finform%', A, ' = cinform%', A )" )       &
          TRIM( inform_r( i ) ), TRIM( inform_r( i ) )
      END DO
      DO i = 1, inform_ns
        WRITE( ciface_unit, "('    finform%', A, ' = cinform%', A )" )       &
          TRIM( inform_s( i ) ), TRIM( inform_s( i ) )
      END DO
    END IF  
    IF ( inform_nl > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Logicals' )" )
      DO i = 1, inform_nl
        WRITE( ciface_unit, "('    finform%', A, ' = cinform%', A )" )       &
          TRIM( inform_l( i ) ), TRIM( inform_l( i ) )
      END DO
    END IF  
    IF ( inform_nt > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Derived types' )" )
      DO i = 1, inform_nt
        len_name = LEN_TRIM( inform_t( i ) ) 
        DO l = 1, len_name
          st1 = inform_t( i )( l : l ) 
          CALL STRING_lower_scalar( st1 )
          depends_lower( l : l ) = st1
        END DO
        IF ( depends_lower( 1 : len_name ) == 'time' ) THEN
          WRITE( ciface_unit,                                                  &
            "('    CALL copy_time_in( cinform%time, finform%time )' )" )
        ELSE
          WRITE( ciface_unit, "('    CALL copy_', A, '_inform_in( cinform%',   &
         &  A, '_inform, finform%', A, '_inform )' )" )                       &
            depends_lower( 1 : len_name ), depends_lower( 1 : len_name ),      &
            depends_lower( 1 : len_name )
        END IF
      END DO
    END IF  
    IF ( inform_nc > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Strings' )" )
      DO i = 1, inform_nc
        WRITE( ciface_unit, "( &
      & '    DO i = 1, ', I0, /, &
      & '      IF ( cinform%', A, '( i ) == C_NULL_CHAR ) EXIT', /, &
      & '      finform%', A, '( i : i ) = cinform%', A, '( i )', /, &
      & '    END DO' &
      & )" ) inform_clen( i ) + 1, TRIM( inform_c( i ) ),                    &
         TRIM( inform_c( i ) ), TRIM( inform_c( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_inform_in', / '' &
   )" )

    WRITE( ciface_unit, "(                                                     &
 & '!  copy fortran inform parameters to C', /, &
 & '', /, &
 & '    SUBROUTINE copy_inform_out( finform, cinform ) ', /, &
 & '    TYPE ( f_', A, '_inform_type ), INTENT( IN ) :: finform', /, &
 & '    TYPE ( ', A, '_inform_type ), INTENT( OUT ) :: cinform', /, &
 & '    INTEGER :: i' &
   )" ) TRIM( package_lower ), TRIM( package_lower )
    IF ( inform_ni > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Integers' )" )
      DO i = 1, inform_ni
        WRITE( ciface_unit, "('    cinform%', A, ' = finform%', A )" )       &
          TRIM( inform_i( i ) ), TRIM( inform_i( i ) )
      END DO
    END IF  
    IF ( inform_nr + inform_ns > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Reals' )" )
      DO i = 1, inform_nr
        WRITE( ciface_unit, "('    cinform%', A, ' = finform%', A )" )       &
          TRIM( inform_r( i ) ), TRIM( inform_r( i ) )
      END DO
      DO i = 1, inform_ns
        WRITE( ciface_unit, "('    cinform%', A, ' = finform%', A )" )       &
          TRIM( inform_s( i ) ), TRIM( inform_s( i ) )
      END DO
    END IF  
    IF ( inform_nl > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Logicals' )" )
      DO i = 1, inform_nl
        WRITE( ciface_unit, "('    cinform%', A, ' = finform%', A )" )       &
          TRIM( inform_l( i ) ), TRIM( inform_l( i ) )
      END DO
    END IF  
    IF ( inform_nt > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Derived types' )" )
      DO i = 1, inform_nt
        len_name = LEN_TRIM( inform_t( i ) ) 
        DO l = 1, len_name
          st1 = inform_t( i )( l : l ) 
          CALL STRING_lower_scalar( st1 )
          depends_lower( l : l ) = st1
        END DO
        IF ( depends_lower( 1 : len_name ) == 'time' ) THEN
          WRITE( ciface_unit,                                                  &
            "('    CALL copy_time_out( finform%time, cinform%time )' )" )
        ELSE
          WRITE( ciface_unit, "('    CALL copy_', A, '_inform_out( finform%',  &
         &  A, '_inform, cinform%', A, '_inform )' )" )                        &
            depends_lower( 1 : len_name ), depends_lower( 1 : len_name ),      &
            depends_lower( 1 : len_name )
         END IF
      END DO
    END IF  
    IF ( inform_nc > 0 ) THEN
      WRITE( ciface_unit, "( '', /, '    ! Strings' )" )
      DO i = 1, inform_nc
        WRITE( ciface_unit, "( &
      & '    DO i = 1, LEN( finform%', A, ' )', /, &
      & '      cinform%', A, '( i ) = finform%', A, '( i : i )', /, &
      & '    END DO', /, &
      & '    cinform%', A, '( LEN( finform%', A, ' ) + 1 ) = C_NULL_CHAR' &
      & )" ) TRIM( inform_c( i ) ), TRIM( inform_c( i ) ),                   &
         TRIM( inform_c( i ) ), TRIM( inform_c( i ) ), TRIM( inform_c( i ) )
      END DO
    END IF  
    WRITE( ciface_unit, "(                                                     &
 & '    RETURN', /, &
 & '', /, &
 & '    END SUBROUTINE copy_inform_out', /, '' &
   )" )
  END IF

  WRITE( ciface_unit, "(                                                       &
 &  '  END MODULE GALAHAD_', A, '_double_ciface' )" ) TRIM( package_upper )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  -------------------------------------', /, &
 & '!  C interface to fortran ', A, '_initialize', /, &
 & '!  -------------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_initialize( cdata, ccontrol, cinform ) BIND( C ) ', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box', /, &
 & '  TYPE ( ', A, '_control_type ), INTENT( OUT ) :: ccontrol', /, &
 & '  TYPE ( ', A, '_inform_type ), INTENT( OUT ) :: cinform', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_full_data_type ), POINTER :: fdata', /, &
 & '  TYPE ( f_', A, '_control_type ) :: fcontrol', /, &
 & '  TYPE ( f_', A, '_inform_type ) :: finform', /, &
 & '  LOGICAL :: f_indexing ', /, &
 & '', /, &
 & '!  allocate fdata', /, &
 & '', /, &
 & '  ALLOCATE( fdata ); cdata = C_LOC( fdata )', /, &
 & '', /, &
 & '!  initialize required fortran types', /, &
 & '', /, &
 & '  CALL f_', A, '_initialize( fdata, fcontrol, finform )', /, &
 & '', /, &
 & '!  C sparse matrix indexing by default', /, &
 & '', /, &
 & '  f_indexing = .FALSE.', /, &
 & '  fdata%f_indexing = f_indexing', /, &
 & '', /, &
 & '!  copy control out ', /, &
 & '', /, &
 & '  CALL copy_control_out( fcontrol, ccontrol, f_indexing )', /, &
 & '', /, &
 & '!  copy inform out', /, &
 & '', /, &
 & '  CALL copy_inform_out( finform, cinform )', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_initialize' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  ----------------------------------------', /, &
 & '!  C interface to fortran ', A, '_read_specfile', /, &
 & '!  ----------------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_read_specfile( ccontrol, cspecfile ) BIND( C )', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  TYPE ( ', A, '_control_type ), INTENT( INOUT ) :: ccontrol', /, &
 & '  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_control_type ) :: fcontrol', /, &
 & '  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile', /, &
 & '  LOGICAL :: f_indexing', /, &
 & '', /, &
 & '!  device unit number for specfile', /, &
 & '', /, &
 & '  INTEGER ( KIND = C_INT ), PARAMETER :: device = 10', /, &
 & '', /, &
 & '!  convert C string to Fortran string', /, &
 & '', /, &
 & '  fspecfile = cstr_to_fchar( cspecfile )', /, &
 & '', /, &
 & '!  copy control in', /, &
 & '', /, &
 & '  CALL copy_control_in( ccontrol, fcontrol, f_indexing )', /, &
 & '  ', /, &
 & '!  open specfile for reading', /, &
 & '', /, &
 & '  OPEN( UNIT = device, FILE = fspecfile )', /, &
 & '  ', /, &
 & '!  read control parameters from the specfile', /, &
 & '', /, &
 & '  CALL f_', A, '_read_specfile( fcontrol, device )', /, &
 & '', /, &
 & '!  close specfile', /, &
 & '', /, &
 & '  CLOSE( device )', /, &
 & '', /, &
 & '!  copy control out', /, &
 & '', /, &
 & '  CALL copy_control_out( fcontrol, ccontrol, f_indexing )', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_read_specfile' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  ---------------------------------', /, &
 & '!  C interface to fortran ', A, '_inport', /, &
 & '!  ---------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_import( ccontrol, cdata, status ) BIND( C )', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status', /, &
 & '  TYPE ( ', A, '_control_type ), INTENT( INOUT ) :: ccontrol', /, &
 & '  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_control_type ) :: fcontrol', /, &
 & '  TYPE ( f_', A, '_full_data_type ), POINTER :: fdata', /, &
 & '  LOGICAL :: f_indexing', /, &
 & '', /, &
 & '!  copy control and inform in', /, &
 & '', /, &
 & '  CALL copy_control_in( ccontrol, fcontrol, f_indexing )', /, &
 & '', /, &
 & '!  associate data pointer', /, &
 & '', /, &
 & '  CALL C_F_POINTER( cdata, fdata )', /, &
 & '', /, &
 & '!  is fortran-style 1-based indexing used?', /, &
 & '', /, &
 & '  fdata%f_indexing = f_indexing', /, &
 & '', /, &
 & '!  handle C sparse matrix indexing', /, &
 & '', /, &
 & '  IF ( .NOT. f_indexing ) THEN', /, &
 & '', /, &
 & '!  import the problem data into the required ', A, ' structure', /, &
 & '', /, &
 & '    CALL f_', A, '_import( fcontrol, fdata, status )', /, &
 & '  ELSE', /, &
 & '    CALL f_', A, '_import( fcontrol, fdata, status )', /, &
 & '  END IF', /, &
 & '', /, &
 & '!  copy control out', /, &
 & '', /, &
 & '  CALL copy_control_out( fcontrol, ccontrol, f_indexing )', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_import' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_upper ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  ---------------------------------------', /, &
 & '!  C interface to fortran ', A, '_reset_control', /, &
 & '!  ----------------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_reset_control( ccontrol, cdata, status ) BIND( C )', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status', /, &
 & '  TYPE ( ', A, '_control_type ), INTENT( INOUT ) :: ccontrol', /, &
 & '  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_control_type ) :: fcontrol', /, &
 & '  TYPE ( f_', A, '_full_data_type ), POINTER :: fdata', /, &
 & '  LOGICAL :: f_indexing', /, &
 & '', /, &
 & '!  copy control in', /, &
 & '', /, &
 & '  CALL copy_control_in( ccontrol, fcontrol, f_indexing )', /, &
 & '', /, &
 & '!  associate data pointer', /, &
 & '', /, &
 & '  CALL C_F_POINTER( cdata, fdata )', /, &
 & '', /, &
 & '!  is fortran-style 1-based indexing used?', /, &
 & '', /, &
 & '  fdata%f_indexing = f_indexing', /, &
 & '', /, &
 & '!  import the control parameters into the required structure', /, &
 & '', /, &
 & '  CALL f_', A, '_reset_control( fcontrol, fdata, status )', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_reset_control' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_upper ), TRIM( package_lower )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  --------------------------------------', /, &
 & '!  C interface to fortran ', A, '_information', /, &
 & '!  --------------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_information( cdata, cinform, status ) BIND( C ) ', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata', /, &
 & '  TYPE ( ', A, '_inform_type ), INTENT( INOUT ) :: cinform', /, &
 & '  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_full_data_type ), pointer :: fdata', /, &
 & '  TYPE ( f_', A, '_inform_type ) :: finform', /, &
 & '', /, &
 & '!  associate data pointer', /, &
 & '', /, &
 & '  CALL C_F_POINTER( cdata, fdata )', /, &
 & '', /, &
 & '!  obtain ', A, ' solution information', /, &
 & '', /, &
 & '  CALL f_', A, '_information( fdata, finform, status )', /, &
 & '', /, &
 & '!  copy inform out', /, &
 & '', /, &
 & '  CALL copy_inform_out( finform, cinform )', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_information' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_upper ), TRIM( package_lower ), TRIM( package_lower )

  WRITE( ciface_unit, "(                                                       &
 & '', /, &
 & '!  ------------------------------------', /, &
 & '!  C interface to fortran ', A, '_terminate', /, &
 & '!  ------------------------------------', /, &
 & '', /, &
 & '  SUBROUTINE ', A, '_terminate( cdata, ccontrol, cinform ) BIND( C ) ', /, &
 & '  USE GALAHAD_', A, '_double_ciface', /, &
 & '  IMPLICIT NONE', /, &
 & '', /, &
 & '!  dummy arguments', /, &
 & '', /, &
 & '  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata', /, &
 & '  TYPE ( ', A, '_control_type ), INTENT( IN ) :: ccontrol', /, &
 & '  TYPE ( ', A, '_inform_type ), INTENT( INOUT ) :: cinform', /, &
 & '', /, &
 & '!  local variables', /, &
 & '', /, &
 & '  TYPE ( f_', A, '_full_data_type ), pointer :: fdata', /, &
 & '  TYPE ( f_', A, '_control_type ) :: fcontrol', /, &
 & '  TYPE ( f_', A, '_inform_type ) :: finform', /, &
 & '  LOGICAL :: f_indexing', /, &
 & '', /, &
 & '!  copy control in', /, &
 & '', /, &
 & '  CALL copy_control_in( ccontrol, fcontrol, f_indexing )', /, &
 & '', /, &
 & '!  copy inform in', /, &
 & '', /, &
 & '  CALL copy_inform_in( cinform, finform )', /, &
 & '', /, &
 & '!  associate data pointer', /, &
 & '', /, &
 & '  CALL C_F_POINTER( cdata, fdata )', /, &
 & '', /, &
 & '!  deallocate workspace', /, &
 & '', /, &
 & '  CALL f_', A, '_terminate( fdata, fcontrol, finform )', /, &
 & '', /, &
 & '!  copy inform out', /, &
 & '', /, &
 & '  CALL copy_inform_out( finform, cinform )', /, &
 & '', /, &
 & '!  deallocate data', /, &
 & '', /, &
 & '  DEALLOCATE( fdata ); cdata = C_NULL_PTR ', /, &
 & '  RETURN', /, &
 & '', /, &
 & '  END SUBROUTINE ', A, '_terminate' &
 & )" )                                                                        &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_upper ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower ), TRIM( package_lower ), TRIM( package_lower ),      &
     TRIM( package_lower )

  CLOSE( UNIT = ciface_unit )
  WRITE( out, "( 1X, A, '_ciface.f90 constructed' )" ) TRIM( package_lower )

  STOP

  CONTAINS

    SUBROUTINE STRING_lower_scalar( string ) ! converts a string to lower case
    CHARACTER, INTENT( INOUT ) :: string
    INTEGER :: letter
    CHARACTER, DIMENSION( 26 ) :: LOWER, UPPER
    DATA LOWER / 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',             &
                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',             &
                 'u', 'v', 'w', 'x', 'y', 'z' /
    DATA UPPER / 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',             &
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',             &
                 'U', 'V', 'W', 'X', 'Y', 'Z' /
    DO letter = 1, 26
      IF ( string == UPPER( letter ) ) THEN
        string = LOWER( letter )
        EXIT
      END IF
    END DO
    RETURN
    END SUBROUTINE STRING_lower_scalar

    SUBROUTINE STRING_upper_scalar( string ) ! converts a string to upper case
    CHARACTER, INTENT( INOUT ) :: string
    INTEGER :: letter
    CHARACTER, DIMENSION( 26 ) :: LOWER, UPPER
    DATA LOWER / 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',             &
                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',             &
                 'u', 'v', 'w', 'x', 'y', 'z' /
    DATA UPPER / 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',             &
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',             &
                 'U', 'V', 'W', 'X', 'Y', 'Z' /
    DO letter = 1, 26
      IF ( string == LOWER( letter ) ) THEN
        string = UPPER( letter )
        EXIT
      END IF
    END DO
    RETURN
    END SUBROUTINE STRING_upper_scalar

    SUBROUTINE contains_dimension( line, past ) ! does string contain DIMENSION(
    integer ::  past
    CHARACTER ( LEN = 80 ) :: line
    integer :: i, len
    len = LEN_TRIM( line )
    past = 0
    IF ( len < 10 ) RETURN
    DO i = 1, len - 10
      IF ( line( i : i + 9 ) == 'DIMENSION(' ) THEN
        past = i + 10
        RETURN
      END IF
    END DO
    RETURN
    END SUBROUTINE contains_dimension

  END PROGRAM extract_types
