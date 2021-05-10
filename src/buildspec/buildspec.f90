! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 3.0. October 23rd 2017


      PROGRAM BUILDSPEC

!         +-----------------------------------------+
!         |  Assemble the algorithmic specification |
!         |  (spec)file for GALAHAD programs        |
!         +-----------------------------------------+

!  use: compiled_file package [update]

!  where compiled_file is the name of the object file created by compiling this
!  program, package is the name of the package whose specfile is required,
!  and the optional word update indicates that non-default values from an
!  existing version of the specfile in $GALAHAD/spec/ should be retained.
!  Here and elsewhere [] means an optional argument, and (value) and (VALUE)
!  indicates the lower- and upper-case conversion of these arguments.

!  The program will process the file $GALAHAD/src/(package)/RUN(PACKAGE).meta
!  one line at a time. The result of processing the meta file will be a
!  so-called problem specification (spec)file, which is written to
!  ~/.galahad/specs//RUN(PACKAGE).SPC. This specfile may subsequently be edited
!  by users to change options that control the performance of the package

!  The meta file may contain the following keywords:

!  MAIN
!  INCLUDE template_name [section_name]
!  REMOVE
!  REPLACE
!  COMMENT
!
!  A keyword remains active until replaced by another one

!  Any other line (except blank lines, which are ignored) will start with
!  the character "!" and provide default and optional values for each
!  package called, or comments as appropriate

!  The MAIN section should occur first (if at all) and announces the start
!  of the options relating to the main program that will be used to call
!  package. Any following "!" lines that occur until the next keyword will
!  define options that control the action of the main program, that will be
!  compiled from $GALAHAD/src/(package)/use_(package).f90. These options will
!  be written to the specfile, and bookended by the lines
!  BEGIN RUN(PACKAGE) SPECIFICATIONS
!  and
!  END RUN(PACKAGE) SPECIFICATIONS

!  An INCLUDE section announces the start of the options relating to a
!  particular package template_name.f90 that is called by use_(package).f90.
!  If section_name is absent, it takes the value template_name. The options
!  for the package in question will be copied, verbatim, from the template
!  file $GALAHAD/src/(TEMPLATE_NAME).template, and appended to the specfile,
!  and bookended by the lines
!  BEGIN (SECTION_NAME) SPECIFICATIONS
!  and
!  END (SECTION_NAME) SPECIFICATIONS

!  A REMOVE section indicates that any subsequent "!" lines that occur until
!  the next keyword will be removed from the current MAIN or INCLUDE section
!  in the final specfile. The string following the "!" will be compared to
!  the set of "!" lines for the current section, and any matches removed.
!  This provides a mechanism to remove a generic option that is not
!  appropriate in the context of the main program

!  A REPLACE section indicates that any subsequent "!" lines that occur until
!  the next keyword will replace any similar line from the current MAIN or
!  INCLUDE section in the final specfile. The string following the "!" will
!  be compared to the set of "!" lines for the current section, and the
!  first match replaced. This provides a mechanism to replace a generic
!  option value that is inappropriate in the context of the main program

!  A COMMENT section indicates that any subsequent "!" lines that occur until
!  the next keyword will be copied verbatim into the specfile.

!  The order of MAIN, INCLUDE and COMMENT sections defines the order that
!  the resulting options will appear in the final specfile

        USE GALAHAD_STRING, ONLY: STRING_lower_word, STRING_upper_word

!  local variables

        INTEGER :: num_args, len1, len2, stat
        LOGICAL :: update
        CHARACTER ( LEN = 20 ) :: action
        CHARACTER ( LEN = 20 ) :: package
        CHARACTER ( LEN = 1000 ) :: galahad
        CHARACTER ( LEN = 1000 ) :: home

!  error file

        INTEGER, PARAMETER :: error = 11
        CHARACTER ( LEN = 16 ), PARAMETER :: error_filename = 'BUILD_SPEC.error'
        OPEN( error, FILE = error_filename, FORM = 'FORMATTED',                &
              STATUS = 'REPLACE' )

!  check that there are at least two command line arguments

        num_args = COMMAND_ARGUMENT_COUNT( )
        IF ( num_args < 1 ) THEN
          WRITE( error, "( ' error: at least 1 command line argument is',      &
         &                 ' required, ', I0, ' provided' )" ) num_args
          GO TO 990
        END IF

!  obtain both arguments, package and root

        CALL GET_COMMAND_ARGUMENT( 1, value = package,                         &
                                   length = len1, status = stat )
        IF ( stat /= 0 ) THEN
          WRITE( error, "( ' error: command line arguments 1 of length ', I0,  &
         &                 ' exceeds privided length of 20' )" ) len1
          GO TO 990
        END IF

!  check to see if there is a second argument, and if it is "update"

        update = .FALSE.
        IF ( num_args > 1 ) THEN
          CALL GET_COMMAND_ARGUMENT( 2, value = action,                        &
                                     length = len2, status = stat )
          IF ( stat /= 0 ) THEN
            WRITE( error, "( ' error: command line arguments 2 of length ',    &
           &                 I0, ' exceeds privided length of 20' )" ) len2
            GO TO 990
          END IF
          CALL STRING_lower_word( action( : len2 ) )
          IF ( action( : len2 ) == "update" ) update = .TRUE.
        END IF

!  find the name of the GALAHAD home directory

        CALL GET_ENVIRONMENT_VARIABLE( 'GALAHAD', value = galahad,             &
                                        status = stat )
        IF ( stat /= 0 ) THEN
          WRITE( error, "( ' error: GALAHAD environment variable not set' )" )
          GO TO 990
        END IF

!  find the name of the user's home directory

        CALL GET_ENVIRONMENT_VARIABLE( 'HOME', value = home, status = stat )
        IF ( stat /= 0 ) THEN
          WRITE( error, "( ' error: HOME environment variable not set' )" )
          GO TO 990
        END IF

!  use a subroutine for the remainder of the work; this is simply a fix to
!  use the trimmed versions of the names package, galahad and home to work
!  around compiler bugs with old fortran compilers

        CALL BUILDSPEC_sub( TRIM( package ), TRIM( galahad ), TRIM( home ),    &
                            len1, update, error )

!  perpare to stop

 990    CONTINUE
        CLOSE( error )
        STOP

      CONTAINS

        SUBROUTINE BUILDSPEC_sub( package, galahad, home, len1, update, error )

!  subroutine to assemble the spec file

        CHARACTER ( LEN = * ), INTENT( IN ) :: package
        CHARACTER ( LEN = * ), INTENT( IN ) :: galahad
        CHARACTER ( LEN = * ), INTENT( IN ) :: home
        INTEGER, INTENT( IN ) :: len1, error
        LOGICAL, INTENT( IN ) :: update

!  local variables

        INTEGER :: i, j, end_section
        INTEGER :: len2, len_spec, len_template_name, len_section_name
        LOGICAL :: is_file, two_args
        LOGICAL :: main_section, include_section, comment_section
        LOGICAL :: remove, replace, old, section_found
        CHARACTER ( LEN = len1 ) :: lower_package, upper_package
        INTEGER, PARAMETER :: meta = 21
        INTEGER, PARAMETER :: template = 22
        INTEGER, PARAMETER :: spec = 23
        INTEGER, PARAMETER :: old_spec = 23
        CHARACTER ( LEN = 120 ) :: blank = REPEAT( ' ', 120 )
        CHARACTER ( LEN = 120 ) :: newline, template_name, section_name
        CHARACTER ( LEN = 120 ) :: name, lower_template_name
        CHARACTER ( LEN = 120 ) :: upper_template_name, upper_section_name
        CHARACTER ( LEN = 120 ), DIMENSION( 3 ) :: dummy, old_dummy
        CHARACTER ( LEN = 120 ), DIMENSION( 1000 ) :: SECTION
        CHARACTER ( LEN = 120 ), DIMENSION( 10000 ) :: NEW_SPEC
        CHARACTER ( LEN = 1040 ) :: meta_file
        CHARACTER ( LEN = 1040 ) :: template_file
        CHARACTER ( LEN = 1040 ) :: spec_file
        CHARACTER ( LEN = 1040 ) :: old_spec_file

!       write(6,*) package
!       write(6,*) galahad
!       write(6,*) home

!  record the package name in upper case

        lower_package = package( : len1 )
        CALL STRING_lower_word( lower_package )
        upper_package = package( : len1 )
        CALL STRING_upper_word( upper_package )

!  search for the spec meta file in "standard" places

!  check first in galahad/src/package/RUNPACKAGE.meta

        meta_file = galahad // "/src/" //                                      &
            lower_package // "/RUN" // upper_package // '.meta'
        INQUIRE( FILE = meta_file, EXIST = is_file )

!  otherwise look in galahad/src/oblivion/package/RUNPACKAGE.meta

        IF ( .NOT. is_file ) THEN
          meta_file = galahad // "/src/oblivion/" //                           &
              lower_package // "/RUN" // upper_package // '.meta'
          INQUIRE( FILE = meta_file, EXIST = is_file )

!  or else  in galahad/src/forthcoming/package/RUNPACKAGE.meta

          IF ( .NOT. is_file ) THEN
            meta_file = galahad // "/src/forthcoming/" //                      &
                lower_package // "/RUN" // upper_package // '.meta'
            INQUIRE( FILE = meta_file, EXIST = is_file )

! there is no appropriate meta file

            IF ( .NOT. is_file ) THEN
              WRITE( error, "( ' error: no meta file for package ', A )" )     &
                lower_package
              RETURN
            END IF
          END IF
        END IF

!  open the meta file

        OPEN( meta, FILE = meta_file, FORM = 'FORMATTED', STATUS = 'OLD' )

!  if an existing spec file is to be updated, search for the file

        IF ( update ) THEN
          old_spec_file = home // "/.galahad/specs/RUN" //                     &
                          upper_package // '.SPC'
          INQUIRE( FILE = old_spec_file, EXIST = old )

!  open the existing spec file if it exists

          IF ( old ) OPEN( old_spec, FILE = old_spec_file, FORM = 'FORMATTED', &
                           STATUS = 'OLD' )

        ELSE
          old = .FALSE.
        END IF

!  process the meta file, one line at a time

        i = 0 ; end_section = 0
        len_spec = 0
        main_section = .FALSE. ; include_section = .FALSE.
        comment_section = .FALSE. ; remove = .FALSE. ; replace = .FALSE.

        DO
          newline = blank
          READ( meta, "( A )", END = 900, ERR = 900 ) newline
          IF ( newline == blank ) CYCLE

!  start the main section
!  -----------------------

          IF ( newline( 1 : 4 ) == 'MAIN' ) THEN
            main_section = .TRUE.
            i = 1 ; end_section = 0
            newline = blank
            newline = 'BEGIN RUN' // upper_package // ' SPECIFICATIONS'
            SECTION( i ) = newline

!  start a new include section
!  ---------------------------

!  first terminate the current section (if any)

          ELSE IF ( newline( 1 : 7 ) == 'INCLUDE' ) THEN
            newline( 1 : 7 ) = "       "
            newline = ADJUSTL( newline )
            IF ( INDEX( TRIM( newline ), ' ' ) > 0 ) THEN
              READ( newline, * ) dummy( 2 : 3 )
            ELSE
              READ( newline, * ) dummy( 2 )
              dummy( 3 ) = dummy( 2 )
            END IF
            IF ( main_section ) THEN
              newline = blank
              newline = 'END RUN' // upper_package // ' SPECIFICATIONS'
              end_section = i + 1
              SECTION( end_section ) = newline
            ELSE IF ( include_section ) THEN
              newline = blank
              newline = 'END ' // upper_section_name( : len_section_name )     &
                               // ' SPECIFICATIONS'
              end_section = i + 1
              SECTION( end_section ) = newline
            ELSE IF ( comment_section ) THEN
              end_section = i
            END IF

! if required, apply non-default modifications from an earlier specfile

            IF ( old .AND. .NOT. comment_section ) THEN
              name = blank
              IF ( main_section ) THEN
                name = 'RUN' // upper_package
              ELSE
                name = upper_section_name( : len_section_name )
              END IF
              REWIND( old_spec )
              section_found = .FALSE.

!  search the old specfile for the earlier incarnation of the previous section

              DO
                READ( old_spec, "( A )", END = 500, ERR = 500 ) newline
                IF ( LEN( TRIM( newline ) )== 0 ) CYCLE
                newline = ADJUSTL( newline )

!  the start of the section has been found

                IF ( INDEX( newline, 'BEGIN ' // TRIM( name ) ) > 0 ) THEN
                   section_found = .TRUE.
                   CYCLE

!  the end of the section has been found

                ELSE IF ( INDEX( newline, 'END ' // TRIM( name ) ) > 0 ) THEN
                  EXIT
                END IF

!  in the current section, find any non-default values

                IF ( section_found ) THEN
                  IF ( newline( 1 : 1 ) == "!" ) CYCLE

!  replace the default with the non default

                  two_args = INDEX( TRIM( newline ), ' ' ) > 0
                  IF ( two_args ) THEN
                    READ( newline, * ) old_dummy( 1 : 2 )
                  ELSE
                    READ( newline, * ) old_dummy( 1 : 1 )
                  END IF
                  DO i = 2, end_section - 1
                    IF ( INDEX( SECTION( i ) // " " ,                          &
                                TRIM( old_dummy( 1 ) ) // " " ) > 0 ) THEN
                      newline = blank
                      newline = "  " // TRIM( old_dummy( 1 ) )
                      IF ( two_args ) THEN
                        len2 = LEN( TRIM( old_dummy( 2 ) ) )
                        newline( 53 : 52 + len2 ) = TRIM( old_dummy( 2 ) )
                      ELSE
                        newline( 53 : 55 ) = 'yes'
                      END IF
                      SECTION( i ) = newline
                      EXIT
                    END IF
                  END DO
                END IF
              END DO
 500          CONTINUE
            END IF

!  append the completed current section to the new spec file

            DO i = 1, end_section
              IF ( TRIM( SECTION( i ) ) /= "! !" ) THEN
                len_spec = len_spec + 1
                NEW_SPEC( len_spec ) = SECTION( i )
              END IF
            END DO

!  initialize the new section

            main_section = .FALSE. ; include_section = .TRUE.
            comment_section = .FALSE. ; remove = .FALSE. ; replace = .FALSE.
            template_name = dummy( 2 )
            section_name = dummy( 3 )
            len_template_name = LEN( TRIM( template_name ) )
            len_section_name = LEN( TRIM( section_name ) )

!  record lower and upper case versions of the template and section name

            lower_template_name( : len_template_name )                         &
              = template_name( : len_template_name )
            CALL STRING_lower_word( lower_template_name( : len_template_name ) )
            upper_template_name( : len_template_name )                         &
              = template_name( : len_template_name )
            CALL STRING_upper_word( upper_template_name( : len_template_name ) )
            upper_section_name( : len_section_name )                           &
              = section_name( : len_section_name )
            CALL STRING_upper_word( upper_section_name( : len_section_name ) )

            i = 2 ; end_section = 0
            newline = blank
            newline = 'BEGIN ' // upper_section_name( : len_section_name )     &
              // ' SPECIFICATIONS'
            SECTION( 1 ) = blank ; SECTION( i ) = newline

!  search for the section's spec template file in "standard" places

!  check first in galahad/src/package/RUNPACKAGE.meta

            template_file = galahad // "/src/" //                              &
                lower_template_name( : len_template_name ) // "/" //           &
                upper_template_name( : len_template_name ) // '.template'
            INQUIRE( FILE = template_file, EXIST = is_file )


!  otherwise look in galahad/src/oblivion/package/RUNPACKAGE.template

            IF ( .NOT. is_file ) THEN
              template_file = galahad // "/src/oblivion/" //                   &
                lower_template_name( : len_template_name ) // "/" //           &
                upper_template_name( : len_template_name ) // '.template'
              INQUIRE( FILE = template_file, EXIST = is_file )

!  or else  in galahad/src/forthcoming/package/RUNPACKAGE.template

              IF ( .NOT. is_file ) THEN
                template_file = galahad // "/src/forthcoming/" //              &
                  lower_template_name( : len_template_name ) // "/" //         &
                  upper_template_name( : len_template_name ) // '.template'
                INQUIRE( FILE = template_file, EXIST = is_file )

! there is no appropriate template file

                IF ( .NOT. is_file ) THEN
                  WRITE( error, "( ' error: no template file for package ',    &
                 &    A )" ) lower_template_name( : len_template_name )
                  RETURN
                END IF
              END IF
            END IF

!  open the template file

            OPEN( template, FILE = template_file, FORM = 'FORMATTED',          &
                  STATUS = 'OLD' )

!  copy the template data to the new section

            DO
              newline = blank
              READ( template, "( A )", END = 600, ERR = 600 ) newline
              IF ( newline == blank ) CYCLE
              i = i + 1
              SECTION( i ) = newline
            END DO

 600        CONTINUE
            CLOSE( template )

!  start a new comment section
!  ---------------------------

!  first terminate any current section

          ELSE IF ( newline( 1  : 7 ) == 'COMMENT' ) THEN
            IF ( main_section ) THEN
              newline = blank
              newline = 'END RUN' // upper_package // ' SPECIFICATIONS'
              end_section = i + 1
              SECTION( end_section ) = newline
            ELSE IF ( include_section ) THEN
              newline = blank
              newline = 'END ' // upper_section_name( : len_section_name )     &
                               // ' SPECIFICATIONS'
              end_section = i + 1
              SECTION( end_section ) = newline
            ELSE IF ( comment_section ) THEN
              end_section = i
            END IF

! if required, apply non-default modifications from an earlier specfile

            IF ( old .AND. .NOT. comment_section ) THEN
              name = blank
              IF ( main_section ) THEN
                name = 'RUN' // upper_package
              ELSE
                name = upper_section_name( : len_section_name )
              END IF
              REWIND( old_spec )
              section_found = .FALSE.

!  search the old specfile for the earlier incarnation of the previous section

              DO
                READ( old_spec, "( A )", END = 700, ERR = 700 ) newline
                IF ( LEN( TRIM( newline ) )== 0 ) CYCLE
                newline = ADJUSTL( newline )

!  the start of the section has been found

                IF ( INDEX( newline, 'BEGIN ' // TRIM( name ) ) > 0 ) THEN
                   section_found = .TRUE.
                   CYCLE

!  the end of the section has been found

                ELSE IF ( INDEX( newline, 'END ' // TRIM( name ) ) > 0 ) THEN
                  EXIT
                END IF

!  in the current section, find any non-default values

                IF ( section_found ) THEN
                  IF ( newline( 1 : 1 ) == "!" ) CYCLE

!  replace the default with the non default

                  two_args = INDEX( TRIM( newline ), ' ' ) > 0
                  IF ( two_args ) THEN
                    READ( newline, * ) old_dummy( 1 : 2 )
                  ELSE
                    READ( newline, * ) old_dummy( 1 : 1 )
                  END IF
                  DO i = 2, end_section - 1
                    IF ( INDEX( SECTION( i ) // " ",                           &
                         TRIM( old_dummy( 1 ) ) // " " ) > 0 ) THEN
                      newline = blank
                      newline = "  " // TRIM( old_dummy( 1 ) )
                      IF ( two_args ) THEN
                        len2 = LEN( TRIM( old_dummy( 2 ) ) )
                        newline( 53 : 52 + len2 ) = TRIM( old_dummy( 2 ) )
                      ELSE
                        newline( 53 : 55 ) = 'yes'
                      END IF
                      SECTION( i ) = newline
                      EXIT
                    END IF
                  END DO
                END IF
              END DO
 700          CONTINUE
            END IF

!  append the completed section to the new spec file

            DO i = 1, end_section
              IF ( TRIM( SECTION( i ) ) /= "! !" ) THEN
                len_spec = len_spec + 1
                NEW_SPEC( len_spec ) = SECTION( i )
              END IF
            END DO

            main_section = .FALSE. ; include_section = .FALSE.
            comment_section = .TRUE. ; remove = .FALSE. ; replace = .FALSE.

            i = 1 ; end_section = 0
            SECTION( i ) = blank

!  remove data from the current section
!  -------------------------------------

          ELSE IF ( newline( 1  : 7 ) == 'REMOVE' ) THEN
            remove = .TRUE. ; replace = .FALSE.

!  replace data from the current section
!  -------------------------------------

          ELSE IF ( newline( 1  : 7 ) == 'REPLACE' ) THEN
            remove = .FALSE. ; replace = .TRUE.
          ELSE

!  remove the line containing the string on this line

            IF ( remove ) THEN
              READ( newline, * ) dummy( 1 : 2 )
              dummy( 3 ) = dummy( 2 )
              DO j = 2, i
                READ( SECTION( j ), * ) dummy( 1 : 2 )
                IF ( TRIM( dummy( 3 ) ) == TRIM( dummy( 2 ) ) ) THEN
                  newline = blank
                  newline( 1 : 3 ) = "! !"
                  SECTION( j ) = newline
                  EXIT
                END IF
              END DO

!  replace the line containing the string on this line with the string

            ELSE IF ( replace ) THEN
              READ( newline, * ) dummy( 1 : 2 )
              dummy( 3 ) = dummy( 2 )
              DO j = 2, i
                READ( SECTION( j ), * ) dummy( 1 : 2 )
                IF ( TRIM( dummy( 3 ) ) == TRIM( dummy( 2 ) ) ) THEN
                  SECTION( j ) = blank
                  SECTION( j ) = newline
                  EXIT
                END IF
              END DO

!  copy data to the main section

            ELSE
              i = i + 1
              SECTION( i ) = newline
            END IF
          END IF
        END DO

! the meta file has been processed. Conclude remaining tasks

 900    CONTINUE

!  terminate the final section

        IF ( main_section ) THEN
          newline = blank
          newline = 'END RUN' // upper_package // ' SPECIFICATIONS'
          end_section = i + 1
          SECTION( end_section ) = newline
        ELSE IF ( include_section ) THEN
          newline = blank
          newline = 'END ' // upper_section_name( : len_section_name )         &
                           // ' SPECIFICATIONS'
          end_section = i + 1
          SECTION( end_section ) = newline
        ELSE IF ( comment_section ) THEN
          end_section = i
        END IF

! if required, apply non-default modifications from an earlier specfile

        IF ( old .AND. .NOT. comment_section ) THEN
          name = blank
          IF ( main_section ) THEN
            name = 'RUN' // upper_package
          ELSE
            name = upper_section_name( : len_section_name )
          END IF
          REWIND( old_spec )
          section_found = .FALSE.

!  search the old specfile for the earlier incarnation of the previous section

          DO
            READ( old_spec, "( A )", END = 910, ERR = 910 ) newline
            IF ( LEN( TRIM( newline ) )== 0 ) CYCLE
            newline = ADJUSTL( newline )

!  the start of the section has been found

            IF ( INDEX( newline, 'BEGIN ' // TRIM( name ) ) > 0 ) THEN
               section_found = .TRUE.
               CYCLE

!  the end of the section has been found

            ELSE IF ( INDEX( newline, 'END ' // TRIM( name ) ) > 0 ) THEN
              EXIT
            END IF

!  in the current section, find any non-default values

            IF ( section_found ) THEN
              IF ( newline( 1 : 1 ) == "!" ) CYCLE

!  replace the default with the non default

              two_args = INDEX( TRIM( newline ), ' ' ) > 0
              IF ( two_args ) THEN
                READ( newline, * ) old_dummy( 1 : 2 )
              ELSE
                READ( newline, * ) old_dummy( 1 : 1 )
              END IF
              DO i = 2, end_section - 1
                IF ( INDEX( SECTION( i ) // " ",                               &
                     TRIM( old_dummy( 1 ) ) // " " ) > 0 ) THEN
                  newline = blank
                  newline = "  " // TRIM( old_dummy( 1 ) )
                  IF ( two_args ) THEN
                    len2 = LEN( TRIM( old_dummy( 2 ) ) )
                    newline( 53 : 52 + len2 ) = TRIM( old_dummy( 2 ) )
                  ELSE
                    newline( 53 : 55 ) = 'yes'
                  END IF
                  SECTION( i ) = newline
                  EXIT
                END IF
              END DO
            END IF
          END DO
 910      CONTINUE
        END IF

!  append the final section to the new spec file

        DO i = 1, end_section
          IF ( TRIM( SECTION( i ) ) /= "! !" ) THEN
            len_spec = len_spec + 1
            NEW_SPEC( len_spec ) = SECTION( i )
          END IF
        END DO

        IF ( old ) CLOSE( old_spec )

!  create the specfile for writing

        spec_file =  home // '/.galahad/specs/RUN'                             &
           // upper_package // '.SPC'
        INQUIRE( FILE = spec_file, EXIST = is_file )
        IF ( is_file ) THEN
          OPEN( spec, FILE = spec_file, FORM = 'FORMATTED', STATUS = 'OLD',    &
                IOSTAT = i )
        ELSE
          OPEN( spec, FILE = spec_file, FORM = 'FORMATTED', STATUS = 'NEW',    &
                IOSTAT = i )
        END IF

!  write the new specfile, and close the file afterwards

        DO i = 1, len_spec
          IF ( i > 1 ) THEN
            WRITE( spec, "( A ) " ) TRIM( NEW_SPEC( i ) )
          ELSE
            IF ( TRIM( NEW_SPEC( i ) ) /= "" )                                 &
              WRITE( spec, "( A ) " ) TRIM( NEW_SPEC( i ) )
          END IF
        END DO
        CLOSE( spec )

        WRITE( error, "( ' specfile RUN', A, '.SPC built successfuly' )" )     &
          upper_package

!  perpare to return

        CLOSE( meta )
        RETURN

!  end of subroutine BUILDSPEC_sub

        END SUBROUTINE BUILDSPEC_sub

!  end of program BUILDSPEC

      END PROGRAM BUILDSPEC
