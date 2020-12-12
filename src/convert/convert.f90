! THIS VERSION: GALAHAD 3.3 - 30/10/2020 AT 16:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ C O N V E R T    M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started June 8th 2014
!   originally released GALAHAD Version 2.6. June 8th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CONVERT_double

!      -------------------------------------------------------------------
!     |                                                                   |
!     | Given a real matrix A stored in one format, convert it to another |
!     |                                                                   |
!      -------------------------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_SMT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CONVERT_read_specfile,                                         &
                CONVERT_between_matrix_formats, CONVERT_to_sparse_row_format,  &
                CONVERT_to_sparse_column_format, CONVERT_to_coordinate_format, &
                CONVERT_to_dense_row_format, CONVERT_to_dense_column_format,   &
                CONVERT_transpose, CONVERT_order, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: real_bytes = 8
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  obtain the transpose of the input matrix?

        LOGICAL :: transpose = .FALSE.

!  add the values of entries in duplicate positions

        LOGICAL :: sum_duplicates = .FALSE.

!  order row or column data by increasing index

        LOGICAL :: order = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

      END TYPE CONVERT_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_time_type

!  total cpu time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

      END TYPE CONVERT_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_inform_type

!  return status. See CONVERT_between_matrix_formats (etc) for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the number of duplicates found (-ve = not checked)

        INTEGER :: duplicates = - 1

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  timings (see above)

        TYPE ( CONVERT_time_type ) :: time

      END TYPE CONVERT_inform_type

   CONTAINS

!-*-*-   C O N V E R T _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-

      SUBROUTINE CONVERT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by CONVERT_initialize could (roughly)
!  have been set as:

! BEGIN CONVERT SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  transpose-matrix                                  F
!  sum-duplicates                                    F
!  order-entries                                     F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END CONVERT SPECIFICATIONS

!  Dummy arguments

      TYPE ( CONVERT_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: transpose = print_level + 1
      INTEGER, PARAMETER :: sum_duplicates = transpose + 1
      INTEGER, PARAMETER :: order = sum_duplicates + 1
      INTEGER, PARAMETER :: space_critical = order + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 7 ), PARAMETER :: specname = 'CONVERT'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

!  Logical key-words

      spec( transpose )%keyword = 'transpose-matrix'
      spec( sum_duplicates )%keyword = 'sum-duplicates'
      spec( order )%keyword = 'order-entries'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( transpose ),                           &
                                  control%transpose,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( sum_duplicates ),                      &
                                  control%sum_duplicates,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( order ),                               &
                                  control%order,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

      RETURN

!  End of CONVERT_read_specfile

      END SUBROUTINE CONVERT_read_specfile

!-*-  C O N V E R T _ T O _ C O L U M N _ F O R M A T   S U B R O U T I N E  -*-

      SUBROUTINE CONVERT_between_matrix_formats( A, output_format, A_out,      &
                                                 control, inform )

!  convert the input matrix or its transpose to sparse-column format.
!
!  If the matrix is to be stored so that the row indices in each column
!  appear in increasing order or if the matrix is to be squeezed to sum
!  duplicate entries, the optional arrays IWORK and WORK of length at
!  least m (n when the transpose is sought) must be provided, with all
!  entries of IWORK set to 0; IWORK will have been reset to 0 on exit.
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!
!    A%m is an INTEGER variable, which must be set to the number of rows of A
!     RESTRICTION: A%m >= 1
!
!    A%n is an INTEGER variable, which must be set to the number of columns of A
!     RESTRICTION: A%n >= 1
!
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%ne         the number of nonzeros used to store A
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column
!       A%row( : )   the row indices of the components of A
!       A%ptr( : )   pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', A%type )
!       (alternatively A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type ) is allowed)
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column,
!                    with each the entries in each column in order of
!                    increasing row indicies.
!
!   ouput_format is a character string that specifies the desired output
!    format for A. Permissible values are:
!
!     'COORDINATE'        the matrix A_out will be as in i) above
!     'SPARSE_BY_ROWS'    the matrix A_out will be as in ii) above
!     'SPARSE_BY_COLUMNS' the matrix A_out will be as in iii) above
!     'DENSE_BY_ROWS'     the matrix A_out will be as in iv) above
!     'DENSE'             a synonym for 'DENSE_BY_ROWS'
!     'DENSE_BY_COLUMNS'  the matrix A_out will be as in v) above
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    (or its transpose, as desired) stored according to the format specified
!    by output_format. The output components provided will be precisely as
!
!   control and inform as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      CHARACTER ( LEN = * ) :: output_format
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: m
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IWORK
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%transpose ) THEN
        m = A%n
      ELSE
        m = A%m
      END IF

!  call appropriate translator

      SELECT CASE( TRIM( output_format ) )

!  output A as a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        CALL CONVERT_to_dense_row_format( A, A_out, control, inform )

!  output A as a dense matrix (stored by columns)

      CASE ( 'DENSE_BY_COLUMNS' )
        CALL CONVERT_to_dense_column_format( A, A_out, control, inform )

!  output A as a sparse matrix (stored by rows)

      CASE ( 'SPARSE_BY_ROWS' )

!  provide workspace if necessary

        IF ( control%sum_duplicates .OR. control%order ) THEN
          CALL SPACE_resize_array( m, IWORK, inform%status,                    &
                                   inform%alloc_status )
          IF ( inform%status /= GALAHAD_ok ) RETURN
          CALL SPACE_resize_array( m, WORK, inform%status,                     &
                                   inform%alloc_status )
          IF ( inform%status /= GALAHAD_ok ) RETURN
          IWORK = 0

          CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,        &
                                             IWORK, WORK )

!  discard workspace

          CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        ELSE
          CALL CONVERT_to_sparse_row_format( A, A_out, control, inform )
        END IF

!  output A as a sparse matrix (stored by columns)

      CASE ( 'SPARSE_BY_COLUMNS' )

!  provide workspace if necessary

        IF ( control%sum_duplicates .OR. control%order ) THEN
          CALL SPACE_resize_array( m, IWORK, inform%status,                    &
                                   inform%alloc_status )
          IF ( inform%status /= GALAHAD_ok ) RETURN
          CALL SPACE_resize_array( m, WORK, inform%status,                     &
                                   inform%alloc_status )
          IF ( inform%status /= GALAHAD_ok ) RETURN
          IWORK = 0

          CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,     &
                                                IWORK, WORK )
!  discard workspace

          CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        ELSE
          CALL CONVERT_to_sparse_column_format( A, A_out, control, inform )
        END IF

!  output A as a sparse matrix (stored by coordinates)

      CASE ( 'COORDINATE' )
        CALL CONVERT_to_coordinate_format( A, A_out, control, inform )

!  desired output format unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** desired output format ',    &
         &  A, ' unknown' )" ) prefix, TRIM( output_format )
      END SELECT

      RETURN

!  end of subroutine CONVERT_between_matrix_formats

      END SUBROUTINE CONVERT_between_matrix_formats

!-*-  C O N V E R T _ T O _ C O L U M N _ F O R M A T   S U B R O U T I N E  -*-

      SUBROUTINE CONVERT_to_sparse_column_format( A, A_out, control,           &
                                                  inform, IWORK, WORK )

!  convert the input matrix or its transpose to sparse-column format.
!
!  If the matrix is to be stored so that the row indices in each column
!  appear in increasing order or if the matrix is to be squeezed to sum
!  duplicate entries, the optional arrays IWORK and WORK of length at
!  least m (n when the transpose is sought) must be provided, with all
!  entries of IWORK set to 0; IWORK will have been reset to 0 on exit.
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_to_sparse_column_format above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    (or its transpose, as desired) stored as a sparse matrix by columns.
!
!    The following values will be set:
!
!       A_out%m   the number of rows of the output A
!       A_out%n   the number of columns of the output A
!       A_out%type( 1 : 17 ) = 'SPARSE_BY_COLUMNS'
!       A_out%val( : )   the values of the components of A, stored
!                            column by columns
!       A_out%row( : )   the row indices of the components of A
!       A_out%ptr( : )   pointers to the start of each column, and past
!                            the end of the last column
!
!   control, inform and the optional arguments as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, OPTIONAL, INTENT( INOUT ), DIMENSION( : ) :: IWORK
      REAL ( KIND = wp ), OPTIONAL, INTENT( INOUT ),                           &
        DIMENSION( : ) :: WORK

!  Local variables

      INTEGER :: i, j, k, l, ll, lu, m, n, ne, order_status
      REAL ( KIND = wp ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  discover the array size

      order_cols = .FALSE.

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = A%m * A%n
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%m + 1 ) - 1
        IF ( control%transpose ) order_cols = .TRUE.
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
        IF ( .NOT. control%transpose ) order_cols = .TRUE.
      CASE ( 'COORDINATE' )
        ne = A%ne
        order_cols = .TRUE.

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF

!  check that optional arrays are present and large enough if needed

      IF ( control%sum_duplicates .OR. ( control%order .AND. order_cols ) ) THEN
        IF ( .NOT. PRESENT( IWORK ) ) THEN
          inform%status = GALAHAD_error_optional
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( ' ', /, A, ' ** missing optional',        &
           &   ' argument(s)' )" ) prefix
          RETURN
        END IF

        IF ( SIZE( IWORK ) < m ) THEN
          inform%status = GALAHAD_error_integer_ws
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( ' ', /, A, ' ** length of iwork must at', &
           & ' at least ', I0 )" ) prefix, m
          RETURN
        END IF

        IF ( control%order .AND. order_cols ) THEN
          IF ( .NOT. PRESENT( WORK ) ) THEN
            inform%status = GALAHAD_error_optional
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( ' ', /, A, ' ** missing optional',      &
             &   ' argument(s)' )" ) prefix
            RETURN
          END IF

          IF ( SIZE( WORK ) < m ) THEN
            inform%status = GALAHAD_error_real_ws
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( ' ', /, A, ' ** length of work must',   &
             & ' be at least ', I0 )" ) prefix, m
            RETURN
          END IF
        END IF
      END IF

!  store A column-wise in A_out, perhaps with the row entries within
!  each column in increasing order

      A_out%m = m ; A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'SPARSE_BY_COLUMNS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_out%ptr'
      CALL SPACE_resize_array( A_out%n + 1, A_out%ptr,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%row'
      CALL SPACE_resize_array( ne, A_out%row,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%val'
      CALL SPACE_resize_array( ne, A_out%val,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          k = 0 ; l = 1
          DO j = 1, n
            A_out%ptr( j ) = l
            DO i = 1, m
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%row( l ) = i ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( n + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 1
          DO j = 1, n
            k = j
            A_out%ptr( j ) = l
            DO i = 1, m
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%row( l ) = i ; A_out%val( l ) = val
                l = l + 1
              END IF
              k = k + n
            END DO
          END DO
          A_out%ptr( n + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%ptr( : n + 1 ) = A%ptr( : n + 1 )
          A_out%row( : A_out%ne ) = A%col( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          CALL CONVERT_transpose( A%m, A%n, ne, A%ptr, A%row, A%val,           &
                                  A_out%ptr, A_out%row, A_out%val )

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%ptr( : n ) = 0
          DO l = 1, A%ne
            j= A%row( l )
            A_out%ptr( j ) = A_out%ptr( j ) + 1
          END DO
          l = 1
          DO j = 1, n
            i = A_out%ptr( j ) + l
            A_out%ptr( j ) = l
            l = i
          END DO
          DO l = 1, A%ne
            j = A%row( l ) ; i = A_out%ptr( j )
            A_out%row( i ) = A%col( l ) ; A_out%val( i ) = A%val( l )
            A_out%ptr( j ) = i + 1
          END DO
          DO j = n, 1, - 1
            A_out%ptr( j + 1 ) = A_out%ptr( j )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 1
          DO j = 1, n
            k = j
            A_out%ptr( j ) = l
            DO i = 1, m
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%row( l ) = i ; A_out%val( l ) = val
                l = l + 1
              END IF
              k = k + n
            END DO
          END DO
          A_out%ptr( n + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          k = 0 ; l = 1
          DO j = 1, n
            A_out%ptr( j ) = l
            DO i = 1, m
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%row( l ) = i ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( n + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          CALL CONVERT_transpose( A%n, A%m, ne, A%ptr, A%col, A%val,           &
                                  A_out%ptr, A_out%row, A_out%val )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%ptr( : n + 1 ) = A%ptr( : n + 1 )
          A_out%row( : A_out%ne ) = A%row( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%ptr( : n ) = 0
          DO l = 1, A%ne
            j = A%col( l )
            A_out%ptr( j ) = A_out%ptr( j ) + 1
          END DO
          l = 1
          DO j = 1, n
            i = A_out%ptr( j ) + l
            A_out%ptr( j ) = l
            l = i
          END DO
          DO l = 1, A%ne
            j = A%col( l ) ; i = A_out%ptr( j )
            A_out%row( i ) = A%row( l ) ; A_out%val( i ) = A%val( l )
            A_out%ptr( j ) = i + 1
          END DO
          DO j = n, 1, - 1
            A_out%ptr( j + 1 ) = A_out%ptr( j )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT
      END IF

!  sum duplicate entries and squeeze the storage space

      IF ( control%sum_duplicates ) THEN

!  consider each column one at a time

        k = 1 ; inform%duplicates = 0
        DO i = 1, n
          ll = k

!  loop over the rows j in the ith column

          DO l = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1
            j = A_out%row( l )

!  if iwork(j) = 0, the entry is not a duplicate, so record the row and value,
!  and flag in iwork(j) the location where the value is stored in A_out%val

            IF ( IWORK( j ) == 0 ) THEN
              A_out%row( k ) = A_out%row( l )
              A_out%val( k ) = A_out%val( l )
              IWORK( j ) = k
              k = k + 1

!  if iwork(j) /= 0, the entry is a duplicate, and its value should be added
!  to A_out%val(iwork(j))

            ELSE
              inform%duplicates = inform%duplicates + 1
              j = IWORK( j )
              A_out%val( j ) = A_out%val( j ) + A_out%val( l )
            END IF
          END DO

!  reset IWORK to zero

          DO l = ll, k - 1
            IWORK(  A_out%row( l ) ) = 0
          END DO

!  reset the pointer to the start of the column

          A_out%ptr( i ) = ll
        END DO
        A_out%ptr( n + 1 ) = k
        A_out%ne = k - 1
      END IF

!  order the row entries within each column in increasing row order

      inform%status = GALAHAD_ok
      IF ( control%order .AND. order_cols ) THEN
        DO i = 1, n
          ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
          IF ( lu > ll ) THEN
            CALL CONVERT_order( m, lu - ll + 1, A_out%row( ll : lu ),          &
                                A_out%val( ll : lu ), order_status,            &
                                IWORK, WORK )
            IF ( order_status == GALAHAD_warning_repeated_entry )              &
              inform%status = order_status
          END IF
        END DO
      END IF

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_to_sparse_column_format

      END SUBROUTINE CONVERT_to_sparse_column_format

!- C O N V E R T _ T O _ S P A R S E _ R O W _ F O R M A T  S U B R O U T I N E

      SUBROUTINE CONVERT_to_sparse_row_format( A, A_out, control,              &
                                               inform, IWORK, WORK )

!  convert the input matrix or its transpose to sparse-row format.
!
!  If the matrix is to be stored so that the column indices in each row
!  appear in increasing order or if the matrix is to be squeezed to sum
!  duplicate entries, the optional arrays IWORK and WORK of length at
!  least m (n when the transpose is sought) must be provided, with all
!  entries of IWORK set to 0; IWORK will have been reset to 0 on exit.
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    (or its transpose, as desired) stored as a sparse matrix by columns.
!
!    The following values will be set:
!
!       A_out%m   the number of rows of the output A
!       A_out%n   the number of columns of the output A
!       A_out%type( 1 : 17 ) = 'SPARSE_BY_COLUMNS'
!       A_out%val( : )   the values of the components of A, stored
!                            column by columns
!       A_out%col( : )   the column indices of the components of A
!       A_out%ptr( : )   pointers to the start of each row, and past
!                            the end of the last row
!
!   control, inform and the optional arguments as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, OPTIONAL, INTENT( INOUT ), DIMENSION( : ) :: IWORK
      REAL ( KIND = wp ), OPTIONAL, INTENT( INOUT ),                           &
        DIMENSION( : ) :: WORK

!  Local variables

      INTEGER :: i, j, k, l, ll, lu, m, n, ne, order_status
      REAL ( KIND = wp ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  discover the array size

      order_cols = .FALSE.

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = A%m * A%n
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%m + 1 ) - 1
        IF ( control%transpose ) order_cols = .TRUE.
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
        IF ( .NOT. control%transpose ) order_cols = .TRUE.
      CASE ( 'COORDINATE' )
        ne = A%ne
        order_cols = .TRUE.

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF

!  check that optional arrays are present and large enough if needed

      IF ( control%sum_duplicates .OR. ( control%order .AND. order_cols ) ) THEN
        IF ( .NOT. PRESENT( IWORK ) ) THEN
          inform%status = GALAHAD_error_optional
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( ' ', /, A, ' ** missing optional',        &
           &   ' argument(s)' )" ) prefix
          RETURN
        END IF

        IF ( SIZE( IWORK ) < m ) THEN
          inform%status = GALAHAD_error_integer_ws
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( ' ', /, A, ' ** length of iwork must at', &
           & ' at least ', I0 )" ) prefix, m
          RETURN
        END IF

        IF ( control%order .AND. order_cols ) THEN
          IF ( .NOT. PRESENT( WORK ) ) THEN
            inform%status = GALAHAD_error_optional
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( ' ', /, A, ' ** missing optional',      &
             &   ' argument(s)' )" ) prefix
            RETURN
          END IF

          IF ( SIZE( WORK ) < m ) THEN
            inform%status = GALAHAD_error_real_ws
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( ' ', /, A, ' ** length of work must',   &
             & ' be at least ', I0 )" ) prefix, m
            RETURN
          END IF
        END IF
      END IF

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%m = m ; A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'SPARSE_BY_ROWS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_out%ptr'
      CALL SPACE_resize_array( A_out%m + 1, A_out%ptr,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%col'
      CALL SPACE_resize_array( ne, A_out%col,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%val'
      CALL SPACE_resize_array( ne, A_out%val,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 1
          DO i = 1, m
            k = i
            A_out%ptr( i ) = l
            DO j = 1, n
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%col( l ) = j ; A_out%val( l ) = val
                l = l + 1
              END IF
              k = k + m
            END DO
          END DO
          A_out%ptr( m + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          k = 0 ; l = 1
          DO i = 1, m
            A_out%ptr( i ) = l
            DO j = 1, n
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%col( l ) = j ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( m + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          CALL CONVERT_transpose( A%n, A%m, ne, A%ptr, A%col, A%val,           &
                                  A_out%ptr, A_out%col, A_out%val )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%ptr( : m + 1 ) = A%ptr( : m + 1 )
          A_out%col( : A_out%ne ) = A%row( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%ptr( : m ) = 0
          DO l = 1, A%ne
            i = A%col( l )
            A_out%ptr( i ) = A_out%ptr( i ) + 1
          END DO
          l = 1
          DO i = 1, m
            j = A_out%ptr( i ) + l
            A_out%ptr( i ) = l
            l = j
          END DO
          DO l = 1, A%ne
            i = A%col( l ) ; j = A_out%ptr( i )
            A_out%col( j ) = A%row( l ) ; A_out%val( j ) = A%val( l )
            A_out%ptr( i ) = j + 1
          END DO
          DO i = m, 1, - 1
            A_out%ptr( i + 1 ) = A_out%ptr( i )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          k = 0 ; l = 1
          DO i = 1, m
            A_out%ptr( i ) = l
            DO j = 1, n
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%col( l ) = j ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( m + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 1
          DO i = 1, m
            k = i
            A_out%ptr( i ) = l
            DO j = 1, n
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_out%col( l ) = j ; A_out%val( l ) = val
                l = l + 1
              END IF
              k = k + m
            END DO
          END DO
          A_out%ptr( m + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%ptr( : m + 1 ) = A%ptr( : m + 1 )
          A_out%col( : A_out%ne ) = A%col( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          CALL CONVERT_transpose( A%m, A%n, ne, A%ptr, A%row, A%val,           &
                                  A_out%ptr, A_out%col, A_out%val )

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%ptr( : m ) = 0
          DO l = 1, A%ne
            i = A%row( l )
            A_out%ptr( i ) = A_out%ptr( i ) + 1
          END DO
          l = 1
          DO i = 1, m
            j = A_out%ptr( i ) + l
            A_out%ptr( i ) = l
            l = j
          END DO
          DO l = 1, A%ne
            i = A%row( l ) ; j = A_out%ptr( i )
            A_out%col( j ) = A%col( l ) ; A_out%val( j ) = A%val( l )
            A_out%ptr( i ) = j + 1
          END DO
          DO i = m, 1, - 1
            A_out%ptr( i + 1 ) = A_out%ptr( i )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT
      END IF

!  sum duplicate entries and squeeze the storage space

      IF ( control%sum_duplicates ) THEN

!  consider each row one at a time

        k = 1 ; inform%duplicates = 0
        DO i = 1, m
          ll = k

!  loop over the columns j in the ith row

          DO l = A_out%ptr( i ), A_out%ptr( i + 1 ) - 1
            j = A_out%col( l )

!  if iwork(j) = 0, the entry is not a duplicate, so record the column and value
!  and flag in iwork(j) the location where the value is stored in A_out%val

            IF ( IWORK( j ) == 0 ) THEN
              A_out%col( k ) = A_out%col( l )
              A_out%val( k ) = A_out%val( l )
              IWORK( j ) = k
              k = k + 1

!  if iwork(j) /= 0, the entry is a duplicate, and its value should be added
!  to A_out%val(iwork(j))

            ELSE
              inform%duplicates = inform%duplicates + 1
              j = IWORK( j )
              A_out%val( j ) = A_out%val( j ) + A_out%val( l )
            END IF
          END DO

!  reset IWORK to zero

          DO l = ll, k - 1
            IWORK( A_out%col( l ) ) = 0
          END DO

!  reset the pointer to the start of the row

          A_out%ptr( i ) = ll
        END DO
        A_out%ptr( m + 1 ) = k
        A_out%ne = k - 1
      END IF

!  order the row entries within each row in increasing column order

      inform%status = GALAHAD_ok
      IF ( control%order .AND. order_cols ) THEN
        DO i = 1, m
          ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
          IF ( lu > ll ) THEN
            CALL CONVERT_order( n, lu - ll + 1, A_out%col( ll : lu ),          &
                                A_out%val( ll : lu ), order_status,            &
                                IWORK, WORK )
            IF ( order_status == GALAHAD_warning_repeated_entry )              &
              inform%status = order_status
          END IF
        END DO
      END IF

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_to_sparse_row_format

      END SUBROUTINE CONVERT_to_sparse_row_format

!- C O N V E R T _ T O _ C O O R D I N A T E _ F O R M A T  S U B R O U T I N E

      SUBROUTINE CONVERT_to_coordinate_format( A, A_out, control, inform )

!  convert the input matrix or its transpose to sparse co-ordinate format.

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!   (or its transpose, as desired) stored as a sparse matrix in co-ordinate form
!
!    The following values will be set:
!
!       A_out%m   the number of rows of the output A
!       A_out%n   the number of columns of the output A
!       A_out%ne  the number of nonzeros used to store the output A
!       A_out%type( 1 : 10 ) = 'COORDINATE'
!       A_out%val( : )   the values of the components of A, stored
!                            column by columns
!       A_out%row( : )   the row indices of the components of A
!       A_out%col( : )   the column indices of the components of A
!
!   control and inform as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, k, l, m, n, ne
      REAL ( KIND = wp ) :: val, time_start, time_now, clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  discover the array size

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = A%m * A%n
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%m + 1 ) - 1
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
      CASE ( 'COORDINATE' )
        ne = A%ne

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%m = m ; A_out%n = n
      CALL SMT_put( A_out%type, 'COORDINATE', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_out%row'
      CALL SPACE_resize_array( ne, A_out%row,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%col'
      CALL SPACE_resize_array( ne, A_out%col,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out%val'
      CALL SPACE_resize_array( ne, A_out%val,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 0 ; k = 0
          DO i = 1, A%m
            DO j = 1, A%n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0 ; k = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          k = 0
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = A%col( l ) ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          k = 0
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = A%row( l )
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          k = 0
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              k = k + 1
              A_out%row( k ) = A%col( l ) ; A_out%col( k ) = A%row( l )
              A_out%val( k ) = val
            END IF
          END DO
          A_out%ne = k
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 0 ; k = 0
          DO i = 1, A%m
            DO j = 1, A%n
              l = l + 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0 ; k = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          k = 0
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = A%col( l )
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          k = 0
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero ) THEN
                k = k + 1
                A_out%row( k ) = A%row( l ) ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          k = 0
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero ) THEN
              k = k + 1
              A_out%row( k ) = A%row( l ) ; A_out%col( k ) = A%col( l )
              A_out%val( k ) = val
            END IF
          END DO
          A_out%ne = k
        END SELECT
      END IF

!  order the row entries within each row in increasing column order

      inform%status = GALAHAD_ok

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_to_coordinate_format

      END SUBROUTINE CONVERT_to_coordinate_format

!- C O N V E R T _ T O _ D E N S E _ R O W _ F O R M A T  S U B R O U T I N E

      SUBROUTINE CONVERT_to_dense_row_format( A, A_out, control, inform )

!  convert the input matrix or its transpose to dense-row format.

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    (or its transpose, as desired) stored as a dense matrix by columns.
!
!    The following values will be set:
!
!       A_out%m   the number of rows of the output A
!       A_out%n   the number of columns of the output A
!       A_out%type( 1 : 13 ) = 'DENSE_BY_ROWS'
!       A_out%val( : )   the values of the components of A, stored
!                            as a dense matrix row by row
!
!   control and inform as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, k, l, m, n, ne
      REAL ( KIND = wp ) :: time_start, time_now, clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  discover the array size

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',                     &
             'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' )

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF
      ne = m * n

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%m = m ; A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'DENSE_BY_ROWS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_out%val'
      CALL SPACE_resize_array( ne, A_out%val,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 0
          DO i = 1, A%m
            DO j = 1, A%n
              l = l + 1
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%val = zero
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%val = zero
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              i = A%row( l )
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%val = zero
          DO l = 1, A%ne
            i = A%row( l )
            j = A%col( l )
            k = A%m * ( j - 1 ) + i
            A_out%val( k ) = A_out%val( k ) + A%val( l )
          END DO
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          A_out%val( 1 : ne ) = A%val( 1 : ne )

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%val = zero
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%val = zero
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              i = A%row( l )
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%val = zero
          DO l = 1, A%ne
            i = A%row( l )
            j = A%col( l )
            k = A%n * ( i - 1 ) + j
            A_out%val( k ) = A_out%val( k ) + A%val( l )
          END DO
        END SELECT
      END IF

!  order the row entries within each row in increasing column order

      inform%status = GALAHAD_ok

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_to_dense_row_format

      END SUBROUTINE CONVERT_to_dense_row_format

!-*-*- C O N V E R T _ T O _ D E N S E _ C O L U M N   S U B R O U T I N E -*-*-

      SUBROUTINE CONVERT_to_dense_column_format( A, A_out, control, inform )

!  convert the input matrix or its transpose to dense-column format.

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    (or its transpose, as desired) stored as a dense matrix by columns.
!
!    The following values will be set:
!
!       A_out%m   the number of rows of the output A
!       A_out%n   the number of columns of the output A
!       A_out%type( 1 : 16 ) = 'DENSE_BY_COLUMNS'
!       A_out%val( : )   the values of the components of A, stored
!                            as a dense matrix column by column
!
!   control and inform as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, k, l, m, n, ne
      REAL ( KIND = wp ) :: time_start, time_now, clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  discover the array size

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',                     &
             'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' )

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF
      ne = m * n

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%m = m ; A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'DENSE_BY_COLUMNS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_out%val'
      CALL SPACE_resize_array( ne, A_out%val,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 0
          DO i = 1, A%m
            DO j = 1, A%n
              l = l + 1
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%val = zero
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%val = zero
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              i = A%row( l )
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%val = zero
          DO l = 1, A%ne
            i = A%row( l )
            j = A%col( l )
            k = A%n * ( i - 1 ) + j
            A_out%val( k ) = A_out%val( k ) + A%val( l )
          END DO
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          l = 0
          DO i = 1, A%m
            DO j = 1, A%n
              l = l + 1
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A%val( l )
            END DO
          END DO

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          A_out%val( 1 : ne ) = A%val( 1 : ne )

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%val = zero
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%val = zero
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              i = A%row( l )
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A_out%val( k ) + A%val( l )
            END DO
          END DO

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          A_out%val = zero
          DO l = 1, A%ne
            i = A%row( l )
            j = A%col( l )
            k = A%m * ( j - 1 ) + i
            A_out%val( k ) = A_out%val( k ) + A%val( l )
          END DO
        END SELECT
      END IF

!  order the row entries within each row in increasing column order

      inform%status = GALAHAD_ok

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_to_dense_column_format

      END SUBROUTINE CONVERT_to_dense_column_format

!-*-*-*-*-  C O N V E R T _ T R A N S P O S E   S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE CONVERT_transpose( m, n, ne, A_ptr, A_ind, A_val,             &
                                    A_transpose_ptr, A_transpose_ind,          &
                                    A_transpose_val )

!  given a matrix A stored by column, compute its transpose stored by column

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, ne
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
      INTEGER, INTENT( OUT ), DIMENSION( m + 1 ) :: A_transpose_ptr
      INTEGER, INTENT( IN ), DIMENSION( ne ) :: A_ind
      INTEGER, INTENT( OUT ), DIMENSION( ne ) :: A_transpose_ind
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ne ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( ne ) :: A_transpose_val

!  Local variables

      INTEGER :: i, j, k, l

!  store A_new row-wise. First count the number of entries in each row

      A_transpose_ptr( 1 : m ) = 0
      DO j = 1, n
        DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
          i = A_ind( l )
          A_transpose_ptr( i ) = A_transpose_ptr( i ) + 1
        END DO
      END DO

!  now set the starting addresses for the rows

      l = 1
      DO i = 1, m
        j = A_transpose_ptr( i )
        A_transpose_ptr( i ) = l
        l = l + j
      END DO

!  next insert the entries into the rows

      DO j = 1, n
        DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
          i = A_ind( l )
          k = A_transpose_ptr( i )
          A_transpose_ind( k ) = j
          A_transpose_val( k ) = A_val( l )
          A_transpose_ptr( i ) = k + 1
        END DO
      END DO

!  finally, reset the row starting addresss

      DO i = m, 1, - 1
        A_transpose_ptr( i + 1 ) = A_transpose_ptr( i )
      END DO
      A_transpose_ptr( 1 ) = 1

      RETURN

!  end of subroutine CONVERT_transpose

      END SUBROUTINE CONVERT_transpose

!-*-*-*-*-*-*-   C O N V E R T _ O R D E R   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE CONVERT_order( n, nz, IND, VAL, status, IW, W )

!  sort the vectors (IND,VAL) of length nnz, for which each component of IND
!  is a unique integer in 1:n, so that on output IND is in increasing order.
!  IW should be set to 0 on entry, and will have been reset to 0 on exit.

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, nz
      INTEGER, INTENT( OUT ) :: status
      INTEGER, INTENT( INOUT ), DIMENSION( nz ) :: IND
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( : ) :: IW
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nz ) :: VAL
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W

!  Local variables

      INTEGER :: i, j, jm1, k
      REAL ( KIND = wp ) :: v
      LOGICAL :: workspace

      workspace = SIZE( W ) >= n .AND. SIZE( IW ) >= n
      status = GALAHAD_ok

!  if nz is large relative to m and we have extra workspace, do a counting sort

      IF ( nz ** 2 >= n .AND. workspace ) THEN
!       IW = 0
        DO i = 1, nz
          j = IND( i )
          IF ( IW( j ) == 0 ) THEN
            IW( j ) = j
          ELSE
            status = GALAHAD_warning_repeated_entry
          END IF
          W( j ) = VAL( i )
        END DO
        k = 0
        DO j = 1, n
          IF ( IW( j ) > 0 ) THEN
            k = k + 1
            IND( k ) = IW( j ) ; VAL( k ) = W( j )
          END IF
        END DO
        DO i = 1, nz
          IW( IND( i ) ) = 0
        END DO

!  otherwise do an exchange sort

      ELSE
        DO k = 2, nz
          DO j = k, 2, - 1
            jm1 = j - 1
            IF ( IND( j ) > IND( jm1 ) ) EXIT
            i = IND( j ) ; IND( j ) = IND( jm1 ) ; IND( jm1 ) = i
            v = VAL( j ) ; VAL( j ) = VAL( jm1 ) ; VAL( jm1 ) = v
          END DO
        END DO
      END IF

      RETURN

!  end of subroutine CONVERT_order

      END SUBROUTINE CONVERT_order

!  end of module GALAHAD_CONVERT_double

    END MODULE GALAHAD_CONVERT_double
