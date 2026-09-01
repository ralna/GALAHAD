! THIS VERSION: GALAHAD 5.6 - 2026-09-01 AT 10:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-  G A L A H A D _ C O N V E R T    M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started June 8th 2014
!   originally released GALAHAD Version 2.6. June 8th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_CONVERT_precision

!      -------------------------------------------------------------------
!     |                                                                   |
!     | Given a real matrix A stored in one format, convert it to another |
!     |                                                                   |
!      -------------------------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok,                                   &
                                 GALAHAD_error_allocate,                       &
                                 GALAHAD_error_deallocate,                     &
                                 GALAHAD_error_restrictions,                   &
                                 GALAHAD_error_optional,                       &
                                 GALAHAD_error_unknown_storage,                &
                                 GALAHAD_error_integer_ws,                     &
                                 GALAHAD_error_real_ws,                        &
                                 GALAHAD_warning_repeated_entry 
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_TYPE, SMT_GET, SMT_PUT
      USE GALAHAD_SORT_precision, ONLY: SORT_heapsort_build,                   &
                                        SORT_heapsort_smallest
      USE GALAHAD_STRING, ONLY: STRING_ordinal

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE
      PUBLIC :: CONVERT_read_specfile,                                         &
                CONVERT_between_matrix_formats, CONVERT_to_sparse_row_format,  &
                CONVERT_to_sparse_column_format, CONVERT_to_coordinate_format, &
                CONVERT_to_dense_row_format, CONVERT_to_dense_column_format,   &
                CONVERT_between_symmetric_formats,                             &
                CONVERT_to_sparse_symmetric_row_format,                        &
                CONVERT_to_sparse_symmetric_column_format,                     &
                CONVERT_to_symmetric_coordinate_format,                        &
                CONVERT_to_dense_symmetric_row_format,                         &
                CONVERT_to_dense_symmetric_column_format,                      &
                CONVERT_transpose, CONVERT_order, CONVERT_information,         &
                CONVERT_map_values, CONVERT_increasing_order,                  &
                CONVERT_compress_duplicates,                                   &
                SMT_type, SMT_put, SMT_get

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

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

        REAL ( KIND = rp_ ) :: total = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

      END TYPE CONVERT_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_inform_type

!  return status. See CONVERT_between_matrix_formats (etc) for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the number of duplicates found (-ve = not checked)

        INTEGER ( KIND = ip_ ) :: duplicates = - 1

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  timings (see above)

        TYPE ( CONVERT_time_type ) :: time

      END TYPE CONVERT_inform_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: CONVERT_full_data_type
        LOGICAL :: f_indexing, explicit_a
        TYPE ( CONVERT_control_type ) :: CONVERT_control
        TYPE ( CONVERT_inform_type ) :: CONVERT_inform
      END TYPE CONVERT_full_data_type

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: transpose = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: sum_duplicates = transpose + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: order = sum_duplicates + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = order + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
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

!-*-  C O N V E R T _ B E T W E E N _ M _ F O R M A T S  S U B R O U T I N E -*-

      SUBROUTINE CONVERT_between_matrix_formats( A, output_format, A_out,      &
                                                 control, inform, MAP )

!  convert the input matrix or its transpose to a specified output format.
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
!    above
!
!   control and inform as above
!
!   MAP is an optional integer array of length SIZE(A%val) that, if present,
!    provides a map between the values in A%val and those in A_out%val. In 
!    particular A_out%val(MAP(i)) = A%val(i) for i = 1, ..., SIZE(A%val)

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      CHARACTER ( LEN = * ) :: output_format
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: m

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
        CALL CONVERT_to_dense_row_format( A, A_out, control, inform,           &
                                          MAP = MAP )

!  output A as a dense matrix (stored by columns)

      CASE ( 'DENSE_BY_COLUMNS' )
        CALL CONVERT_to_dense_column_format( A, A_out, control, inform,        &
                                             MAP = MAP )

!  output A as a sparse matrix (stored by rows)

      CASE ( 'SPARSE_BY_ROWS' )

        CALL CONVERT_to_sparse_row_format( A, A_out, control, inform,          &
                                           MAP = MAP )

!  output A as a sparse matrix (stored by columns)

      CASE ( 'SPARSE_BY_COLUMNS' )

!  provide workspace if necessary

        CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,       &
                                              MAP = MAP )

!  output A as a sparse matrix (stored by coordinates)

      CASE ( 'COORDINATE' )
        CALL CONVERT_to_coordinate_format( A, A_out, control, inform,          &
                                           MAP = MAP )

!  desired output format unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_unknown_storage
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** desired output format ',    &
         &  A, ' unknown' )" ) prefix, TRIM( output_format )
      END SELECT

      RETURN

!  end of subroutine CONVERT_between_matrix_formats

      END SUBROUTINE CONVERT_between_matrix_formats

!-*-  C O N V E R T _ T O _ C O L U M N _ F O R M A T   S U B R O U T I N E  -*-

      SUBROUTINE CONVERT_to_sparse_column_format( A, A_out, control, inform,   &
                                                  MAP )

!  convert the input matrix or its transpose to sparse-column format.
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!    (see CONVERT_to_sparse_column_format above)
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
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, ll, lu, m, n, ne, order_status
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols, sum_duplicates, map_vals
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK, INV
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  set the output type

      CALL SMT_put( A_out%type, 'SPARSE_BY_COLUMNS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  discover the array size

      order_cols = .FALSE.
      IF ( A%m > 0 ) THEN
        sum_duplicates = control%sum_duplicates 
      ELSE
        sum_duplicates = .FALSE.
      END IF
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
        IF ( A%m > 0 ) order_cols = .TRUE.

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

!  store A column-wise in A_out, perhaps with the row entries within
!  each column in increasing order

      A_out%m = m ; A_out%n = n ; A_out%ne = ne

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

!  special case - m = 0

      IF ( A%m == 0 ) THEN
        A_out%ptr( : A_out%n + 1 ) = 1
        inform%status = GALAHAD_ok ; GO TO 890
      END IF

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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
          IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          CALL CONVERT_transpose( A%m, A%n, ne, A%ptr, A%row, A%val,           &
                                  A_out%ptr, A_out%row, A_out%val, MAP = MAP )

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
            IF ( map_vals ) MAP( l ) = i
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
                A_out%row( l ) = i ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( n + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          CALL CONVERT_transpose( A%n, A%m, ne, A%ptr, A%col, A%val,           &
                                  A_out%ptr, A_out%row, A_out%val, MAP = MAP )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%ptr( : n + 1 ) = A%ptr( : n + 1 )
          A_out%row( : A_out%ne ) = A%row( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )
          IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

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
            IF ( map_vals ) MAP( l ) = i
          END DO
          DO j = n, 1, - 1
            A_out%ptr( j + 1 ) = A_out%ptr( j )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT
      END IF

!  if the mapping array is present, make sure that it is adjusted if
!  the matrix is to be stored so that the row indices in each column appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries

      IF ( map_vals ) THEN

!  compute the inverse, INV, to the map array, MAP, and workspace IWORK, 
!  and initialize INV

        array_name = 'CONVERT: INV'
        CALL SPACE_resize_array( ne, INV,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( m, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO i = 1, ne
          INV( MAP( i ) ) = i 
        END DO

!  if required, reorder so that the row indices in each column appear
!  in increasing order 

        IF ( control%order .AND. order_cols ) THEN
          CALL CONVERT_increasing_order( n, A_out%ne, A_out%ptr, A_out%row,    &
                                         A_out%val, MAP, INV )
        END IF

!  sum duplicate entries and squeeze the storage space

        CALL CONVERT_compress_duplicates( m, n, A_out%ne, A_out%ptr,           &
                                          A_out%row, A_out%val, MAP, INV,      &
                                          IWORK )
        A_out%ne = A_out%ptr( n + 1 ) - 1

!  deallocate workspace

        CALL SPACE_dealloc_array( INV, inform%status, inform%alloc_status )

!  skip to the exit if there is no need to order rows within columns or to
!  sum dupicates

      ELSE 
        IF ( .NOT. ( sum_duplicates .OR.                                       &
                    ( control%order .AND. order_cols ) ) ) GO TO 890

!  if the matrix is to be stored so that the row indices in each column appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries, but no mapping s required, the arrays IWORK and WORK of length m 
!  (n when the transpose is sought) are used, with IWORK initialised as 0

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( m, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        IWORK( : m ) = 0

!  sum duplicate entries and squeeze the storage space

        IF ( sum_duplicates ) THEN

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
              IWORK( A_out%row( l ) ) = 0
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
          array_name = 'CONVERT: WORK'
          CALL SPACE_resize_array( m, WORK,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO i = 1, n
            ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
            IF ( lu > ll ) THEN
              CALL CONVERT_order( m, lu - ll + 1, A_out%row( ll : lu ),        &
                                  A_out%val( ll : lu ), order_status,          &
                                  IWORK, WORK )
              IF ( order_status == GALAHAD_warning_repeated_entry )            &
                inform%status = order_status
            END IF
          END DO
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        END IF
      END IF

!  deallocate workspace

      CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )

!  record the total time taken

  890 CONTINUE
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

      SUBROUTINE CONVERT_to_sparse_row_format( A, A_out, control, inform, MAP )

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
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, ll, lu, m, n, ne, order_status
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols, sum_duplicates, map_vals
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK, INV
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  discover the array size

      order_cols = .FALSE.
      IF ( A%m > 0 ) THEN
        sum_duplicates = control%sum_duplicates 
      ELSE
        sum_duplicates = .FALSE.
      END IF
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
        IF ( A%m > 0 ) order_cols = .TRUE.

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

!  special case - m = 0

      IF ( A%m == 0 ) THEN
        A_out%ptr( : A_out%m + 1 ) = 1
        inform%status = GALAHAD_ok ; GO TO 890
      END IF

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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
                A_out%col( l ) = j ; A_out%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_out%ptr( m + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          CALL CONVERT_transpose( A%n, A%m, ne, A%ptr, A%col, A%val,           &
                                  A_out%ptr, A_out%col, A_out%val, MAP = MAP )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_out%ptr( : m + 1 ) = A%ptr( : m + 1 )
          A_out%col( : A_out%ne ) = A%row( : A_out%ne )
          A_out%val( : A_out%ne ) = A%val( : A_out%ne )
          IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

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
            IF ( map_vals ) MAP( l ) = j
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
              IF ( map_vals ) MAP( k ) = l
              IF ( val /= zero .OR. map_vals ) THEN
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
          IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          CALL CONVERT_transpose( A%m, A%n, ne, A%ptr, A%row, A%val,           &
                                  A_out%ptr, A_out%col, A_out%val, MAP = MAP )

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
            IF ( map_vals ) MAP( l ) = j
          END DO
          DO i = m, 1, - 1
            A_out%ptr( i + 1 ) = A_out%ptr( i )
          END DO
          A_out%ptr( 1 ) = 1
        END SELECT
      END IF

!  if the mapping array is present, make sure that it is adjusted if the matrix
!  is to be stored so that the column indices in each row appear in increasing 
!  order or if the matrix is to be squeezed to sum duplicate  entries

      IF ( map_vals ) THEN

!  compute the inverse, INV, to the map array, MAP, and workspace IWORK, 
!  and initialize INV

        array_name = 'CONVERT: INV'
        CALL SPACE_resize_array( ne, INV,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO i = 1, ne
          INV( MAP( i ) ) = i 
        END DO

!  if required, reorder so that the column indices in each row appear
!  in increasing order 

        IF ( control%order .AND. order_cols ) THEN
          CALL CONVERT_increasing_order( m, A_out%ne, A_out%ptr, A_out%col,    &
                                         A_out%val, MAP, INV )
        END IF

!  sum duplicate entries and squeeze the storage space

        CALL CONVERT_compress_duplicates( n, m, A_out%ne, A_out%ptr,           &
                                          A_out%col, A_out%val, MAP, INV,      &
                                          IWORK )
        A_out%ne = A_out%ptr( m + 1 ) - 1

!  deallocate workspace

        CALL SPACE_dealloc_array( INV, inform%status, inform%alloc_status )

!  skip to the exit if there is no need to order rows within columns or to
!  sum dupicates

      ELSE 
        IF ( .NOT. ( sum_duplicates .OR.                                       &
                    ( control%order .AND. order_cols ) ) ) GO TO 890

!  if the matrix is to be stored so that the column indices in each row appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries, but no mapping is required, the arrays IWORK and WORK of length m 
!  (n when the transpose is sought) are used, with IWORK initialised as 0

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        IWORK( : n ) = 0

!  sum duplicate entries and squeeze the storage space

        IF ( sum_duplicates ) THEN

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

!  order the row entries within each column in increasing row order

        inform%status = GALAHAD_ok
        IF ( control%order .AND. order_cols ) THEN
          array_name = 'CONVERT: WORK'
          CALL SPACE_resize_array( n, WORK,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO i = 1, m
            ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
            IF ( lu > ll ) THEN
              CALL CONVERT_order( n, lu - ll + 1, A_out%col( ll : lu ),        &
                                  A_out%val( ll : lu ), order_status,          &
                                  IWORK, WORK )
              IF ( order_status == GALAHAD_warning_repeated_entry )            &
                inform%status = order_status
            END IF
          END DO
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        END IF
      END IF

!  deallocate workspace

      CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )

!  record the total time taken

  890 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

  900 CONTINUE
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

      SUBROUTINE CONVERT_to_coordinate_format( A, A_out, control, inform, MAP )

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
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, m, n, ne
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INV

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 .OR. A%m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n and A%m must be +ve' )")&
            prefix
        RETURN
      END IF

!  if it required to sum duplicates, first convert to sparse column format 

      IF ( A%m > 0 .AND. control%sum_duplicates ) THEN
        CALL CONVERT_to_sparse_column_format( A, A_out, control, inform,       &
                                              MAP = MAP )

!  now provide space for and extract the column number for each component

        ne = A_out%ptr( A_out%n + 1 ) - 1
        array_name = 'CONVERT: A_out%col'
        CALL SPACE_resize_array( ne, A_out%col,                                &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO j = 1, A_out%n
          A_out%col( A_out%ptr( j ) : A_out%ptr( j + 1 ) - 1 ) = j
        END DO

!  deallocate the column pointers

        CALL SPACE_dealloc_array( A_out%ptr, inform%status,                    &
                                  inform%alloc_status )
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

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
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
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
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          k = 0
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = A%col( l ) ; A_out%col( k ) = i
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          k = 0
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = j ; A_out%col( k ) = A%row( l )
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          k = 0
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = A%col( l ) ; A_out%col( k ) = A%row( l )
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
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
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
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
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          k = 0
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = i ; A_out%col( k ) = A%col( l )
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          k = 0
          DO j = 1, A%n
            DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
              val = A%val( l )
              IF ( val /= zero .OR. map_vals ) THEN
                k = k + 1
                A_out%row( k ) = A%row( l ) ; A_out%col( k ) = j
                A_out%val( k ) = val
              END IF
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO
          A_out%ne = k

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' )
          k = 0
          DO l = 1, A%ne
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = A%row( l ) ; A_out%col( k ) = A%col( l )
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
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

      SUBROUTINE CONVERT_to_dense_row_format( A, A_out, control, inform, MAP )

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
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, m, n, ne
      REAL ( KIND = rp_ ) :: time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INV

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

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

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
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A%val( l )
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
            IF ( map_vals ) MAP( l ) = k
          END DO
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE', 'DENSE_BY_ROWS' )
          A_out%val( 1 : ne ) = A%val( 1 : ne )
          IF ( map_vals ) MAP( 1 : ne ) = [ ( i, i = 1, ne ) ]

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 0
          DO j = 1, A%n
            DO i = 1, A%m
              l = l + 1
              k = A%n * ( i - 1 ) + j
              A_out%val( k ) = A%val( l )
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
            IF ( map_vals ) MAP( l ) = k
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

      SUBROUTINE CONVERT_to_dense_column_format( A, A_out, control, inform,    &
                                                 MAP )

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
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, m, n, ne
      REAL ( KIND = rp_ ) :: time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INV

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

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
            IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
            END DO
          END DO

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          A_out%val( 1 : ne ) = A%val( 1 : ne )
          IF ( map_vals ) MAP( 1 : ne ) = [ ( i, i = 1, ne ) ]

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_out%val = zero
          DO i = 1, A%m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              k = A%m * ( j - 1 ) + i
              A_out%val( k ) = A_out%val( k ) + A%val( l )
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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
              IF ( map_vals ) MAP( l ) = k
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

!-*-  C O N V E R T _ B E T W E E N _ S _ F O R M A T S  S U B R O U T I N E -*-

      SUBROUTINE CONVERT_between_symmetric_formats( A, output_format, A_out,   &
                                                    control, inform, MAP )

!  convert the input SYMMETRIC matrix to a specified output format.
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
!    ** Only the LOWER TRIANGLE of the matrix should be provided, the
!       structure and values of the upper triangle will be deduced 
!       from the assumed symmetry. When we say A below, we always mean 
!       the lower triangle of A **
!
!    A%n is an INTEGER variable, which must be set to the number of 
!     rows/columns of A
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
!    above
!
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      CHARACTER ( LEN = * ) :: output_format
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: n

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  call appropriate translator

      n = A%n
      SELECT CASE( TRIM( output_format ) )

!  output A as a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        CALL CONVERT_to_dense_symmetric_row_format( A, A_out, control,         &
                                                    inform, MAP = MAP )

!  output A as a dense matrix (stored by columns)

      CASE ( 'DENSE_BY_COLUMNS' )
        CALL CONVERT_to_dense_symmetric_column_format( A, A_out, control,      &
                                                       inform, MAP = MAP )

!  output A as a sparse matrix (stored by rows)

      CASE ( 'SPARSE_BY_ROWS' )

!  provide workspace if necessary

        CALL CONVERT_to_sparse_symmetric_row_format( A, A_out, control,        &
                                                     inform, MAP = MAP )

!  output A as a sparse matrix (stored by columns)

      CASE ( 'SPARSE_BY_COLUMNS' )

!  provide workspace if necessary

        CALL CONVERT_to_sparse_symmetric_column_format( A, A_out, control,     &
                                                        inform, MAP = MAP )

!  output A as a sparse matrix (stored by coordinates)

      CASE ( 'COORDINATE' )
        CALL CONVERT_to_symmetric_coordinate_format( A, A_out, control,        &
                                                     inform, MAP = MAP )

!  desired output format unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_unknown_storage
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** desired output format ',    &
         &  A, ' unknown' )" ) prefix, TRIM( output_format )
      END SELECT

      RETURN

!  end of subroutine CONVERT_between_symmetric_formats

      END SUBROUTINE CONVERT_between_symmetric_formats

! - C O N V E R T _ T O _ S _ C O L U M N _ F O R M A T   S U B R O U T I N E  -

      SUBROUTINE CONVERT_to_sparse_symmetric_column_format( A, A_out, control, &
                                                            inform, MAP )

!  convert the input symmetric matrix to sparse-column format
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_to_sparse_column_format above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    stored as a sparse matrix by columns.
!
!    The following values will be set:
!
!       A_out%n   the number of rows/columns of the output A
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
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, ll, lu, n, ne, order_status
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols, sum_duplicates, map_vals
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK, INV
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
            prefix
        RETURN
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  discover the array size

      order_cols = .FALSE.
      sum_duplicates = control%sum_duplicates
      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = ( A%n * ( A%n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%n + 1 ) - 1
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
        order_cols = .TRUE.
      CASE ( 'COORDINATE' )
        ne = A%ne
        order_cols = .TRUE.
      CASE ( 'DIAGONAL' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'SCALED-IDENTITY' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'IDENTITY' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'ZERO', 'NONE' )
        ne = 0
        sum_duplicates = .FALSE.

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      n = A%n

!  store A column-wise in A_out, perhaps with the row entries within
!  each column in increasing order

      A_out%n = n ; A_out%ne = ne
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

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        k = 0 ; l = 1
        A_out%ptr( 2 : n + 1 ) = 0
        DO i = 1, n
          DO j = 1, i
            k = k + 1
            val = A%val( k )
            IF ( map_vals ) MAP( k ) = l
            IF ( val /= zero .OR. map_vals ) THEN
              A_out%row( l ) = i ; A_out%val( l ) = val
              A_out%ptr( j + 1 ) = A_out%ptr( j + 1 ) + 1
              l = l + 1
            END IF
          END DO
        END DO
        A_out%ptr( 1 ) = 1
        DO i = 2, n + 1
          A_out%ptr( i ) = A_out%ptr( i ) + A_out%ptr( i - 1 )
        END DO

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )
        k = 0 ; l = 1
        DO j = 1, n
          A_out%ptr( j ) = l
          DO i = j, n
            k = k + 1
            val = A%val( k )
            IF ( map_vals ) MAP( k ) = l
            IF ( val /= zero .OR. map_vals ) THEN
              A_out%row( l ) = i ; A_out%val( l ) = val
              l = l + 1
            END IF
          END DO
        END DO
        A_out%ptr( n + 1 ) = l

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )
        CALL CONVERT_transpose( A%n, A%n, ne, A%ptr, A%col, A%val,             &
                                A_out%ptr, A_out%row, A_out%val, MAP = MAP )

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )
        A_out%ptr( : n + 1 ) = A%ptr( : n + 1 )
        A_out%row( : A_out%ne ) = A%row( : A_out%ne )
        A_out%val( : A_out%ne ) = A%val( : A_out%ne )
        IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

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
          IF ( map_vals ) MAP( l ) = i
        END DO
        DO j = n, 1, - 1
          A_out%ptr( j + 1 ) = A_out%ptr( j )
        END DO
        A_out%ptr( 1 ) = 1

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 : n )
        IF ( map_vals ) MAP( : n ) = [ ( i, i = 1, n ) ]
      CASE ( 'SCALED-IDENTITY' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 )
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'IDENTITY' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = 1.0_rp_
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'ZERO', 'NONE' )
        A_out%ptr( 1 : n + 1 ) = 1
      END SELECT

!  if the mapping array is present, make sure that it is adjusted if
!  the matrix is to be stored so that the row indices in each column appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries

      IF ( map_vals ) THEN

!  compute the inverse, INV, to the map array, MAP, and workspace IWORK, 
!  and initialize INV

        array_name = 'CONVERT: INV'
        CALL SPACE_resize_array( ne, INV,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO i = 1, ne
          INV( MAP( i ) ) = i 
        END DO

!  if required, reorder so that the row indices in each column appear
!  in increasing order 

        IF ( control%order .AND. order_cols ) THEN
          CALL CONVERT_increasing_order( n, A_out%ne, A_out%ptr, A_out%row,    &
                                         A_out%val, MAP, INV )
        END IF

!  sum duplicate entries and squeeze the storage space

        CALL CONVERT_compress_duplicates( n, n, A_out%ne, A_out%ptr,           &
                                          A_out%row, A_out%val, MAP, INV,      &
                                          IWORK )
        A_out%ne = A_out%ptr( n + 1 ) - 1

!  deallocate workspace

        CALL SPACE_dealloc_array( INV, inform%status, inform%alloc_status )

!  skip to the exit if there is no need to order rows within columns or to
!  sum dupicates

      ELSE 
        IF ( .NOT. ( sum_duplicates .OR.                                       &
                    ( control%order .AND. order_cols ) ) ) GO TO 890

!  if the matrix is to be stored so that the row indices in each column appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries, but no mapping s required, the arrays IWORK and WORK of length m 
!  (n when the transpose is sought) are used, with IWORK initialised as 0

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        IWORK( : n ) = 0

!  sum duplicate entries and squeeze the storage space

        IF ( sum_duplicates ) THEN

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
              IWORK( A_out%row( l ) ) = 0
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
          array_name = 'CONVERT: WORK'
          CALL SPACE_resize_array( n, WORK,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO i = 1, n
            ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
            IF ( lu > ll ) THEN
              CALL CONVERT_order( n, lu - ll + 1, A_out%row( ll : lu ),        &
                                  A_out%val( ll : lu ), order_status,          &
                                  IWORK, WORK )
              IF ( order_status == GALAHAD_warning_repeated_entry )            &
                inform%status = order_status
            END IF
          END DO
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        END IF
      END IF

!  deallocate workspace

      CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )

!  record the total time taken

 890  CONTINUE
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

!  end of subroutine CONVERT_to_sparse_symmetric_column_format

      END SUBROUTINE CONVERT_to_sparse_symmetric_column_format

! C O N V E R T _ T O _ S P A R S E _ S _ R O W _ F O R M A T  S U B R O U T INE

      SUBROUTINE CONVERT_to_sparse_symmetric_row_format( A, A_out, control,    &
                                                         inform, MAP )

!  convert the input symmetric matrix to sparse-row format.
!
!  If the matrix is to be stored so that the column indices in each row
!  appear in increasing order or if the matrix is to be squeezed to sum
!  duplicate entries, the optional arrays IWORK and WORK of length at
!  least n must be provided, with all entries of IWORK set to 0; IWORK 
!  will have been reset to 0 on exit.
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    stored as a sparse matrix by columns.
!
!    The following values will be set:
!
!       A_out%n   the number of rows/columns of the output A
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
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, ll, lu, n, ne, order_status
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: order_cols, sum_duplicates, map_vals
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK, INV
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
            prefix
        RETURN
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  discover the array size

      order_cols = .FALSE.
      sum_duplicates = control%sum_duplicates
      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = ( A%n * ( A%n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%n + 1 ) - 1
        order_cols = .TRUE.
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
        order_cols = .TRUE.
      CASE ( 'COORDINATE' )
        ne = A%ne
        order_cols = .TRUE.
      CASE ( 'DIAGONAL' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'SCALED-IDENTITY' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'IDENTITY' )
        ne = A%n
        sum_duplicates = .FALSE.
      CASE ( 'ZERO', 'NONE' )
        ne = 0
        sum_duplicates = .FALSE.

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      n = A%n

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'SPARSE_BY_ROWS', inform%alloc_status )
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

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        k = 0 ; l = 1
        DO i = 1, n
          A_out%ptr( i ) = l
          DO j = 1, i
            k = k + 1
            val = A%val( k )
            IF ( map_vals ) MAP( k ) = l
            IF ( val /= zero .OR. map_vals ) THEN
              A_out%col( l ) = j ; A_out%val( l ) = val
              l = l + 1
            END IF
          END DO
        END DO
        A_out%ptr( n + 1 ) = l

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )
        k = 0 ; l = 1
        A_out%ptr( 2 : n + 1 ) = 0
        DO j = 1, n
          DO i = j, n
            k = k + 1
            val = A%val( k )
            IF ( map_vals ) MAP( k ) = l
            IF ( val /= zero .OR. map_vals ) THEN
              A_out%col( l ) = j ; A_out%val( l ) = val
              A_out%ptr( i + 1 ) = A_out%ptr( i + 1 ) + 1
              l = l + 1
            END IF
          END DO
        END DO
        A_out%ptr( 1 ) = 1
        DO i = 2, n + 1
          A_out%ptr( i ) = A_out%ptr( i ) + A_out%ptr( i - 1 )
        END DO

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )
        A_out%ptr( : n + 1 ) = A%ptr( : n + 1 )
        A_out%col( : A_out%ne ) = A%col( : A_out%ne )
        A_out%val( : A_out%ne ) = A%val( : A_out%ne )
        IF ( map_vals ) MAP( : A_out%ne ) = [ ( i, i = 1, A_out%ne ) ]

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )
        CALL CONVERT_transpose( A%n, A%n, ne, A%ptr, A%row, A%val,             &
                                A_out%ptr, A_out%col, A_out%val, MAP = MAP )

!  A is a sparse matrix stored by its co-ordinates

      CASE ( 'COORDINATE' )
        A_out%ptr( : n ) = 0
        DO l = 1, A%ne
          i = A%row( l )
          A_out%ptr( i ) = A_out%ptr( i ) + 1
        END DO
        l = 1
        DO i = 1, n
          j = A_out%ptr( i ) + l
          A_out%ptr( i ) = l
          l = j
        END DO
        DO l = 1, A%ne
          i = A%row( l ) ; j = A_out%ptr( i )
          A_out%col( j ) = A%col( l ) ; A_out%val( j ) = A%val( l )
          A_out%ptr( i ) = j + 1
          IF ( map_vals ) MAP( l ) = j
        END DO
        DO i = n, 1, - 1
          A_out%ptr( i + 1 ) = A_out%ptr( i )
        END DO
        A_out%ptr( 1 ) = 1

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 : n )
        IF ( map_vals ) MAP( : n ) = [ ( i, i = 1, n ) ]
      CASE ( 'SCALED-IDENTITY' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 )
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'IDENTITY' )
        A_out%ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = 1.0_rp_
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'ZERO', 'NONE' )
        A_out%ptr( 1 : n + 1 ) = 1
      END SELECT

!  if the mapping array is present, make sure that it is adjusted if the matrix
!  is to be stored so that the column indices in each row appear in increasing 
!  order or if the matrix is to be squeezed to sum duplicate  entries

      IF ( map_vals ) THEN

!  compute the inverse, INV, to the map array, MAP, and workspace IWORK, 
!  and initialize INV

        array_name = 'CONVERT: INV'
        CALL SPACE_resize_array( ne, INV,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO i = 1, ne
          INV( MAP( i ) ) = i 
        END DO

!  if required, reorder so that the column indices in each row appear
!  in increasing order 

        IF ( control%order .AND. order_cols ) THEN
          CALL CONVERT_increasing_order( n, A_out%ne, A_out%ptr, A_out%col,    &
                                         A_out%val, MAP, INV )
        END IF

!  sum duplicate entries and squeeze the storage space

        CALL CONVERT_compress_duplicates( n, n, A_out%ne, A_out%ptr,           &
                                          A_out%col, A_out%val, MAP, INV,      &
                                          IWORK )
        A_out%ne = A_out%ptr( n + 1 ) - 1

!  deallocate workspace

        CALL SPACE_dealloc_array( INV, inform%status, inform%alloc_status )

!  skip to the exit if there is no need to order rows within columns or to
!  sum dupicates

      ELSE 
        IF ( .NOT. ( sum_duplicates .OR.                                       &
                    ( control%order .AND. order_cols ) ) ) GO TO 890

!  if the matrix is to be stored so that the column indices in each row appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate 
!  entries, but no mapping is required, the arrays IWORK and WORK of length n
!  are used, with IWORK initialised as 0

        array_name = 'CONVERT: IWORK'
        CALL SPACE_resize_array( n, IWORK,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        IWORK( : n ) = 0

!  sum duplicate entries and squeeze the storage space

        IF ( sum_duplicates ) THEN

!  consider each row one at a time

          k = 1 ; inform%duplicates = 0
          DO i = 1, n
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
          A_out%ptr( n + 1 ) = k
          A_out%ne = k - 1
        END IF

!  order the row entries within each column in increasing row order

        inform%status = GALAHAD_ok
        IF ( control%order .AND. order_cols ) THEN
          array_name = 'CONVERT: WORK'
          CALL SPACE_resize_array( n, WORK,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO i = 1, n
            ll =  A_out%ptr( i ) ; lu = A_out%ptr( i + 1 ) - 1
            IF ( lu > ll ) THEN
              CALL CONVERT_order( n, lu - ll + 1, A_out%col( ll : lu ),        &
                                  A_out%val( ll : lu ), order_status,          &
                                  IWORK, WORK )
              IF ( order_status == GALAHAD_warning_repeated_entry )            &
                inform%status = order_status
            END IF
          END DO
          CALL SPACE_dealloc_array( WORK, inform%status, inform%alloc_status )
        END IF
      END IF

!  deallocate workspace

      CALL SPACE_dealloc_array( IWORK, inform%status, inform%alloc_status )

!  record the total time taken

 890  CONTINUE
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

!  end of subroutine CONVERT_to_sparse_symmetric_row_format

      END SUBROUTINE CONVERT_to_sparse_symmetric_row_format

! C O N V E R T _ T O _ S _ C O O R D I N A T E _ F O R M A T  S U B R O U T INE

      SUBROUTINE CONVERT_to_symmetric_coordinate_format( A, A_out, control,    &
                                                         inform, MAP )

!  convert the input symmetric matrix to sparse co-ordinate format.

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!   stored as a sparse matrix in co-ordinate form
!
!    The following values will be set:
!
!       A_out%n   the number of rows/columns of the output A
!       A_out%ne  the number of nonzeros used to store the output A
!       A_out%type( 1 : 10 ) = 'COORDINATE'
!       A_out%val( : )   the values of the components of A, stored
!                            column by columns
!       A_out%row( : )   the row indices of the components of A
!       A_out%col( : )   the column indices of the components of A
!
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, n, ne
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK, INV
!     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
            prefix
        RETURN
      END IF

!  if it required to sum duplicates, first convert to sparse column format 

      IF ( control%sum_duplicates ) THEN
        CALL CONVERT_to_sparse_symmetric_column_format( A, A_out, control,     &
                                                        inform, MAP = MAP )

!  now provide space for and extract the column number for each component

        ne = A_out%ptr( A_out%n + 1 ) - 1
        array_name = 'CONVERT: A_out%col'
        CALL SPACE_resize_array( ne, A_out%col,                                &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO j = 1, A_out%n
          A_out%col( A_out%ptr( j ) : A_out%ptr( j + 1 ) - 1 ) = j
        END DO

!  deallocate the column pointers

        CALL SPACE_dealloc_array( A_out%ptr, inform%status,                    &
                                  inform%alloc_status )
      END IF

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  discover the array size

      SELECT CASE( SMT_get( A%type ) )
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = ( A%n * ( A%n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( A%n + 1 ) - 1
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( A%n + 1 ) - 1
      CASE ( 'COORDINATE' )
        ne = A%ne
      CASE ( 'DIAGONAL' )
        ne = A%n
      CASE ( 'SCALED-IDENTITY' )
        ne = A%n
      CASE ( 'IDENTITY' )
        ne = A%n
      CASE ( 'ZERO', 'NONE' )
        ne = 0

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      n = A%n

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%n = n
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

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        l = 0 ; k = 0
        DO i = 1, n
          DO j = 1, i
            l = l + 1
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = i ; A_out%col( k ) = j
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO
        A_out%ne = k

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )
        l = 0 ; k = 0
        DO j = 1, n
          DO i = j, n
            l = l + 1
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = i ; A_out%col( k ) = j
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO
        A_out%ne = k

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )
        k = 0
        DO i = 1, n
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = i ; A_out%col( k ) = A%col( l )
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO
        A_out%ne = k

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )
        k = 0
        DO j = 1, A%n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            val = A%val( l )
            IF ( val /= zero .OR. map_vals ) THEN
              k = k + 1
              A_out%row( k ) = A%row( l ) ; A_out%col( k ) = j
              A_out%val( k ) = val
            END IF
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO
        A_out%ne = k

!  A is a sparse matrix stored by its co-ordinates

      CASE ( 'COORDINATE' )
        k = 0
        DO l = 1, A%ne
          val = A%val( l )
          IF ( val /= zero .OR. map_vals ) THEN
            k = k + 1
            A_out%row( k ) = A%row( l ) ; A_out%col( k ) = A%col( l )
            A_out%val( k ) = val
          END IF
          IF ( map_vals ) MAP( l ) = k
        END DO
        A_out%ne = k

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 : n )
        MAP( 1 : n ) = [ ( i, i = 1, n ) ]
      CASE ( 'SCALED-IDENTITY' )
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = A%val( 1 )
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'IDENTITY' )
        A_out%row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%col( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out%val( 1 : n ) = 1.0_rp_
        IF ( map_vals ) MAP( : n ) = n + 1
      CASE ( 'ZERO', 'NONE' )
        A_out%ptr( 1 : n + 1 ) = 1
      END SELECT

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

!  end of subroutine CONVERT_to_symmetric_coordinate_format

      END SUBROUTINE CONVERT_to_symmetric_coordinate_format

! C O N V E R T _ T O _ D E N S E _ S _ R O W _ F O R M A T  S U B R O U T I N E

      SUBROUTINE CONVERT_to_dense_symmetric_row_format( A, A_out, control,     &
                                                        inform, MAP )

!  convert the input symmetric matrix to dense-row format.

!  if the lower trangular part of a symmetric matrix is stored as a vector 
!  by rows, the (i,j)th entry is in position 
!    i(i-1)/2 + j

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    stored as a dense matrix by columns.
!
!    The following values will be set:
!
!       A_out%n   the number of rows/columns of the output A
!       A_out%type( 1 : 13 ) = 'DENSE_BY_ROWS'
!       A_out%val( : )   the values of the components of A, stored
!                            as a dense matrix row by row
!
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, n, ne
      REAL ( KIND = rp_ ) :: time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INV

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
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

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  set initial values

      n = A%n ; ne = ( n * ( n + 1 ) ) / 2

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%n = n ; A_out%ne = ne
      CALL SMT_put( A_out%type, 'DENSE_BY_ROWS', inform%alloc_status )
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

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        A_out%val( 1 : ne ) = A%val( 1 : ne )
        IF ( map_vals ) MAP( 1 : ne ) = [ ( i, i = 1, ne ) ]

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )
        l = 0
        DO j = 1, n
          DO i = j, n
            l = l + 1
            k = ( i * ( i - 1 ) ) / 2 + j
            A_out%val( k ) = A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )
        A_out%val = zero
        DO i = 1, n
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l )
            k = ( i * ( i - 1 ) ) / 2 + j
            A_out%val( k ) = A_out%val( k ) + A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )
        A_out%val = zero
        DO j = 1, A%n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            i = A%row( l )
            k = ( i * ( i - 1 ) ) / 2 + j
            A_out%val( k ) = A_out%val( k ) + A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a sparse matrix stored by its co-ordinates

      CASE ( 'COORDINATE' )
        A_out%val = zero
        DO l = 1, A%ne
          i = A%row( l )
          j = A%col( l )
          k = ( i * ( i - 1 ) ) / 2 + j
          A_out%val( k ) = A_out%val( k ) + A%val( l )
          IF ( map_vals ) MAP( l ) = k
        END DO

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = A%val( j )
          IF ( map_vals ) MAP( j ) = k
        END DO
      CASE ( 'SCALED-IDENTITY' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = A%val( 1 )
          IF ( map_vals ) MAP( j ) = n + 1
        END DO
      CASE ( 'IDENTITY' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = 1.0_rp_
          IF ( map_vals ) MAP( j ) = 0
        END DO
      CASE ( 'ZERO', 'NONE' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = 0.0_rp_
          IF ( map_vals ) MAP( j ) = 0
        END DO
      END SELECT

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

!  end of subroutine CONVERT_to_dense_symmetric_row_format

      END SUBROUTINE CONVERT_to_dense_symmetric_row_format

!-*- C O N V E R T _ T O _ D E N S E _ S _ C O L U M N   S U B R O U T I N E -*-

      SUBROUTINE CONVERT_to_dense_symmetric_column_format( A, A_out, control,  &
                                                           inform, MAP )

!  convert the input matrix to dense-column format.

!  if the lower trangular part of a symmetric matrix is stored as a vector 
!  by columns, the (i,j)th entry is in position i(i-1)/2+j
!    n(j-1) - j(j-1)/2 + i

!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_between_matrix_formats above)
!
!   A_out is a structure of type SMT_type used to hold the output matrix A
!    stored as a dense matrix by columns.
!
!    The following values will be set:
!
!       A_out%n   the number of rows/columns of the output A
!       A_out%type( 1 : 16 ) = 'DENSE_BY_COLUMNS'
!       A_out%val( : )   the values of the components of A, stored
!                            as a dense matrix column by column
!
!   control, inform and MAP as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_out
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                        DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, n, ne
      REAL ( KIND = rp_ ) :: time_start, time_now, clock_start, clock_now
      LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INV

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
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

!  ensure that all values are recorded if MAP is present

      map_vals = PRESENT( MAP )

!  set initial values

      n = A%n ; ne = ( n * ( n + 1 ) ) / 2

!  store A row-wise in A_out, perhaps with the column entries within
!  each row in increasing order

      A_out%n = n ; A_out%ne = ne
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

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )
        l = 0
        DO i = 1, n
          DO j = 1, i
            l = l + 1
            k = n * ( j - 1 ) - ( j * ( j - 1 ) ) / 2 + i
            A_out%val( k ) = A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )
        A_out%val( 1 : ne ) = A%val( 1 : ne )
        IF ( map_vals ) MAP( 1 : ne ) = [ ( i, i = 1, ne ) ]

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )
        A_out%val = zero
        DO i = 1, n
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            j = A%col( l )
            k = n * ( j - 1 ) - ( j * ( j - 1 ) ) / 2 + i
!           k = n * ( j - 1 ) + i
            A_out%val( k ) = A_out%val( k ) + A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )
        A_out%val = zero
        DO j = 1, n
          DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
            i = A%row( l )
            k = n * ( j - 1 ) - ( j * ( j - 1 ) ) / 2 + i
!           k = n * ( j - 1 ) + i
            A_out%val( k ) = A_out%val( k ) + A%val( l )
            IF ( map_vals ) MAP( l ) = k
          END DO
        END DO

!  A is a sparse matrix stored by its co-ordinates

      CASE ( 'COORDINATE' )
        A_out%val = zero
        DO l = 1, A%ne
          i = A%row( l )
          j = A%col( l )
          k = n * ( j - 1 ) - ( j * ( j - 1 ) ) / 2 + i
          A_out%val( k ) = A_out%val( k ) + A%val( l )
          IF ( map_vals ) MAP( l ) = k
        END DO

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out%val( : ne ) = zero
        DO i = 1, n
          k = n * ( i - 1 ) - ( i * ( i - 1 ) ) / 2 + i
          A_out%val( k ) = A%val( i )
          IF ( map_vals ) MAP( i ) = k
        END DO
      CASE ( 'SCALED-IDENTITY' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = A%val( 1 )
          IF ( map_vals ) MAP( j ) = n + 1
        END DO
      CASE ( 'IDENTITY' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = 1.0_rp_
          IF ( map_vals ) MAP( j ) = 0
        END DO
      CASE ( 'ZERO', 'NONE' )
        A_out%val( : ne ) = zero
        DO j = 1, n
          k = ( j * ( j - 1 ) ) / 2 + j
          A_out%val( k ) = 0.0_rp_
          IF ( map_vals ) MAP( j ) = 0
        END DO
      END SELECT

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

!  end of subroutine CONVERT_to_dense_symmetric_column_format

      END SUBROUTINE CONVERT_to_dense_symmetric_column_format

!-*-*-*-*-  C O N V E R T _ T R A N S P O S E   S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE CONVERT_transpose( m, n, ne, A_ptr, A_ind, A_val,             &
                                    A_transpose_ptr, A_transpose_ind,          &
                                    A_transpose_val, MAP )

!  given a matrix A stored by column, compute its transpose stored by column

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, ne
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
      INTEGER ( KIND = ip_ ), INTENT( OUT ),                                   &
                                DIMENSION( m + 1 ) :: A_transpose_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: A_ind
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( ne ) :: A_transpose_ind
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ne ) :: A_val
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( ne ) :: A_transpose_val
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
                                       DIMENSION( SIZE( A_val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l
      LOGICAL :: map_vals

      map_vals = PRESENT( MAP )

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
          IF ( map_vals ) MAP( l ) = k
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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( nz ) :: IND
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), OPTIONAL, DIMENSION( : ) :: IW
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( nz ) :: VAL
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: W

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, jm1, k
      REAL ( KIND = rp_ ) :: v
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

!-*-*- C O N V E R T _ I N C R E A S I N G _ O R D E R  S U B R O U T I N E -*-

      SUBROUTINE CONVERT_increasing_order( ns, ne, PTR, IND, VAL, MAP, INV )

!  given a mapping array, MAP, from a list of ne elements into parallel arrays
!  of indices and values, IND and VAL, suppose that IND is partitioned into
!  sub-arrays IND(PTR(i):PTR(i+1)-1) for i = 1, ..., ns. The aim is to rearrange
!  the entries in each sub-array (and its corresponding VAL(PTR(i):PTR(i+1)-1)),
!  so that the indices in each sub-array are in increasing order, and to adjust
!  MAP accordingly. To help, we are given the inverse, INV, to MAP, that is 
!  INV(MAP(i)) = i for i = 1, ..., ne, and on return, INV is also adjusted 
!  accordingly

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ns, ne
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ns + 1 ) :: PTR
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ne ) :: IND, MAP, INV
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ne ) :: VAL

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, is, ism1, j, k, inform, nes, nesmj1
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IP

!  compute the largest dimension of the sub-arrays, and allocate workspace

      nes = MAXVAL( PTR( 2 : ns + 1 ) - PTR( 1 : ns ) )
      ALLOCATE( IP( nes ), STAT = inform ) 

!  loop over the sub-arrays

!write(6,"( ' map', 20( 1X, I2 ) )" ) MAP
!write(6,"( ' inv', 20( 1X, I2 ) )" ) INV
!write(6,"( ' ind', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) IND
!write(6,"( ' val', 10( 1X, F4.1 ), /, ( '    ',  10( 1X, F4.1 ) ) )" ) VAL

      DO i = 1, ns

!       WRITE( 6, "( ' sub-array ', I0 )" ) i

!  record the number of entries, nes, in the i-th sub-array

        is = PTR( i ) ; ism1 = is - 1
        nes = PTR( i + 1 ) - is

!  initialize the permutation array

        IP( 1 : nes ) = [ ( j, j = 1, nes ) ]

!  build the heap

!write(6,*) IND( is : is + nes - 1 )
!write(6,*) ' inv ', INV( is : is + nes - 1 )

        CALL SORT_heapsort_build( nes, IND( is : ism1 + nes ), inform,         &
                                  ix = IP( : nes ),                            &
                                  rx = VAL( is : ism1 + nes ),                 &
                                  largest = .TRUE. )

!  reorder the values

         DO j = 1, nes
           nesmj1 = nes - j + 1
           CALL SORT_heapsort_smallest( nesmj1, IND( is : ism1 + nesmj1 ),     &
                                        inform, ix = IP( : nesmj1 ),           &
                                        rx = VAL( is : ism1 + nesmj1 ),        &
                                        largest = .TRUE. )
!          write(6,*) IND( is : is + nes - 1 )

!          WRITE( 6, "( ' The ', I2, '-', A2, ' largest value, IND(', I0,      &
!         & ') is ', I0, ' corresponding value is ', G0, 1X, F4.1 ) " )        &
!           j, STRING_ordinal( j ), IP( nesmj1 ), IND( ism1 + nesmj1 ),        &
!           IND( ism1 + nesmj1 ), VAL( ism1 + nesmj1 )
         END DO
!write(6,*) ' ip ', IP( 1 : nes ) + ism1

         DO j = 1, nes
           k = INV( ism1 + IP( j ) )
           MAP( k ) = ism1 + j
!write(6,*) ' j, ip, k, map ', j, IP(j), k, MAP(k)
           IP( j ) = k
         END DO

         DO j = 1, nes
           k = IP( j )
           INV( MAP( k ) ) = k
         END DO
      END DO
!write(6,"( ' map', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) MAP
!write(6,"( ' ind', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) IND
!write(6,"( ' val ', 10( F4.1, 1X ), /, ( '     ',  10( F4.1, 1X ) ) )" ) VAL
!write(6,"( ' inv', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) INV

      DO i = 1, ne
        MAP( INV( i ) ) = i
      END DO

!write(6,"( ' inv', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) INV
!write(6,"( ' map', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) MAP
!write(6,"( ' map(inv) ', 20( I2, 1X ) )" ) MAP(INV)
!write(6,*) ' leaving CONVERT_increasing_order'

!  deallocate workspace

      DEALLOCATE( IP, STAT = inform ) 

      RETURN

!  end of subroutine CONVERT_increasing_order

      END SUBROUTINE CONVERT_increasing_order

!- C O N V E R T _ C O M P R E S S _ D U P L I C A T E S  S U B R O U T I N E -

      SUBROUTINE CONVERT_compress_duplicates( n, ns, ne, PTR, IND, VAL,        &
                                              MAP, INV, IW )

!  the array MAP points from the positions (we refer to these as the "origin"s)
!  in the given input matrix A_in, described in some format via the tied arrays
!  (row_in,col_in,val_in), to the positions (known as the "target"s) in the 
!  tied arrays (row_out,col_out,val_out) in the re-ordered matrix A_out in some
!  other format. That is, origin entry i in the storage arrays for the original
!  matrix will be mapped to target entry MAP(i) in the reordered matrix. In
!  particular, normally (val_in)_i will be mapped to (val_out)_MAP(i). We do 
!  not presume that both row_in and col_in are are given, just that at least
!  one is, and this is tied to val_in. For example, in a compressed sparse
!  row (CSR) format, col_in and val_in will be tied, and row_in will be
!  implied by requiring that column entries in each row occur successively,
!  and row_in may be deduced from starting and ending pointers to each row.

!  If there are duplicate entries in the storage format for A_in, that is 
!  there are duplicate entries in (row_in,col_in), the assumption is that 
!  repeats are to be summed in A_out. To cope with this, if enties in orinin 
!  positions i < j correspond to a repeat (i.e., (row_in)_i = (row_in)_j and  
!  (col_in)_i = (col_in)_j), then map(i) will give the target position in A_out
!  of the entry (A_out)_map(i), and map(j) = - map(i), where the minus sign
!  says that (val_out)_map(i) = (val_in)_i + (val_in)_j); there will then be
!  fewer components of (row_out,col_out,val_out) than of (row_in,col_in,val_in).

!  Given a mapping array, MAP (along with its inverse INV, for which 
!  INV(MAP(i)) = i for i = 1, ..., ne) from a list of ne input elements into 
!  tied arrays of indices and values, IND and VAL, suppose that IND in [1:n]
!  is  partitioned into sub-arrays IND(PTR(i):PTR(i+1)-1) for i = 1, ..., ns. 
!  The aim is to compress IND and VAL to remove (add together) duplicate 
!  entries, and to adjust MAP, INV and PTR, as described above

!  map from (IND_in,VAL_in) to (IND,VAL) via
!  (IND,VAL)(MAP(i)) = (IND_in,VAL_in)(i)
!  (IND,VAL)(i) = (IND_in,VAL_in)(INV(i))

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ns, ne
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ns + 1 ) :: PTR
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( ne ) :: IND, MAP, INV
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: IW
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ne ) :: VAL

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, ip, ips, j, j_old, k, k_old, l

!write(6,*) ' entering CONVERT_compress_duplicates'
!write(6,"( ' map ', 20( I2, 1X ) )" ) MAP
!write(6,"( ' inv', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) INV
!write(6,"( ' ind', 10( 1X, I4 ), /, ( '    ',  10( 1X, I4 ) ) )" ) IND
!write(6,"( ' val ', 10( F4.1, 1X ), /, ( '     ',  10( F4.1, 1X ) ) )" ) VAL
!write(6,"( ' map(inv) ', 20( I2, 1X ) )" ) MAP(INV)


!  ip gives the first unoccupied position in IND and VAL once the reordering
!  takes place

      ip = 1

!  initialize the array IW that records entries in each segment

      IW( : n ) = 0

!  loop over the segements

      DO l = 1, ns

!       WRITE( 6, "( ' sub-array ', I0 )" ) l

!  segment l has indices i = IND(j), where targets j = PTR(i):PTR(i+1)-1),
!  and k = INV(j) records the origin entry in (row_in,col_in,val_in) for 
!  which MAP(k) = j

!write(6,"( ' inv ', 20( I2, 1X ) )" ) INV(PTR( l ): PTR( l + 1 ) - 1)
        DO j = PTR( l ), PTR( l + 1 ) - 1
          i = IND( j ) ; k = INV( j )

!  first pass: within each segment, reorder MAP if necesssary so that if 
!  IND(j_1) = IND(j_2) with j_1 < j_2, then INV(j_1) < INV(j_2)

!  if this is the first occurence of index i in segment l, record the origin 
!  entry in (row_in,col_in,val_in) that points at this target in 
!  (row_out,col_out,val_out)

          IF ( IW( i ) == 0 ) THEN
            IW( i ) = k

!  if this is not the the first instance of index i in the segment, recover
!  the origin index, k_old, of the "best" previous instance, and its target
!  j_old = MAP(k_old)

          ELSE
            k_old = IW( i ) ; j_old = MAP( k_old )

!  swap the two entries if necessary to ensure that IW(i) records the smaller
!  of k and k_old

!write(6,*) ' j, i = ', j, i, ' k, k_old ', k, k_old
            IF ( k < k_old ) THEN
!write(6,*) ' switch'
             MAP( k ) = j_old
             MAP( k_old ) = j
             INV( j_old ) = k
             INV( j ) = k_old
             IW( i ) = k
            END IF
          END IF
        END DO
!write(6,"( ' inv ', 20( I2, 1X ) )" ) INV(PTR( l ): PTR( l + 1 ) - 1)

!  reset IW

        DO j = PTR( l ), PTR( l + 1 ) - 1
          IW ( IND( j ) ) = 0
        END DO

!  second pass: reposition IND and VAL to account for the compression

        ips = ip
        DO j = PTR( l ), PTR( l + 1 ) - 1
          i = IND( j ) ; k = INV( j )

!  record IND and VAL for the first occurence of index i in this segment of
!  the target, and set MAP to point to this

          IF ( IW( i ) == 0 ) THEN
            IW( i ) = ip
            IND( ip ) = i
            VAL( ip ) = VAL( j )
            MAP( k ) = ip
!           INV( ip ) = k
            ip = ip + 1

!  if index i is a dulpicate, sum the values into the target, and set MAP to
!  point to this by including the negative of the target for the first occurence

          ELSE
            VAL( IW( i ) ) = VAL( IW( i ) ) + VAL( j )
            MAP( k ) = - IW( i )
          END IF
!write(6,"( ' i, k, IW, MAP, VAL', 4I4, F5.1 )" ) &
! i, k, IW( i ), MAP( k ), VAL( j )
        END DO

!  record the new pointer for the start of l-th segment, and reset IW

        PTR( l ) = ips
        IW( IND( ips : ip - 1 ) ) = 0

      END DO

!  record the new pointer beyond the ns-th segment

      PTR( ns + 1 ) = ip

!do l = 1, ns
!  write(6,"( ' segment ', I0, ' ( ind, val ) =' )" ) l
!  write(6, "( : 4( 1X, '(', I2, F5.1, ')' : ) )" ) &
!    ( IND( i ), VAL( i ), i = PTR( l ), PTR( l + 1 ) - 1 )
!end do
!write(6,"( ' map', 20( 1X, I3 ) )" ) MAP

      RETURN

!  end of subroutine CONVERT_compress_duplicates

      END SUBROUTINE CONVERT_compress_duplicates

!-*-*-*-*-*- C O N V E R T _ M A P _ V A L U E S   S U B R O U T I N E -*-*-*-*-

      SUBROUTINE CONVERT_map_values( ne_in, VAL_in, ne_out, VAL_out, MAP )

!  map the values in the input VAL_in to the appropriate positions in the 
!  output VAL_out under the control of the mapping array, MAP
!
!  Arguments:
!
!  all arguments as above

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ne_in, ne_out
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ne_in ) :: VAL_in
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( ne_out ) :: VAL_out
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne_in ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, n
 
!  map the values
     
      IF ( ne_in == 1 ) THEN

!  special case: scaled identity

        IF ( MAP( ne_in ) > ne_in ) THEN
          n = MAP( ne_in )
          VAL_out( : ne_out ) = VAL_in( 1 )
          RETURN
        END IF
      END IF

!  normal case

      DO i = 1, ne_in
        j = MAP( i )
!       WRITE( 6, "( ' i, j, val ', I0, 1X, I0, 1X, F4.1 )" ) i, j, VAL_in( i )
        IF ( j > 0 ) THEN
          VAL_out( j ) = VAL_in( i )
        ELSE
          VAL_out( - j ) = VAL_out( - j ) + VAL_in( i )
        END IF
      END DO

      RETURN

!  end of subroutine CONVERT_map_values

      END SUBROUTINE CONVERT_map_values

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-  G A L A H A D - C O N V E R T _ i n f o r m a t i o n  S U B R O U T I N E -

     SUBROUTINE CONVERT_information( data, inform, status )

!  return conversion information during or after application of CONVERT
!  See CONVERT_solve for a description of the required arguments
!
!  Arguments:
!
!  data     private internal data
!  inform   a structure containing output information. See preamble
!  status   return status

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( CONVERT_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( CONVERT_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%convert_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine CONVERT_information

     END SUBROUTINE CONVERT_information

!  end of module GALAHAD_CONVERT_precision

    END MODULE GALAHAD_CONVERT_precision




















!!$   BPMPD (Meszaros)
!!$   H - lower triangle by rows
!!$
!!$   BQPD (Fletcher)
!!$   H - upper triangle by co-ordinates
!!$
!!$   e04nqf (NAG)
!!$   H - lower triangle by columns
!!$
!!$   HiGHS (Edinburgh)
!!$   H - lower triangle by columns, zero based
!!$
!!$   OSQP (Oxford)
!!$   H - upper triangle by columns
!!$
!!$   QPALM (Leuven)
!!$   H - upper triangle by columns
!!$
!!$   qpOASES (Heidelberg) 
!!$   H - both triangles by columns
!!$
!!$
!!$
!!$   H - upper triangle by co-ordinates
!!$   H - upper triangle by columns (or lower triangle by rows)
!!$   H - lower triangle by columns
!!$   H - lower triangle by columns
!!$   H - both triangles by columns

