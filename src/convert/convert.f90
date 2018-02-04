! THIS VERSION: GALAHAD 2.6 - 08/06/2014 AT 14:30 GMT.

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
!     | (currently, only convert to column format, others in due course)  |
!     |                                                                   |
!      -------------------------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_SMT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CONVERT_read_specfile, CONVERT_to_column_format,               &
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

!  return status. See CONVERT_form_and_factorize for details

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

      SUBROUTINE CONVERT_to_column_format( A, A_by_cols, control, inform,      &
                                           IWORK, len_iwork, WORK, len_work )

!  convert the input matrix or its transpose to column format. If the matrix 
!  is to stored so that the row indices in each column appear in increasing
!  order, the optional arrays IWORK and WORK of length m (n when the transpose
!  is sought) must be provided, with all entries of IWORK set to 0; IWORK
!  will have been reset to 0 on exit

!  Dummy arguments

      INTEGER, INTENT( IN ) :: len_iwork, len_work
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A_by_cols
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( INOUT ), DIMENSION( len_iwork ) :: IWORK
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( len_work ) :: WORK

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

      SELECT CASE( SMT_get( A%type ) ) 
      CASE ( 'DENSE', 'DENSE_BY_COLUMNS' )
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

!  store A column-wise in A_by_cols, perhaps with the row entries within 
!  each column in increasing order

      A_by_cols%m = m ; A_by_cols%n = n ; A_by_cols%ne = ne
      CALL SMT_put( A_by_cols%type, 'SPARSE_BY_COLUMNS', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate
        GO TO 900
      END IF

      array_name = 'CONVERT: A_by_cols%ptr'
      CALL SPACE_resize_array( A_by_cols%n + 1, A_by_cols%ptr,                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_by_cols%row'
      CALL SPACE_resize_array( ne, A_by_cols%row,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_by_cols%val'
      CALL SPACE_resize_array( ne, A_by_cols%val,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A

      order_cols = .FALSE.

!  the transpose is required

      IF ( control%transpose ) THEN

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) ) 

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE' )
          k = 0 ; l = 1
          DO j = 1, n
            A_by_cols%ptr( j ) = l
            DO i = 1, m
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_by_cols%row( l ) = i ; A_by_cols%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_by_cols%ptr( n + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          l = 1
          DO j = 1, n
            k = j
            A_by_cols%ptr( j ) = l
            DO i = 1, m
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_by_cols%row( l ) = i ; A_by_cols%val( l ) = val
                l = l + 1
              END IF
              k = k + n
            END DO
          END DO
          A_by_cols%ptr( n + 1 ) = l

!         A_by_cols%ptr( : n ) = 0
!         k = 0
!         DO j = 1, m
!           DO i = 1, n
!             k = k + 1
!             IF ( A%val( k ) /= zero )                                        &
!               A_by_cols%ptr( i ) = A_by_cols%ptr( i ) + 1
!           END DO
!         END DO
!         l = 1
!         DO j = 1, n
!           i = A_by_cols%ptr( j ) + l
!           A_by_cols%ptr( j ) = l
!           l = i
!         END DO
!         k = 0
!         DO j = 1, m
!           DO i = 1, n
!             k = k + 1
!             val = A%val( k )
!             IF ( val /= zero ) THEN
!               l = A_by_cols%ptr( i )
!               A_by_cols%row( l ) = j ; A_by_cols%val( l ) = val
!               A_by_cols%ptr( i ) = l + 1
!             END IF
!           END DO
!         END DO
!         DO j = n, 1, - 1
!           A_by_cols%ptr( j + 1 ) = A_by_cols%ptr( j ) 
!         END DO
!         A_by_cols%ptr( 1 ) = 1

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          A_by_cols%ptr( : n + 1 ) = A%ptr( : n + 1 )
          A_by_cols%row( : A_by_cols%ne ) = A%col( : A_by_cols%ne )
          A_by_cols%val( : A_by_cols%ne ) = A%val( : A_by_cols%ne )
          order_cols = .TRUE.

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          CALL CONVERT_transpose( A%m, A%n, A%ne, A%ptr, A%row, A%val,         &
                                  A_by_cols%ptr, A_by_cols%row, A_by_cols%val )

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' ) 
          A_by_cols%ptr( : n ) = 0
          DO l = 1, A%ne
            j = A%row( l )
            A_by_cols%ptr( j ) = A_by_cols%ptr( j ) + 1
          END DO
          l = 1
          DO j = 1, n
            i = A_by_cols%ptr( j ) + l
            A_by_cols%ptr( j ) = l
            l = i
          END DO
          DO l = 1, A%ne
            j = A%row( l ) ; i = A_by_cols%ptr( j )
            A_by_cols%row( i ) = A%col( l ) ; A_by_cols%val( i ) = A%val( l )
            A_by_cols%ptr( j ) = i + 1
          END DO
          DO j = n, 1, - 1
            A_by_cols%ptr( j + 1 ) = A_by_cols%ptr( j ) 
          END DO
          A_by_cols%ptr( 1 ) = 1
          order_cols = .TRUE.
        END SELECT

!  the transpose is not required

      ELSE

!  consider the input storage scheme

        SELECT CASE( SMT_get( A%type ) ) 

!  A is a dense matrix (stored by rows)

        CASE ( 'DENSE' )
          l = 1
          DO j = 1, n
            k = j
            A_by_cols%ptr( j ) = l
            DO i = 1, m
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_by_cols%row( l ) = i ; A_by_cols%val( l ) = val
                l = l + 1
              END IF
              k = k + n
            END DO
          END DO
          A_by_cols%ptr( n + 1 ) = l

!  A is a dense matrix stored by columns

        CASE ( 'DENSE_BY_COLUMNS' )
          k = 0 ; l = 1
          DO j = 1, n
            A_by_cols%ptr( j ) = l
            DO i = 1, m
              k = k + 1
              val = A%val( k )
              IF ( val /= zero ) THEN
                A_by_cols%row( l ) = i ; A_by_cols%val( l ) = val
                l = l + 1
              END IF
            END DO
          END DO
          A_by_cols%ptr( n + 1 ) = l

!  A is a sparse matrix stored by rows

        CASE ( 'SPARSE_BY_ROWS' )
          CALL CONVERT_transpose( A%n, A%m, A%ne, A%ptr, A%col, A%val,         &
                                  A_by_cols%ptr, A_by_cols%row, A_by_cols%val )

!  A is a sparse matrix stored by columns

        CASE ( 'SPARSE_BY_COLUMNS' )
          A_by_cols%ptr( : n + 1 ) = A%ptr( : n + 1 )
          A_by_cols%row( : A_by_cols%ne ) = A%row( : A_by_cols%ne )
          A_by_cols%val( : A_by_cols%ne ) = A%val( : A_by_cols%ne )
          order_cols = .TRUE.

!  A is a sparse matrix stored by its co-ordinates

        CASE ( 'COORDINATE' ) 
          A_by_cols%ptr( : n ) = 0
          DO l = 1, A%ne
            j = A%col( l )
            A_by_cols%ptr( j ) = A_by_cols%ptr( j ) + 1
          END DO
          l = 1
          DO j = 1, n
            i = A_by_cols%ptr( j ) + l
            A_by_cols%ptr( j ) = l
            l = i
          END DO
          DO l = 1, A%ne
            j = A%col( l ) ; i = A_by_cols%ptr( j )
            A_by_cols%row( i ) = A%row( l ) ; A_by_cols%val( i ) = A%val( l )
            A_by_cols%ptr( j ) = i + 1
          END DO
          DO j = n, 1, - 1
            A_by_cols%ptr( j + 1 ) = A_by_cols%ptr( j ) 
          END DO
          A_by_cols%ptr( 1 ) = 1
          order_cols = .TRUE.
        END SELECT
      END IF

!  sum duplicate entries and squeeze the storage space

      IF ( control%sum_duplicates ) THEN

!  ensure that there is sufficient workspace

        IF ( len_iwork < n ) THEN
          inform%status = GALAHAD_error_restrictions
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( ' ', /, A, ' ** len_iwork must be at',    &
           & ' least ', I0 )" ) prefix, m
          RETURN 
        END IF

!  consider each column one at a time

        k = 1 ; inform%duplicates = 0
        DO i = 1, n
          ll = k

!  loop over the rows j in the ith column

          DO l = A_by_cols%ptr( i ), A_by_cols%ptr( i + 1 ) - 1
            j = A_by_cols%row( l )

!  if iwork(j) = 0, the entry is not a duplicate, so record the row and value,
!  and flag in iwork(j) the location where the value is stored in A_by_cols%val

            IF ( IWORK( j ) == 0 ) THEN
              A_by_cols%row( k ) = A_by_cols%row( l )
              A_by_cols%val( k ) = A_by_cols%val( l )
              IWORK( j ) = k
              k = k + 1

!  if iwork(j) /= 0, the entry is a duplicate, and its value should be added
!  to A_by_cols%val(iwork(j))

            ELSE
              inform%duplicates = inform%duplicates + 1
              j = IWORK( j )
              A_by_cols%val( j ) = A_by_cols%val( j ) + A_by_cols%val( l )
            END IF
          END DO

!  reset IWORK to zero

          DO l = ll, k - 1
            IWORK(  A_by_cols%row( l ) ) = 0
          END DO

!  reset the pointer to the start of the column

          A_by_cols%ptr( i ) = ll
        END DO
        A_by_cols%ptr( n + 1 ) = k
        A_by_cols%ne = k - 1
      END IF

!  order the row entries within each column in increasing row order

      inform%status = GALAHAD_ok
      IF ( control%order .AND. order_cols ) THEN
        DO i = 1, n
          ll =  A_by_cols%ptr( i ) ; lu =  A_by_cols%ptr( i + 1 ) - 1
          IF ( lu > ll ) THEN
            CALL CONVERT_order( m, lu - ll + 1, A_by_cols%row( ll : lu ),      &
                                A_by_cols%val( ll : lu ), order_status,        &
                                IWORK, len_iwork, WORK, len_work )
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

!  end of subroutine CONVERT_to_column_format

      END SUBROUTINE CONVERT_to_column_format

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

      SUBROUTINE CONVERT_order( n, nz, IND, VAL, status, IW, len_iw, W, len_w )

!  sort the vectors (IND,VAL) of length nnz, for which each  component of IND 
!  is a unique integer in 1:n, so that on output IND is in increasing order.
!  IW should be set to 0 on entry, and will have been reset to 0 on exit.

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, nz, len_iw, len_w
      INTEGER, INTENT( OUT ) :: status
      INTEGER, INTENT( INOUT ), DIMENSION( nz ) :: IND
      INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( len_iw ) :: IW
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nz ) :: VAL
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( len_w ) :: W

!  Local variables

      INTEGER :: i, j, jm1, k
      REAL ( KIND = wp ) :: v
      LOGICAL :: workspace

      workspace = len_w >= n .AND. len_iw >= n
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



