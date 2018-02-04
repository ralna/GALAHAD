! THIS VERSION: GALAHAD 2.8 - 02/11/2015 AT 13:50 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ B S C   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started July 27th 2008 as an extract from SBLS
!   originally released GALAHAD Version 2.3. August 1st 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BSC_double

!      ------------------------------------
!     |                                    |
!     | Given matrices A and (diagonal) D, |
!     | build the "Schur complement"       |
!     |                                    |
!     |         S  =  A D A^T              |
!     |                                    |
!      ------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double, ONLY: QPT_keyword_A
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BSC_initialize, BSC_read_specfile, BSC_form,                   &
                BSC_terminate, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BSC_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!  maximum permitted number of nonzeros in a column of A; -ve means unlimited

        INTEGER :: max_col = - 1

!  how much has A changed since last factorization:
!   0 = not changed, 1 = values changed, 2 = structure changed
!   3 = structure changed but values not required

        INTEGER :: new_a = 2

!  how much extra space is to be allocated in S above that needed to
!   hold the Schur complement

        INTEGER :: extra_space_s = 0

!  should s%ptr also be set to indicate the first entry in each column of S?

        LOGICAL :: s_also_by_column = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '
      END TYPE BSC_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BSC_data_type
        PRIVATE
        TYPE ( SMT_type ) :: S
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_col_ptr
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_by_rows
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row_ptr
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_by_cols
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW2
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
      END TYPE BSC_data_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BSC_inform_type

!  return status. See SBLS_form_and_factorize for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the maximum number of entries in a column of A

        INTEGER :: max_col_a = - 1

!  the number of columns of A that have more than control%max_col entries

        INTEGER :: exceeds_max_col = 0

!  the total CPU time spent in the package

       REAL ( KIND = wp ) :: time = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_time = 0.0

      END TYPE BSC_inform_type

   CONTAINS

!-*-*-*-*-*-   B S C  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE BSC_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BSC. This routine should be called before BSC_form
!
!  -----------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( BSC_control_type ), INTENT( OUT ) :: control
      TYPE ( BSC_data_type ), INTENT( INOUT ) :: data
      TYPE ( BSC_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of BSC_initialize

      END SUBROUTINE BSC_initialize

!-*-*-*-   B S C _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE BSC_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BSC_initialize could (roughly)
!  have been set as:

!  BEGIN BSC SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   has-a-changed                                   2
!   maximum-column-nonzeros-in-schur-complement     -1
!   extra-space-in-s                                0
!   also-store-s-by-column                          no
!   deallocate-error-fatal                          no
!   output-line-prefix                              ""
!  END BSC SPECIFICATIONS

!  Dummy arguments

      TYPE ( BSC_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: new_a = print_level + 1
      INTEGER, PARAMETER :: max_col = new_a + 1
      INTEGER, PARAMETER :: space_critical = max_col + 1
      INTEGER, PARAMETER :: extra_space_s = space_critical + 1
      INTEGER, PARAMETER :: s_also_by_column = extra_space_s + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = s_also_by_column + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'BSC'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( new_a )%keyword = 'has-a-changed'
      spec( max_col )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec( extra_space_s )%keyword = 'extra-space-in-s'

!  Logical key-words


      spec( s_also_by_column )%keyword = 'also-store-s-by-column'
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
      CALL SPECFILE_assign_value( spec( new_a ),                               &
                                  control%new_a,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_col ),                             &
                                  control%max_col,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( extra_space_s ),                       &
                                  control%extra_space_s,                       &
                                  control%error )

!  Set logical values


      CALL SPECFILE_assign_value( spec( s_also_by_column ),                    &
                                  control%s_also_by_column,                    &
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

      END SUBROUTINE BSC_read_specfile

!-*-*-*-*-*-*-*-*-*-   B S C _ F O R M   S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE BSC_form( m, n, A, S, data, control, inform, D )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Form the "Schur-complement" matrix S = A D A^T
!
!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_BSC.
!
!   ** NB. default real/complex means double precision real/complex in
!   ** GALAHAD_BSC_double
!
!  m is a scalar integer variable that must hold the number of rows of A.
!
!  n is a scalar integer variable that must hold the number of columns of A.
!
!  A is scalar variable of type SMT_TYPE that must hold the matrix A. A may be
!   input in dense, sparse co-ordinate or sparse row-wise form. The following
!   components are used here:
!
!   A%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of A%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of A%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme,
!    and the first fourteen components of A%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into A%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%A%type, 'COORDINATE' )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   A%ne is a scalar variable of type default integer, that holds the number of
!    entries in the matrix A in the sparse co-ordinate storage scheme. It need
!    not be set for any of the other schemes.
!
!   A%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the matrix A in any of the available storage
!    schemes.
!
!   A%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the matrix A in the sparse co-ordinate storage scheme.
!    It need not be allocated for any of the other two schemes.
!
!   A%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the matrix A in either the sparse
!    co-ordinate, or the sparse row-wise storage scheme. It need not
!    be allocated when the dense scheme is used.
!
!   A%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of each row of the matrix A,
!    as well as the total number of entries plus one, in the sparse row-wise
!    storage scheme. It need not be allocated when the other schemes are used.
!
!  S is scalar variable of type SMT_TYPE that will hold the matrix S in
!   sparse co-ordinate form. Since S is symmetric ONLY THE ENTRIES IN THE LOWER
!   TRIANGULAR PART OF S WILL BE STORED. The entries will occur in column
!   order. The following components will be set.

!   S%ne is a scalar variable of type default integer, that holds the number of
!    entries in the LOWER TRIANGULAR PART of the matrix.
!
!   S%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries in the LOWER TRIANGULAR PART of the matrix S.
!
!   S%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the entries in the LOWER TRIANGULAR PART of the matrix
!    S in the same order as in S%val.
!
!   S%col is a rank-one allocatable array of type default integer, that holds
!    the column indices of the entries in the LOWER TRIANGULAR PART of the
!    matrix S in the same order as in S%val.
!
!   S%ptr is a rank-one allocatable array of type default integer, whose ith
!    component gives the positions in S%row (etc) of the first entry of column
!    i of S, and S%ptr(m + 1) = S%ne+1, when control%s_also_by_column has been
!    set .TRUE. It will not have been set if control%s_also_by_column = .FALSE.
!
!  control is a scalar variable of type BSC_control_type. See BSC_initialize
!   for details
!
!  inform is a scalar variable of type BSC_inform_type. On initial entry,
!   inform%status should be set to 1. On exit, the following components will
!   have been set:
!
!   status is a scalar variable of type default integer, that gives
!    the exit status from the package. Possible values are:
!
!      0. The run was succesful
!
!     -1. An allocation error occurred. A message indicating the offending
!         array is written on unit control%error, and the returned allocation
!         status and a string containing the name of the offending array
!         are held in inform%alloc_status and inform%bad_alloc respectively.
!     -2. A deallocation error occurred.  A message indicating the offending
!         array is written on unit control%error and the returned allocation
!         status and a string containing the name of the offending array
!         are held in inform%alloc_status and inform%bad_alloc respectively.
!     -3. The restriction nlp%n > 0 or requirement that prob%A_type contains
!         its relevant string 'DENSE', 'COORDINATE' or 'SPARSE_BY_ROWS'
!         has been violated.
!
!   alloc_status is a scalar variable of type default integer, that gives
!    the status of the last attempted array allocation or deallocation.
!    This will be 0 if status = 0.
!
!   bad_alloc is a scalar variable of type default character
!    and length 80, that  gives the name of the last internal array
!    for which there were allocation or deallocation errors.
!    This will be the null string if status = 0.
!
!   time is a scalar variable of type real, that gives the total time taken
!
!   clock_time is a scalar variable of type real, that gives the total
!    clock time taken

!  data is a scalar variable of type BSC_data_type used for internal data.
!
!  D is an optional rank-one array of type default real and length n, that if
!   present must hold the diagonal entries of the matrix D. If D is absent,
!   the matrix D will be assumed to be the identity matrix.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: S
      TYPE ( BSC_data_type ), INTENT( INOUT ) :: data
      TYPE ( BSC_control_type ), INTENT( IN ) :: control
      TYPE ( BSC_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: D

!  Local variables

      INTEGER :: i, ii, j, k, kk, l, out, new_a, max_col
      INTEGER :: nnz_col_j, nnz_adat_old, nnz_adat, new_pos, a_ne
      REAL ( KIND = wp ) :: time_start, time_end, clock_start, clock_end
      REAL ( KIND = wp ) :: al
      LOGICAL :: printi, got_d
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Set default information values

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''

      out = control%out
      printi = control%print_level >= 1 .AND. out >= 0

!  Check for faulty dimensions

      IF ( n <= 0 .OR. m < 0 .OR. .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 990
      END IF

      IF ( control%out >= 0 .AND. control%print_level >= 1 ) THEN
        WRITE( control%out,                                                    &
          "( /, A, ' n = ', I0, ', m = ', I0 )" ) prefix, n, m
      END IF

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne
      END IF

      max_col = control%max_col
      IF ( max_col < 0 ) max_col = m
      new_a = control%new_a
      got_d = PRESENT( d )

!   ======================
!    FORM SCHUR COMPLEMENT
!   ======================

      array_name = 'bsc: data%IW'
      CALL SPACE_resize_array( n, data%IW,                                    &
         inform%status, inform%alloc_status, array_name = array_name,         &
         deallocate_error_fatal = control%deallocate_error_fatal,             &
         exact_size = control%space_critical,                                 &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 980

!  Check to see if there are not too many entries in any column of A. Find the
!  number of entries in each column - only do this for sparse_by_row and
!  coordinate storage)

      IF ( SMT_get( A%type ) /= ' DENSE' ) THEN
        array_name = 'bsc: data%A_col_ptr'
        CALL SPACE_resize_array( n + 1, data%A_col_ptr,                      &
          inform%status, inform%alloc_status, array_name = array_name,       &
          deallocate_error_fatal = control%deallocate_error_fatal,           &
          exact_size = control%space_critical,                               &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 980
      END IF

      IF ( new_a >= 2 ) THEN
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          inform%max_col_a = m
          IF ( m > max_col ) THEN
            inform%exceeds_max_col = n
          ELSE
            inform%exceeds_max_col = 0
          END IF
        CASE ( 'SPARSE_BY_ROWS' )
          data%A_col_ptr( 2 : ) = 0
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l ) + 1
              data%A_col_ptr( j ) = data%A_col_ptr( j ) + 1
            END DO
          END DO
          inform%max_col_a = MAXVAL( data%A_col_ptr( 2 : ) )
          inform%exceeds_max_col = COUNT( data%A_col_ptr( 2 : ) > max_col )
        CASE ( 'COORDINATE' )
          data%A_col_ptr( 2 : ) = 0
          DO l = 1, A%ne
            j = A%col( l ) + 1
            data%A_col_ptr( j ) = data%A_col_ptr( j ) + 1
          END DO
          inform%max_col_a = MAXVAL( data%A_col_ptr( 2 : ) )
          inform%exceeds_max_col = COUNT( data%A_col_ptr( 2 : ) > max_col )
        END SELECT

        IF ( printi ) WRITE( out, "( A,                                        &
       &  ' maximum, average column lengths of A = ', I0, ', ', F0.1, /,       &
       &  A, ' number of columns of A longer than maxcol = ', I0,              &
       &     ' is ',  I0 )" ) prefix, inform%max_col_a,                        &
          float( SUM( data%A_col_ptr( 2 : ) ) ) / float( n ), prefix,          &
          max_col, inform%exceeds_max_col

!  Exit if the column with the largest number of entries exceeds max_col

        IF ( inform%max_col_a > max_col ) THEN
          IF ( printi ) WRITE( out, "(                                         &
         &  A, ' - abandon the Schur-complement' )" ) prefix
          inform%status = GALAHAD_error_schur_complement
          GO TO 990
        END IF

!  Now store A by rows. First find the number of entries in each row
!  (Only do this for coordinate storage, as it is already available
!  for storage by rows!)

        IF ( SMT_get( A%type ) == 'COORDINATE' ) THEN
          array_name = 'bsc: data%A_row_ptr'
          CALL SPACE_resize_array( m + 1, data%A_row_ptr,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980

          array_name = 'bsc: data%A_by_rows'
          CALL SPACE_resize_array( a_ne, data%A_by_rows,                       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980

          data%A_row_ptr( 2 : ) = 0
          DO l = 1, A%ne
            i = A%row( l ) + 1
            data%A_row_ptr( i ) = data%A_row_ptr( i ) + 1
          END DO

!  Next assign row pointers

          data%A_row_ptr( 1 ) = 1
          DO i = 2, m + 1
            data%A_row_ptr( i ) = data%A_row_ptr( i ) + data%A_row_ptr( i - 1 )
          END DO

!  Now record where the entries in each row occur in the original matrix

          DO l = 1, A%ne
            i = A%row( l )
            new_pos = data%A_row_ptr( i )
            data%A_by_rows( new_pos ) = l
            data%A_row_ptr( i ) = new_pos + 1
          END DO

!  Finally readjust the row pointers

          DO i = m + 1, 2, - 1
            data%A_row_ptr( i ) = data%A_row_ptr( i - 1 )
          END DO
          data%A_row_ptr( 1 ) = 1
        END IF

!  Also store A by columns, but with the entries sorted in increasing
!  row order within each column. First assign column pointers

        IF ( SMT_get( A%type ) /= 'DENSE' ) THEN
          data%A_col_ptr( 1 ) = 1
          DO j = 2, n + 1
            data%A_col_ptr( j ) = data%A_col_ptr( j ) + data%A_col_ptr( j - 1 )
          END DO

!  Now record where the entries in each colum occur in the original matrix

          array_name = 'bsc: data%A_by_cols'
          CALL SPACE_resize_array( a_ne, data%A_by_cols,                       &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'SPARSE_BY_ROWS' )

            array_name = 'bsc: data%A_row'
            CALL SPACE_resize_array( a_ne, data%A_row,                         &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 980

            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                new_pos = data%A_col_ptr( j )
                data%A_row( new_pos ) = i
                data%A_by_cols( new_pos ) = l
                data%A_col_ptr( j ) = new_pos + 1
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO i = 1, m
              DO k = data%A_row_ptr( i ), data%A_row_ptr( i + 1 ) - 1
                l = data%A_by_rows( k )
                j = A%col( l )
                new_pos = data%A_col_ptr( j )
                data%A_by_cols( new_pos ) = l
                data%A_col_ptr( j ) = new_pos + 1
              END DO
            END DO
          END SELECT

!  Finally readjust the column pointers

          DO j = n + 1, 2, - 1
            data%A_col_ptr( j ) = data%A_col_ptr( j - 1 )
          END DO
          data%A_col_ptr( 1 ) = 1
        END IF

!  Now build the sparsity structure of S = A D A^T

        IF ( SMT_get( A%type ) == 'DENSE' ) THEN
          array_name = 'bsc: data%W'
          CALL SPACE_resize_array( n, data%W,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = .FALSE.,                                             &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980
        ELSE
          array_name = 'bsc: data%IW'
          CALL SPACE_resize_array( m, data%IW,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = .FALSE.,                                             &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980

          array_name = 'bsc: data%IW2'
          CALL SPACE_resize_array( m, data%IW2,                                &
             inform%status, inform%alloc_status,  array_name = array_name,     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980
        END IF

!  Compute the total storage for the (lower triangle) of S = A D A^T

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          nnz_adat = m * ( m + 1 ) / 2
        CASE ( 'SPARSE_BY_ROWS' )
          nnz_adat = 0
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            nnz_col_j = 0
            DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( k )
              DO kk = data%A_col_ptr( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = data%A_row( kk )
                IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                  nnz_col_j = nnz_col_j + 1
                  data%IW2( i ) = j
                END IF
              END DO
            END DO
            nnz_adat = nnz_adat + nnz_col_j
          END DO
        CASE ( 'COORDINATE' )
          nnz_adat = 0
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            nnz_col_j = 0
            DO k = data%A_row_ptr( j ), data%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( data%A_by_rows( k ) )
              DO kk = data%A_col_ptr( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = A%row( data%A_by_cols( kk ) )
                IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                  nnz_col_j = nnz_col_j + 1
                  data%IW2( i ) = j
                END IF
              END DO
            END DO
            nnz_adat = nnz_adat + nnz_col_j
          END DO
        END SELECT

!  Allocate space to hold A D A^T in S

        S%m = m ; S%n = m ; S%ne = nnz_adat
        CALL SMT_put( S%type, 'COORDINATE', inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = GALAHAD_error_allocate ; GO TO 980
        END IF

        array_name = 'bsc: S%row'
        CALL SPACE_resize_array( S%ne + control%extra_space_s, S%row,          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 980

        array_name = 'bsc: S%col'
        CALL SPACE_resize_array( S%ne + control%extra_space_s, S%col,          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 980

        array_name = 'bsc: S%val'
        CALL SPACE_resize_array( S%ne + control%extra_space_s, S%val,          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 980

        IF ( control%s_also_by_column ) THEN
          array_name = 'bsc: S%ptr'
          CALL SPACE_resize_array( S%n + 1, S%ptr,                             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 980
        END IF
!     END IF

!  -------------------
!  New structure for S
!  -------------------

!     IF ( new_a >= 2 ) THEN

!  Now insert the (row/col/val) entries of A D A^T into S

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          nnz_adat = 0
          l = 0
          DO i = 1, m
            IF ( control%s_also_by_column ) S%ptr( i ) = nnz_adat + 1
            IF ( got_d ) THEN
              data%W = A%val( l + 1 : l + n ) * D
            ELSE
              data%W = A%val( l + 1 : l + n )
            END IF
            k = 0
            DO j = 1, i
              nnz_adat = nnz_adat + 1
              S%row( nnz_adat ) = i
              S%col( nnz_adat ) = j
              IF ( new_a == 2 ) S%val( nnz_adat ) =                            &
                                  DOT_PRODUCT( data%W, A%val( k + 1 : k + n ) )
              k = k + n
            END DO
            l = l + n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          nnz_adat_old = 0
          nnz_adat = 0
          data%IW( : n ) = data%A_col_ptr( : n )
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            IF ( control%s_also_by_column ) S%ptr( j ) = nnz_adat + 1
            DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( k )
              IF ( got_d ) THEN
                al = A%val( k ) * D( l )
              ELSE
                al = A%val( k )
              END IF
              DO kk = data%IW( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = data%A_row( kk )

!  ... and which are in lower-triangular part

                IF ( i >= j ) THEN

!  The first entry in this position ...

                  IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_adat = nnz_adat + 1
                    data%IW2( i ) = j + nnz_adat
                    S%row( nnz_adat ) = i
                    S%col( nnz_adat ) = j
                    IF ( new_a == 2 ) S%val( nnz_adat ) =                      &
                                        al * A%val( data%A_by_cols( kk ) )

!  ... or a subsequent one

                  ELSE
                    ii = data%IW2( i ) - j
                    IF ( new_a == 2 ) S%val( ii ) = S%val( ii ) +              &
                                        al * A%val( data%A_by_cols( kk ) )
                  END IF

!  IW is incremented since all entries above lie in the upper triangle

                ELSE
                  data%IW( l ) = data%IW( l ) + 1
                END IF
              END DO
            END DO
            DO l = nnz_adat_old + 1, nnz_adat
              data%IW2( S%row( l ) ) = j
            END DO
            nnz_adat_old  = nnz_adat
          END DO
        CASE ( 'COORDINATE' )
          nnz_adat_old = 0
          nnz_adat = 0
          data%IW( : n ) = data%A_col_ptr( : n )
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            IF ( control%s_also_by_column ) S%ptr( j ) = nnz_adat + 1
            DO k = data%A_row_ptr( j ), data%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( data%A_by_rows( k ) )
              IF ( got_d ) THEN
                al = A%val( data%A_by_rows( k ) ) * D( l )
              ELSE
                al = A%val( data%A_by_rows( k ) )
              END IF
              DO kk = data%IW( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = A%row( data%A_by_cols( kk ) )

!  ... and which are in lower-triangular part

                IF ( i >= j ) THEN

!  The first entry in this position ...

                  IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_adat = nnz_adat + 1
                    data%IW2( i ) = j + nnz_adat
                    S%row( nnz_adat ) = i
                    S%col( nnz_adat ) = j
                    IF ( new_a == 2 ) S%val( nnz_adat ) =                      &
                                        al * A%val( data%A_by_cols( kk ) )

!  ... or a subsequent one

                  ELSE
                    ii = data%IW2( i ) - j
                    IF ( new_a == 2 ) S%val( ii ) = S%val( ii ) +              &
                      al * A%val( data%A_by_cols( kk ) )
                  END IF

!  IW is incremented since all entries above lie in the upper triangle

                ELSE
                  data%IW( l ) = data%IW( l ) + 1
                END IF
              END DO
            END DO
            DO l = nnz_adat_old + 1, nnz_adat
              data%IW2( S%row( l ) ) = j
            END DO
            nnz_adat_old  = nnz_adat
          END DO
        END SELECT
        IF ( control%s_also_by_column ) S%ptr( m + 1 ) = nnz_adat + 1

!  ------------------------
!  Existing structure for S
!  ------------------------

      ELSE

!  Now insert the (val) entries of A D A^T into S

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          nnz_adat = 0
          l = 0
          DO i = 1, m
            IF ( got_d ) THEN
              data%W = A%val( l + 1 : l + n ) * D
            ELSE
              data%W = A%val( l + 1 : l + n )
            END IF
            k = 0
            DO j = 1, i
              nnz_adat = nnz_adat + 1
              S%val( nnz_adat ) = DOT_PRODUCT( data%W, A%val( k + 1 : k + n ) )
              k = k + n
            END DO
            l = l + n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          nnz_adat_old = 0
          nnz_adat = 0
          data%IW( : n ) = data%A_col_ptr( : n )
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( k )
              IF ( got_d ) THEN
                al = A%val( k ) * D( l )
              ELSE
                al = A%val( k )
              END IF
              DO kk = data%IW( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = data%A_row( kk )

!  ... and which are in lower-triangular part

                IF ( i >= j ) THEN

!  The first entry in this position ...

                  IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_adat = nnz_adat + 1
                    data%IW2( i ) = j + nnz_adat
                    S%val( nnz_adat ) = al * A%val( data%A_by_cols( kk ) )

!  ... or a subsequent one

                  ELSE
                    ii = data%IW2( i ) - j
                    S%val( ii ) = S%val( ii ) +                                &
                      al * A%val( data%A_by_cols( kk ) )
                  END IF

!  IW is incremented since all entries above lie in the upper triangle

                ELSE
                  data%IW( l ) = data%IW( l ) + 1
                END IF
              END DO
            END DO
            DO l = nnz_adat_old + 1, nnz_adat
              data%IW2( S%row( l ) ) = j
            END DO
            nnz_adat_old  = nnz_adat
          END DO
        CASE ( 'COORDINATE' )
          nnz_adat_old = 0
          nnz_adat = 0
          data%IW( : n ) = data%A_col_ptr( : n )
          data%IW2 = 0

!  For the j-th column of A D A^T ...

          DO j = 1, m
            DO k = data%A_row_ptr( j ), data%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

              l = A%col( data%A_by_rows( k ) )
              IF ( got_d ) THEN
                al = A%val( data%A_by_rows( k ) ) * D( l )
              ELSE
                al = A%val( data%A_by_rows( k ) )
              END IF
              DO kk = data%IW( l ), data%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                i = A%row( data%A_by_cols( kk ) )

!  ... and which are in lower-triangular part

                IF ( i >= j ) THEN

!  The first entry in this position ...

                  IF ( data%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_adat = nnz_adat + 1
                    data%IW2( i ) = j + nnz_adat
                    S%val( nnz_adat ) = al * A%val( data%A_by_cols( kk ) )

!  ... or a subsequent one

                  ELSE
                    ii = data%IW2( i ) - j
                    S%val( ii ) = S%val( ii ) +                                &
                      al * A%val( data%A_by_cols( kk ) )
                  END IF

!  IW is incremented since all entries above lie in the upper triangle

                ELSE
                  data%IW( l ) = data%IW( l ) + 1
                END IF
              END DO
            END DO
            DO l = nnz_adat_old + 1, nnz_adat
              data%IW2( S%row( l ) ) = j
            END DO
            nnz_adat_old  = nnz_adat
          END DO
        END SELECT
      END IF

!     WRITE( out, "( ' S: m, nnz ', 2I4 )" ) S%n, S%ne
!     WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( S%row )
!     WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( S%col )
!     WRITE( out, "( A, /, ( F7.2) )" ) ' vals =', ( S%val )

      IF ( printi ) WRITE( out,                                                &
         "( A, ' time to form matrix ', F6.2 )") prefix, time_end - time_start

      inform%status = GALAHAD_ok
      RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_TIME( time_end ) ; inform%time = time_end - time_start
     CALL CLOCK_time( clock_end ) ; inform%clock_time = clock_end - clock_start
     RETURN

 990 CONTINUE
     CALL CPU_TIME( time_end ) ; inform%time = time_end - time_start
     CALL CLOCK_time( clock_end ) ; inform%clock_time = clock_end - clock_start
     IF ( printi ) WRITE( out, "( A, ' Inform = ', I0, ' Stopping ' )" )       &
       prefix, inform%status
     RETURN

!  End of subroutine BSC_form

      END SUBROUTINE BSC_form

!-*-*-*-*-*-   B S C _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE BSC_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine BSC_initialize
!   control see Subroutine BSC_initialize
!   inform  see Subroutine BSC_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( BSC_control_type ), INTENT( IN ) :: control
      TYPE ( BSC_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( BSC_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'bsc: data%IW'
      CALL SPACE_dealloc_array( data%IW,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%A_col_ptr'
      CALL SPACE_dealloc_array( data%A_col_ptr,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%A_row_ptr'
      CALL SPACE_dealloc_array( data%A_row_ptr,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%A_by_rows'
      CALL SPACE_dealloc_array( data%A_by_rows,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%A_by_cols'
      CALL SPACE_dealloc_array( data%A_by_cols,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%A_row'
      CALL SPACE_dealloc_array( data%A_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%IW'
      CALL SPACE_dealloc_array( data%IW,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bsc: data%IW2'
      CALL SPACE_dealloc_array( data%IW2,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine BSC_terminate

      END SUBROUTINE BSC_terminate

!  End of module BSC

   END MODULE GALAHAD_BSC_double
