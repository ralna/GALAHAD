! THIS VERSION: GALAHAD 2.5 - 09/04/2013 AT 14:45 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S H A   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 8th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SHA_double

!    ------------------------------------------------
!   |                                                |
!   | SHA: find an approximation to a sparse Hessian |
!   |      using componentwise secant approximation  |
!   |                                                |
!    ------------------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_LAPACK_interface, ONLY : GETRF, GETRS, GELSS, GELSD, GELSY

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SHA_initialize, SHA_read_specfile, SHA_analyse, SHA_estimate,   &
               SHA_count, SHA_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SHA_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!  which approximation algorithm should be used?
!    0 : unsymmetric (alg 2.1 in paper)
!    1 : symmetric (alg 2.2 in paper)
!    2 : composite (alg 2.3 in paper)
!    3 : composite 2 (alg 2.2/3 in paper)

       INTEGER :: approximation_algorithm = 2

!  which dense linear equation solver should be used?
!    1 : Gaussian elimination
!    2 : QR factorization
!    3 : singular-value decomposition
!    4 : singular-value decomposition with divide-and-conquor

       INTEGER :: dense_linear_solver = 1

!  the maximum sparse degree if the combined version is used

       INTEGER :: max_sparse_degree = 50

!  if space is critical, ensure allocated arrays are no bigger than needed

       LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE SHA_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SHA_inform_type

!  return status. See SHA_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the maximum degree in the adgacency graph

       INTEGER :: max_degree = - 1

!  the number of differences that will be needed

       INTEGER :: differences_needed = - 1

!  the maximum reduced degree in the adgacency graph

       INTEGER :: max_reduced_degree = - 1

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

     END TYPE SHA_inform_type

!  - - - - - - - - - - - - -
!   matrix_data derived type
!  - - - - - - - - - - - - -

     TYPE, PUBLIC :: SHA_solve_system_data_type

       INTEGER :: out = 0
       INTEGER :: la_save1 = - 1
       INTEGER :: la_save2 = - 1
       INTEGER :: lb_save = - 1
       INTEGER :: lwork = - 1
       INTEGER :: liwork = - 1
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IWORK
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S, WORK
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_save
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: U, VT
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: A_save
     END TYPE SHA_solve_system_data_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: SHA_data_type

!  local variables

       INTEGER :: n, nz, nb, unsym_rows, approximation_algorithm_used

       INTEGER :: dense_linear_solver = - 1

!  initial array sizes

       INTEGER :: la1 = - 1
       INTEGER :: la2 = - 1
       INTEGER :: lb1 = - 1
       INTEGER :: ls = - 1

!  SHA_analyse_called is true once SHA_analyse has been called

       LOGICAL :: SHA_analyse_called = .FALSE.

       INTEGER, DIMENSION( 1 ) :: IWORK_1
       REAL ( KIND = wp ), DIMENSION( 1 ) :: WORK_1

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IN_DEGREE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: DEGREE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: FIRST
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LAST
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: COUNT
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM_inv
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PTR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PTR_sym
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PU
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PK
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: A
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: B
       TYPE ( SHA_solve_system_data_type ) :: solve_system_data
     END TYPE SHA_data_type

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------


   CONTAINS

!-*-*-  G A L A H A D -  S H A _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE SHA_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SHA controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SHA_data_type ), INTENT( INOUT ) :: data
     TYPE ( SHA_control_type ), INTENT( OUT ) :: control
     TYPE ( SHA_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initial private data

     data%SHA_analyse_called = .FALSE.

     RETURN

!  End of subroutine SHA_initialize

     END SUBROUTINE SHA_initialize

!-*-*-*-*-   S H A _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE SHA_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by SHA_initialize could (roughly)
!  have been set as:

! BEGIN SHA SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  approximation-algorithm                         1
!  dense-linear-solver                             1
!  maximum-degree-considered-sparse                50
!  space-critical                                  F
!  deallocate-error-fatal                          F
!  output-line-prefix                              ""
! END SHA SPECIFICATIONS

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     TYPE ( SHA_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: approximation_algorithm = print_level + 1
     INTEGER, PARAMETER :: dense_linear_solver = approximation_algorithm + 1
     INTEGER, PARAMETER :: max_sparse_degree = dense_linear_solver + 1
     INTEGER, PARAMETER :: space_critical = max_sparse_degree + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'SHA '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( approximation_algorithm )%keyword = 'approximation-algorithm'
     spec( dense_linear_solver )%keyword = 'dense-linear-solver'
     spec( max_sparse_degree )%keyword = 'maximum-degree-considered-sparse'

!  Logical key-words

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

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( approximation_algorithm ),              &
                                 control%approximation_algorithm,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dense_linear_solver ),                  &
                                 control%dense_linear_solver,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_sparse_degree ),                    &
                                 control%max_sparse_degree,                    &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     RETURN

!  End of subroutine SHA_read_specfile

     END SUBROUTINE SHA_read_specfile

!-*-*-*-  G A L A H A D -  S H A _ a n a l y s e  S U B R O U T I N E -*-*-*-

      SUBROUTINE SHA_analyse( n, nz, ROW, COL, data, control, inform )
!
!  ***************************************************************
!  *                                                             *
!  *   Compute a permutation of a symmetric matrix to try to     *
!  *   minimize the number of super-diagonal entries in each row *
!  *                                                             *
!  ***************************************************************
!
!  the Hessian stucture is given by n, nz, ROW and COL, where

!     n is the number of variables
!     nz is the number of nonzero elements in the UPPER TRIANGULAR
!        part of the matrix
!     ROW(i), COL(i) are the row and column indices of these entries
!        i = 1, .., nz

!  the analysed and permuted structure and the groups are stored in the
!  derived type data (see preface)

!  action of the subroutine is controlled by components of the derived type
!  control, while information about the progress of the subroutine is reported
!  in inform (again, see preface). Success or failure is flagged by the
!  component inform%status -
!     0 if no error was detected
!    -1 the allocation of workspace array inform%bad_alloc failed with status
!       inform%alloc_status
!    -3 invalid values input for n or nz
!   -23 if there was an error in the inform%bad_row-th row or column

!  ***********************************************************************

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER, INTENT( IN ) :: n, nz
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: ROW, COL

      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, j, j1, jj, k, k1, kk, l, ll, r, c, max_row, deg, min_degree
      CHARACTER ( LEN = 80 ) :: array_name
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix

!  test for errors in the input data

      IF ( n <= 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 900
      END IF

      data%n = n ; data%nz = nz
      inform%status = GALAHAD_ok

!  allocate workspace

      array_name = 'SHA: data%PK'
      CALL SPACE_resize_array( n + 1, data%PK,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%PU'
      CALL SPACE_resize_array( n, data%PU,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  determine how many nonzeros there are in each row

      data%PU = 0
      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%PU( i ) = data%PU( i ) + 1
        IF ( i /= j ) data%PU( j ) = data%PU( j ) + 1
      END DO

!  now set the starting addresses PK for each row in the array PTR

      data%PK( 1 ) = 1
      DO i = 1, n
        data%PK( i + 1 ) = data%PK( i ) + data%PU( i )
      END DO

!  compute the maximum degree

      max_row = MAXVAL( data%PU( 1 : n ) )
      inform%max_degree = max_row

!  allocate space for the list of degrees, as well the first and last
!  positions for those of a given degree and pointers from the rows to
!  the list of degrees

      array_name = 'SHA: data%DEGREE'
      CALL SPACE_resize_array( n, data%DEGREE,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%IN_DEGREE'
      CALL SPACE_resize_array( n, data%IN_DEGREE,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%FIRST'
      CALL SPACE_resize_array( 0, max_row, data%FIRST,                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%LAST'
      CALL SPACE_resize_array( 0, MAX( max_row, control%max_sparse_degree ),   &
             data%LAST,                                                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  count the number of rows with each degree

      data%LAST( 0 : max_row ) = 0
      DO i = 1, n
        data%LAST( data%PU( i ) ) = data%LAST( data%PU( i ) ) + 1
      END DO
      WRITE( control%out, "( ' (row size, # with this size):' )" )
      CALL SHA_write_nonzero_list( control%out, max_row, data%LAST )

!  set the start and finish positions for each degree in DEGREE

      data%FIRST( 0 ) = 1
      DO i = 0, max_row - 1
        data%FIRST( i + 1 ) = data%FIRST( i ) + data%LAST( i )
        data%LAST( i ) = data%FIRST( i + 1 ) - 1
      END DO
      data%LAST( max_row ) = data%FIRST( max_row ) + data%LAST( max_row ) - 1

!  now sort the rows by increaseing degree ...

      DO i = 1, n
        deg = data%PU( i )
        data%DEGREE( data%FIRST( deg ) ) = i
        data%IN_DEGREE( i ) = data%FIRST( deg )
        data%FIRST( deg ) = data%FIRST( deg ) + 1
      END DO

!  .. and reset the starting positions

      DO i = max_row - 1, 0, - 1
        data%FIRST( i + 1 ) =  data%FIRST( i )
      END DO
      data%FIRST( 0 ) = 1

!  compute the minimum degree

      DO j = 0, max_row + 1
        IF ( data%FIRST( j ) <= data%LAST( j ) ) THEN
          min_degree = j
          EXIT
        END IF
      END DO

!DO i = 0, max_row
!  write(6,"( ' degree ', I0, ' rows:' )" ) i
!  write(6,"( 6( 1X, I0 ) )" ) &
!   ( data%DEGREE( l ), l = data%FIRST( i ), data%LAST( i ) )
!END DO
!WRITE(6,"( 10( 1X, I0 ) )" ) ( data%IN_DEGREE( i ), i = 1, n )

!  allocate space for PTR to hold mappings from the rows back to the
!  coordinate storage, and the "shadow" PTR_sym set so that entries k and
!  PTR_sym(k) of PTR correspond to entries (i,j) and (j,i)

      array_name = 'SHA: data%PTR'
      CALL SPACE_resize_array( data%PK( n + 1 ) - 1, data%PTR,                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%PTR_sym'
      CALL SPACE_resize_array( data%PK( n + 1 ) - 1, data%PTR_sym,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  now set the PTR map ...

      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%PTR( data%PK( i ) ) = l
        data%PK( i ) = data%PK( i ) + 1
        IF ( i /= j ) THEN
          data%PTR( data%PK( j ) ) = l
          data%PK( j ) = data%PK( j ) + 1
          data%PTR_sym( data%PK( j ) - 1 ) = data%PK( i ) - 1
          data%PTR_sym( data%PK( i ) - 1 ) = data%PK( j ) - 1
        ELSE
          data%PTR_sym( data%PK( i ) - 1 ) = data%PK( i ) - 1
        END IF
      END DO

!  ... and reset the starting addresses

      DO i = n - 1, 1, - 1
        data%PK( i + 1 ) = data%PK( i )
      END DO
      data%PK( 1 ) = 1

!DO i = 1, n
!  write(6,"( ' row ', I0 )" ) i
!  write(6,"( 6( '(' I0, ',', I0 ')' ) )" ) ( ROW( data%PTR( l ) ),     &
!    COL( data%PTR( l ) ), l = data%PK( i ),  data%PK( i + 1 ) - 1 )
!end do
!stop

!DO l = 1, data%PK( n + 1 ) - 1
!  write( 6, "( I0, 2( ' (', I0, ',', I0 ')' ) )" ) l, &
!  ROW( data%PTR( l ) ), COL( data%PTR( l ) ), &
!  ROW( data%PTR( data%PTR_sym( l ) ) ), COL( data%PTR( data%PTR_sym( l ) ) )
!END DO
!stop

!  allocate further workspace

      array_name = 'SHA: data%COUNT'
      CALL SPACE_resize_array( n, data%COUNT,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%PERM_inv'
      CALL SPACE_resize_array( n, data%PERM_inv,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize row counts

      DO i = 1, n
        data%COUNT( i ) = data%PU( i )
        data%PU( i ) =  data%PK( i )
      END DO

      data%approximation_algorithm_used = control%approximation_algorithm

!  ----------------------------------
!  algorithms 0-2 (aka paper 2.1-2.3)
!  ----------------------------------

      IF ( control%approximation_algorithm <= 2 ) THEN

        DO l = 1, n

!  find a row with the lowest count

          i = data%DEGREE( MAX( data%FIRST( min_degree ), l ) )
          data%FIRST( min_degree ) = MAX( data%FIRST( min_degree ), l ) + 1
          IF ( min_degree > 0 ) THEN
            data%LAST( min_degree - 1 ) = data%FIRST( min_degree ) - 1
            data%FIRST( min_degree - 1 ) = data%FIRST( min_degree )
          END IF

!  update the minumum degree

          IF ( l < n ) THEN
            DO j = min_degree, max_row
              IF ( data%FIRST( j ) <= data%LAST( j ) ) THEN
                min_degree = j
                EXIT
              END IF
            END DO

!  reduce the row counts for all other rows that have an entry in column i

            DO k = data%PU( i ), data%PK( i + 1 ) - 1
              kk = data%PTR( k )
              r = ROW( kk ) ; c = COL( kk )
              IF ( r == c ) CYCLE

!  determine which of row( kk ) or col( kk ) gives the column number j

              IF ( c == i ) THEN
                j = r
              ELSE
                j = c
              END IF

!  upgrade DEGREE and its pointers

              deg = data%COUNT( j )
              kk = data%IN_DEGREE( j )
              jj = MAX( data%FIRST( deg ), l )
              IF ( jj /= kk ) THEN
                data%DEGREE( kk ) = data%DEGREE( jj )
                data%DEGREE( jj ) = j
                data%IN_DEGREE( j ) = jj
                data%IN_DEGREE( data%DEGREE( kk ) ) = kk
              END IF
              data%FIRST( deg ) = jj + 1
              data%LAST( deg - 1 ) = data%LAST( deg - 1 ) + 1
              min_degree = MIN( min_degree, deg - 1 )

!DO jj = 0, max_row
!  write(6,"( ' degree ', I0, ' rows:' )" ) jj
!  write(6,"( 6( 1X, I0 ) )" ) &
!   ( data%DEGREE( kk ), kk = data%FIRST( jj ), data%LAST( jj ) )
!END DO
!WRITE(6,"( 10( 1X, I0 ) )" ) ( data%IN_DEGREE( jj ), jj = 1, n )

!  reduce the count for row j

             data%COUNT( j ) = deg - 1

!  interchange entries jj = PU(j) and kk = data%PTR_sym( k ) of PTR
!  and their shadows

              jj = data%PU( j )
              kk = data%PTR_sym( k )
              IF ( jj /= kk ) THEN
                k1 = data%PTR_sym( kk )
                j1 = data%PTR_sym( jj )
                data%PTR_sym( jj ) = k1
                data%PTR_sym( k1 ) = jj
                data%PTR_sym( kk ) = j1
                data%PTR_sym( j1 ) = kk
                ll = data%PTR( jj )
                data%PTR( jj ) = data%PTR( kk )
                data%PTR( kk ) = ll
              END IF
              data%PU( j ) = data%PU( j ) + 1
            END DO
          END IF
          data%COUNT( i ) = n + 1
          data%PERM_inv( l ) = i

!DO ll = 1, data%PK( n + 1 ) - 1
!  write( 6, "( I0, 2( ' (', I0, ',', I0 ')' ) )" ) ll, &
!  ROW( data%PTR( ll ) ), COL( data%PTR( ll ) ), &
!  ROW( data%PTR( data%PTR_sym( ll ) ) ), COL( data%PTR( data%PTR_sym( ll ) ) )
!END DO

        END DO

!write(6,*) ' inv perm ', data%PERM_inv( : n )

!DO i = 1, n
!   write(6,"( ' row ', I0 )" ) i
!   write(6,"( 6( '(', I0, ',', I0, ')' ) )" ) ( ROW( data%PTR( l ) ),         &
!     COL( data%PTR( l ) ), l = data%PU( i ),  data%PK( i + 1 ) - 1 )
!END DO

!DO i = 1, n
!  write(6,"( 3I0 )" ) i, data%PU( i ), data%PK( i + 1 ) - 1
!END DO

        IF ( control%approximation_algorithm == 1 ) THEN
          inform%differences_needed                                            &
            = MAXVAL( data%PK( 2 : n + 1 ) -  data%PU( 1 : n ) )
        ELSE IF ( control%approximation_algorithm == 2 ) THEN
          inform%differences_needed = 0
          DO i = 1, n
            IF ( data%PK( i + 1 ) - data%PK( i ) <=                            &
                 control%max_sparse_degree ) THEN
              inform%differences_needed = MAX( inform%differences_needed,      &
                                               data%PK( i + 1 ) - data%PK( i ) )
            ELSE
              inform%differences_needed = MAX( inform%differences_needed,      &
                                               data%PK( i + 1 ) - data%PU( i ) )
            END IF
          END DO
        ELSE
          inform%differences_needed                                            &
            = MAXVAL( data%PK( 2 : n + 1 ) -  data%PK( 1 : n ) )
        END IF

!  report the numbers of each block size

        data%LAST( 0 : inform%differences_needed ) = 0
        DO i = 1, n
          l = data%PK( i + 1 ) - data%PU( i )
          data%LAST( l ) = data%LAST( l ) + 1
        END DO
        WRITE( control%out, "( ' (block size, # with this size):' )" )
        CALL SHA_write_nonzero_list( control%out, inform%differences_needed,   &
                                     data%LAST )

!  -----------------------------
!  algorithm 3 (aka paper 2.2/3)
!  -----------------------------

      ELSE
        data%unsym_rows = 0
        DO i = 1, n

!  skip rows that have more than max_sparse_degree entries

          IF ( data%PK( i + 1 ) -  data%PK( i ) >                              &
               control%max_sparse_degree ) CYCLE
          data%unsym_rows = data%unsym_rows + 1
          data%PERM_inv( data%unsym_rows ) = i

!  reduce the row counts for all other rows that have an entry in column i

          IF ( data%unsym_rows < n ) THEN
            DO k = data%PU( i ), data%PK( i + 1 ) - 1
              kk = data%PTR( k )
              r = ROW( kk ) ; c = COL( kk )
              IF ( r == c ) CYCLE

!  determine which of row( kk ) or col( kk ) gives the column number j

              IF ( c == i ) THEN
                j = r
              ELSE
                j = c
              END IF

!  interchange entries jj = PU(j) and kk = data%PTR_sym( k ) of PTR
!  and their shadows

              jj = data%PU( j )
              kk = data%PTR_sym( k )
              IF ( jj /= kk ) THEN
                k1 = data%PTR_sym( kk )
                j1 = data%PTR_sym( jj )
                data%PTR_sym( jj ) = k1
                data%PTR_sym( k1 ) = jj
                data%PTR_sym( kk ) = j1
                data%PTR_sym( j1 ) = kk
                ll = data%PTR( jj )
                data%PTR( jj ) = data%PTR( kk )
                data%PTR( kk ) = ll
              END IF
              data%PU( j ) = data%PU( j ) + 1
            END DO
          END IF
          data%COUNT( i ) = n + 1
        END DO

        inform%max_reduced_degree = 0
        inform%differences_needed = 0
        j = data%unsym_rows
        DO i = 1, n
          IF (  data%COUNT( i ) == n + 1 ) THEN
            data%PU( i ) = data%PK( i )
            inform%differences_needed =                                        &
              MAX( inform%differences_needed, data%PK( i + 1 ) -  data%PU( i ) )
          ELSE
            inform%max_reduced_degree =                                        &
              MAX( inform%max_reduced_degree, data%PK( i + 1 ) -  data%PU( i ) )
            j = j + 1
            data%PERM_inv( j ) = i
          END IF
        END DO

        inform%differences_needed                                              &
          = MAX( inform%differences_needed, inform%max_reduced_degree )

!       WRITE( 6, "( ' maximum degree in the connectivity graph = ', I0 )" )   &
!          inform%max_degree
!       WRITE( 6, "( 1X, I0, ' symmetric differences required ' )" )           &
!          inform%differences_needed
!       WRITE( 6, "( ' max reduced degree = ', I0 )" ) inform%max_reduced_degree
      END IF

!  prepare to return

 900  CONTINUE
      IF ( inform%status == GALAHAD_ok ) THEN
        data%SHA_analyse_called = .TRUE.

!  report error returns if required

      ELSE
        IF ( control%out > 0 .AND. control%print_level > 0 ) THEN
          IF ( LEN( TRIM( control%prefix ) ) > 2 )                             &
            prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
          WRITE( control%out, "( A, ' error in SHA_analyse, status = ',        &
         &  I0 ) " ) prefix, inform%status
          IF ( inform%status == GALAHAD_error_restrictions ) THEN
            WRITE ( control%out, "( A, ' illegal values',                      &
           &   ' for n or nz = ', I0, 1X, I0 )" ) prefix, n, nz
          ELSE IF ( inform%status == GALAHAD_error_allocate ) THEN
            WRITE( control%out,                                                &
              "( A, ' Allocation error, for ', A, /, A, ' status = ', I0 ) " ) &
              prefix, inform%bad_alloc, inform%alloc_status
          END IF
        END IF
      END IF

!  deallocate some workspace arrays

      array_name = 'SHA: data%IN_DEGREE'
      CALL SPACE_dealloc_array( data%IN_DEGREE,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%DEGREE'
      CALL SPACE_dealloc_array( data%DEGREE,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%FIRST'
      CALL SPACE_dealloc_array( data%FIRST,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%LAST'
      CALL SPACE_dealloc_array( data%LAST,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%COUNT'
      CALL SPACE_dealloc_array( data%COUNT,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%PTR_sym'
      CALL SPACE_dealloc_array( data%PTR_sym,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  internal subroutine for writing a selective list of nonzeros

      CONTAINS

        SUBROUTINE SHA_write_nonzero_list( out, length, LIST )
        INTEGER, INTENT( IN ) :: out, length
        INTEGER, INTENT( IN ), DIMENSION( 0 : length ) :: LIST
        INTEGER :: i, pos
        pos = 1
        DO i = 0, length
          IF ( LIST( i ) /= 0 ) THEN
            IF ( pos == 6 ) THEN
              WRITE( out, "( 1X, '(', I0, ',', I0, ')' )" ) i, LIST( i )
              pos = 1
            ELSE
              WRITE( out, "( 1X, '(', I0, ',', I0, ')' )", ADVANCE = 'no' )    &
                i, LIST( i )
              pos = pos + 1
            END IF
          END IF
        END DO
        IF ( pos /= 1 ) WRITE( out, "( '' )" )
        END SUBROUTINE SHA_write_nonzero_list

!  End of subroutine SHA_analyse

      END SUBROUTINE SHA_analyse

!-*-*-*-  G A L A H A D -  S H A _ e s t i m a t e  S U B R O U T I N E -*-*-*-

      SUBROUTINE SHA_estimate( n, nz, ROW, COL, m_max, m, RD, ls1, ls2, S,     &
                               ly1, ly2, Y, VAL, data, control, inform )

!     ********************************************************
!     *                                                      *
!     *   Estimation of a sparse Hessian matrix to try to    *
!     *    satisfy compentwise secant equations H s = y      *
!     *                                                      *
!     ********************************************************
!
!   The Hessian stucture given by n, nz, ROW and COL is described in
!   SHA_analyse and should not have been changed since the last call to
!   SHA_analyse. Additional arguments are

!     m_max is the maximum number of differences that might arise
!     m is the number of differences provided
!     RD(i), i=1:m gives the columns of S and Y of the ith most recent diffs
!     ls1,ls2 are the declared leading and trailing dimensions of S
!     S(i,j) (i=1:n,j=RD(1:m)) are the steps
!     ly1,ly2 are the declared leading and trailing dimensions of Y
!     Y(i,j) (i=1:n,j=RD(1:m)) are the differences in gradients
!     VAL(i) is the i-th nonzero in the estimated Hessian matrix.(i=1,nz)

!   The analysed and permuted structure and the groups are stored in the
!   derived type data (see preface)

!   Action of the subroutine is controlled by components of the derived type
!   control, while information about the progress of the subroutine is reported
!   in inform (again, see preface). Success or failure is flagged by the
!   component inform%status -
!     0 if no error was detected
!    -3 invalid values input for n or nz
!   -23 if there was an error in the inform%bad_row-th row or column
!   -31 if the call to SHA_estimate was not preceded by a call to SHA_analyse

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER, INTENT( IN ) :: n, nz, m_max, m, ls1, ls2, ly1, ly2
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: ROW, COL
      INTEGER, INTENT( IN ), DIMENSION( m ) :: RD
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ls1 , ls2 ) :: S
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ly1 , ly2 ) :: Y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( nz ) :: VAL
      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, ii, info, j, jj, k, kk, nn
      INTEGER :: liwork, lwork, mu, nu, min_mn, max_mn, rank, status
      REAL ( KIND = wp ) :: rcond
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      CHARACTER ( LEN = 80 ) :: array_name
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  test for an error in the input data

      IF ( .NOT. data%SHA_analyse_called ) THEN
        inform%status = GALAHAD_error_call_order ; GO TO 900
      END IF

      IF ( ( control%approximation_algorithm >= 3 .AND.                        &
             data%approximation_algorithm_used < 3 ) .OR.                      &
           ( control%approximation_algorithm < 3 .AND.                         &
             data%approximation_algorithm_used >= 3 ) ) THEN
        WRITE(6, "( ' incorrect approximation algorithm following analysis' )" )
        inform%status = - 116 ; GO TO 900
      END IF

      IF( control%out > 0 .AND. control%print_level > 2 ) THEN
        data%solve_system_data%out = control%out
      ELSE
        data%solve_system_data%out = 0
      END IF

!  allocate workspace

      nn = inform%differences_needed
      min_mn = MIN( m_max, nn ) ; max_mn = MAX( m_max, n, 1 )

!  generic solver workspace

      IF ( data%la1  < m .OR. data%la2  < nn ) THEN
        data%la1 = m_max ; data%la2 = nn
        array_name = 'SHA: data%A'
        CALL SPACE_resize_array( data%la1, data%la2, data%A,                   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%lb1 < max_mn ) THEN
        data%lb1 = max_mn
        array_name = 'SHA: data%B'
        CALL SPACE_resize_array( data%lb1, 1, data%B,                          &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  solver-specific workspace

      IF ( data%dense_linear_solver /= control%dense_linear_solver ) THEN
        data%dense_linear_solver = control%dense_linear_solver

!  allocate space to hold a copy of A if needed

        IF ( control%dense_linear_solver <= 2 ) THEN
          IF ( data%solve_system_data%la_save1 < m_max .OR.                    &
               data%solve_system_data%la_save2 < nn + 1 ) THEN
            data%solve_system_data%la_save1 = m_max
            data%solve_system_data%la_save2 = nn + 1
            array_name = 'SHA: data%A_save'
            CALL SPACE_resize_array( data%solve_system_data%la_save1,          &
               data%solve_system_data%la_save2, data%solve_system_data%A_save, &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  allocate space to hold a copy of b if needed

          IF ( data%solve_system_data%lb_save < m_max ) THEN
            data%solve_system_data%lb_save = m_max
            array_name = 'SHA: data%B_save'
            CALL SPACE_resize_array( data%solve_system_data%lb_save,           &
                   data%solve_system_data%B_save, inform%status,               &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF
        END IF

!  allocate space to hold the singular values if needed

        IF ( data%ls < min_mn ) THEN
          data%ls = min_mn
          array_name = 'SHA: data%solve_syetem_data%S'
          CALL SPACE_resize_array( data%ls, data%solve_system_data%S,          &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  discover how much temporary integer and real storage may be needed

        IF ( control%dense_linear_solver == 4 ) THEN
          CALL GELSD( m_max, nn, 1, data%A, data%la1, data%B, data%lb1,        &
                      data%solve_system_data%S, rcond, rank,                   &
                      data%WORK_1, - 1, data%IWORK_1, status )
          lwork = INT( data%WORK_1( 1 ) ) ; liwork = INT( data%IWORK_1( 1 ) )
        ELSE
          CALL GELSS( m_max, nn, 1, data%A, data%la1, data%B, data%lb1,        &
                      data%solve_system_data%S, rcond, rank, data%WORK_1, - 1, &
                      status )
          lwork = INT( data%WORK_1( 1 ) ) ; liwork = nn
        END IF

!  allocate temporary integer storage

        IF ( control%dense_linear_solver /= 3 ) THEN
          IF ( data%solve_system_data%liwork  < liwork ) THEN
            data%solve_system_data%liwork = liwork
            array_name = 'SHA: data%solve_system_data%IWORK'
            CALL SPACE_resize_array( data%solve_system_data%liwork,            &
                   data%solve_system_data%IWORK, inform%status,                &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF
        END IF

!  discover how much additional temporary real storage may be needed

        IF ( control%dense_linear_solver == 2 ) THEN
          CALL GELSY( m_max, n, 1, data%A, data%la1, data%B, data%lb1,         &
                      data%solve_system_data%IWORK, rcond, rank,               &
                      data%WORK_1, - 1, status )
          lwork = MAX( lwork, INT( data%WORK_1( 1 ) ) )
        END IF

!  allocate temporary real storage

        IF ( data%solve_system_data%lwork  < lwork ) THEN
          data%solve_system_data%lwork = lwork
          array_name = 'SHA: data%solve_system_data%WORK'
          CALL SPACE_resize_array( data%solve_system_data%lwork,               &
                 data%solve_system_data%WORK,                                  &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF
      END IF
      inform%status = GALAHAD_ok

!  for permuted row i:

!     - ----------- ------------------ -
! PTR  | known     |  unknown         | .    (ROW(kk),COL(kk),VAL(kk)),
!     - ----------- ------------------ -      kk=PTR(k) gives entries in row(i)
!       ^           ^                  ^      for k=PK(i),..,P(i+1)-1
!       |           |                  |      with those for k=PU(i),..,P(i+1)-1
!     PK(i)        PU(i)            PK(i+1)   still to be determined

!  -----------------------------
!  method that exploits symmetry
!  -----------------------------

      IF ( control%approximation_algorithm == 1 .OR.                           &
           control%approximation_algorithm == 3 ) THEN

!  run through the rows finding the unknown entries

        DO ii = 1, n
          i = data%PERM_inv( ii )
          nu = data%PK( i + 1 ) - data%PU( i )
          IF ( nu == 0 ) CYCLE
          mu = MIN( m, nu )
!         IF ( nu > m ) THEN
!           IF ( control%out > 0 .AND. control%print_level >= 1 )              &
!             WRITE( control%out, "( I0, ' entries to be found in row ', I0,   &
!            &  ' but only ', I0, ' differences supplied' )" ) nu, i, m
!           inform%status = - 111
!           RETURN
!         END IF

!  compute the unknown entries B_{ij}, j in I_i^-, to satisfy
!    sum_{j in I_i^-} B_{ij} s_{jl}  = y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}
!  for l = 1,.., |I_i^+|, where
!    I_i^+ = { j : j \in I_i and B_{ji} is already known }
!    I_i^- = I_i \ I_i^+ and
!    I_i = { j : B_{ij} /= 0}

!  compute the right-hand side y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}

!  initialize b to Y(i,l)

          data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  loop over the known entries

          DO k = data%PK( i ), data%PU( i ) - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

            data%B( 1 : mu, 1 )                                               &
              = data%B( 1 : mu, 1 ) - VAL( kk ) * S( j, RD( 1 : mu ) )
          END DO

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

          jj = 1
          DO k = data%PU( i ), data%PK( i + 1 ) - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )

!  set the entries of A

            data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
            jj = jj + 1
          END DO

!  solve A x = b

          CALL SHA_solve_system( control%dense_linear_solver, mu, nu, data%A,  &
                                 data%la1, data%B, data%lb1,                   &
                                 data%solve_system_data, i, info )
!write(6,*) ' ii, i, info ', ii, i, info
          IF ( info /= 0 ) THEN
            inform%status = GALAHAD_error_factorization ; GO TO 900
          END IF
!write(6,*) ' pass ', inform%status

!  finally, set the unknown B_{ij}

          jj = 1
          DO k = data%PU( i ), data%PK( i + 1 ) - 1
            VAL( data%PTR( k ) ) = data%B( jj, 1 )
            jj = jj + 1
          END DO
        END DO

!  ---------------------------------------
!  method that partially exploits symmetry
!  ---------------------------------------

      ELSE IF ( control%approximation_algorithm == 2 ) THEN

!  run through the rows finding the unknown entries

        DO ii = 1, n
          i = data%PERM_inv( ii )
!write(6,*) ' ii, i ', ii, i
          nu = data%PK( i + 1 ) - data%PK( i )
          IF ( nu == 0 ) CYCLE

!  if there is sufficient data, compute all of the entries in the row afresh

          IF ( nu <= m ) THEN
            mu = nu

!  compute the unknown entries B_{ij}, j in I_i, to satisfy
!    sum_{j in I_i} B_{ij} s_{jl}  = y_{il}
!  for l = 1,.., |I_i^+|, where
!    I_i = { j : B_{ij} /= 0}

!  store the right-hand side y_{il}

!  initialize b to Y(i,l)

            data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 1
            DO k = data%PK( i ), data%PK( i + 1 ) - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  set the entries of A

!write(6,*) ' RD ', RD( : mu )
              data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
!write(6,*) data%A( 1 : mu, jj )
              jj = jj + 1
            END DO

!  solve A x = b

!write(6,*) ' solver ', control%dense_linear_solver
            CALL SHA_solve_system( control%dense_linear_solver, mu, nu,        &
                                   data%A, data%la1, data%B, data%lb1,         &
                                   data%solve_system_data, i, info )
            IF ( info /= 0 ) THEN
              inform%status = GALAHAD_error_factorization ; GO TO 900
            END IF

!  finally, set the unknown B_{ij}

            jj = 1
            DO k = data%PK( i ), data%PK( i + 1 ) - 1
              VAL( data%PTR( k ) ) = data%B( jj, 1 )
              jj = jj + 1
            END DO

!  if there is insufficient data, compute only the unknown entries in the row

          ELSE
!write(6,*) nu, m
            nu = data%PK( i + 1 ) - data%PU( i )
            IF ( nu == 0 ) CYCLE
            mu = MIN( m, nu )
!           IF ( nu > m ) THEN
!             IF ( control%out > 0 .AND. control%print_level >= 1 )            &
!               WRITE( control%out, "( I0, ' entries to be found in row ', I0, &
!              &  ' but only ', I0, ' differences supplied' )" ) nu, i, m
!             inform%status = - 111
!             RETURN
!           END IF

!  compute the unknown entries B_{ij}, j in I_i^-, to satisfy
!    sum_{j in I_i^-} B_{ij} s_{jl}  = y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}
!  for l = 1,.., |I_i^+|, where
!    I_i^+ = { j : j \in I_i and B_{ji} is already known }
!    I_i^- = I_i \ I_i^+ and
!    I_i = { j : B_{ij} /= 0}

!  compute the right-hand side y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}

!  initialize b to Y(i,l)

            data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  loop over the known entries

            DO k = data%PK( i ), data%PU( i ) - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

              data%B( 1 : mu, 1 )                                              &
                = data%B( 1 : mu, 1 ) - VAL( kk ) * S( j, RD( 1 : mu ) )
            END DO

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 1
            DO k = data%PU( i ), data%PK( i + 1 ) - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  set the entries of A

              data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
              jj = jj + 1
            END DO

!  solve A x = b

            CALL SHA_solve_system( control%dense_linear_solver, mu, nu,        &
                                   data%A, data%la1, data%B, data%lb1,         &
                                   data%solve_system_data, i, info )
            IF ( info /= 0 ) THEN
              inform%status = GALAHAD_error_factorization ; GO TO 900
            END IF

!  finally, set the unknown B_{ij}

            jj = 1
            DO k = data%PU( i ), data%PK( i + 1 ) - 1
              VAL( data%PTR( k ) ) = data%B( jj, 1 )
              jj = jj + 1
            END DO
          END IF
        END DO

!  ------------------------------------------
!  naive method that doesn't exploit symmetry
!  ------------------------------------------

      ELSE

!  run through the rows finding the unknown entries

!       DO ii = 1, n
        DO ii = n, 1, - 1
          i = data%PERM_inv( ii )
          nu = data%PK( i + 1 ) - data%PK( i )
          IF ( nu == 0 ) CYCLE
          mu = MIN( m, nu )
!         IF ( nu > m ) THEN
!           IF ( control%out > 0 .AND. control%print_level >= 1 )              &
!             WRITE( control%out, "( I0, ' entries to be found in row ', I0,   &
!            &  ' but only ', I0, ' differences supplied' )" ) nu, i, m
!           inform%status = - 111
!           RETURN
!         END IF

!  compute the unknown entries B_{ij}, j in I_i, to satisfy
!    sum_{j in I_i} B_{ij} s_{jl}  = y_{il}
!  for l = 1,.., |I_i^+|, where
!    I_i = { j : B_{ij} /= 0}

!  store the right-hand side y_{il}

!  initialize b to Y(i,l)

          data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

          jj = 1
          DO k = data%PK( i ), data%PK( i + 1 ) - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )

!  set the entries of A

!write(6,*) ' RD ', RD( : mu )
            data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
!write(6,*) data%A( 1 : mu, jj )
            jj = jj + 1
          END DO

!  solve A x = b

          CALL SHA_solve_system( control%dense_linear_solver, mu, nu, data%A,  &
                                 data%la1, data%B, data%lb1,                   &
                                 data%solve_system_data, i, info )
          IF ( info /= 0 ) THEN
            inform%status = GALAHAD_error_factorization ; GO TO 900
          END IF

!  finally, set the unknown B_{ij}

          jj = 1
          DO k = data%PK( i ), data%PK( i + 1 ) - 1
            VAL( data%PTR( k ) ) = data%B( jj, 1 )
            jj = jj + 1
          END DO
        END DO
      END IF

!  prepare to return

 900  CONTINUE

!  report error returns if required

      IF ( inform%status /= GALAHAD_ok ) THEN
        IF( control%out > 0 .AND. control%print_level > 0 ) THEN
          IF ( LEN( TRIM( control%prefix ) ) > 2 )                             &
            prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
          WRITE( control%out, "( A, ' error in SHA_estimate, status = ',       &
        &        I0 )" ) prefix, inform%status
          IF ( inform%status == GALAHAD_error_restrictions ) THEN
           WRITE ( control%out, "( A, ' illegal',                              &
         &   ' values for n or nz = ', I0, 1X, I0 )" ) prefix, n, nz
          ELSE IF ( inform%status == GALAHAD_error_call_order ) THEN
           WRITE ( control%out, "( A, ' call to SHA_estimate',                 &
         &  ' must be preceded by a call to SHA_analyse' )" ) prefix
          ELSE IF ( inform%status == GALAHAD_error_factorization ) THEN
            WRITE( control%out, "( A, ' factorize in row ', I0,                &
               &  ' returns with error, info = ', I0 )" ) prefix, i, info
          ELSE IF ( inform%status == GALAHAD_error_allocate ) THEN
            WRITE( control%out,                                                &
              "( A, ' Allocation error, for ', A, /, A, ' status = ', I0 ) " ) &
              prefix, inform%bad_alloc, inform%alloc_status
          END IF
        END IF
      END IF

      RETURN

!  End of subroutine SHA_estimate

      END SUBROUTINE SHA_estimate

!-*-  G A L A H A D -  S H A _ s o l v e _ s y s t e m  S U B R O U T I N E  -*-

      SUBROUTINE SHA_solve_system( dense_linear_solver, m, n, A, la1, B, lb1,  &
                                   data, row, status )

      INTEGER, INTENT( IN ) :: dense_linear_solver, m, n, la1, lb1, row
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( la1, n ) :: A
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lb1, 1 ) :: B
      TYPE ( SHA_solve_system_data_type ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: rank
      REAL ( KIND = wp ) :: rcond

      rcond = - one

!  solve A x = b using Gaussian elimination; A is copied to A_save as a
!  precaution

      IF ( dense_linear_solver == 1 ) THEN
        IF ( m == n ) THEN
          data%A_save( : m, : n ) = A( : m, : n )
          CALL GETRF( m, n, data%A_save, data%la_save1, data%IWORK, status )
          IF ( status == 0 ) THEN
            data%B_save( : m ) = B( : m, 1 )
            CALL GETRS( 'N', n, 1, data%A_save, data%la_save1, data%IWORK,     &
                        B, lb1, status )
            IF ( status == 0 ) RETURN
            B( : m, 1 ) = data%B_save( : m )
          END IF
        END IF

!  solve A x = b using a QR factorization; A is copied to A_save as a precaution

      ELSE IF ( dense_linear_solver == 2 ) THEN
        data%A_save( : m, : n ) = A( : m, : n )
        data%B_save( : m ) = B( : m, 1 )
        CALL GELSY( m, n, 1, data%A_save, data%la_save1, B, lb1, data%IWORK,   &
                    rcond, rank, data%WORK, data%lwork, status )
        IF ( status == 0 ) RETURN
        B( : m, 1 ) = data%B_save( : m )
      END IF

!  solve A x = b using a singular-value decomposition

      IF ( dense_linear_solver == 4 ) THEN
        CALL GELSD( m, n, 1, A, la1, B, lb1, data%S, rcond, rank,              &
                    data%WORK, data%lwork, data%IWORK, status )
      ELSE
        CALL GELSS( m, n, 1, A, la1, B, lb1, data%S, rcond, rank,              &
                    data%WORK, data%lwork, status )
      END IF

      IF ( data%out > 0 ) THEN
        IF ( rank > 0 ) THEN
          write( 6, "( ' row ', I8, ' m ', I8, ' n ', I8, ' rank ', I8,        &
        & ' kappa ', ES11.4 )" ) row, m, n, rank, data%S( rank ) / data%S( 1 )
        ELSE
          write( 6, "( ' row ', I8, ' m ', I8, ' n ', I8, ' rank ', I8 )" )    &
          row, m, n, rank
        END IF
      END IF

      RETURN

!  End of subroutine SHA_solve_system

      END SUBROUTINE SHA_solve_system

!-*-*-*-*-  G A L A H A D -  S H A _ c o u n t  S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE SHA_count( n, nz, ROW, COL, ROW_COUNT )

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER, INTENT( IN ) :: n, nz
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: ROW, COL
      INTEGER, INTENT( OUT ), DIMENSION( n ) :: ROW_COUNT

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, j, l

!  determine how many nonzeros there are in each row

      ROW_COUNT = 0
      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        ROW_COUNT( i ) = ROW_COUNT( i ) + 1
        IF ( i /= j ) ROW_COUNT( j ) = ROW_COUNT( j ) + 1
      END DO

      END SUBROUTINE SHA_count

!-*-*-  G A L A H A D -  S H A _ t e r m i n a t e  S U B R O U T I N E  -*-*-

      SUBROUTINE SHA_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data
      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'SHA: data%PERM_inv'
      CALL SPACE_dealloc_array( data%PERM_inv,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%PTR'
      CALL SPACE_dealloc_array( data%PTR,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%PU'
      CALL SPACE_dealloc_array( data%PU,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%PK'
      CALL SPACE_dealloc_array( data%PK,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%A'
      CALL SPACE_dealloc_array( data%A,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%solve_system_data%IWORK'
      CALL SPACE_dealloc_array( data%solve_system_data%IWORK,                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%solve_system_data%WORK'
      CALL SPACE_dealloc_array( data%solve_system_data%WORK,                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%solve_system_data%A_save'
      CALL SPACE_dealloc_array( data%solve_system_data%A_save,                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%solve_system_data%B_save'
      CALL SPACE_dealloc_array( data%solve_system_data%B_save,                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  re-initial private data

      data%SHA_analyse_called = .FALSE.
      data%la1 = - 1
      data%la2 = - 1
      data%lb1 = - 1
      data%ls = - 1
      data%solve_system_data%la_save1 = - 1
      data%solve_system_data%la_save2 = - 1
      data%solve_system_data%lb_save = - 1
      data%solve_system_data%lwork = - 1
      data%solve_system_data%liwork = - 1
      data%dense_linear_solver = - 1

      RETURN

!  End of subroutine SHA_terminate

      END SUBROUTINE SHA_terminate

!  End of module GALAHAD_SHA

   END MODULE GALAHAD_SHA_double
