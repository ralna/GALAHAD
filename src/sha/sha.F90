! THIS VERSION: GALAHAD 4.1 - 2023-05-05 AT 08:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S H A   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. April 8th 2013

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SHA_precision

!    ------------------------------------------------
!   |                                                |
!   | SHA: find an approximation to a sparse Hessian |
!   |      using componentwise secant approximation  |
!   |                                                |
!    ------------------------------------------------

     USE GALAHAD_KINDS_precision
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_LAPACK_interface, ONLY : GETRF, GETRS, GELSS, GELSD, GELSY

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SHA_initialize, SHA_read_specfile, SHA_analyse,                 &
               SHA_estimate, SHA_count, SHA_terminate,                         &
               SHA_full_initialize, SHA_full_terminate, SHA_information

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SHA_initialize
       MODULE PROCEDURE SHA_initialize, SHA_full_initialize
     END INTERFACE SHA_initialize

     INTERFACE SHA_terminate
       MODULE PROCEDURE SHA_terminate, SHA_full_terminate
     END INTERFACE SHA_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: eps_singular = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SHA_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  which approximation algorithm should be used?
!    0 : unsymmetric (alg 2.1 in paper)
!    1 : symmetric (alg 2.2 in paper)
!    2 : composite (alg 2.3 in paper)
!    3 : composite 2 (alg 2.2/3 in paper)

       INTEGER ( KIND = ip_ ) :: approximation_algorithm = 2

!  which dense linear equation solver should be used?
!    1 : Gaussian elimination
!    2 : QR factorization
!    3 : singular-value decomposition
!    4 : singular-value decomposition with divide-and-conquer

       INTEGER ( KIND = ip_ ) :: dense_linear_solver = 3

!  the maximum sparse degree if the combined version is used

       INTEGER ( KIND = ip_ ) :: max_sparse_degree = 50

!  if available use an addition extra_differences differences

       INTEGER ( KIND = ip_ ) :: extra_differences = 0

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

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the maximum degree in the adgacency graph

       INTEGER ( KIND = ip_ ) :: max_degree = - 1

!  the number of differences that will be needed (more may be helpful)

       INTEGER ( KIND = ip_ ) :: differences_needed = - 1

!  the maximum reduced degree in the adgacency graph

       INTEGER ( KIND = ip_ ) :: max_reduced_degree = - 1

!  a failure occured when forming the bad_row-th row or column (0 = no failure)

       INTEGER ( KIND = ip_ ) :: bad_row = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

     END TYPE SHA_inform_type

!  - - - - - - - - - - - - -
!   matrix_data derived type
!  - - - - - - - - - - - - -

     TYPE, PUBLIC :: SHA_solve_system_data_type

       INTEGER ( KIND = ip_ ) :: out = 0
       INTEGER ( KIND = ip_ ) :: lwork = - 1
       INTEGER ( KIND = ip_ ) :: liwork = - 1
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IWORK
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S, WORK
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: U, VT
     END TYPE SHA_solve_system_data_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: SHA_data_type

!  local variables

       INTEGER ( KIND = ip_ ) :: n, nz, nb, unsym_rows
       INTEGER ( KIND = ip_ ) :: approximation_algorithm_used
       INTEGER ( KIND = ip_ ) :: dense_linear_solver = - 1
       INTEGER ( KIND = ip_ ) :: differences_needed = - 1

!  initial array sizes

       INTEGER ( KIND = ip_ ) :: la1 = - 1
       INTEGER ( KIND = ip_ ) :: la2 = - 1
       INTEGER ( KIND = ip_ ) :: lb1 = - 1
       INTEGER ( KIND = ip_ ) :: la_save1 = - 1
       INTEGER ( KIND = ip_ ) :: la_save2 = - 1
       INTEGER ( KIND = ip_ ) :: lb_save = - 1
       INTEGER ( KIND = ip_ ) :: ls = - 1

!  SHA_analyse_called is true once SHA_analyse has been called

       LOGICAL :: SHA_analyse_called = .FALSE.

       INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK_1
       REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK_1

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: DEGREE_inv
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: DEGREE
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: FIRST
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LAST
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COUNT
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PERM_inv
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PTR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PTR_lower
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PU
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PK
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: A
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: A_save
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: B
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: B_save
       TYPE ( SHA_solve_system_data_type ) :: solve_system_data
     END TYPE SHA_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: SHA_full_data_type
        LOGICAL :: f_indexing
        TYPE ( SHA_data_type ) :: SHA_data
        TYPE ( SHA_control_type ) :: SHA_control
        TYPE ( SHA_inform_type ) :: SHA_inform
      END TYPE SHA_full_data_type

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

!- G A L A H A D -  S H A _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE SHA_full_initialize( data, control, inform )

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

     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SHA_control_type ), INTENT( OUT ) :: control
     TYPE ( SHA_inform_type ), INTENT( OUT ) :: inform

     CALL SHA_initialize( data%sha_data, control, inform )

     RETURN

!  End of subroutine SHA_full_initialize

     END SUBROUTINE SHA_full_initialize

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
!  extra-differences                               0
!  space-critical                                  F
!  deallocate-error-fatal                          F
!  output-line-prefix                              ""
! END SHA SPECIFICATIONS

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     TYPE ( SHA_control_type ), INTENT( INOUT ) :: control
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: approximation_algorithm              &
                                            = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: dense_linear_solver                  &
                                            = approximation_algorithm + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_sparse_degree                    &
                                            = dense_linear_solver + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: extra_differences                    &
                                            = max_sparse_degree + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                       &
                                            = extra_differences + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
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
     spec( extra_differences )%keyword = 'extra-differences'

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
     CALL SPECFILE_assign_value( spec( extra_differences ),                    &
                                 control%extra_differences,                    &
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
!  ****************************************************************
!  *                                                              *
!  *   Compute a permutation of a symmetric matrix to try to      *
!  *   minimize the number of super-diagonal entries in each row  *
!  *                                                              *
!  ****************************************************************
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
!   -11 if there was an error when forming the inform%bad_row-th row or column

!  ***********************************************************************

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: ROW, COL

      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER ( KIND = ip_ ) :: i, j, j1, jj, k, k1, kk, l, ll, r, c
      INTEGER ( KIND = ip_ ) :: max_row, deg, min_degree
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  test for errors in the input data

      IF ( n <= 0 .OR. nz < 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 900
      END IF

      data%n = n ; data%nz = nz
      inform%status = GALAHAD_ok

!  allocate space for starting address PK and row count COUNT

      array_name = 'SHA: data%PK'
      CALL SPACE_resize_array( n + 1, data%PK,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%COUNT'
      CALL SPACE_resize_array( n, data%COUNT,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  determine how many nonzeros there are in each row of the whole matrix 
!  (both upper and lower parts)

      data%COUNT = 0
      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%COUNT( i ) = data%COUNT( i ) + 1
        IF ( i /= j ) data%COUNT( j ) = data%COUNT( j ) + 1
      END DO

!  now set the starting addresses PK for each row of the whole matrix in the 
!  pointer array PTR

      data%PK( 1 ) = 1
      DO i = 1, n
        data%PK( i + 1 ) = data%PK( i ) + data%COUNT( i )
      END DO

!  compute the maximum degree (row length)

      max_row = MAXVAL( data%COUNT( 1 : n ) )
      inform%max_degree = max_row

!  allocate space for the list of rows of increasing degrees (DEGREE), as well 
!  the first and last positions for those of a given degree (FIRST & LAST) and 
!  pointers from the rows to the list of degrees (DEGREE_inv)

      array_name = 'SHA: data%DEGREE'
      CALL SPACE_resize_array( n, data%DEGREE,                                 &
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

      array_name = 'SHA: data%DEGREE_inv'
      CALL SPACE_resize_array( n, data%DEGREE_inv,                             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  count the number of rows with each degree 

      data%LAST( 0 : max_row ) = 0
      DO i = 1, n
        data%LAST( data%COUNT( i ) ) = data%LAST( data%COUNT( i ) ) + 1
      END DO
      IF ( control%out > 0 .AND. control%print_level > 0 ) THEN
        WRITE( control%out, "( A, ' (row size, # with this size):' )" ) prefix
        CALL SHA_write_nonzero_list( control%out, max_row, data%LAST )
      END IF

!  set the start and finish positions for each degree in DEGREE

      data%FIRST( 0 ) = 1
      DO i = 0, max_row - 1
        data%FIRST( i + 1 ) = data%FIRST( i ) + data%LAST( i )
        data%LAST( i ) = data%FIRST( i + 1 ) - 1
      END DO
      data%LAST( max_row ) = data%FIRST( max_row ) + data%LAST( max_row ) - 1

!  now sort the rows by increasing degree (and record their inverses) ...

      DO i = 1, n
        deg = data%COUNT( i )
        data%DEGREE( data%FIRST( deg ) ) = i
        data%DEGREE_inv( i ) = data%FIRST( deg )
        data%FIRST( deg ) = data%FIRST( deg ) + 1
      END DO

!  .. and reset the starting positions

      DO i = max_row - 1, 0, - 1
        data%FIRST( i + 1 ) =  data%FIRST( i )
      END DO
      data%FIRST( 0 ) = 1

!  compute the minimum degree (row length)

      DO j = 0, max_row + 1
        IF ( data%FIRST( j ) <= data%LAST( j ) ) THEN
          min_degree = j
          EXIT
        END IF
      END DO

!  print row statistics if required

      IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
        DO i = 0, max_row
          WRITE( control%out, "( ' degree ', I0, ' rows:' )" ) i
          IF ( data%FIRST( i ) <= data%LAST( i ) )                             &
            WRITE( control%out, "( 10( :, 1X, I0 ) )" )                        &
             ( data%DEGREE( l ), l = data%FIRST( i ), data%LAST( i ) )
        END DO
        WRITE( control%out, "( ' degree_inv:', /, 10( 1X, I0 ) )" )            &
          data%DEGREE_inv( 1 : n )
      END IF

!  allocate space for PTR to hold mappings from the rows back to the
!  coordinate storage, and its "shadow" PTR_lower set so that entries k and
!  PTR_lower(k) of PTR correspond to the "upper" and "lower" triangular entries 
!  (i,j) and (j,i)

      array_name = 'SHA: data%PTR'
      CALL SPACE_resize_array( data%PK( n + 1 ) - 1, data%PTR,                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%PTR_lower'
      CALL SPACE_resize_array( data%PK( n + 1 ) - 1, data%PTR_lower,           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  now set the PTR and PTR_lower maps ...

      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%PTR( data%PK( i ) ) = l
        data%PK( i ) = data%PK( i ) + 1
        IF ( i /= j ) THEN
          data%PTR( data%PK( j ) ) = l
          data%PK( j ) = data%PK( j ) + 1
          data%PTR_lower( data%PK( j ) - 1 ) = data%PK( i ) - 1
          data%PTR_lower( data%PK( i ) - 1 ) = data%PK( j ) - 1
        ELSE
          data%PTR_lower( data%PK( i ) - 1 ) = data%PK( i ) - 1
        END IF
      END DO

!  ... and reset the starting addresses

      DO i = n - 1, 1, - 1
        data%PK( i + 1 ) = data%PK( i )
      END DO
      data%PK( 1 ) = 1

!  print more row statistics if required

      IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
        DO i = 1, n
          WRITE( control%out, "( ' row ', I0 )" ) i
          WRITE( control%out, "( 1X, 6( '(', I0, ',', I0, ')', : ) )" )        &
            ( ROW( data%PTR( l ) ), COL( data%PTR( l ) ), l = data%PK( i ),    &
              data%PK( i + 1 ) - 1 )
        END DO
      END IF

      IF ( control%out > 0 .AND. control%print_level > 2 ) THEN
        WRITE( control%out, "( ' matrix:' )" )
        DO l = 1, data%PK( n + 1 ) - 1
          WRITE( control%out, "( 1X, I0, 2( ' (', I0, ',', I0, ')' ) )" ) l,   &
            ROW( data%PTR( l ) ), COL( data%PTR( l ) ),                        &
            COL( data%PTR( data%PTR_lower( l ) ) ),                            &
            ROW( data%PTR( data%PTR_lower( l ) ) )
        END DO
      END IF

!  allocate further workspace to record row (inverse) permutations and the
!  starting addresses for undetermined entries in each row

      array_name = 'SHA: data%PERM_inv'
      CALL SPACE_resize_array( n, data%PERM_inv,                               &
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

!  initialize undetermined row entry pointers

      data%PU( 1 : n ) = data%PK( 1 : n )

      data%approximation_algorithm_used = control%approximation_algorithm

!  ----------------------------------
!  algorithms 0-2 (aka paper 2.1-2.3)
!  ----------------------------------

      IF ( control%approximation_algorithm <= 2 ) THEN

!  find a row with the lowest count

        DO l = 1, n
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

!  reduce the row counts for each other row j that has an entry in column i

            DO k = data%PU( i ), data%PK( i + 1 ) - 1
              kk = data%PTR( k )
              r = ROW( kk ) ; c = COL( kk )
              IF ( r == c ) CYCLE

!  determine which of row(kk) or col(kk) gives the row number j

              IF ( c == i ) THEN
                j = r
              ELSE
                j = c
              END IF

!  upgrade DEGREE and its pointers

              deg = data%COUNT( j )
              kk = data%DEGREE_inv( j )
              jj = MAX( data%FIRST( deg ), l )
              IF ( jj /= kk ) THEN
                data%DEGREE( kk ) = data%DEGREE( jj )
                data%DEGREE( jj ) = j
                data%DEGREE_inv( j ) = jj
                data%DEGREE_inv( data%DEGREE( kk ) ) = kk
              END IF
              data%FIRST( deg ) = jj + 1
              data%LAST( deg - 1 ) = data%LAST( deg - 1 ) + 1
              min_degree = MIN( min_degree, deg - 1 )

!             DO jj = 0, max_row
!               write( control%out,"( ' degree ', I0, ' rows:' )" ) jj
!               write( control%out,"( 6( 1X, I0 ) )" ) &
!                ( data%DEGREE( kk ), kk = data%FIRST( jj ), data%LAST( jj ) )
!             END DO
!             WRITE( control%out,"( 10( 1X, I0 ) )" )                          &
!               ( data%DEGREE_inv( jj ), jj = 1, n )

!  reduce the count for row j

              data%COUNT( j ) = deg - 1

!  interchange entries jj = PU(j) and kk = data%PTR_lower(k) of PTR
!  and their shadows

              jj = data%PU( j )
              kk = data%PTR_lower( k )
              IF ( jj /= kk ) THEN
                k1 = data%PTR_lower( kk )
                j1 = data%PTR_lower( jj )
                data%PTR_lower( jj ) = k1
                data%PTR_lower( k1 ) = jj
                data%PTR_lower( kk ) = j1
                data%PTR_lower( j1 ) = kk
                ll = data%PTR( jj )
                data%PTR( jj ) = data%PTR( kk )
                data%PTR( kk ) = ll
              END IF
              data%PU( j ) = data%PU( j ) + 1
            END DO
          END IF
          data%COUNT( i ) = n + 1
          data%PERM_inv( l ) = i

!         DO ll = 1, data%PK( n + 1 ) - 1
!           write(  control%out, "( I0, 2( ' (', I0, ',', I0 ')' ) )" ) ll,    &
!             ROW( data%PTR( ll ) ), COL( data%PTR( ll ) ),                    &
!             ROW( data%PTR( data%PTR_lower( ll ) ) ),                         &
!             COL( data%PTR( data%PTR_lower( ll ) ) )
!         END DO

        END DO

!write( control%out,*) ' inv perm ', data%PERM_inv( : n )

!      DO i = 1, n
!        write( control%out,"( ' row ', I0 )" ) i
!        write( control%out,"( 6( '(', I0, ',', I0, ')' ) )" )                 &
!          ( ROW( data%PTR( l ) ), COL( data%PTR( l ) ),                       &
!            l = data%PU( i ),  data%PK( i + 1 ) - 1 )
!      END DO
!      DO i = 1, n
!        write( control%out,"( 3I0 )" ) i, data%PU( i ), data%PK( i + 1 ) - 1
!      END DO

        IF ( control%approximation_algorithm == 1 ) THEN
          data%differences_needed                                              &
            = MAXVAL( data%PK( 2 : n + 1 ) -  data%PU( 1 : n ) )
        ELSE IF ( control%approximation_algorithm == 2 ) THEN
          data%differences_needed = 0
          DO i = 1, n
            IF ( data%PK( i + 1 ) - data%PK( i ) <=                            &
                 control%max_sparse_degree ) THEN
              data%differences_needed = MAX( data%differences_needed,          &
                                               data%PK( i + 1 ) - data%PK( i ) )
            ELSE
              data%differences_needed = MAX( data%differences_needed,          &
                                               data%PK( i + 1 ) - data%PU( i ) )
            END IF
          END DO
        ELSE
          data%differences_needed                                              &
            = MAXVAL( data%PK( 2 : n + 1 ) -  data%PK( 1 : n ) )
        END IF

!  report the numbers of each block size

        data%LAST( 0 : data%differences_needed ) = 0
        DO i = 1, n
          l = data%PK( i + 1 ) - data%PU( i )
          data%LAST( l ) = data%LAST( l ) + 1
        END DO
        IF ( control%out > 0 .AND. control%print_level > 0 ) THEN
          WRITE( control%out, "( A, ' (block size, # with this size):' )" )    &
            prefix
          CALL SHA_write_nonzero_list( control%out, data%differences_needed,   &
                                       data%LAST )
        END IF

!  -----------------------------
!  algorithm 3 (aka paper 2.2/3)
!  -----------------------------

      ELSE
        data%unsym_rows = 0
        DO i = 1, n

!  skip rows that have more than max_sparse_degree entries

          IF ( data%PK( i + 1 ) - data%PK( i ) >                               &
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

!  interchange entries jj = PU(j) and kk = data%PTR_lower( k ) of PTR
!  and their shadows

              jj = data%PU( j )
              kk = data%PTR_lower( k )
              IF ( jj /= kk ) THEN
                k1 = data%PTR_lower( kk )
                j1 = data%PTR_lower( jj )
                data%PTR_lower( jj ) = k1
                data%PTR_lower( k1 ) = jj
                data%PTR_lower( kk ) = j1
                data%PTR_lower( j1 ) = kk
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
        data%differences_needed = 0
        j = data%unsym_rows
        DO i = 1, n
          IF (  data%COUNT( i ) == n + 1 ) THEN
            data%PU( i ) = data%PK( i )
            data%differences_needed =                                          &
              MAX( data%differences_needed, data%PK( i + 1 ) - data%PU( i ) )
          ELSE
            inform%max_reduced_degree =                                        &
              MAX( inform%max_reduced_degree, data%PK( i + 1 ) - data%PU( i ) )
            j = j + 1
            data%PERM_inv( j ) = i
          END IF
        END DO

        data%differences_needed                                                &
          = MAX( data%differences_needed, inform%max_reduced_degree )

        IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
!         WRITE(  control%out, "( ' maximum degree in the connectivity',       &
!      &   ' graph = ', I0, /, 1X, I0, ' symmetric differences required ',     &
!      &   /, ' max reduced degree = ', I0 )" )                                &
!         inform%max_degree, data%differences_needed, inform%max_reduced_degree
        END IF
      END IF
      inform%differences_needed = data%differences_needed

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

      array_name = 'SHA: data%DEGREE_inv'
      CALL SPACE_dealloc_array( data%DEGREE_inv,                               &
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

      array_name = 'SHA: data%PTR_lower'
      CALL SPACE_dealloc_array( data%PTR_lower,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  internal subroutine for writing a selective list of nonzeros

      CONTAINS

        SUBROUTINE SHA_write_nonzero_list( out, length, LIST )
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: out, length
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( 0 : length ) :: LIST
        INTEGER ( KIND = ip_ ) :: i, pos
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

      SUBROUTINE SHA_estimate( n, nz, ROW, COL, m_available, RD, ls1, ls2, S,  &
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

!     m_available is the number of differences provided; ideally this should
!       be as large as inform%differences_needed computed by sha_analyse
!     RD(i), i=1:m gives the index of the column of S and Y of the i-th
!       most recent differences
!     ls1, ls2 are the declared leading and trailing dimensions of S
!     S(i,j) (i=1:n,j=RD(1:m_avaiable)) are the steps
!     ly1, ly2 are the declared leading and trailing dimensions of Y
!     Y(i,j) (i=1:n,j=RD(1:m_available)) are the differences in gradients
!     VAL(i) is the i-th nonzero in the estimated Hessian matrix.(i=1,nz)

!   The analysed and permuted structure and the groups are stored in the
!   derived type data (see preface)

!   Action of the subroutine is controlled by components of the derived type
!   control, while information about the progress of the subroutine is reported
!   in inform (again, see preface). Success or failure is flagged by the
!   component inform%status -
!     0 if no error was detected
!    -3 invalid values input for n or nz
!   -11 if there was an error when forming the inform%bad_row-th row or column
!   -31 if the call to SHA_estimate was not preceded by a call to SHA_analyse

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz, m_available
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: ls1, ls2, ly1, ly2
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: ROW, COL
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m_available ) :: RD
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( nz ) :: VAL
      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER ( KIND = ip_ ) :: i, ii, info, j, jj, k, kk, n_max, rank
      INTEGER ( KIND = ip_ ) :: m_max, liwork, lwork, mu, nu, min_mn
      INTEGER ( KIND = ip_ ) :: m_needed, m_used, pki, pkip1, pui, status
      INTEGER ( KIND = ip_ ) :: ii_start, ii_end, ii_stride
      LOGICAL :: sym
!     LOGICAL :: debug_residuals = .TRUE.
      LOGICAL :: debug_residuals = .FALSE.
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

!  recall the number of differences needed to reproduce a fixed Hessian

      m_needed = data%differences_needed

! add %extra_differences to accommodate a singularity precaution if possible

!     m_max = MIN( m_needed + control%extra_differences, m_available )
      m_max = m_needed + control%extra_differences
      n_max = m_needed
      min_mn = MIN( m_max, n_max )

!  allocate workspace

!  generic solver workspace

      IF ( data%la1 < m_max .OR. data%la2 < n_max ) THEN
        data%la1 = m_max ; data%la2 = n_max
        array_name = 'SHA: data%A'
        CALL SPACE_resize_array( data%la1, data%la2, data%A,                   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%lb1 < m_max ) THEN
        data%lb1 = m_max
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

        IF ( data%la_save1 < m_max .OR. data%la_save2 < n_max ) THEN
          data%la_save1 = m_max ; data%la_save2 = n_max
          array_name = 'SHA: data%A_save'
          CALL SPACE_resize_array( data%la_save1, data%la_save2,               &
             data%A_save, inform%status, inform%alloc_status,                  &
             array_name = array_name,                                          &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  allocate space to hold a copy of b if needed

        IF ( data%lb_save < m_needed ) THEN
          data%lb_save = m_max
          array_name = 'SHA: data%B_save'
          CALL SPACE_resize_array( data%lb_save, 1,                            &
                 data%B_save, inform%status, inform%alloc_status,              &
                 array_name = array_name,                                      &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  discover how much additional temporary real storage may be needed by LU / LQ

        IF ( control%dense_linear_solver == 1 ) THEN
          liwork = min_mn
        ELSE IF ( control%dense_linear_solver == 2 ) THEN
          m_used = m_needed
          CALL GELSY( m_used, n, 1, data%A, data%la1, data%B, data%lb1,        &
                      data%solve_system_data%IWORK, eps_singular, rank,        &
                      data%WORK_1, - 1, status )
          lwork = INT( data%WORK_1( 1 ) ) ; liwork = n_max

!  allocate space to hold the singular values if needed

        ELSE
          IF ( data%ls < min_mn ) THEN
            data%ls = min_mn
            array_name = 'SHA: data%solve_syetem_data%S'
            CALL SPACE_resize_array( data%ls, data%solve_system_data%S,        &
                   inform%status, inform%alloc_status,                         &
                   array_name = array_name,                                    &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  discover how much temporary integer and real storage may be needed by SVD

          m_used = m_needed
          IF ( control%dense_linear_solver == 4 ) THEN
            CALL GELSD( m_used, n_max, 1, data%A, data%la1, data%B, data%lb1,  &
                        data%solve_system_data%S, eps_singular, rank,          &
                        data%WORK_1, - 1, data%IWORK_1, status )
            lwork = INT( data%WORK_1( 1 ) ) ; liwork = INT( data%IWORK_1( 1 ) )
          ELSE
            CALL GELSS( m_used, n_max, 1, data%A, data%la1, data%B, data%lb1,  &
                        data%solve_system_data%S, eps_singular, rank,          &
                        data%WORK_1, - 1, status )
            lwork = INT( data%WORK_1( 1 ) ) ; liwork = n_max
          END IF
        END IF

!  allocate temporary integer storage

        IF ( control%dense_linear_solver /= 3 ) THEN
          IF ( data%solve_system_data%liwork < liwork ) THEN
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

!  allocate temporary real storage

        IF ( control%dense_linear_solver /= 1 ) THEN
          IF ( data%solve_system_data%lwork < lwork ) THEN
            data%solve_system_data%lwork = lwork
            array_name = 'SHA: data%solve_system_data%WORK'
            CALL SPACE_resize_array( data%solve_system_data%lwork,             &
                   data%solve_system_data%WORK, inform%status,                 &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF
        END IF
      END IF
      inform%status = GALAHAD_ok

!  for permuted row i:

!     - ----------- ------------------ -
! PTR  | known     |  unknown         | .    (ROW(kk),COL(kk),VAL(kk)),
!     - ----------- ------------------ -      kk=PTR(k) gives entries in row(i)
!       ^           ^                  ^      for k=PK(i),..,P(i+1)-1 with
!       |           |                  |      those for k=PU(i),..,P(i+1)-1
!     PK(i)        PU(i)            PK(i+1)   still to be determined

!  run through the rows finding the unknown entries (backwards when
!  not exploiting symmetry)

      IF ( control%approximation_algorithm > 0 ) THEN
        ii_start = 1 ; ii_end = n ; ii_stride = 1
      ELSE
        ii_start = n ; ii_end = 1 ; ii_stride = - 1
      END IF

      DO ii = ii_start, ii_end, ii_stride
        i = data%PERM_inv( ii )
        pki = data%PK( i ) ; pkip1 = data%PK( i + 1 ) ; nu = pkip1 - pki
        IF ( nu == 0 ) CYCLE

!  decide whether to exploit symmetry or not

        sym = control%approximation_algorithm == 1 .OR.                        &
              control%approximation_algorithm == 3 .OR.                        &
            ( control%approximation_algorithm == 2 .AND. nu > m_available )

!  -----------------------------
!  methods that exploit symmetry
!  -----------------------------

        IF ( sym ) THEN
          pui = data%PU( i ) ; nu = pkip1 - pui
          IF ( nu == 0 ) CYCLE
          mu = MIN( nu + control%extra_differences, m_available )

!  compute the unknown entries B_{ij}, j in I_i^-, to satisfy
!    sum_{j in I_i^-} B_{ij} s_{jl}  = y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}
!  for l = 1,.., |I_i^+|, where
!    I_i^+ = { j : j \in I_i and B_{ji} is already known }
!    I_i^- = I_i \ I_i^+ and I_i = { j : B_{ij} /= 0}

!  compute the right-hand side y_{il} - sum_{j in I_i^+} B_{ij} s_{jl},
!  initialize b to Y(i,l)

          data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  loop over the known entries

          jj = 0
          DO k = pki, pui - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )
            jj = jj + 1 ; data%COUNT( jj ) = j            

!  subtract B_{ij} s_{jl} from b

            data%B( 1 : mu, 1 )                                                &
              = data%B( 1 : mu, 1 ) - VAL( kk ) * S( j, RD( 1 : mu ) )
          END DO
          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, "( ' known', 9( 1X, I0 ), /, ( 1X, 10I6 ) )" ) &
              data%COUNT( : jj )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

          jj = 0
          DO k = pui, pkip1 - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )
            jj = jj + 1 ; data%COUNT( jj ) = j            

!  set the entries of A

            data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
          END DO

! make a copy of A and b as a precaution for possible use later

          IF ( mu + 1 <= m_max ) THEN
            data%A_save( 1 : mu, 1 : nu ) = data%A( 1 : mu, 1 : nu )
            data%B_save( 1 : mu, 1 ) = data%B( 1 : mu, 1 )
          END IF
          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, "( ' unknown', 9( 1X, I0 ), /, ( 1X, 10I6 ) )")&
              data%COUNT( : jj )

!  solve A x = b

          CALL SHA_solve_system( control%dense_linear_solver, mu, nu,          &
                                 data%A, data%la1, data%B, data%lb1,           &
                                 data%solve_system_data, i,                    &
                                 control%out, control%print_level, info )

!  if A appears to be singular, add an extra row if there is one, and
!  solve the system as a least-squares problem

          IF ( info == MAX( nu, mu ) + 1 .AND. mu + 1 <= m_max ) THEN

!  initialize b to Y(i,l)

            data%B( mu + 1, 1 ) = Y( i, RD( mu + 1 ) )

!  loop over the known entries

            DO k = pki, pui - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

              data%B( mu + 1, 1 )                                              &
                = data%B( mu + 1, 1 ) - VAL( kk ) * S( j, RD( mu + 1 ) )
            END DO

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 0
            DO k = pui, pkip1 - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  set the entries of A

              jj = jj + 1
              data%A( mu + 1, jj ) = S( j, RD( mu + 1 ) )
            END DO

!  solve A x = b

            CALL SHA_solve_system( control%dense_linear_solver, mu + 1, nu,    &
                                   data%A, data%la1, data%B, data%lb1,         &
                                   data%solve_system_data, i,                  &
                                   control%out, control%print_level, info )

!  check for errors

          ELSE IF ( info /= 0 ) THEN
            inform%status = GALAHAD_error_factorization
            inform%bad_row = i ; GO TO 900
          END IF

!  finally, set the unknown B_{ij}

          jj = 1
          DO k = pui, pkip1 - 1
            VAL( data%PTR( k ) ) = data%B( jj, 1 )
            jj = jj + 1
          END DO

!  if required, compute and print the residuals

          IF ( debug_residuals ) THEN

!  initialize b to Y(i,l)

            data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  loop over the known entries

            DO k = pki, pkip1 - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

              data%B( 1 : mu, 1 )                                              &
                = data%B( 1 : mu, 1 ) - VAL( kk ) * S( j, RD( 1 : mu ) )
            END DO
            write(6, "( ' max error row is ', ES12.4, ' in row ', I0 )" )      &
              MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i
          END IF

!  -----------------------------------
!  methods that don't exploit symmetry
!  -----------------------------------

        ELSE
          mu = MIN( nu + control%extra_differences, m_available )

!  compute the unknown entries B_{ij}, j in I_i, to satisfy
!    sum_{j in I_i} B_{ij} s_{jl}  = y_{il}
!  for l = 1,.., |I_i|, where I_i = { j : B_{ij} /= 0}

!  store the right-hand side y_{il}, initialize b to Y(i,l)

          data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

          jj = 0
          DO k = pki, pkip1 - 1
            kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

            j = COL( kk )
            IF ( j == i ) j = ROW( kk )

!  set the entries of A

            jj = jj + 1
            data%A( 1 : mu, jj ) = S( j, RD( 1 : mu ) )
          END DO

! make a copy of A and b as a precaution for possible use later

          IF ( mu + 1 <= m_max ) THEN
            data%A_save( 1 : mu, 1 : nu ) = data%A( 1 : mu, 1 : nu )
            data%B_save( 1 : mu, 1 ) = data%B( 1 : mu, 1 )
          END IF

!  solve A x = b

          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, "( ' vars ', 9I6, /, ( 10I6 ) )" )             &
              COL( data%PTR( pki : pkip1 - 1 ) )
          CALL SHA_solve_system( control%dense_linear_solver, mu, nu, data%A,  &
                                 data%la1, data%B, data%lb1,                   &
                                 data%solve_system_data, i,                    &
                                 control%out, control%print_level, info )

!  if A appears to be singular, add an extra row if there is one, and
!  solve the system as a least-squares problem

          IF ( info == MAX( nu, mu ) + 1 .AND. mu + 1 <= m_max ) THEN

!  store the right-hand side y_{il}, initialize b to Y(i,l)

            data%B( mu + 1, 1 ) = Y( i, RD( mu + 1 ) )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 0
            DO k = pki, pkip1 - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  set the entries of A

              jj = jj + 1
              data%A( mu + 1, jj ) = S( j, RD( mu + 1 ) )
            END DO

!  solve A x = b

            CALL SHA_solve_system( control%dense_linear_solver, mu + 1, nu,    &
                                   data%A, data%la1, data%B, data%lb1,         &
                                   data%solve_system_data, i,                  &
                                   control%out, control%print_level, info )
!  check for errors

          ELSE IF ( info /= 0 ) THEN
            inform%status = GALAHAD_error_factorization
            inform%bad_row = i ; GO TO 900
          END IF

!  finally, set the unknown B_{ij}

          jj = 0
          DO k = pki, pkip1 - 1
            jj = jj + 1
            VAL( data%PTR( k ) ) = data%B( jj, 1 )
          END DO

!  if required, compute and print the residuals

          IF ( debug_residuals ) THEN

!  initialize b to Y(i,l)

            data%B( 1 : mu, 1 ) = Y( i, RD( 1 : mu ) )

!  loop over the known entries

            DO k = pki, pkip1 - 1
              kk = data%PTR( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

              data%B( 1 : mu, 1 )                                              &
                = data%B( 1 : mu, 1 ) - VAL( kk ) * S( j, RD( 1 : mu ) )
            END DO
            write(6, "( ' max error row is ', ES12.4, ' in row ', I0 )" )      &
              MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i
          END IF
        END IF
      END DO

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
                                   data, row, out, print_level, status )

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: dense_linear_solver, m, n
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: la1, lb1, row, out, print_level
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( la1, n ) :: A
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( lb1, 1 ) :: B
      TYPE ( SHA_solve_system_data_type ) :: data

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER ( KIND = ip_ ) :: i, rank
      REAL ( KIND = rp_ ), DIMENSION( la1, n ) :: A_save
      LOGICAL :: printi

      printi = out > 0 .AND. print_level > 0

!  solve A x = b using Gaussian elimination; A is copied to A_save as a
!  precaution

      IF ( dense_linear_solver == 1 ) THEN
        IF ( m == n ) THEN
          CALL GETRF( m, n, A, la1, data%IWORK, status )
          IF ( status == 0 ) THEN
            CALL GETRS( 'N', n, 1, A, la1, data%IWORK, B, lb1, status )
            IF ( status == 0 ) RETURN
          END IF
        END IF

!  solve A x = b using a QR factorization; A is copied to A_save as a precaution

      ELSE IF ( dense_linear_solver == 2 ) THEN
        CALL GELSY( m, n, 1, A, la1, B, lb1, data%IWORK, eps_singular, rank,   &
                    data%WORK, data%lwork, status )
        IF ( status == 0 ) RETURN

!  solve A x = b using a singular-value decomposition

      ELSE
        IF ( printi ) A_save( : m, : n ) = A( : m, : n )
        IF ( dense_linear_solver == 4 ) THEN
          CALL GELSD( m, n, 1, A, la1, B, lb1, data%S, eps_singular, rank,     &
                      data%WORK, data%lwork, data%IWORK, status )
        ELSE ! dense_linear_solver == 3
          CALL GELSS( m, n, 1, A, la1, B, lb1, data%S, eps_singular, rank,     &
                      data%WORK, data%lwork, status )
        END IF
!       IF ( data%S( MIN( m, n ) ) / data%S( 1 ) <= eps_singular ) THEN
!         status = MAX( m, n ) + 1
          IF( printi ) THEN
            WRITE( out, "( ' matrix singular, sigma_min/sigma_1 = ',           &
           &       ES11.4 )" )  data%S( MIN( m, n ) ) / data%S( 1 )
            IF ( print_level > 1 ) THEN
              WRITE( out, "( ' row ', I0, ', solver status = ',                &
             &       I0, /, ' matrix =' )" ) row, status
              DO i = 1, n
                WRITE( out, "( ' column ', I0, ' = ', ( 5ES12.4 ) )" )         &
                  i, A_save( : m, i )
              END DO
              WRITE( out, "( ' sigma = ', ( 5ES12.4 ) )" )                     &
                data%S( 1 : MIN( m, n ) )
              WRITE( out, "( ' b = ', ( 5ES12.4 ) )" ) B( 1 : n, 1 )
            END IF
          END IF
        END IF
!     END IF
      IF ( printi ) THEN
        IF ( rank > 0 ) THEN
          WRITE( out, "( ' row ', I0, ', m ', I0, ', n ', I0, ', rank ',       &
         &       I0, ', kappa ', ES11.4 )" )                                   &
           row, m, n, rank, data%S( rank ) / data%S( 1 )
        ELSE
          WRITE( out, "( ' row ', I0, ' m ', I0, ' n ', I0, ' rank ', I0 )" )  &
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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nz
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: ROW, COL
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: ROW_COUNT

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER ( KIND = ip_ ) :: i, j, l

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

      array_name = 'SHA: data%COUNT'
      CALL SPACE_dealloc_array( data%COUNT,                                    &
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
      CALL SPACE_dealloc_array( data%A_save,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%solve_system_data%B_save'
      CALL SPACE_dealloc_array( data%B_save,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  re-initial private data

      data%SHA_analyse_called = .FALSE.
      data%la1 = - 1 ; data%la2 = - 1 ; data%lb1 = - 1 ; data%ls = - 1
      data%la_save1 = - 1 ; data%la_save2 = - 1 ; data%lb_save = - 1
      data%solve_system_data%lwork = - 1
      data%solve_system_data%liwork = - 1
      data%dense_linear_solver = - 1
      data%differences_needed = - 1

      RETURN

!  End of subroutine SHA_terminate

      END SUBROUTINE SHA_terminate

! -  G A L A H A D -  S H A _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE SHA_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SHA_control_type ), INTENT( IN ) :: control
     TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

     CALL SHA_terminate( data%sha_data, control, inform )

     RETURN

!  End of subroutine SHA_full_terminate

     END SUBROUTINE SHA_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-  G A L A H A D -  S H A _ i n f o r m a t i o n   S U B R O U T I N E  -*-

     SUBROUTINE SHA_information( data, inform, status )

!  return solver information during or after solution by SHA
!  See SHA_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SHA_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%sha_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SHA_information

     END SUBROUTINE SHA_information

!  End of module GALAHAD_SHA

   END MODULE GALAHAD_SHA_precision
