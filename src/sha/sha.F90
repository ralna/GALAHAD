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
               SHA_full_initialize, SHA_full_terminate, SHA_information,       &
               SHA_analyse_matrix, SHA_recover_matrix, SHA_reset_control

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
!    for each iteration, >= 3 gives increasingly verbose (debu

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  which approximation algorithm should be used?
!    1 : unsymmetric (alg 2.1 in paper)
!    2 : symmetric (alg 2.2 in paper)
!    3 : composite (alg 2.3 in paper)
!    4 : composite 2 (alg 2.4 in paper)
!    5 : cautious (alg 2.5 in paper)

       INTEGER ( KIND = ip_ ) :: approximation_algorithm = 4

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

!  the maximum degree in the adjacency graph

       INTEGER ( KIND = ip_ ) :: max_degree = - 1

!  the number of differences that will be needed (more may be helpful)

       INTEGER ( KIND = ip_ ) :: differences_needed = - 1

!  the maximum reduced degree in the adjacency graph

       INTEGER ( KIND = ip_ ) :: max_reduced_degree = - 1

!  the actual approximation algorithm used

       INTEGER ( KIND = ip_ ) :: approximation_algorithm_used = - 1

!  a failure occurred when forming the bad_row-th row or column (0 = no failure)

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
       INTEGER ( KIND = ip_ ) :: l_sparse
       INTEGER ( KIND = ip_ ) :: singular_matrices

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

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: STR
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: STU
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LIST
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: FIRST
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: LAST
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COUNT
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PERM_inv
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ROWS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP_lower
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
       INTEGER ( KIND = ip_ ) :: n, ne
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ROW
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COL
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER
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
!  approximation-algorithm                         4
!  dense-linear-solver                             3
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
      LOGICAL :: printi
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
      printi = control%out > 0 .AND. control%print_level > 0

!  allocate space for row starting addresses STR and unsymmetric row counts 
!  COUNT

      array_name = 'SHA: data%STR'
      CALL SPACE_resize_array( n + 1, data%STR,                                &
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

!  record the number of nonzeros in each row of the whole matrix (both upper 
!  and lower parts) in COUNT

      data%COUNT = 0
      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%COUNT( i ) = data%COUNT( i ) + 1
        IF ( i /= j ) data%COUNT( j ) = data%COUNT( j ) + 1
      END DO

!  now set the starting addresses for each row of the whole matrix in STR

      data%STR( 1 ) = 1
      DO i = 1, n
        data%STR( i + 1 ) = data%STR( i ) + data%COUNT( i )
      END DO

!  compute the maximum degree (row length)

      max_row = MAXVAL( data%COUNT( 1 : n ) )
      inform%max_degree = max_row

!  allocate space for the list of rows ordered by increasing degree (LIST), 
!  as well the first and last positions for those of a given degree (FIRST 
!  and LAST) and pointers from the the list of degrees to the rows (ROWS)

      array_name = 'SHA: data%LIST'
      CALL SPACE_resize_array( n, data%LIST,                                   &
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

      array_name = 'SHA: data%ROWS'
      CALL SPACE_resize_array( n, data%ROWS,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize the number of rows with each degree in LAST

      data%LAST( 0 : max_row ) = 0
      DO i = 1, n
        data%LAST( data%COUNT( i ) ) = data%LAST( data%COUNT( i ) ) + 1
      END DO

!  make sure that the approximation algorithm requested is possible

      IF ( control%approximation_algorithm < 1 .OR.                            &
           control%approximation_algorithm > 5 ) THEN
        inform%approximation_algorithm_used = 4
      ELSE
        inform%approximation_algorithm_used = control%approximation_algorithm
      END IF

      IF ( printi ) THEN
        WRITE( control%out, "( ' Algorithm ', I0 )" )                          &
          inform%approximation_algorithm_used
        WRITE( control%out, "( A, ' (row size, # with this size):' )" ) prefix
        CALL SHA_write_nonzero_list( control%out, max_row, data%LAST )
      END IF

!  set the start (FIRST) and finish (LAST) positions for each degree in the
!  list LIST

      data%FIRST( 0 ) = 1
      DO i = 0, max_row - 1
        data%FIRST( i + 1 ) = data%FIRST( i ) + data%LAST( i )
        data%LAST( i ) = data%FIRST( i + 1 ) - 1
      END DO
      data%LAST( max_row ) = data%FIRST( max_row ) + data%LAST( max_row ) - 1

!  now sort the rows by increasing degree (LIST) (and record their inverses) 
!  (ROWS) ...

      DO i = 1, n
        deg = data%COUNT( i )
        data%LIST( data%FIRST( deg ) ) = i
        data%ROWS( i ) = data%FIRST( deg )
        data%FIRST( deg ) = data%FIRST( deg ) + 1
      END DO

!  .. and reset the starting positions (FIRST)

      DO i = max_row - 1, 0, - 1
        data%FIRST( i + 1 ) = data%FIRST( i )
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
          WRITE( control%out, "( ' degree ', I0, ' involves rows:' )" ) i
          IF ( data%FIRST( i ) <= data%LAST( i ) )                             &
            WRITE( control%out, "( 10( :, 1X, I0 ) )" )                        &
             ( data%LIST( l ), l = data%FIRST( i ), data%LAST( i ) )
        END DO
        WRITE( control%out, "( ' ROWS:', /, 10( 1X, I0 ) )" )                  &
          data%ROWS( 1 : n )
      END IF

!  allocate space for MAP to hold mappings from the rows back to the
!  coordinate storage, and its "shadow" MAP_lower set so that entries k and
!  MAP_lower(k) of MAP correspond to the "upper" and "lower" triangular entries
!  (i,j) and (j,i), i <= j

      array_name = 'SHA: data%MAP'
      CALL SPACE_resize_array( data%STR( n + 1 ) - 1, data%MAP,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%MAP_lower'
      CALL SPACE_resize_array( data%STR( n + 1 ) - 1, data%MAP_lower,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  now set the MAP and MAP_lower maps ...

      DO l = 1, nz
        i = ROW( l ) ; j = COL( l )
        data%MAP( data%STR( i ) ) = l
        IF ( i /= j ) THEN
          data%MAP( data%STR( j ) ) = l
          data%MAP_lower( data%STR( j ) ) = data%STR( i )
          data%MAP_lower( data%STR( i ) ) = data%STR( j )
          data%STR( j ) = data%STR( j ) + 1
        ELSE
          data%MAP_lower( data%STR( i ) ) = data%STR( i )
        END IF
        data%STR( i ) = data%STR( i ) + 1
      END DO

!  ... and reset the row starting addresses

      DO i = n - 1, 1, - 1
        data%STR( i + 1 ) = data%STR( i )
      END DO
      data%STR( 1 ) = 1

!  print more row statistics if required

      IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
        DO i = 1, n
          WRITE( control%out, "( ' row ', I0, ' has entries' )" ) i
          WRITE( control%out, "( 1X, 6( '(', I0, ',', I0, ')', : ) )" )        &
            ( ROW( data%MAP( l ) ), COL( data%MAP( l ) ), l = data%STR( i ),   &
              data%STR( i + 1 ) - 1 )
        END DO
      END IF

      IF ( control%out > 0 .AND. control%print_level > 2 ) THEN
        WRITE( control%out, "( ' matrix:' )" )
        DO l = 1, data%STR( n + 1 ) - 1
          WRITE( control%out, "( 1X, I0, 2( ' (', I0, ',', I0, ')' ) )" ) l,   &
            ROW( data%MAP( l ) ), COL( data%MAP( l ) ),                        &
            COL( data%MAP( data%MAP_lower( l ) ) ),                            &
            ROW( data%MAP( data%MAP_lower( l ) ) )
        END DO
      END IF
!stop

!  allocate further workspace to record row (inverse) permutations (PERM_inv)
!  and the starting addresses for undetermined entries in each row (STU)

      array_name = 'SHA: data%PERM_inv'
      CALL SPACE_resize_array( n, data%PERM_inv,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'SHA: data%STU'
      CALL SPACE_resize_array( n, data%STU,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize undetermined row entry pointers

      data%STU( 1 : n ) = data%STR( 1 : n )

      SELECT CASE ( inform%approximation_algorithm_used )

!  ----------------------------------
!  algorithms 1-3 (aka paper 2.1-2.3)
!  ----------------------------------

      CASE ( 1 : 3 )

!  find a row with the lowest count

        DO l = 1, n
          i = data%LIST( MAX( data%FIRST( min_degree ), l ) )
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

            DO k = data%STU( i ), data%STR( i + 1 ) - 1
              kk = data%MAP( k )
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
              kk = data%ROWS( j )
              jj = MAX( data%FIRST( deg ), l )
              IF ( jj /= kk ) THEN
                data%LIST( kk ) = data%LIST( jj )
                data%LIST( jj ) = j
                data%ROWS( j ) = jj
                data%ROWS( data%LIST( kk ) ) = kk
              END IF
              data%FIRST( deg ) = jj + 1
              data%LAST( deg - 1 ) = data%LAST( deg - 1 ) + 1
              min_degree = MIN( min_degree, deg - 1 )

!             DO jj = 0, max_row
!               write( control%out,"( ' degree ', I0, ' rows:' )" ) jj
!               write( control%out,"( 6( 1X, I0 ) )" ) &
!                ( data%LIST( kk ), kk = data%FIRST( jj ), data%LAST( jj ) )
!             END DO
!             WRITE( control%out,"( 10( 1X, I0 ) )" )                          &
!               ( data%ROWS( jj ), jj = 1, n )

!  reduce the count for row j

              data%COUNT( j ) = deg - 1

!  interchange entries jj = PU(j) and kk = data%MAP_lower(k) of MAP
!  and their shadows

              jj = data%STU( j )
              kk = data%MAP_lower( k )
              IF ( jj /= kk ) THEN
                k1 = data%MAP_lower( kk )
                j1 = data%MAP_lower( jj )
                data%MAP_lower( jj ) = k1
                data%MAP_lower( k1 ) = jj
                data%MAP_lower( kk ) = j1
                data%MAP_lower( j1 ) = kk
                ll = data%MAP( jj )
                data%MAP( jj ) = data%MAP( kk )
                data%MAP( kk ) = ll
              END IF
              data%STU( j ) = data%STU( j ) + 1
            END DO
          END IF
          data%COUNT( i ) = n + 1
          data%PERM_inv( l ) = i

!         DO ll = 1, data%STR( n + 1 ) - 1
!           write(  control%out, "( I0, 2( ' (', I0, ',', I0 ')' ) )" ) ll,    &
!             ROW( data%MAP( ll ) ), COL( data%MAP( ll ) ),                    &
!             ROW( data%MAP( data%MAP_lower( ll ) ) ),                         &
!             COL( data%MAP( data%MAP_lower( ll ) ) )
!         END DO

        END DO

!write( control%out,*) ' inv perm ', data%PERM_inv( : n )

!      DO i = 1, n
!        write( control%out,"( ' row ', I0 )" ) i
!        write( control%out,"( 6( '(', I0, ',', I0, ')' ) )" )                 &
!          ( ROW( data%MAP( l ) ), COL( data%MAP( l ) ),                       &
!            l = data%STU( i ),  data%STR( i + 1 ) - 1 )
!      END DO
!      DO i = 1, n
!        write( control%out,"( 3I0 )" ) i, data%STU( i ), data%STR( i + 1 ) - 1
!      END DO

        IF ( inform%approximation_algorithm_used == 2 ) THEN
          data%differences_needed                                              &
            = MAXVAL( data%STR( 2 : n + 1 ) -  data%STU( 1 : n ) )
        ELSE IF ( inform%approximation_algorithm_used == 3 ) THEN
          data%differences_needed = 0
          DO i = 1, n
            IF ( data%STR( i + 1 ) - data%STR( i ) <=                          &
                 control%max_sparse_degree ) THEN
              data%differences_needed = MAX( data%differences_needed,          &
                                             data%STR( i + 1 ) - data%STR( i ) )
            ELSE
              data%differences_needed = MAX( data%differences_needed,          &
                                             data%STR( i + 1 ) - data%STU( i ) )
            END IF
          END DO
        ELSE
          data%differences_needed                                              &
            = MAXVAL( data%STR( 2 : n + 1 ) -  data%STR( 1 : n ) )
        END IF

!  report the numbers of each block size

        data%LAST( 0 : data%differences_needed ) = 0
        DO i = 1, n
          l = data%STR( i + 1 ) - data%STU( i )
          data%LAST( l ) = data%LAST( l ) + 1
        END DO
        IF ( printi ) THEN
          WRITE( control%out, "( A, ' (block size, # with this size):' )" )    &
            prefix
          CALL SHA_write_nonzero_list( control%out, data%differences_needed,   &
                                       data%LAST )
        END IF

!  ---------------------------
!  algorithm 4 (aka paper 2.4)
!  ---------------------------

      CASE ( 4 )
        data%unsym_rows = 0
        DO i = 1, n

!  skip rows that have more than max_sparse_degree entries

          IF ( data%STR( i + 1 ) - data%STR( i ) >                             &
               control%max_sparse_degree ) CYCLE
          data%unsym_rows = data%unsym_rows + 1
          data%PERM_inv( data%unsym_rows ) = i

!  reduce the row counts for all other rows that have an entry in column i

          IF ( data%unsym_rows < n ) THEN
            DO k = data%STU( i ), data%STR( i + 1 ) - 1
              kk = data%MAP( k )
              r = ROW( kk ) ; c = COL( kk )
              IF ( r == c ) CYCLE

!  determine which of row(kk) or col(kk) gives the column number j

              IF ( c == i ) THEN
                j = r
              ELSE
                j = c
              END IF

!  interchange entries jj = PU(j) and kk = data%MAP_lower( k ) of MAP
!  and their shadows

              jj = data%STU( j )
              kk = data%MAP_lower( k )
              IF ( jj /= kk ) THEN
                k1 = data%MAP_lower( kk )
                j1 = data%MAP_lower( jj )
                data%MAP_lower( jj ) = k1
                data%MAP_lower( k1 ) = jj
                data%MAP_lower( kk ) = j1
                data%MAP_lower( j1 ) = kk
                ll = data%MAP( jj )
                data%MAP( jj ) = data%MAP( kk )
                data%MAP( kk ) = ll
              END IF
              data%STU( j ) = data%STU( j ) + 1
            END DO
          END IF
          data%COUNT( i ) = n + 1
        END DO

        inform%max_reduced_degree = 0
        data%differences_needed = 0
        j = data%unsym_rows
        DO i = 1, n
          IF (  data%COUNT( i ) == n + 1 ) THEN
            data%STU( i ) = data%STR( i )
            data%differences_needed =                                          &
              MAX( data%differences_needed, data%STR( i + 1 ) - data%STU( i ) )
          ELSE
            inform%max_reduced_degree =                                        &
             MAX( inform%max_reduced_degree, data%STR( i + 1 ) - data%STU( i ) )
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

!  ---------------------------
!  algorithm 5 (aka paper 2.5)
!  ---------------------------

      CASE ( 5 )
 
        data%differences_needed = 0
        data%l_sparse = data%LAST( MIN( max_row, control%max_sparse_degree ) )
!write(6,*) MIN( max_row, control%max_sparse_degree )
!write(6,*) data%LAST( MIN( max_row, control%max_sparse_degree ) )
!stop
        data%LAST( 0 : max_row ) = 0
!  loop over the rows by increasing counts

        DO l = 1, data%l_sparse
          i = data%LIST( l )
          data%PERM_inv( l ) = i
          ll = data%STR( i + 1 ) - data%STR( i )
          data%differences_needed = MAX( data%differences_needed, ll )
          IF ( printi ) data%LAST( ll ) = data%LAST( ll ) + 1
!         write( 6, "( ' ------ i, nz ------', 2I8 )" ) & 
!           i, data%STR( i + 1 ) -  data%STR( i )

!  loop over the entries in the chosen row

!         DO k = data%STR( i ), data%STR( i + 1 ) - 1
          DO k = data%STU( i ), data%STR( i + 1 ) - 1
            kk = data%MAP( k )
            r = ROW( kk ) ; c = COL( kk )
!           write( 6, "( ' i, r, c, kk ', 4I8 )" ) i, r, c, kk
            IF ( r == c ) CYCLE

!  determine which of row(kk) or col(kk) gives the row number j

            IF ( c == i ) THEN
              j = r
            ELSE
              j = c
            END IF

!  interchange entries jj = PU(j) and kk = data%MAP_lower(k) of MAP
!  and their shadows

            jj = data%STU( j )
            kk = data%MAP_lower( k )
            IF ( jj /= kk ) THEN
              k1 = data%MAP_lower( kk )
              j1 = data%MAP_lower( jj )
              data%MAP_lower( jj ) = k1
              data%MAP_lower( k1 ) = jj
              data%MAP_lower( kk ) = j1
              data%MAP_lower( j1 ) = kk
              ll = data%MAP( jj )
              data%MAP( jj ) = data%MAP( kk )
              data%MAP( kk ) = ll
            END IF
            data%STU( j ) = jj + 1
          END DO
        END DO

        DO l = data%l_sparse + 1, n
          i = data%LIST( l )
          data%PERM_inv( l ) = i
          ll = data%STR( i + 1 ) - data%STU( i )
          data%differences_needed = MAX( data%differences_needed, ll )
          IF ( printi ) data%LAST( ll ) = data%LAST( ll ) + 1
        END DO
!      write( 6, * ) ' n, l_sparse, differences_needed ',  &
!        n, data%l_sparse, data%differences_needed
!      write( 6, "( '    l    i        st        su        s+' )" )
!      DO l = 1, n
!        i = data%LIST( l )
!        write(6,"(2I5,3I10)") l, i, data%STR(i), data%STU(i), data%STR(i+1)
!       END DO

!  report the numbers of each block size

        IF ( printi ) THEN
          WRITE( control%out, "( A, ' (block size, # with this size):' )" )    &
            prefix
          CALL SHA_write_nonzero_list( control%out, data%differences_needed,   &
                                       data%LAST )
        END IF
      END SELECT
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

      array_name = 'SHA: data%ROWS'
      CALL SPACE_dealloc_array( data%ROWS,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%LIST'
      CALL SPACE_dealloc_array( data%LIST,                                     &
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

      array_name = 'SHA: data%MAP_lower'
      CALL SPACE_dealloc_array( data%MAP_lower,                                &
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

      SUBROUTINE SHA_estimate( n, nz, ROW, COL, m_available, S, ls1, ls2,      &
                               Y, ly1, ly2, VAL, data, control, inform,        &
                               ORDER, VAL_true )

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
!     ls1, ls2 are the declared leading and trailing dimensions of S
!     S(i,j) (i=1:n,j=ORDER(1:m_available)) are the steps
!     ly1, ly2 are the declared leading and trailing dimensions of Y
!     Y(i,j) (i=1:n,j=ORDER(1:m_available)) are the gradient differences
!     VAL(i) is the i-th nonzero in the estimated Hessian matrix.(i=1,nz)
!
!   in addition, optional arguments are

!     ORDER(i), i=1:m gives the index of the column of S and Y of the i-th
!       most recent differences. If absent, the index will be i, i=1:m
!     VAL_true(i) is the i-th nonzero in the true Hessian matrix.(i=1,nz),
!       and only used for testing

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
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( nz ) :: VAL
      TYPE ( SHA_control_type ), INTENT( IN ) :: control
      TYPE ( SHA_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SHA_data_type ), INTENT( INOUT ) :: data

!  optional arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                          &
                              DIMENSION( m_available ) :: ORDER
      REAL ( KIND = rp_ ), INTENT( IN ), OPTIONAL, DIMENSION( nz ) :: VAL_true

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER ( KIND = ip_ ) :: i, ii, info, j, jj, k, kk, n_max, rank
      INTEGER ( KIND = ip_ ) :: m_max, liwork, lwork, mu, nu, min_mn
      INTEGER ( KIND = ip_ ) :: m_needed, m_used, stri, strip1, stui, status
      INTEGER ( KIND = ip_ ) :: ii_start, ii_end, ii_stride, pass
      INTEGER ( KIND = ip_ ) :: dense_linear_solver
      LOGICAL :: sym, order_present
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

      data%singular_matrices = 0
      order_present = PRESENT( order )

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
        IF ( data%dense_linear_solver < 1 .OR.                                 &
             data%dense_linear_solver > 4 ) data%dense_linear_solver = 3

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

        IF ( data%dense_linear_solver == 1 ) THEN
          liwork = min_mn
        ELSE IF ( data%dense_linear_solver == 2 ) THEN
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
          IF ( data%dense_linear_solver == 4 ) THEN
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

        IF ( data%dense_linear_solver /= 3 ) THEN
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

        IF ( data%dense_linear_solver /= 1 ) THEN
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
! MAP  | known     |  unknown         | .    (ROW(kk),COL(kk),VAL(kk)),
!     - ----------- ------------------ -      kk=MAP(k) gives entries in row(i)
!       ^           ^                  ^      for k=STR(i),..,STR(i+1)-1 with
!       |           |                  |      those for k=STU(i),..,STR(i+1)-1
!     PK(i)        PU(i)            PK(i+1)   still to be determined
!     = stri       = stui           = strip1
!

!  split the main loop into two passes so that algorithm 2.5 can march
!  backwards over separate segments (the second pass is skipped for the
!  other algorithms)

      DO pass = 1, 2

!  run through the rows finding the unknown entries (backwards when
!  not exploiting symmetry)

        IF ( pass == 1 ) THEN
          IF ( inform%approximation_algorithm_used == 1 ) THEN
            ii_start = n ; ii_end = 1 ; ii_stride = - 1
          ELSE IF ( inform%approximation_algorithm_used == 5 ) THEN
            ii_start = data%l_sparse ; ii_end = 1 ; ii_stride = - 1
          ELSE
            ii_start = 1 ; ii_end = n ; ii_stride = 1
          END IF
        ELSE
          IF ( inform%approximation_algorithm_used < 5 ) EXIT
          ii_start = n ; ii_end = data%l_sparse + 1 ; ii_stride = - 1
        END IF
!write(6,*) ' pass ', pass
        DO ii = ii_start, ii_end, ii_stride
          i = data%PERM_inv( ii )
          stri = data%STR( i ) ; strip1 = data%STR( i + 1 ) ; nu = strip1 - stri
          IF ( nu == 0 ) CYCLE

!  decide whether to exploit symmetry or not

          sym = inform%approximation_algorithm_used == 2 .OR.                  &
              ( inform%approximation_algorithm_used == 3 .AND.                 &
                nu > m_available ) .OR.                                        &
                inform%approximation_algorithm_used == 4 .OR.                  &
              ( inform%approximation_algorithm_used == 5 .AND. pass == 2 )

!  -----------------------------
!  methods that exploit symmetry
!  -----------------------------

          IF ( sym ) THEN
            stui = data%STU( i )

!  find nu new components of B given mu (s,y) data pairs

            nu = strip1 - stui
            IF ( nu == 0 ) CYCLE
!           mu = MIN( nu + control%extra_differences, m_available )
            mu = MIN( nu, m_available )

!write(6,*) ' m_available, mu, nu ', m_available, mu, nu

!  compute the unknown entries B_{ij}, j in I_i^-, to satisfy
!    sum_{j in I_i^-} B_{ij} s_{jl}  = y_{il} - sum_{j in I_i^+} B_{ij} s_{jl}
!  for l = 1,.., |I_i^+|, where
!    I_i^+ = { j : j \in I_i and B_{ji} is already known }
!    I_i^- = I_i \ I_i^+ and I_i = { j : B_{ij} /= 0}

!  compute the right-hand side y_{il} - sum_{j in I_i^+} B_{ij} s_{jl},
!  initialize b to Y(i,l)

            IF ( order_present ) THEN
              data%B( 1 : mu, 1 ) = Y( i, ORDER( 1 : mu ) )
            ELSE
              data%B( 1 : mu, 1 ) = Y( i, 1 : mu )
            END IF

!  loop over the known entries

            jj = 0
            DO k = stri, stui - 1
              kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )
              jj = jj + 1 ; data%COUNT( jj ) = j

!  subtract B_{ij} s_{jl} from b

              IF ( order_present ) THEN
                data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                      &
                  - VAL( kk ) * S( j, ORDER( 1 : mu ) )
              ELSE
                data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                      &
                  - VAL( kk ) * S( j, 1 : mu )
              END IF
            END DO
            IF ( control%out > 0 .AND. control%print_level > 1 )               &
              WRITE( control%out, "( ' known', 9( 1X, I0 ), /,                 &
             &  ( 1X, 10I6 ) )" ) data%COUNT( : jj )

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 0
            DO k = stui, strip1 - 1
              kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )
              jj = jj + 1 ; data%COUNT( jj ) = j

!  set the entries of A

              IF ( order_present ) THEN
                data%A( 1 : mu, jj ) = S( j, ORDER( 1 : mu ) )
              ELSE
                data%A( 1 : mu, jj ) = S( j, 1 : mu )
              END IF
            END DO

! make a copy of A and b as a precaution for possible use later

            IF ( mu + 1 <= m_max ) THEN
              data%A_save( 1 : mu, 1 : nu ) = data%A( 1 : mu, 1 : nu )
              data%B_save( 1 : mu, 1 ) = data%B( 1 : mu, 1 )
            END IF
            IF ( control%out > 0 .AND. control%print_level > 1 )               &
              WRITE( control%out, "( ' unknown', 9( 1X, I0 ), /,               &
             &    ( 1X, 10I6 ) )" ) data%COUNT( : jj )

!  solve A x = b

            IF ( mu == nu ) THEN
              dense_linear_solver = data%dense_linear_solver
            ELSE
              dense_linear_solver = MAX( data%dense_linear_solver, 3 )
            END IF
            CALL SHA_solve_system( dense_linear_solver, mu, nu,                &
                                   data%A, data%la1, data%B, data%lb1,         &
                                   data%solve_system_data, i,                  &
                                   control%out, control%print_level, info )

!  if A appears to be singular, add an extra row if there is one, and
!  solve the system as a least-squares problem

            IF ( info == MAX( nu, mu ) + 1 .AND. mu + 1 <= m_max ) THEN
              data%singular_matrices = data%singular_matrices + 1
write(6,*) ' singular pass, ii, i ', pass, ii, i

!  initialize b to Y(i,l)

              IF ( order_present ) THEN
                data%B( mu + 1, 1 ) = Y( i, ORDER( mu + 1 ) )
              ELSE
                data%B( mu + 1, 1 ) = Y( i, mu + 1 )
              END IF

!  loop over the known entries

              DO k = stri, stui - 1
                kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

                j = COL( kk )
                IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

                IF ( order_present ) THEN
                  data%B( mu + 1, 1 ) = data%B( mu + 1, 1 )                    &
                    - VAL( kk ) * S( j, ORDER( mu + 1 ) )
                ELSE
                  data%B( mu + 1, 1 ) = data%B( mu + 1, 1 )                    &
                    - VAL( kk ) * S( j, mu + 1 )
                END IF
              END DO

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

              jj = 0
              DO k = stui, strip1 - 1
                kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

                j = COL( kk )
                IF ( j == i ) j = ROW( kk )

!  set the entries of A

                jj = jj + 1
                IF ( order_present ) THEN
                  data%A( mu + 1, jj ) = S( j, ORDER( mu + 1 ) )
                ELSE
                  data%A( mu + 1, jj ) = S( j, mu + 1 )
                END IF
              END DO

!  solve A x = b

              IF ( mu + 1 == nu ) THEN
                dense_linear_solver = data%dense_linear_solver
              ELSE
                dense_linear_solver = MAX( data%dense_linear_solver, 3 )
              END IF
              CALL SHA_solve_system( dense_linear_solver, mu + 1, nu,          &
                                     data%A, data%la1, data%B, data%lb1,       &
                                     data%solve_system_data, i,                &
                                     control%out, control%print_level, info )

!  check for errors

            ELSE IF ( info /= 0 ) THEN
              inform%status = GALAHAD_error_factorization
              inform%bad_row = i ; GO TO 900
            END IF

!  finally, set the unknown B_{ij}

            jj = 1
            DO k = stui, strip1 - 1
              VAL( data%MAP( k ) ) = data%B( jj, 1 )
              jj = jj + 1
            END DO

!  if required, compute and print the residuals

            IF ( debug_residuals ) THEN

!  initialize b to Y(i,l)

              IF ( order_present ) THEN
                data%B( 1 : mu, 1 ) = Y( i, ORDER( 1 : mu ) )
              ELSE
                data%B( 1 : mu, 1 ) = Y( i, 1 : mu )
              END IF

!  loop over the known entries

              DO k = stri, strip1 - 1
                kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

                j = COL( kk )
                IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

                IF ( order_present ) THEN
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, ORDER( 1 : mu ) )
                ELSE
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, 1 : mu )
                END IF
              END DO
              write(6, "( ' max error is ', ES12.4, ' in row ', I0, 1X, I0 )" )&
                MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i, ii
            END IF

!  -----------------------------------
!  methods that don't exploit symmetry
!  -----------------------------------

          ELSE
!           mu = MIN( nu + control%extra_differences, m_available )
            mu = MIN( nu, m_available )

!  compute the unknown entries B_{ij}, j in I_i, to satisfy
!    sum_{j in I_i} B_{ij} s_{jl}  = y_{il}
!  for l = 1,.., |I_i|, where I_i = { j : B_{ij} /= 0}

!  store the right-hand side y_{il}, initialize b to Y(i,l)

            IF ( order_present ) THEN
              data%B( 1 : mu, 1 ) = Y( i, ORDER( 1 : mu ) )
            ELSE
              data%B( 1 : mu, 1 ) = Y( i, 1 : mu )
            END IF

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

            jj = 0
            DO k = stri, strip1 - 1
              kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

              j = COL( kk )
              IF ( j == i ) j = ROW( kk )

!  set the entries of A

              jj = jj + 1
              IF ( order_present ) THEN
                data%A( 1 : mu, jj ) = S( j, ORDER( 1 : mu ) )
              ELSE
                data%A( 1 : mu, jj ) = S( j, 1 : mu )
              END IF
            END DO

! make a copy of A and b as a precaution for possible use later

            IF ( mu + 1 <= m_max ) THEN
              data%A_save( 1 : mu, 1 : nu ) = data%A( 1 : mu, 1 : nu )
              data%B_save( 1 : mu, 1 ) = data%B( 1 : mu, 1 )
            END IF

!  solve A x = b

            IF ( control%out > 0 .AND. control%print_level > 1 )               &
              WRITE( control%out, "( ' vars ', 9I6, /, ( 10I6 ) )" )           &
                COL( data%MAP( stri : strip1 - 1 ) )
            IF ( mu == nu ) THEN
              dense_linear_solver = data%dense_linear_solver
            ELSE
              dense_linear_solver = MAX( data%dense_linear_solver, 3 )
            END IF
            CALL SHA_solve_system( dense_linear_solver, mu, nu, data%A,        &
                                   data%la1, data%B, data%lb1,                 &
                                   data%solve_system_data, i,                  &
                                   control%out, control%print_level, info )
!write(6,*) ' mu, nu, info ', mu, nu, info
!  if A appears to be singular, add an extra row if there is one, and
!  solve the system as a least-squares problem

            IF ( info == MAX( nu, mu ) + 1 .AND. mu + 1 <= m_max ) THEN
              data%singular_matrices = data%singular_matrices + 1
!write(6,"(' singular pass, ii, i ', 5(1X, I0))" ) pass, ii, i, stri, strip1
!  store the right-hand side y_{il}, initialize b to Y(i,l)

!write(6,*)  mu + 1, m_max, ORDER( mu + 1 )
              IF ( order_present ) THEN
                data%B( mu + 1, 1 ) = Y( i, ORDER( mu + 1 ) )
              ELSE
                data%B( mu + 1, 1 ) = Y( i, mu + 1 )
              END IF

!  now form the matrix A whose l,jjth entry is s_{jl}, j in I_i^-, where
!  B_{ij} is the jjth unknown

              jj = 0
              DO k = stri, strip1 - 1
                kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

                j = COL( kk )
                IF ( j == i ) j = ROW( kk )

!  set the entries of A

                jj = jj + 1
                IF ( order_present ) THEN
                  data%A( mu + 1, jj ) = S( j, ORDER( mu + 1 ) )
                ELSE
                  data%A( mu + 1, jj ) = S( j, mu + 1 )
                END IF
              END DO

!  solve A x = b

              IF ( mu + 1 == nu ) THEN
                dense_linear_solver = data%dense_linear_solver
              ELSE
                dense_linear_solver = MAX( data%dense_linear_solver, 3 )
              END IF
              CALL SHA_solve_system( dense_linear_solver, mu + 1, nu,          &
                                     data%A, data%la1, data%B, data%lb1,       &
                                     data%solve_system_data, i,                &
                                     control%out, control%print_level, info )

!  check for errors

            ELSE IF ( info /= 0 ) THEN
              inform%status = GALAHAD_error_factorization
              inform%bad_row = i ; GO TO 900
            END IF

!  finally, set the unknown B_{ij}

            jj = 0
            DO k = stri, strip1 - 1
              jj = jj + 1
              VAL( data%MAP( k ) ) = data%B( jj, 1 )
            END DO

!  if required, compute and print the residuals

            IF ( debug_residuals ) THEN

!  initialize b to Y(i,l)

              IF ( order_present ) THEN
                data%B( 1 : mu, 1 ) = Y( i, ORDER( 1 : mu ) )
              ELSE
                data%B( 1 : mu, 1 ) = Y( i, 1 : mu )
              END IF

!  loop over the known entries

              DO k = stri, strip1 - 1
                kk = data%MAP( k )

!  determine which of row( kk ) or col( kk ) gives the column number j

                j = COL( kk )
                IF ( j == i ) j = ROW( kk )

!  subtract B_{ij} s_{jl} from b

                IF ( order_present ) THEN
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, ORDER( 1 : mu ) )
                ELSE
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, 1 : mu )
                END IF
              END DO
              write(6, "( ' max error row is ', ES12.4, ' in row ', I0 )" )    &
                MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i
            END IF
          END IF
        END DO
      END DO

!  report how many block matrices were singular

      IF ( control%out > 0 .AND. control%print_level > 0 .AND.                 &
           data%singular_matrices > 0 ) WRITE( control%out, "( A, ' *** ', I0, &
     & ' block matrices were singular')" ) prefix, data%singular_matrices

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
      REAL ( KIND = rp_ ) :: b_norm, x_norm
      LOGICAL :: printi

      printi = out > 0 .AND. print_level > 1
!     b_norm = MAXVAL( ABS( B( : , 1 ) ) )

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

!       x_norm = MAXVAL( ABS( B( : , 1 ) ) )
!       WRITE( out, "( ' b, x, sigma_min, sigma_max =   ', 4ES11.4 )" )  &
!         b_norm, x_norm, data%S( MIN( m, n ) ), data%S( 1 )

        IF ( data%S( MIN( m, n ) ) / data%S( 1 ) <= eps_singular ) THEN
          status = MAX( m, n ) + 1
          IF ( printi ) THEN
            WRITE( out, "( ' matrix singular, sigma_min/sigma_1 = ',           &
           &       ES11.4 )" )  data%S( MIN( m, n ) ) / data%S( 1 )
            IF ( print_level > 2 ) THEN
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
      END IF
      IF ( .FALSE. ) THEN
!     IF ( printi ) THEN
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

      array_name = 'SHA: data%MAP'
      CALL SPACE_dealloc_array( data%MAP,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%MAP_lower'
      CALL SPACE_dealloc_array( data%MAP_lower,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%STU'
      CALL SPACE_dealloc_array( data%STU,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%STR'
      CALL SPACE_dealloc_array( data%STR,                                      &
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

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

      CALL SHA_terminate( data%sha_data, control, inform )

!  deallocate all remaining allocated arrays

      array_name = 'SHA: data%ROW'
      CALL SPACE_dealloc_array( data%ROW,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%COL'
      CALL SPACE_dealloc_array( data%COL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%ORDER'
      CALL SPACE_dealloc_array( data%ORDER,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

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

!-  G A L A H A D -  S H A _ r e s e t _ c o n t r o l   S U B R O U T I N E  -

     SUBROUTINE SHA_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SHA_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SHA_control_type ), INTENT( IN ) :: control
     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%sha_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SHA_reset_control

     END SUBROUTINE SHA_reset_control

!-  G A L A H A D -  S H A _ a n a l y s e _ m a t r i x _ S U B R O U T I N E -

     SUBROUTINE SHA_analyse_matrix( control, data, status, n,                  &
                                    matrix_ne, matrix_row, matrix_col, m )

!  import structural matrix data into internal storage, and analyse the
!  structure prior to recovery of the Hessian approximation

!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SHA_solve
!
!  data is a scalar variable of type SHA_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import and analysis. Possible values are:
!
!    0. The analysis was succesful, and the package is ready for the
!       factorization phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in statusrm.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0 or matrix_ne >= 0 has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   rows (and columns) of the matrix H
!
!  matrix_ne is a scalar variable of type default integer, that holds the
!   number of entries in the upper triangular part of H in the sparse
!   co-ordinate storage scheme
!
!  matrix_row is a rank-one array of type default integer, whose first 
!   matrix_ne entries holds the row indices of the upper triangular part 
!   of H in the sparse co-ordinate storage scheme
!
!  matrix_col is a rank-one array of type default integer, whose first 
!   matrix_ne entries holds the column indices of the upper triangular 
!   part of H in the sparse co-ordinate scheme
!
!  m is a scalar variable of type default integer, that gives the minimum 
!   number of (s^(k),y^(k)) pairs that will be needed to recover a good 
!   Hessian approximation
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SHA_control_type ), INTENT( INOUT ) :: control
     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, matrix_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, m
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: matrix_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: matrix_col

!  copy control to data

     data%SHA_control = control

!  store the structure of H

     data%n = n ; data%ne = matrix_ne

     CALL SPACE_resize_array( data%ne, data%ROW,                               &
            data%sha_inform%status, data%sha_inform%alloc_status )
     IF ( data%sha_inform%status /= 0 ) GO TO 900

     CALL SPACE_resize_array( data%ne, data%COL,                               &
            data%sha_inform%status, data%sha_inform%alloc_status )
     IF ( data%sha_inform%status /= 0 ) GO TO 900

     IF ( data%f_indexing ) THEN
       data%ROW( : data%ne ) = matrix_row( : data%ne )
       data%COL( : data%ne ) = matrix_col( : data%ne )
     ELSE
       data%ROW( : data%ne ) = matrix_row( : data%ne ) + 1
       data%COL( : data%ne ) = matrix_col( : data%ne ) + 1
     END IF

!  analyse the sparsity structure of the matrix prior to factorization

     CALL SHA_analyse( n, data%ne, data%ROW, data%COL,                         &
                       data%sha_data, data%sha_control, data%sha_inform )

     m = data%sha_inform%differences_needed
     status = data%sha_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%sha_inform%status
     RETURN

!  End of subroutine SHA_analyse_matrix

     END SUBROUTINE SHA_analyse_matrix

! G A L A H A D - S H A _ r e c o v e r _ m a t r i x  S U B R O U T I N E -

     SUBROUTINE SHA_recover_matrix( data, status, m, s, y, matrix_val,         &
                                    order )

!  recover the values of the Hessian matrix from m pairs of differences (s,y)

!  Arguments are as follows:

!  data is a scalar variable of type SHA_full_data_type used for internal data
!
!  status is a scalar variable of type default integer that indicates the
!   success or otherwise of the factorization. Possible values are:
!
!    0. The factorization was successful, and the package is ready for the
!       solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!
!  m is a scalar variable of type default integer, that specifies the 
!  number of (s,y) pairs that are available
!
!  s is a rank-two array of type default real, that holds the step vectors s^(k)
!   s_i^(k) should be stored in s(i,k)
!
!  y is a rank-two array of type default real, that holds the gradient 
!   differencestep vectors y^(k). y_i^(k) should be stored in y(i,k)
!
!  matrix_val is a rank-one array of type default real, that holds the
!   values of the upper triangular part of H input in precisely the same
!   order as those for the row and column indices in SHA_analyse_matrix
!
!  order is a rank-one array of type default integer, whose components
!   give the preferred order of access for the pairs (s^(k),y^(k)). 
!   The $k$-th component of order specifies the column number of s and y 
!   that will be used as the $k$-th most favoured.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SHA_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: m
     REAL ( KIND = rp_ ), DIMENSION( : , : ), INTENT( IN ) :: s, y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: matrix_val
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ),                     &
                                             OPTIONAL :: order

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

     INTEGER ( KIND = ip_ ) :: ls1, ls2, ly1, ly2
     LOGICAL :: allocate_order

!  record the lengths of each dimesion of s and y

     ls1 = SIZE( s, 1 ) ; ls2 = SIZE( s, 2 )
     ly1 = SIZE( y, 1 ) ; ly2 = SIZE( y, 2 )

!  store the preferred ordering for the (s,y) pairs

     IF ( PRESENT( order ) ) THEN
       IF ( ALLOCATED( data%order ) ) THEN
         IF ( SIZE( data%order ) < m ) THEN
           allocate_order = .TRUE.
         ELSE
           allocate_order = .FALSE.
         END IF
       ELSE
         allocate_order = .TRUE.
       END IF
       IF ( allocate_order ) CALL SPACE_resize_array( m, data%ORDER, &
              data%sha_inform%status, data%sha_inform%alloc_status )
       IF ( data%sha_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%ORDER( : m ) = order( : m )
       ELSE
         data%ORDER( : m ) = order( : m ) + 1
       END IF

!  factorize the matrix

       CALL SHA_estimate( data%n, data%ne, data%row, data%col, m,              &
                          S, ls1, ls2, Y, ly1, ly2, matrix_val,                &
                          data%sha_data, data%sha_control, data%sha_inform,    &
                          ORDER = data%ORDER )
     ELSE
       CALL SHA_estimate( data%n, data%ne, data%row, data%col, m,              &
                          S, ls1, ls2, Y, ly1, ly2, matrix_val,                &
                          data%sha_data, data%sha_control, data%sha_inform )
     END IF 

     status = data%sha_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%sha_inform%status
     RETURN

!  end of subroutine SHA_recover_matrix

     END SUBROUTINE SHA_recover_matrix

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
