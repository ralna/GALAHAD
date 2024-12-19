! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 14:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S H A   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

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
!$   USE omp_lib
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_LAPACK_inter_precision, ONLY : GETRF, GETRS, GELSS,           &
                                                GELSD, GELSY

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

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
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
!    for each iteration, >= 3 gives increasingly verbose (debug)

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  which approximation algorithm should be used?
!  (5-8 are retained for archival purposes, and not recommended)
!    1 : unsymmetric, parallel (alg 2.1 in paper)
!    2 : symmetric (alg 2.2 in paper)
!    3 : composite, parallel (alg 2.3 in paper)
!    4 : composite, block parallel (alg 2.4 in paper)
!    5 : original unsymmetric (alg 2.1 in old paper)
!    6 : original composite (alg 2.3 in old paper)
!    7 : original composite 2 (alg 2.4 in old paper)
!    8 : original cautious (alg 2.5 in old paper)

       INTEGER ( KIND = ip_ ) :: approximation_algorithm = 4

!  which dense linear equation solver should be used?
!    1 : Gaussian elimination
!    2 : QR factorization
!    3 : singular-value decomposition
!    4 : singular-value decomposition with divide-and-conquer

       INTEGER ( KIND = ip_ ) :: dense_linear_solver = 3

!  if available use an addition extra_differences differences

       INTEGER ( KIND = ip_ ) :: extra_differences = 1

!  rows with no more that sparse_row entries are considered sparse

       INTEGER ( KIND = ip_ ) :: sparse_row = 100

!  if a recursive algorithm is used (Alg 2.4), limit on the maximum number
!    of levels of recursion

       INTEGER ( KIND = ip_ ) :: recursion_max = 25

!  if a recursive algorithm is used (Alg 2.4), recursion can only occur for a
!   (reduced) row if it has at least %recursion_allowed entries

       INTEGER ( KIND = ip_ ) :: recursion_entries_required = 10

!  compute the average value of off diagonal entries rather than those
!  simply from the upper triangle (when necessary, algorithms 1, 3 & 4 only)

       LOGICAL :: average_off_diagonals = .FALSE.

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

!  the maximum difference between estimated off-diagonal pairs if
!  control%average_off_diagonals is .TRUE. and algorithm 1, 3 or 4 employed

       REAL ( KIND = rp_ ) :: max_off_diagonal_difference = zero

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

       INTEGER ( KIND = ip_ ) :: n, nz, nb, unsym_rows, nnz_unsym
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
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COL_unsym
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PTR_unsym
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP_lower
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INFO
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: SPARSE_INDS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: DENSE_INDS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: TMP_INDS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: NSOLVED
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: VAL_unsym
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
!  extra-differences                               1
!  maximum-degree-considered-sparse                100
!  maximum-recursion-levels                        25
!  recursion-entries-required                      10
!  average-off-diagonals                           F
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
     INTEGER ( KIND = ip_ ), PARAMETER :: extra_differences                    &
                                            = dense_linear_solver + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: sparse_row                           &
                                            = extra_differences + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: recursion_max = sparse_row + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: recursion_entries_required           &
                                            = recursion_max + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: average_off_diagonals                &
                                            = recursion_entries_required + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                       &
                                            = average_off_diagonals + 1
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
     spec( extra_differences )%keyword = 'extra-differences'
     spec( sparse_row )%keyword = 'maximum-degree-considered-sparse'
     spec( recursion_max )%keyword = 'maximum-recursion-levels'
     spec( recursion_entries_required )%keyword = 'recursion-entries-required'

!  Logical key-words

     spec( average_off_diagonals )%keyword = 'average-off-diagonals'
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
     CALL SPECFILE_assign_value( spec( extra_differences ),                    &
                                 control%extra_differences,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sparse_row ),                           &
                                 control%sparse_row,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( recursion_max ),                        &
                                 control%recursion_max,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec(recursion_entries_required ),            &
                                 control%recursion_entries_required,           &
                                 control%error )

!  Set logical values


     CALL SPECFILE_assign_value( spec( average_off_diagonals ),                &
                                 control%average_off_diagonals,                &
                                 control%error )
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

!  make sure that the approximation algorithm requested is possible

      IF ( control%approximation_algorithm < 1 .OR.                            &
           control%approximation_algorithm > 8 ) THEN
        inform%approximation_algorithm_used = 4
      ELSE
        inform%approximation_algorithm_used = control%approximation_algorithm
      END IF
      IF ( printi ) WRITE( control%out, "( ' Algorithm ', I0 )" )              &
          inform%approximation_algorithm_used

!  branch depending on the algorithm specified

      SELECT CASE ( inform%approximation_algorithm_used )

!  ------------------------------------------------
!  algorithms 1, 3 and 4 (aka paper 2.1, 2.3 & 2.4)
!  ------------------------------------------------

      CASE ( 1, 3, 4 )

!  make space for pointers to the start of each row in the complete
!  (i.e., both triangles together) matrix

        array_name = 'SHA: data%PTR_unsym'
        CALL SPACE_resize_array( n + 1, data%PTR_unsym,                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  count the number of nonzeros in each row of the complete matrix

        data%PTR_unsym( 2 : n + 1 ) = 0
        DO l = 1, nz
          i = ROW( l ) + 1 ; j = COL( l ) + 1
          data%PTR_unsym( i ) = data%PTR_unsym( i ) + 1
          IF ( i /= j ) THEN
            data%PTR_unsym( j ) = data%PTR_unsym( j ) + 1
          END IF
        END DO

!  compute the starting position for each row of the complete matrix

        data%PTR_unsym( 1 ) = 1
        DO i = 1, n
          data%PTR_unsym( i + 1 )                                              &
            = data%PTR_unsym( i ) + data%PTR_unsym( i + 1 )
        END DO

!  allocate space for the column indices and values of the complete matrix,
!  as well as a map from the lower triangle to the complete matrix

        data%nnz_unsym = data%PTR_unsym( n + 1 ) - 1
        array_name = 'SHA: data%COL_unsym'
        CALL SPACE_resize_array( data%nnz_unsym, data%COL_unsym,               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%VAL_unsym'
        CALL SPACE_resize_array( data%nnz_unsym, data%VAL_unsym,               &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%MAP'
        CALL SPACE_resize_array( nz, data%MAP,                                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  now fill the column indices into each row of the complete matrix. If
!  averages of off-diagonal estimated values are required, record the
!  positions of both upper and lower entries (in MAP and MAP_lower)

        IF ( control%average_off_diagonals ) THEN
          array_name = 'SHA: data%MAP_lower'
          CALL SPACE_resize_array( nz, data%MAP_lower,                         &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO l = 1, nz
            i = ROW( l ) ; j = COL( l )
            data%MAP( l ) = data%PTR_unsym( i )
            data%COL_unsym( data%PTR_unsym( i ) ) = j
            IF ( i /= j ) THEN
              data%MAP_lower( l ) = data%PTR_unsym( j )
              data%COL_unsym( data%PTR_unsym( j ) ) = i
              data%PTR_unsym( j ) = data%PTR_unsym( j ) + 1
            ELSE
              data%MAP_lower( l ) = data%PTR_unsym( i )
            END IF
!write(6,*) data%MAP( l ), data%MAP_lower( l )
            data%PTR_unsym( i ) = data%PTR_unsym( i ) + 1
          END DO
        ELSE
          DO l = 1, nz
            i = ROW( l ) ; j = COL( l )
            data%MAP( l ) = data%PTR_unsym( i )
            data%COL_unsym( data%PTR_unsym( i ) ) = j
            data%PTR_unsym( i ) = data%PTR_unsym( i ) + 1
            IF ( i /= j ) THEN
              data%COL_unsym( data%PTR_unsym( j ) ) = i
              data%PTR_unsym( j ) = data%PTR_unsym( j ) + 1
            END IF
          END DO
        END IF

!  restore the row starting positions

        DO i = n, 1, - 1
          data%PTR_unsym( i + 1 ) = data%PTR_unsym( i )
        END DO
        data%PTR_unsym( 1 ) = 1

!  set up workspace

        CALL SPACE_resize_array( n, data%INFO,                                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        IF ( inform%approximation_algorithm_used >= 3 ) THEN
          CALL SPACE_resize_array( n, data%SPARSE_INDS,                        &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          CALL SPACE_resize_array( n, data%DENSE_INDS,                         &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

        IF ( inform%approximation_algorithm_used == 4 ) THEN
          CALL SPACE_resize_array( n, data%TMP_INDS,                           &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          CALL SPACE_resize_array( n, data%NSOLVED,                            &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

        SELECT CASE ( inform%approximation_algorithm_used )
        CASE ( 1 )
          inform%differences_needed                                            &
            = SHA_differences_needed_2_1( n, data%PTR_unsym )
        CASE ( 3 : 4 )
          inform%differences_needed                                            &
            = SHA_differences_needed_2_3( n, data%nnz_unsym,                   &
                                          data%PTR_unsym, data%COL_unsym,      &
                                          data%SPARSE_INDS, data%DENSE_INDS,   &
                                          control%sparse_row )
!       CASE ( 4 )
        END SELECT


!  ---------------------------------------------------------
!  algorithm 2, 5-8 (aka paper 2.2, old paper 2.1 & 2.3-2.5)
!  ---------------------------------------------------------

      CASE ( 2, 5 : 8 )

!  allocate space for row starting addresses STR and unsymmetric row counts
!  COUNT

        array_name = 'SHA: data%STR'
        CALL SPACE_resize_array( n + 1, data%STR,                              &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%COUNT'
        CALL SPACE_resize_array( n, data%COUNT,                                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
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
        CALL SPACE_resize_array( n, data%LIST,                                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%FIRST'
        CALL SPACE_resize_array( 0_ip_, max_row, data%FIRST,                   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%LAST'
        CALL SPACE_resize_array( 0_ip_,                                        &
               MAX( max_row, control%sparse_row), data%LAST,                   &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%ROWS'
        CALL SPACE_resize_array( n, data%ROWS,                                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize the number of rows with each degree in LAST

        data%LAST( 0 : max_row ) = 0
        DO i = 1, n
          data%LAST( data%COUNT( i ) ) = data%LAST( data%COUNT( i ) ) + 1
        END DO

        IF ( printi ) THEN
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
            IF ( data%FIRST( i ) <= data%LAST( i ) )                           &
              WRITE( control%out, "( 10( :, 1X, I0 ) )" )                      &
               ( data%LIST( l ), l = data%FIRST( i ), data%LAST( i ) )
          END DO
          WRITE( control%out, "( ' ROWS:', /, 10( 1X, I0 ) )" )                &
            data%ROWS( 1 : n )
        END IF

!  allocate space for MAP to hold mappings from the rows back to the
!  coordinate storage, and its "shadow" MAP_lower set so that entries k and
!  MAP_lower(k) of MAP correspond to the "upper" and "lower" triangular entries
!  (i,j) and (j,i), i <= j

        array_name = 'SHA: data%MAP'
        CALL SPACE_resize_array( data%STR( n + 1 ) - 1, data%MAP,              &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%MAP_lower'
        CALL SPACE_resize_array( data%STR( n + 1 ) - 1, data%MAP_lower,        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
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
            WRITE( control%out, "( 1X, 6( '(', I0, ',', I0, ')', : ) )" )      &
              ( ROW( data%MAP( l ) ), COL( data%MAP( l ) ), l = data%STR( i ), &
                data%STR( i + 1 ) - 1 )
          END DO
        END IF

        IF ( control%out > 0 .AND. control%print_level > 2 ) THEN
          WRITE( control%out, "( ' matrix:' )" )
          DO l = 1, data%STR( n + 1 ) - 1
            WRITE( control%out, "( 1X, I0, 2( ' (', I0, ',', I0, ')' ) )" ) l, &
              ROW( data%MAP( l ) ), COL( data%MAP( l ) ),                      &
              COL( data%MAP( data%MAP_lower( l ) ) ),                          &
              ROW( data%MAP( data%MAP_lower( l ) ) )
          END DO
        END IF

!  allocate further workspace to record row (inverse) permutations (PERM_inv)
!  and the starting addresses for undetermined entries in each row (STU)

        array_name = 'SHA: data%PERM_inv'
        CALL SPACE_resize_array( n, data%PERM_inv,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'SHA: data%STU'
        CALL SPACE_resize_array( n, data%STU,                                  &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  initialize undetermined row entry pointers

        data%STU( 1 : n ) = data%STR( 1 : n )

        SELECT CASE ( inform%approximation_algorithm_used )

!  --------------------------------------------------------------
!  algorithms 2, 5 and 6 (aka paper 2.2 and old paper 2.1 &  2.3)
!  --------------------------------------------------------------

        CASE ( 2, 5 : 6 )

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
          END DO

          IF ( inform%approximation_algorithm_used == 2 ) THEN
            data%differences_needed                                            &
              = MAXVAL( data%STR( 2 : n + 1 ) -  data%STU( 1 : n ) )
          ELSE IF ( inform%approximation_algorithm_used == 6 ) THEN
            data%differences_needed = 0
            DO i = 1, n
              IF ( data%STR( i + 1 ) - data%STR( i ) <=                        &
                   control%sparse_row ) THEN
                data%differences_needed = MAX( data%differences_needed,        &
                                             data%STR( i + 1 ) - data%STR( i ) )
              ELSE
                data%differences_needed = MAX( data%differences_needed,        &
                                             data%STR( i + 1 ) - data%STU( i ) )
              END IF
            END DO
          ELSE
            data%differences_needed                                            &
              = MAXVAL( data%STR( 2 : n + 1 ) -  data%STR( 1 : n ) )
          END IF

!  report the numbers of each block size

          data%LAST( 0 : data%differences_needed ) = 0
          DO i = 1, n
            l = data%STR( i + 1 ) - data%STU( i )
            data%LAST( l ) = data%LAST( l ) + 1
          END DO
          IF ( printi ) THEN
            WRITE( control%out, "( A, ' (block size, # with this size):' )" )  &
              prefix
            CALL SHA_write_nonzero_list( control%out, data%differences_needed, &
                                         data%LAST )
          END IF

!  -------------------------------
!  algorithm 7 (aka old paper 2.4)
!  -------------------------------

        CASE ( 7 )
          data%unsym_rows = 0
          DO i = 1, n

!  skip rows that have more than sparse_row entries

            IF ( data%STR( i + 1 ) - data%STR( i ) >                           &
                 control%sparse_row ) CYCLE
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
              data%differences_needed =                                        &
                MAX( data%differences_needed, data%STR( i + 1 )-data%STU( i ) )
            ELSE
              inform%max_reduced_degree =                                      &
               MAX( inform%max_reduced_degree, data%STR( i + 1 )-data%STU( i ) )
              j = j + 1
              data%PERM_inv( j ) = i
            END IF
          END DO

          data%differences_needed                                              &
            = MAX( data%differences_needed, inform%max_reduced_degree )

          IF ( control%out > 0 .AND. control%print_level > 1 ) THEN
!           WRITE(  control%out, "( ' maximum degree in the connectivity',     &
!        &   ' graph = ', I0, /, 1X, I0, ' symmetric differences required ',   &
!        &   /, ' max reduced degree = ', I0 )" ) inform%max_degree,           &
!           data%differences_needed, inform%max_reduced_degree
          END IF

!  -------------------------------
!  algorithm 8 (aka old paper 2.5)
!  -------------------------------

        CASE ( 8 )

          data%differences_needed = 0
          data%l_sparse = data%LAST( MIN( max_row, control%sparse_row ) )
          data%LAST( 0 : max_row ) = 0

!  loop over the rows by increasing counts

          DO l = 1, data%l_sparse
            i = data%LIST( l )
            data%PERM_inv( l ) = i
            ll = data%STR( i + 1 ) - data%STR( i )
            data%differences_needed = MAX( data%differences_needed, ll )
            IF ( printi ) data%LAST( ll ) = data%LAST( ll ) + 1

!  loop over the entries in the chosen row

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

!  report the numbers of each block size

          IF ( printi ) THEN
            WRITE( control%out, "( A, ' (block size, # with this size):' )" )  &
              prefix
            CALL SHA_write_nonzero_list( control%out, data%differences_needed, &
                                         data%LAST )
          END IF
        END SELECT
        inform%differences_needed = data%differences_needed
      END SELECT

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
      INTEGER ( KIND = ip_ ) :: dense_linear_solver, warning
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

!  branch depending on the algorithm specified

      SELECT CASE ( inform%approximation_algorithm_used )

!  ------------------------------------------------
!  algorithms 1, 3 and 4 (aka paper 2.1, 2.3 & 2.4)
!  ------------------------------------------------

      CASE ( 1, 3, 4 )

        SELECT CASE( inform%approximation_algorithm_used )

!  ------------------------------------------------
!  algorithms 1 (aka paper 2.1)
!  ------------------------------------------------

        CASE ( 1 )
          CALL SHA_estimate_2_1( n, data%nnz_unsym, data%PTR_unsym,            &
                                 data%COL_unsym, m_available,                  &
                                 S, ls1, ls2, Y, ly1, ly2,                     &
                                 data%VAL_unsym, control%extra_differences,    &
                                 inform%status, data%INFO, ORDER )

!  ------------------------------------------------
!  algorithms 3 (aka paper 2.3)
!  ------------------------------------------------

        CASE ( 3 )
          CALL SHA_estimate_2_3( n, data%nnz_unsym, data%PTR_unsym,            &
                                 data%COL_unsym, m_available,                  &
                                 S, ls1, ls2, Y, ly1, ly2,                     &
                                 data%VAL_unsym, control%extra_differences,    &
                                 control%sparse_row,                           &
                                 inform%status, data%SPARSE_INDS,              &
                                 data%DENSE_INDS, data%INFO, ORDER )

!  ------------------------------------------------
!  algorithms 4 (aka paper 2.4)
!  ------------------------------------------------

        CASE ( 4 )
          CALL SHA_estimate_2_4( n, data%nnz_unsym, data%PTR_unsym,            &
                                 data%COL_unsym, m_available,                  &
                                 S, ls1, ls2, Y, ly1, ly2,                     &
                                 data%VAL_unsym, control%extra_differences,    &
                                 control%recursion_max,                        &
                                 control%recursion_entries_required,           &
                                 inform%status,                                &
                                 data%SPARSE_INDS, data%DENSE_INDS,            &
                                 data%TMP_INDS, data%NSOLVED,                  &
                                 data%INFO, ORDER )

        END SELECT
        IF ( inform%status < GALAHAD_ok ) THEN
          IF ( inform%status == GALAHAD_error_allocate .OR.                    &
               inform%status == GALAHAD_error_deallocate ) THEN
            inform%alloc_status = - 1
            inform%bad_alloc = 'workspace array from SHA_estimate subroutine'
          END IF
          GO TO 900
        END IF

!  copy the entries from the complete matrix back to its lower triangle

        IF ( control%average_off_diagonals ) THEN
          inform%max_off_diagonal_difference = zero
          DO i = 1, nz
            inform%max_off_diagonal_difference =                               &
              MAX( inform%max_off_diagonal_difference,                         &
                   ABS( data%VAL_unsym( data%MAP( i ) ) -                      &
                        data%VAL_unsym( data%MAP_lower( i ) ) ) )
            VAL( i ) = half * ( data%VAL_unsym( data%MAP( i ) ) +              &
                                data%VAL_unsym( data%MAP_lower( i ) ) )
          END DO
        ELSE
          DO i = 1, nz
            VAL( i ) = data%VAL_unsym( data%MAP( i ) )
          END DO
        END IF

!  ---------------------------------------------------------
!  algorithm 2, 5-8 (aka paper 2.2, old paper 2.1 & 2.3-2.5)
!  ---------------------------------------------------------

      CASE ( 2, 5 : 8 )

!  recall the number of differences needed to reproduce a fixed Hessian

        m_needed = data%differences_needed

!  warn if there is insufficient data

        IF ( m_needed > m_available ) THEN
!         WRITE( *, * ) ( ' Warning: insufficient data pairs are available')
          warning = GALAHAD_warning_data
        ELSE
          warning = GALAHAD_ok
        END IF

! add %extra_differences to accommodate a singularity precaution if possible

!       m_max = MIN( m_needed + control%extra_differences, m_available )
        m_max = MIN( m_needed + MAX( control%extra_differences, 1 ),           &
                     m_available )
        n_max = m_needed
        min_mn = MIN( m_max, n_max )

        data%singular_matrices = 0
        order_present = PRESENT( order )

!  allocate workspace

!  generic solver workspace

        IF ( data%la1 < m_max .OR. data%la2 < n_max ) THEN
          data%la1 = m_max ; data%la2 = n_max
!write(6,*) ' la1, la2 ', data%la1, data%la2
          array_name = 'SHA: data%A'
          CALL SPACE_resize_array( data%la1, data%la2, data%A,                 &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

        IF ( data%lb1 < m_max ) THEN
          data%lb1 = MAX( m_max, n_max )
!write(6,*) ' lb1 ', data%lb1
          array_name = 'SHA: data%B'
          CALL SPACE_resize_array( data%lb1, 1_ip_, data%B,                    &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  solver-specific workspace

        IF ( data%dense_linear_solver /= control%dense_linear_solver ) THEN
          data%dense_linear_solver = control%dense_linear_solver
          IF ( data%dense_linear_solver < 1 .OR.                               &
               data%dense_linear_solver > 4 ) data%dense_linear_solver = 3

!  allocate space to hold a copy of A if needed

          IF ( data%la_save1 < m_max .OR. data%la_save2 < n_max ) THEN
            data%la_save1 = m_max ; data%la_save2 = n_max
            array_name = 'SHA: data%A_save'
            CALL SPACE_resize_array( data%la_save1, data%la_save2,             &
               data%A_save, inform%status, inform%alloc_status,                &
               array_name = array_name,                                        &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  allocate space to hold a copy of b if needed

          IF ( data%lb_save < m_needed ) THEN
            data%lb_save = m_max
            array_name = 'SHA: data%B_save'
            CALL SPACE_resize_array( data%lb_save, 1_ip_,                      &
                   data%B_save, inform%status, inform%alloc_status,            &
                   array_name = array_name,                                    &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  discover how much additional temporary real storage may be needed by LU / LQ

          IF ( data%dense_linear_solver == 1 ) THEN
            liwork = min_mn
          ELSE IF ( data%dense_linear_solver == 2 ) THEN
            m_used = m_needed
            CALL GELSY( m_used, n, 1_ip_, data%A, data%la1, data%B, data%lb1,  &
                        data%solve_system_data%IWORK, eps_singular, rank,      &
                        data%WORK_1, -1_ip_, status )
            lwork = INT( data%WORK_1( 1 ) ) ; liwork = n_max

!  allocate space to hold the singular values if needed

          ELSE
            IF ( data%ls < min_mn ) THEN
!             data%ls = min_mn
              data%ls = min_mn + 1
              array_name = 'SHA: data%solve_syetem_data%S'
              CALL SPACE_resize_array( data%ls, data%solve_system_data%S,      &
                     inform%status, inform%alloc_status,                       &
                     array_name = array_name,                                  &
                     deallocate_error_fatal = control%deallocate_error_fatal,  &
                     exact_size = control%space_critical,                      &
                     bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= GALAHAD_ok ) GO TO 900
            END IF

!  discover how much temporary integer and real storage may be needed by SVD

!           m_used = m_needed
            m_used = m_max
            IF ( data%dense_linear_solver == 4 ) THEN
              CALL GELSD( m_used, n_max, 1_ip_, data%A, data%la1, data%B,      &
                          data%lb1, data%solve_system_data%S, eps_singular,    &
                          rank, data%WORK_1, - 1_ip_, data%IWORK_1, status )
              lwork = INT( data%WORK_1( 1 ) )
              liwork = INT( data%IWORK_1( 1 ) )
            ELSE
              CALL GELSS( m_used, n_max, 1_ip_, data%A, data%la1, data%B,      &
                          data%lb1, data%solve_system_data%S, eps_singular,    &
                          rank, data%WORK_1, - 1_ip_, status )
              lwork = INT( data%WORK_1( 1 ) ) ; liwork = n_max
            END IF
          END IF

!  allocate temporary integer storage

          IF ( data%dense_linear_solver /= 3 ) THEN
            IF ( data%solve_system_data%liwork < liwork ) THEN
              data%solve_system_data%liwork = liwork
              array_name = 'SHA: data%solve_system_data%IWORK'
              CALL SPACE_resize_array( data%solve_system_data%liwork,          &
                     data%solve_system_data%IWORK, inform%status,              &
                     inform%alloc_status, array_name = array_name,             &
                     deallocate_error_fatal = control%deallocate_error_fatal,  &
                     exact_size = control%space_critical,                      &
                     bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= GALAHAD_ok ) GO TO 900
            END IF
          END IF

!  allocate temporary real storage

          IF ( data%dense_linear_solver /= 1 ) THEN
            IF ( data%solve_system_data%lwork < lwork ) THEN
              data%solve_system_data%lwork = lwork
              array_name = 'SHA: data%solve_system_data%WORK'
              CALL SPACE_resize_array( data%solve_system_data%lwork,           &
                     data%solve_system_data%WORK, inform%status,               &
                     inform%alloc_status, array_name = array_name,             &
                     deallocate_error_fatal = control%deallocate_error_fatal,  &
                     exact_size = control%space_critical,                      &
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
            IF ( inform%approximation_algorithm_used == 5 ) THEN
              ii_start = n ; ii_end = 1 ; ii_stride = - 1
            ELSE IF ( inform%approximation_algorithm_used == 8 ) THEN
              ii_start = data%l_sparse ; ii_end = 1 ; ii_stride = - 1
            ELSE
              ii_start = 1 ; ii_end = n ; ii_stride = 1
            END IF
          ELSE
            IF ( inform%approximation_algorithm_used < 8 ) EXIT
            ii_start = n ; ii_end = data%l_sparse + 1 ; ii_stride = - 1
          END IF
          DO ii = ii_start, ii_end, ii_stride
            i = data%PERM_inv( ii )
            stri = data%STR( i ) ; strip1 = data%STR( i + 1 )

!  there are nu unknowns for this pass

            nu = strip1 - stri
            IF ( nu == 0 ) CYCLE

!  decide whether to exploit symmetry or not

            sym = inform%approximation_algorithm_used == 2 .OR.                &
                ( inform%approximation_algorithm_used == 6 .AND.               &
                  nu > control%sparse_row ) .OR.                               &
                  inform%approximation_algorithm_used == 7 .OR.                &
                ( inform%approximation_algorithm_used == 8 .AND. pass == 2 )
!                 mu > m_needed ) .OR.                                         &
!                 nu > m_available ) .OR.                                      &

!  -----------------------------
!  methods that exploit symmetry
!  -----------------------------

            IF ( sym ) THEN
              stui = data%STU( i )

!  find nu new components of B given mu (s,y) data pairs

              nu = strip1 - stui
              IF ( nu == 0 ) CYCLE

!  acknowledge that there may not be sufficient data to find all nu components,
!  and reset nu to the largest possible (an overdetermined least-squares
!  problem will be solved instead)

!             mu = MIN( nu + control%extra_differences, m_available )
              mu = MIN( nu, m_available )
!if ( mu > data%la1 ) write(6,*) mu, data%la1

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
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, ORDER( 1 : mu ) )
                ELSE
                  data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                    &
                    - VAL( kk ) * S( j, 1 : mu )
                END IF
              END DO
              IF ( control%out > 0 .AND. control%print_level > 1 )             &
                WRITE( control%out, "( ' known', 9( 1X, I0 ), /,               &
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
              IF ( control%out > 0 .AND. control%print_level > 1 )             &
                WRITE( control%out, "( ' unknown', 9( 1X, I0 ), /,             &
               &    ( 1X, 10I6 ) )" ) data%COUNT( : jj )

!  solve A x = b

              IF ( mu == nu ) THEN
                dense_linear_solver = data%dense_linear_solver
              ELSE
                dense_linear_solver = MAX( data%dense_linear_solver, 3 )
              END IF
              CALL SHA_solve_system( dense_linear_solver, mu, nu,              &
                                     data%A, data%la1, data%B, data%lb1,       &
                                     data%solve_system_data, i,                &
                                     control%out, control%print_level, info )

!  if A appears to be singular, add an extra row if there is one, and
!  solve the system as a least-squares problem

              IF ( info == MAX( nu, mu ) + 1 .AND. mu + 1 <= m_max ) THEN
                data%singular_matrices = data%singular_matrices + 1

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
                    data%B( mu + 1, 1 ) = data%B( mu + 1, 1 )                  &
                      - VAL( kk ) * S( j, ORDER( mu + 1 ) )
                  ELSE
                    data%B( mu + 1, 1 ) = data%B( mu + 1, 1 )                  &
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
                CALL SHA_solve_system( dense_linear_solver, mu + 1, nu,        &
                                       data%A, data%la1, data%B, data%lb1,     &
                                       data%solve_system_data, i,              &
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
                    data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                  &
                      - VAL( kk ) * S( j, ORDER( 1 : mu ) )
                  ELSE
                    data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                  &
                      - VAL( kk ) * S( j, 1 : mu )
                  END IF
                END DO
              WRITE( 6, "( ' max error is ', ES12.4, ' in row ', I0, 1X, I0 )")&
                MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i, ii
            END IF

!  -----------------------------------
!  methods that don't exploit symmetry
!  -----------------------------------

            ELSE
!             mu = MIN( nu + control%extra_differences, m_available )
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

              IF ( control%out > 0 .AND. control%print_level > 1 )             &
                WRITE( control%out, "( ' vars ', 9I6, /, ( 10I6 ) )" )         &
                  COL( data%MAP( stri : strip1 - 1 ) )
              IF ( mu == nu ) THEN
                dense_linear_solver = data%dense_linear_solver
              ELSE
                dense_linear_solver = MAX( data%dense_linear_solver, 3 )
              END IF
              CALL SHA_solve_system( dense_linear_solver, mu, nu, data%A,      &
                                     data%la1, data%B, data%lb1,               &
                                     data%solve_system_data, i,                &
                                     control%out, control%print_level, info )

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
                CALL SHA_solve_system( dense_linear_solver, mu + 1, nu,        &
                                       data%A, data%la1, data%B, data%lb1,     &
                                       data%solve_system_data, i,              &
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
                    data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                  &
                      - VAL( kk ) * S( j, ORDER( 1 : mu ) )
                  ELSE
                    data%B( 1 : mu, 1 ) = data%B( 1 : mu, 1 )                  &
                      - VAL( kk ) * S( j, 1 : mu )
                  END IF
                END DO
                write(6, "( ' max error row is ', ES12.4, ' in row ', I0 )" )  &
                  MAXVAL( ABS( data%B( 1 : mu, 1 ) ) ), i
              END IF
            END IF
          END DO
        END DO

!  report how many block matrices were singular

        IF ( control%out > 0 .AND. control%print_level > 0 .AND.               &
           data%singular_matrices > 0 ) WRITE( control%out, "( A, ' *** ', I0, &
       & ' block matrices were singular')" ) prefix, data%singular_matrices
        IF ( inform%status == GALAHAD_ok ) inform%status = warning
      END SELECT

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
!     REAL ( KIND = rp_ ) :: b_norm, x_norm
      LOGICAL :: printi

      printi = out > 0 .AND. print_level > 1
!     b_norm = MAXVAL( ABS( B( : , 1 ) ) )

!  solve A x = b using Gaussian elimination; A is copied to A_save as a
!  precaution

      IF ( dense_linear_solver == 1 ) THEN
        IF ( m == n ) THEN
          CALL GETRF( m, n, A, la1, data%IWORK, status )
          IF ( status == 0 ) THEN
            CALL GETRS( 'N', n, 1_ip_, A, la1, data%IWORK, B, lb1, status )
            IF ( status == 0 ) RETURN
          END IF
        END IF

!  solve A x = b using a QR factorization; A is copied to A_save as a precaution

      ELSE IF ( dense_linear_solver == 2 ) THEN
        CALL GELSY( m, n, 1_ip_, A, la1, B, lb1, data%IWORK, eps_singular,     &
                    rank, data%WORK, data%lwork, status )
        IF ( status == 0 ) RETURN

!  solve A x = b using a singular-value decomposition

      ELSE
        IF ( printi ) A_save( : m, : n ) = A( : m, : n )
        IF ( dense_linear_solver == 4 ) THEN
          CALL GELSD( m, n, 1_ip_, A, la1, B, lb1, data%S, eps_singular, rank, &
                      data%WORK, data%lwork, data%IWORK, status )
        ELSE ! dense_linear_solver == 3
          CALL GELSS( m, n, 1_ip_, A, la1, B, lb1, data%S, eps_singular, rank, &
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

!-*-  G A L A H A D -  S H A _ differences _ needed _ 2 _ 1  F U C T I O N  -*-

    FUNCTION SHA_differences_needed_2_1( n, PTR )
    INTEGER ( KIND = ip_ ) :: SHA_differences_needed_2_1

!****************************************************************************
!
!  Algorithm 2.1: Find the number of differences needed for the Sparse Hessian
!    approximation (parallel unsymmetric variant)
!
!  Arguments:
!  n - number of variables
!  PTR - row pointer indices of the COMPLETE sparse Hessian (CSR format)
!
!****************************************************************************

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR

!  local variables

    INTEGER ( KIND = ip_ ) :: i, differences_needed

    differences_needed = 0
    DO i = 1, n
      differences_needed = MAX( differences_needed, PTR( i + 1 ) - PTR( i ) )
    END DO
    SHA_differences_needed_2_1 = differences_needed
    RETURN

!  end of FUNCTION SHA_differences_needed_2_1

    END FUNCTION SHA_differences_needed_2_1

!-*-  G A L A H A D -  S H A _ e s t i m a t e _ 2 _ 1  S U B R O U T I N E  -*-

    SUBROUTINE SHA_estimate_2_1( n, ne, PTR, COL, m_available,                 &
                                 S, ls1, ls2, Y, ly1, ly2, VAL,                &
                                 extra_differences, status, INFO, ORDER )

!****************************************************************************
!
!  Algorithm 2.1: Sparse Hessian approximation (parallel unsymmetric variant)
!
!  Arguments:
!  n - number of variables
!  ne - number of nonzero elements in the COMPLETE sparse Hessian matrix
!  PTR - row pointer indices of the COMPLETE sparse Hessian (CSR format)
!  COL - column indices of the COMPLETE sparse Hessian (CSR format)
!  m_available - number of data pairs k available for Hessian estimation
!  S - n x m_available array of optimization steps
!  Y - n x m_available array of optimization gradient differences
!  VAL - resulting estimated entries of the COMPLETE sparse Hessian (CSR format)
!  extra_differences - # extra data values allowed when solving linear systems
!  status - return status, = 0 for success, < 0 failure
!  INFO - workspace array of length n
!  ORDER (optional) - ORDER(i), i=1:m_available gives the index of the
!    column of S and Y of the i-th most recent steps/differences.
!    If absent, the index  will be i, i=1:m_available
!
!  Written by: Jaroslav Fowkes
!
!****************************************************************************

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne, m_available
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ne ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: INFO

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: i, idx

!  solve for each (sparse) row, determine unknowns and solve linear system

!$omp parallel do private(i)
    DO idx = 1, n
      i = idx
      CALL SHA_solve_unknown_2_1( i, PTR, COL, m_available,                    &
                                  S, ls1, ls2, Y, ly1, ly2, VAL,               &
                                  extra_differences, INFO( i ), ORDER )
    END DO
!$omp end parallel do

    status = MINVAL( INFO )
    IF ( status == 0 ) status = MAXVAL( INFO )
    RETURN

!  end of SUBROUTINE SHA_estimate_2_1

    END SUBROUTINE SHA_estimate_2_1

!-*-  G A L A H A D -  S H A _ s o l v e _ unknown_2_1  S U B R O U T I N E  -*-

    SUBROUTINE SHA_solve_unknown_2_1( i, PTR, COL, m_available,                &
                                      S, ls1, ls2, Y, ly1, ly2, VAL,           &
                                      extra_differences, info, ORDER )

!  "solve" the system whose j-th equation is
!     sum_{unknown i} s_ji val_i = y_j

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: i, m_available, ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: PTR, COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: k, kk, l, nei, m, rank, ptrs, lwork, warning
    INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK1
    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK1
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: IWORK
    REAL ( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: SV, WORK
    REAL ( KIND = rp_ ), DIMENSION( :, : ), ALLOCATABLE :: C, A

!  nonzeros in row

    nei = PTR( i + 1 ) - PTR( i )

!  handle case of null row

    IF ( nei == 0 ) THEN
      info = GALAHAD_ok ; RETURN
    END IF

!  number of data pairs

    m = nei + extra_differences

!  check sufficient data pairs are available

    IF ( m > m_available ) THEN
!     WRITE( *, * ) ( ' Warning: insufficient data pairs are available')
      m = m_available ; warning = GALAHAD_warning_data
    ELSE
      warning = GALAHAD_ok
    END IF

!  allocate storage for RHS and Matrix

    ALLOCATE( C( MAX( m, nei ), 1 ), A( m, nei ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  populate RHS and Matrix

    ptrs = PTR( i ) - 1
    IF ( PRESENT( ORDER ) ) THEN
      DO k = 1, m
        kk = ORDER( k )
        C( k, 1 ) = Y( i, kk )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), kk )
        END DO
      END DO
    ELSE
      DO k = 1, m
        C( k, 1 ) = Y( i, k )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), k )
        END DO
      END DO
    END IF

!  find space required to "solve" the linear system using LAPACK gelsd

    ALLOCATE( SV( MIN( m, nei ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, C, MAX( m, nei ), SV, - one,              &
                rank, WORK1, - 1_ip_, IWORK1, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  "solve" the linear system using gelsd (SVD divide and conquer)

    lwork = INT( WORK1( 1 ) )
    ALLOCATE( WORK( lwork ), IWORK( IWORK1( 1 ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, C, MAX( m, nei ), SV, - one,              &
                rank, WORK, lwork, IWORK, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  populate val with solution

    DO l = 1, nei
      VAL( ptrs + l ) = C( l, 1 )
    END DO

!  deallocate workspace

    DEALLOCATE( C, A, SV, WORK, IWORK, STAT = info )
    IF ( info /= 0 ) info = GALAHAD_error_deallocate
    IF ( info == 0 ) info = warning
    RETURN

!  end of SUBROUTINE SHA_solve_unknown_2_1

    END SUBROUTINE SHA_solve_unknown_2_1

!-*-  G A L A H A D -  S H A _ differences _ needed _ 2 _ 3  F U C T I O N  -*-

    FUNCTION SHA_differences_needed_2_3( n, ne, PTR, COL, SPARSE_INDS,         &
                                         DENSE_INDS, sparse_row )

    INTEGER ( KIND = ip_ ) :: SHA_differences_needed_2_3

!****************************************************************************
!
!  Algorithm 2.3: Find the number of differences needed for the Sparse-dense
!    Hessian approximation (block parallel variant)
!
!  Arguments:
!  n - number of variables
!  PTR - row pointer indices of the COMPLETE sparse Hessian (CSR format)
!
!****************************************************************************

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne, sparse_row
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: COL
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: SPARSE_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: DENSE_INDS

!  local variables

    INTEGER ( KIND = ip_ ) :: i, idx, j, l, differences_needed, m, ns, nd

!  determine if each row is sparse or dense

    ns = 0 ; nd = 0
    DO i = 1, n
      IF ( PTR( i + 1 ) - PTR( i ) <= sparse_row ) THEN ! sparse row
        ns = ns + 1
        SPARSE_INDS( ns ) = i
      ELSE ! dense row
        nd = nd + 1
        DENSE_INDS( nd ) = i
      END IF
    END DO

!  find the maximum size for the sparse rows

    differences_needed = 0
    DO idx = 1, ns
      i = SPARSE_INDS( idx )
      differences_needed = MAX( differences_needed, PTR( i + 1 ) - PTR( i ) )
    END DO

!  find the maximum row size for the dense rows

    DO idx = 1, nd
      i = DENSE_INDS( idx ) ; m = 0
      DO l = PTR( i ), PTR( i + 1 ) - 1
        j = COL( l )
        IF ( j == i .OR. PTR( j + 1 ) - PTR( j ) > sparse_row ) m = m + 1
      END DO
      differences_needed = MAX( differences_needed, m )
    END DO
    SHA_differences_needed_2_3 = differences_needed
    RETURN

!  end of FUNCTION SHA_differences_needed_2_3

    END FUNCTION SHA_differences_needed_2_3

!-*-  G A L A H A D -  S H A _ e s t i m a t e _ 2 _ 3  S U B R O U T I N E  -*-

    SUBROUTINE SHA_estimate_2_3( n, ne, PTR, COL, m_available,                 &
                                 S, ls1, ls2, Y, ly1, ly2, VAL,                &
                                 extra_differences, sparse_row, status,        &
                                 SPARSE_INDS, DENSE_INDS, INFO, ORDER )

!****************************************************************************
!
!  Algorithm 2.3: Sparse-Dense Hessian approximation (block parallel)
!
!  Arguments:
!  n - number of variables
!  ne - number of nonzero elements in the COMPLETE sparse Hessian matrix
!  PTR - row pointer indices of the COMPLETE sparse Hessian (CSR format)
!  COL - column indices of the COMPLETE sparse Hessian (CSR format)
!  m_available - number of data pairs k available for Hessian estimation
!  S - n x m_available array of optimization steps
!  Y - n x m_available array of optimization gradient differences
!  VAL - resulting estimated entries of the COMPLETE sparse Hessian (CSR format)
!  extra_differences - # extra data values allowed when solving linear systems
!  sparse_row - rows with no more that sparse_row entries are considered sparse
!  status - return status, = 0 for success, < 0 failure
!  SPARSE_INDS, DENSE_INDS, INFO - workspace arrays of length n
!  ORDER (optional) - ORDER(i), i=1:m_available gives the index of the
!    column of S and Y of the i-th most recent steps/differences.
!    If absent, the index  will be i, i=1:m_available
!
!  Written by: Jaroslav Fowkes
!
!****************************************************************************

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne, sparse_row, m_available
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ne ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: SPARSE_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: DENSE_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: INFO

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                           &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: i, idx, ns, nd

!  determine if each row is sparse or dense

    ns = 0 ; nd = 0
    DO i = 1, n
      IF ( PTR( i + 1 ) - PTR( i ) <= sparse_row ) THEN ! sparse row
        ns = ns + 1
        SPARSE_INDS( ns ) = i
      ELSE ! dense row
        nd = nd + 1
        DENSE_INDS( nd ) = i
      END IF
    END DO

!  Firstly solve for each sparse row

!$omp parallel do private( i )
    DO idx = 1, ns
      i = SPARSE_INDS( idx )

!  determine unknowns and solve linear system

      CALL SHA_solve_unknown_2_3( i, PTR, COL, m_available,                    &
                                  S, ls1, ls2, Y, ly1, ly2, VAL,               &
                                  extra_differences , INFO( i ), ORDER )
    END DO
!$omp end parallel do

!  check for errors

    status = MINVAL( INFO( SPARSE_INDS( 1 : ns ) ) )
    IF ( status /= GALAHAD_ok ) RETURN

!  then solve for each dense row

!$omp parallel do private( i )
    DO idx = 1, nd
      i = DENSE_INDS( idx )

! determine sparse knowns and unknowns and solve linear system

      CALL SHA_solve_known_unknown_2_3( i, PTR, COL, m_available,              &
                                        S, ls1, ls2, Y, ly1, ly2, VAL,         &
                                        sparse_row, extra_differences,         &
                                        INFO( i ), ORDER )
    END DO
!$omp end parallel do

!  check for errors

    status = MINVAL( INFO( DENSE_INDS( 1 : nd ) ) )
    RETURN

!  end of SUBROUTINE SHA_estimate_2_3

    END SUBROUTINE SHA_estimate_2_3

!-*-  G A L A H A D -  S H A _ s o l v e _ unknown_2_3  S U B R O U T I N E  -*-

    SUBROUTINE SHA_solve_unknown_2_3( i, PTR, COL, m_available,                &
                                      S, ls1, ls2, Y, ly1, ly2, VAL,           &
                                      extra_differences, info, ORDER )

!  "solve" the system whose j-th equation is
!     sum_{unknown i} s_ji val_i = y_j

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: i, m_available, ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: PTR, COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                          &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: k, kk, l, nei, m, ptrs, rank, lwork, warning
    INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK1
    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK1
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: IWORK
    REAL ( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: SV, WORK
    REAL ( KIND = rp_ ), DIMENSION( : , : ), ALLOCATABLE :: C, A

!  nonzeros in row

    nei = PTR( i + 1 ) - PTR( i )

!  handle case of null row

    IF ( nei == 0 ) THEN
      info = GALAHAD_ok ; RETURN
    END IF

!  number of data pairs

    m = nei + extra_differences

!  check sufficient data pairs are available

    IF ( m > m_available ) THEN
!     WRITE( *, * ) ( 'Warning: insufficient data pairs are available' )
      m = m_available ; warning = GALAHAD_warning_data
    ELSE
      warning = GALAHAD_ok
    END IF

!  allocate storage for RHS and Matrix

    ALLOCATE( C( MAX( m, nei ), 1 ), A( m, nei ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  populate RHS and Matrix

    ptrs = PTR( i ) - 1
    IF ( PRESENT( ORDER ) ) THEN
      DO k = 1, m
        kk = ORDER( k )
        C( k, 1 ) = Y( i, kk )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), kk )
        END DO
      END DO
    ELSE
      DO k = 1, m
        C( k, 1 ) = Y( i, k )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), k )
        END DO
      END DO
    END IF

!  find space required to "solve" the linear system using LAPACK gelsd

    ALLOCATE( SV( MIN( m, nei ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, c, MAX( m, nei ), SV, - 1.0_rp_,          &
                rank, WORK1, - 1_ip_, IWORK1, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  "solve" the linear system using gelsd (SVD divide and conquer)

    lwork = INT( WORK1( 1 ) )
    ALLOCATE( WORK( lwork ), IWORK( IWORK1( 1 ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, c, MAX( m, nei ), SV, - 1.0_rp_,          &
                rank, WORK, lwork, IWORK, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  populate val with solution

    DO l = 1, nei
      VAL( ptrs + l ) = C( l, 1 )
    END DO

!  deallocate workpace arrays

    DEALLOCATE( C, A, SV, WORK, IWORK, STAT = info )
    IF ( info /= 0 ) info = GALAHAD_error_deallocate
    IF ( info == 0 ) info = warning
    RETURN

!  end of SUBROUTINE SHA_solve_unknown_2_3

    END SUBROUTINE SHA_solve_unknown_2_3

!-*-  G A L A H A D -  S H A _ solve_known_unknown_2_3  S U B R O U T I N E  -*-

    SUBROUTINE SHA_solve_known_unknown_2_3( i, PTR, COL, m_available,          &
                                            S, ls1, ls2, Y, ly1, ly2,          &
                                            VAL, sparse_row,                   &
                                            extra_differences, info, ORDER )

!  "solve" the system whose j-th equation is
!     sum_{unknown i} s_ji val_i = y_j - sum_{known i} s_ji val_i

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: i, sparse_row, m_available
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: PTR, COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: j, k, kk, l, ip, nei, kn_nei, un_nei, un_m
    INTEGER ( KIND = ip_ ) :: rank, lwork, warning
    INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK1
    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK1
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: KN_COL, UN_COL
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: UN_PTR, IWORK
    REAL ( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: KN_VAL, SV, WORK
    REAL ( KIND = rp_ ), DIMENSION( : , : ), ALLOCATABLE :: C, A

!  nonzeros in row

    nei = PTR( i + 1 ) - PTR( i )

!  handle case of null row

    IF ( nei == 0 ) THEN
      info = GALAHAD_ok ; RETURN
    END IF

!  allocate storage for known (kn_) and unknown (un_) indices and values

    kn_nei = 0 ; un_nei = 0 ! known and unknown nonzeros
    ALLOCATE( KN_COL( nei ), KN_VAL( nei ), UN_COL( nei ), UN_PTR( nei ),      &
              STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  find already known symmetric entries

    DO l = PTR( i ), PTR( i + 1 ) - 1 ! for each dense row entry
      j = COL( l ) ! its column index
      ! only an off-diagonal entry can already exist
      IF ( j /= i .AND. PTR( j + 1 ) - PTR( j ) <= sparse_row ) THEN
        ! row j is sparse so entry exists
        DO ip = PTR( j ), PTR( j + 1 ) - 1 ! for each sparse row entry
          IF ( COL( ip ) == i ) THEN ! dense row index matches sparse
                                     ! column index
            kn_nei = kn_nei + 1
            KN_COL( kn_nei ) = j ! need to know where it is
            KN_VAL( kn_nei ) = VAL( ip ) ! need to know its value
            VAL( l ) = VAL( ip ) ! populate val with symmetric entry
            EXIT
          END IF
        END DO
      ELSE ! row j is dense so entry doesn't already exist,
           ! need to solve for it
        un_nei = un_nei + 1
        UN_COL( un_nei ) = j ! need to know where it is
        UN_PTR( un_nei ) = l ! need to know where to put it
      END IF
    END DO

!  number of data pairs

    un_m = un_nei + extra_differences

!  check sufficient data pairs are available

    IF ( un_m > m_available ) THEN
!     WRITE( *, * ) ( ' Warning: insufficient data pairs are available' )
      un_m = m_available ; warning = GALAHAD_warning_data
    ELSE
      warning = GALAHAD_ok
    END IF

!  allocate storage for RHS and Matrix

    ALLOCATE( C( MAX( un_m, un_nei ), 1 ), A( un_m, un_nei ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  populate RHS and Matrix

    IF ( PRESENT( ORDER ) ) THEN
      DO k = 1, un_m
        kk = ORDER( k )
        C( k, 1 ) = Y( i, kk )
        DO l = 1, kn_nei ! known entries in RHS
          C( k, 1 ) = C( k, 1 ) - S( KN_COL( l ), kk ) * KN_VAL( l )
        END DO
        DO l = 1, un_nei
          A( k, l ) = S( UN_COL( l ), kk )
        END DO
      END DO
    ELSE
      DO k = 1, un_m
        C( k, 1 ) = Y( i, k )
        DO l = 1, kn_nei ! known entries in RHS
          C( k, 1 ) = C( k, 1 ) - S( KN_COL( l ), k ) * KN_VAL( l )
        END DO
        DO l = 1, un_nei
          A( k, l ) = S( UN_COL( l ), k )
        END DO
      END DO
    END IF

!  find space required to "solve" the linear system using LAPACK gelsd

    ALLOCATE( SV( MIN( un_m,un_nei ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( un_m, un_nei, 1_ip_, A, un_m, C, MAX( un_m,un_nei), SV,        &
                - one, rank, WORK1, -1_ip_, IWORK1, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  "solve" the linear system using gelsd (SVD divide and conquer)

    lwork = INT( WORK1( 1 ) )
    ALLOCATE( WORK( lwork ), IWORK( iwork1( 1 ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( un_m, un_nei, 1_ip_, A, un_m, C, MAX( un_m,un_nei), SV,        &
                - one, rank, WORK, lwork, IWORK, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  populate val with solution

    DO l = 1, un_nei
      VAL( UN_PTR( l ) ) = C( l, 1 )
    END DO

!  Deallocate workspace arrays

    DEALLOCATE( C, A, SV, WORK, IWORK, KN_COL, KN_VAL, UN_COL, UN_PTR,         &
                STAT = info )
    IF ( info /= 0 ) info = GALAHAD_error_deallocate
    IF ( info == 0 ) info = warning
    RETURN

!  end of SUBROUTINE SHA_solve_known_unknown_2_3

    END SUBROUTINE SHA_solve_known_unknown_2_3

!-*-  G A L A H A D -  S H A _ e s t i m a t e _ 2 _ 4  S U B R O U T I N E  -*-

    SUBROUTINE SHA_estimate_2_4( n, ne, PTR, COL, m_available,                 &
                                 S, ls1, ls2, Y, ly1, ly2, VAL,                &
                                 extra_differences, recursion_max,             &
                                 recursion_entries_required, status,           &
                                 SPARSE_INDS, DENSE_INDS, TMP_INDS, NSOLVED,   &
                                 INFO, ORDER )

!****************************************************************************
!
!  Algorithm 2.4: Sparse-Dense Hessian approximation (recursive block parallel)
!
!  Arguments:
!  n - number of variables
!  ne - number of nonzero elements in the COMPLETE sparse Hessian matrix
!  PTR - row pointer indices of the COMPLETE sparse Hessian (CSR format)
!  COL - column indices of the COMPLETE sparse Hessian (CSR format)
!  m_available - number of data pairs k available for Hessian estimation
!  S - n x m_available array of optimization steps
!  Y - n x m_available array of optimization gradient differences
!  VAL - resulting estimated entries of the COMPLETE sparse Hessian (CSR format)
!  extra_differences - # extra data values allowed when solving linear systems
!  recursion_max - maximum number of levels of recursion
! recursion_entries_required - recursion can only occur for a (reduced) row if
!     it has at least recursion_entries_required entries
!  status - return status, = 0 for success, < 0 failure
!  SPARSE_INDS, DENSE_INDS, TMP_INDS, NSOLVED, INFO - wkspace arrays of length n
!  ORDER (optional) - ORDER(i), i=1:m_available gives the index of the
!    column of S and Y of the i-th most recent steps/differences.
!    If absent, the index  will be i, i=1:m_available
!
!  Written by: Jaroslav Fowkes
!
!****************************************************************************

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne, m_available
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: ls1, ls2, ly1, ly2, recursion_max
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: recursion_entries_required
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n + 1 ) :: PTR
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: COL
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( ne ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: SPARSE_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: DENSE_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: TMP_INDS
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: NSOLVED
    INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: INFO

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  Local variables

    INTEGER ( KIND = ip_ ) :: net, nei, i, recursion, idx, ns, nd, td

!  initialize solved symmetric entry counter

    NSOLVED( : n ) = 0

!  Set sparse/dense row threshold

    net = m_available - extra_differences

!  determine if each row is truly sparse or dense

    ns = 0 ; nd = 0
    DO i = 1, n
      IF ( PTR( i + 1 ) - PTR( i ) <= net ) THEN ! truly sparse row
        ns = ns + 1
        SPARSE_INDS( ns ) = i
      ELSE ! truly dense row
        nd = nd + 1
        DENSE_INDS( nd ) = i
      END IF
    END DO

!  firstly solve for each truly sparse row

!$omp parallel do private( i )
    DO idx = 1, ns
      i = SPARSE_INDS( idx )

!  determine unknowns and solve linear system

      CALL SHA_solve_unknown_2_4( i, PTR, COL, m_available,                    &
                                  S, ls1, ls2, Y, ly1, ly2, VAL,               &
                                  NSOLVED, extra_differences, INFO( i ), ORDER )
    END DO
!$omp end parallel do

!  check for errors

    IF ( ns > 0 ) THEN
      status = MINVAL( INFO( SPARSE_INDS( 1 : ns ) ) )
    ELSE
      status = GALAHAD_ok
    END IF
    IF ( status /= GALAHAD_ok ) RETURN

!  initialise recursion depth counter

    recursion = 1

!  now recurse

    DO WHILE( recursion <= recursion_max )

! determine which remaining rows are sparse excluding solved symmetric entries

      ns = 0 ; td = 0
      DO idx = 1, nd
        i = DENSE_INDS( idx )
        nei = PTR( i + 1 ) - PTR( i ) - NSOLVED( i )
        IF ( nei <= net .AND.                                                  &
             nei >= recursion_entries_required ) THEN ! newly sparse row
          ns = ns + 1
          SPARSE_inds( ns ) = i
        ELSE ! newly dense row
          td = td + 1
          TMP_INDS( td ) = i
        END IF
      END DO
      nd = td
      DENSE_INDS( : nd ) = TMP_INDS( : nd )

      IF ( ns == 0 ) EXIT ! no more newly sparse rows

!  solve for each newly sparse row

!$omp parallel do private( i )
      DO idx = 1, ns
        i = SPARSE_INDS( idx )

!  determine sparse knowns and unknowns and solve linear system

        CALL SHA_solve_known_unknown_2_4( i, PTR, COL, m_available,            &
                                          S, ls1, ls2, Y, ly1, ly2,            &
                                          VAL, NSOLVED, extra_differences,     &
                                          INFO( i ), ORDER )
      END DO
!$omp end parallel do

!  check for errors

    IF ( ns > 0 ) status = MINVAL( INFO( SPARSE_INDS( 1 : ns ) ) )
    IF ( status /= GALAHAD_ok ) RETURN

!  increment recursion depth counter

      recursion = recursion + 1
    END DO

!  finally solve for each truly dense row

!$omp parallel do private( i )
    DO idx = 1, nd
      i = DENSE_INDS( idx )

!  determine sparse knowns and unknowns and solve linear system

      CALL SHA_solve_known_unknown_2_4( i, PTR, COL, m_available,              &
                                        S, ls1, ls2, Y, ly1, ly2,              &
                                        VAL, NSOLVED, extra_differences,       &
                                        INFO( i ), ORDER )
    END DO
!$omp end parallel do

!  check for errors

    IF ( nd > 0 ) status = MINVAL( INFO( DENSE_INDS( 1 : nd ) ) )
    RETURN

!  end of SUBROUTINE SHA_estimate_2_4

    END SUBROUTINE SHA_estimate_2_4

!-*-  G A L A H A D -  S H A _ s o l v e _ unknown_2_4  S U B R O U T I N E  -*-

    SUBROUTINE SHA_solve_unknown_2_4( i, PTR, COL, m_available,                &
                                      S, ls1, ls2, Y, ly1, ly2, VAL,           &
                                      NSOLVED, extra_differences, info, ORDER )

!  "solve" the system whose j-th equation is
!     sum_{unknown i} s_ji val_i = y_j

!  Dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: i, m_available, ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: PTR, COL
    INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: NSOLVED
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  Local variables

    INTEGER ( KIND = ip_ ) :: k, kk, l, nei, m, ptrs, rank, lwork, warning
    INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK1
    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK1
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: IWORK
    REAL ( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: SV, WORK
    REAL ( KIND = rp_ ), DIMENSION( : , : ), ALLOCATABLE :: C, A

!  Nonzeros in row

    nei = PTR( i + 1 ) - PTR( i )

!  Handle case of null row

    IF ( nei == 0 ) THEN
      info = GALAHAD_ok ; RETURN
    END IF

!  Number of data pairs

    m = nei + extra_differences

!  check sufficient data pairs are available

    IF ( m > m_available ) then
!     WRITE( *, * ) ( ' Warning: insufficient data pairs are available ')
      m = m_available ; warning = GALAHAD_warning_data
    ELSE
      warning = GALAHAD_ok
    END IF

!  allocate storage for RHS and Matrix

    ALLOCATE( C( MAX( m, nei ), 1 ), A( m, nei ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  populate RHS and Matrix

    ptrs = PTR( i ) - 1
    IF ( PRESENT( ORDER ) ) THEN
      DO k = 1, m
        kk = ORDER( k )
        C( k, 1 ) = Y( i, kk )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), kk )
        END DO
      END DO
    ELSE
      DO k = 1, m
        C( k, 1 ) = Y( i, k )
        DO l = 1, nei
          A( k, l ) = S( COL( ptrs + l ), k )
        END DO
      END DO
    END IF

!  find space required to "solve" the linear system using LAPACK gelsd

    ALLOCATE( SV( MIN( m, nei ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, C, MAX( m, nei ), SV, - one,              &
                rank, WORK1, -1_ip_, IWORK1, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  "solve" the linear system using gelsd (SVD divide and conquer)

    lwork = INT( WORK1( 1 ) )
    ALLOCATE( WORK( lwork ), IWORK( IWORK1( 1 ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( m, nei, 1_ip_, A, m, C, MAX( m, nei ), SV, - one,              &
                rank, WORK, lwork, IWORK, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  populate val with solution and record number of solved symmetric entries

    DO l = 1, nei
      k = ptrs + l
      VAL( k ) = C( l, 1 )
      NSOLVED( COL( k ) ) = NSOLVED( COL( k ) ) + 1
    END DO

!  deallocate workspace arrays

    DEALLOCATE( C, A, SV, WORK, IWORK, STAT = info )
    IF ( info /= 0 ) info = GALAHAD_error_deallocate
    IF ( info == 0 ) info = warning
    RETURN

!  end of SUBROUTINE SHA_solve_unknown_2_4

    END SUBROUTINE SHA_solve_unknown_2_4

!-*-  G A L A H A D -  S H A _ solve_known_unknown_2_4  S U B R O U T I N E  -*-

    SUBROUTINE SHA_solve_known_unknown_2_4( i, PTR, COL, m_available,          &
                                            S, ls1, ls2, Y, ly1, ly2,          &
                                            VAL, NSOLVED, extra_differences,   &
                                            info, ORDER )

!  "solve" the system whose j-th equation is
!     sum_{unknown i} s_ji val_i = y_j - sum_{known i} s_ji val_i

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ) :: i, m_available, ls1, ls2, ly1, ly2
    INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: PTR, COL
    INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: NSOLVED
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ls1, ls2 ) :: S
    REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( ly1, ly2 ) :: Y
    REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: VAL
    INTEGER ( KIND = ip_ ), INTENT( IN ) :: extra_differences
    INTEGER ( KIND = ip_ ), INTENT( OUT ) :: info

!  optional arguments

    INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL,                            &
                            DIMENSION( m_available ) :: ORDER

!  local variables

    INTEGER ( KIND = ip_ ) :: net, j, k, kk, l, ip, nei, kn_nei, un_nei, un_m
    INTEGER ( KIND = ip_ ) :: rank, lwork, warning
    INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: IWORK1
    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: WORK1

    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: KN_COL, UN_COL
    INTEGER ( KIND = ip_ ), DIMENSION( : ), ALLOCATABLE :: UN_PTR, IWORK
    REAL ( KIND = rp_ ), DIMENSION( : ), ALLOCATABLE :: KN_VAL, SV, WORK
    REAL ( KIND = rp_ ), DIMENSION( : , : ), ALLOCATABLE :: C, A

!  set sparse/dense row threshold

    net = m_available - extra_differences

!  nonzeros in row

    nei = PTR( i + 1 ) - PTR( i )

!  handle case of null row

    IF ( nei == 0 ) THEN
      info = GALAHAD_ok ; RETURN
    END IF

!  allocate storage for known (kn_) and unknown (un_) indices and values

    kn_nei = 0 ; un_nei = 0 ! known and unknown nonzeros
    ALLOCATE( KN_COL( nei ), KN_VAL( nei ), UN_COL( nei ), UN_PTR( nei ),      &
              STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  find already known symmetric entries

    DO l = PTR( i ), PTR( i + 1 ) - 1 ! for each dense row entry
      j = COL( l ) ! its column index
      ! only an off-diagonal entry can already exist
      IF ( j /= i .AND. PTR( j + 1 ) - PTR( j ) <= net ) THEN ! row j is sparse
                                                              ! so entry exists
        DO ip = PTR( j ), PTR( j + 1 ) - 1 ! for each sparse row entry
          IF ( COL( ip ) == i ) THEN ! dense row index matches
                                     ! sparse column index
            kn_nei = kn_nei + 1
            KN_COL( kn_nei ) = j ! need to know where it is
            KN_VAL( kn_nei ) = VAL( ip ) ! need to know its value
            VAL( l ) = VAL( ip ) ! populate val with symmetric entry
            EXIT
          END IF
        END DO
      ELSE ! row j is dense so entry doesn't already exist,
           ! need to solve for it
        un_nei = un_nei + 1
        UN_COL( un_nei ) = j ! need to know where it is
        UN_PTR( un_nei ) = l ! need to know where to put it
      END IF
    END DO

!  number of data pairs

    un_m = un_nei + extra_differences

!  check sufficient data pairs are available

    IF ( un_m > m_available ) THEN
!     WRITE( *, * ) ( ' Warning: insufficient data pairs are available' )
      un_m = m_available ; warning = GALAHAD_warning_data
    ELSE
      warning = GALAHAD_ok
    END IF

!  allocate storage for RHS and Matrix

    ALLOCATE( C( MAX( un_m,un_nei ), 1 ), A( un_m, un_nei ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

!  populate RHS and Matrix

    IF ( PRESENT( ORDER ) ) THEN
      DO k = 1, un_m
        kk = ORDER( k )
        C( k, 1 ) = Y( i, kk )
        DO l = 1, kn_nei ! known entries in RHS
          C( k, 1 ) = C(k, 1 ) - S( KN_COL( l ), kk ) * KN_VAL( l )
        END DO
        DO l = 1, un_nei
          A( k, l ) =  S( UN_COL( l ), kk )
        END DO
      END DO
    ELSE
      DO k = 1, un_m
        C( k, 1 ) = Y( i, k )
        DO l = 1, kn_nei ! known entries in RHS
          C( k, 1 ) = C(k, 1 ) - S( KN_COL( l ), k ) * KN_VAL( l )
        END DO
        DO l = 1, un_nei
          A( k, l ) =  S( UN_COL( l ), k )
        END DO
      END DO
    END IF

!  find space required to "solve" the linear system using LAPACK gelsd

    ALLOCATE( SV( MIN( un_m, un_nei ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( un_m, un_nei, 1_ip_, A, un_m, C, MAX( un_m, un_nei ), SV,      &
                - one, rank, WORK1, - 1_ip_, IWORK1, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  "solve" the linear system using gelsd (SVD divide and conquer)

    lwork = INT( WORK1( 1 ) )
    ALLOCATE( WORK( lwork ), IWORK( IWORK1( 1 ) ), STAT = info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_allocate ; RETURN
    END IF

    CALL GELSD( un_m, un_nei, 1_ip_, A, un_m, C, MAX( un_m, un_nei ), SV,      &
                - one, rank, WORK, lwork, IWORK, info )
    IF ( info /= 0 ) THEN
      info = GALAHAD_error_lapack ; RETURN
    END IF

!  populate val with solution and record number of solved symmetric entries

    DO l = 1, un_nei
      k =  UN_PTR( l )
      VAL( k ) = C( l, 1 )
      NSOLVED( COL( k ) ) = NSOLVED( COL( k ) ) + 1
    END DO

!  deallocate workspace arrays

    DEALLOCATE( C, A, SV, WORK, IWORK, KN_COL, KN_VAL, UN_COL, UN_PTR,         &
                STAT = info )
    IF ( info /= 0 ) info = GALAHAD_error_deallocate
    IF ( info == 0 ) info = warning
    RETURN

!  end of SUBROUTINE SHA_solve_known_unknown_2_4

    END SUBROUTINE SHA_solve_known_unknown_2_4

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

      array_name = 'SHA: data%PTR_unsym'
      CALL SPACE_dealloc_array( data%PTR_unsym,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%COL_unsym'
      CALL SPACE_dealloc_array( data%COL_unsym,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%VAL_unsym'
      CALL SPACE_dealloc_array( data%VAL_unsym,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%SPARSE_INDS'
      CALL SPACE_dealloc_array( data%SPARSE_INDS,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%DENSE_INDS'
      CALL SPACE_dealloc_array( data%DENSE_INDS,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%TMP_INDS'
      CALL SPACE_dealloc_array( data%TMP_INDS,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%NSOLVED'
      CALL SPACE_dealloc_array( data%NSOLVED,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'SHA: data%INFO'
      CALL SPACE_dealloc_array( data%INFO,                                     &
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
