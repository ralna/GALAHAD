! THIS VERSION: GALAHAD 2.5 - 15/07/2012 AT 09:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ F D H   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould and Philippe Toint

!  History -
!   fortran 77 version by Philippe Toint (1980) for Harwell Subroutine Library
!   originally released GALAHAD Version 2.5. July 15th 2012

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FDH_double

!    -----------------------------------------------------------------------
!   |                                                                       |
!   | FDH: find an approximation to a sparse Hessian by finite differences  |
!   |                                                                       |
!    -----------------------------------------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: FDH_initialize, FDH_read_specfile, FDH_analyse, FDH_estimate,   &
               FDH_terminate, NLPT_userdata_type

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

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FDH_control_type

!   error and warning diagnostics occur on stream error 
   
       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!  if space is critical, ensure allocated arrays are no bigger than needed

       LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE FDH_control_type

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: FDH_inform_type

!  return status. See FDH_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  row in which bad data appeared

       INTEGER :: bad_row = 0

!  the number of gradient differences that will be needed per Hessian estimate

       INTEGER :: products = - 1

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

     END TYPE FDH_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: FDH_data_type

!  local variables

       INTEGER :: branch = 0
       INTEGER :: eval_status, ibnd, istop, n, ng, nz

!  np is the number of gradient differences that will be needed for one 
!  Hessian estimation

       INTEGER :: np

!  FDH_analyse_called is true once FDH_analyse has been called

       LOGICAL :: FDH_analyse_called = .FALSE.

!  DIAG_perm(i) is the analogue of DIAG(i), but for the permuted structure 
!  (i=1,n)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: DIAG_perm

!  ROW_perm(i) is the analogue of ROW(i), but for the permuted structure 
!  (i=1,nz)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_perm

!  OLD(I) is the position IN ROW (original stucture) of the i-th element of 
!  ROW_perm (permuted structure) (i=1,2*n)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: OLD

!  PERM(i) is the position in the original structure of the i-th row and 
!  column of the permuted one (i=1,n) (permutation of the integers 1 to n)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM

!  GROUP(i) is the group to which the i-th column belongs (i=1,n)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: GROUP

!  IWK(i) is integer workspace (i=1,n)

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IWK

!  X(i) and G(i) is real workspace (i=1,n)

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G

     END TYPE FDH_data_type

   CONTAINS

!-*-*-  G A L A H A D -  F D H _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE FDH_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for FDH controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FDH_data_type ), INTENT( INOUT ) :: data
     TYPE ( FDH_control_type ), INTENT( OUT ) :: control
     TYPE ( FDH_inform_type ), INTENT( OUT ) :: inform        

     inform%status = GALAHAD_ok

!  initial private data

     data%FDH_analyse_called = .FALSE.
     data%branch = 0

     RETURN

!  End of subroutine FDH_initialize

     END SUBROUTINE FDH_initialize

!-*-*-*-*-   F D H _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE FDH_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by FDH_initialize could (roughly) 
!  have been set as:

! BEGIN FDH SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  space-critical                                  F
!  deallocate-error-fatal                          F
!  output-line-prefix                              ""
! END FDH SPECIFICATIONS

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     TYPE ( FDH_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: space_critical = print_level + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'FDH '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level' 

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

!  End of subroutine FDH_read_specfile

     END SUBROUTINE FDH_read_specfile

!-*-*-*-  G A L A H A D -  F D H _ a n a l y s e  S U B R O U T I N E -*-*-*-

      SUBROUTINE FDH_analyse( n, nz, ROW, DIAG, data, control, inform )
!
!   **************************************************************
!   *                                                            *
!   *   Estimation of a sparse Hessian matrix by differences     *
!   *                                                            *
!   *   Structure analysis and group forming for the lower       *
!   *    triangular method                                       *
!   *                                                            *
!   *   See M.J.D. Powell and Ph.L. Toint, "On the estimation    *
!   *    of sparse Hessian matrices", SIAM J. Numer. Analysis    *
!   *    Vol. 16, No. 6, (1979) pp. 1060–1074                    *
!   *                                                            *
!   **************************************************************
!
!   The Hessian stucture is given by n, nz, ROW and DIAG, where

!      n is the number of variables
!      nz is the number of nonzero elements in the lower triangular part of 
!         the matrix
!      ROW(i) is the row number of the i-th nonzero element of the lower 
!         triangle of the matrix, where these elements have been arranged 
!         in a list scanning the successive columns (i=1,nz)
!      DIAG(i) is the position of the i-th diagonal elenent in the list 
!        defined by ROW (i=1,n)
 
!   The analysed and permuted structure and the groups are stored in the
!   derived type data (see preface)
 
!   Action of the subroutine is controlled by components of the derived type
!   control, while information about the progress of the subroutine is reported
!   in inform (again, see preface). Success or failure is flagged by the 
!   component inform%status -
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
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: DIAG
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: ROW

      TYPE ( FDH_control_type ), INTENT( IN ) :: control
      TYPE ( FDH_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( FDH_data_type ), INTENT( INOUT ) :: data

!   Programming: F77 version by Philippe Toint (1980) with mods by 
!   Iain Duff (1980) and rewrite in F2003 by Nick Gould (2012)

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, i1, i2, ia, iact, ic, ic1, icons, id2
      INTEGER :: idum, iend, ihgst, ii, ijmx, ikicon, il
      INTEGER :: ipass, ipos, ir, ir1, ir2, row_permi, ivic, ivir
      INTEGER :: iwi, j, j1, j2, jj, jpos, jptr, jsw, lwst, minrow, minum
      INTEGER :: n1, nbmx, nbnd, nga, nm2, nnp1mi, nrl, nshift, ntst, numj
      CHARACTER ( LEN = 80 ) :: array_name
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix

!     logical :: test = .TRUE.
!  test for errors in the input data

      IF ( nz < n .OR. n <= 0 .OR. nz > ( n * ( n + 1 ) ) / 2 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 900
      END IF

      DO i = 1, n
        IF ( ROW( DIAG( i ) ) /= i ) THEN
          inform%status = GALAHAD_error_upper_entry ; inform%bad_row = i
          GO TO 900
        END IF
      END DO

      data%n = n ; data%nz = nz
      inform%status = GALAHAD_ok

!   *******************
!   *                 *
!   *   permutation   *
!   *                 *
!   *******************

!  allocate integer workspace

      array_name = 'fdh: data%DIAG_perm'
      CALL SPACE_resize_array( n, data%DIAG_perm,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'fdh: data%ROW_perm'
      CALL SPACE_resize_array( nz, data%ROW_perm,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'fdh: data%OLD'
      CALL SPACE_resize_array( MAX( 2 * n, nz ), data%OLD,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'fdh: data%PERM'
      CALL SPACE_resize_array( n, data%PERM,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'fdh: data%GROUP'
      CALL SPACE_resize_array( n, data%GROUP,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'fdh: data%IWK'
      CALL SPACE_resize_array( n, data%IWK,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900


      IF ( .FALSE. ) THEN
        data%GROUP( : n ) = 0
        DO i = 1, n - 1
         do j = DIAG( i ),DIAG(i+1)-1
           if ( data%GROUP( ROW( j ) ) /= 0 ) THEN
             write(6,*) i, data%GROUP( ROW( j ) ), j
             write(6,*)  ROW( data%GROUP( ROW( j ) ) ),  ROW( j )
             WRITE(6,*) ' row ', i, ' entries ', ROW(DIAG( i ):DIAG(i+1)-1)
             stop
           end if
           data%GROUP( ROW( j ) ) = j
         end do
         data%GROUP( ROW(DIAG( i ):DIAG(i+1)-1) ) = 0
        END DO

       do j = DIAG( n ),nz
         if ( data%GROUP( ROW( j ) ) /= 0 ) THEN
           write(6,*) i, data%GROUP( ROW( j ) ), j
           write(6,*)  ROW( data%GROUP( ROW( j ) ) ),  ROW( j )
           WRITE(6,*) ' row ', n, ' entries ', ROW(DIAG( n ):nz)
           stop
         end if
         data%GROUP( ROW( j ) ) = j
       end do
       data%GROUP( ROW(DIAG( n ):nz )) = 0
      stop
    END IF

!  initialize

!     nshift = 10 * n
      nshift = 100 * n
      data%ibnd = 0
      ijmx = 0
      data%PERM( : n ) = 0
      data%GROUP( : n ) = 0
      data%IWK( : n ) = 0

!  compute the number of unknowns in each row (in IWK) and its maximum

      DO ic = 1, n
        i1 = DIAG( ic )
        IF ( ic /= n ) THEN
          i2 = DIAG( ic + 1 ) - 1
        ELSE
          i2 = nz
        END IF
        DO ii = i1, i2
          ir = ROW( ii )
          IF ( ir < ic ) THEN
            inform%status = - ic ; GO TO 900
          END IF
          ijmx = MAX( ijmx, ir - ic )
          data%PERM( ir ) = data%PERM( ir ) + 1
          data%IWK( ir ) = data%IWK( ir ) + 1
          IF ( ir /= ic ) data%PERM( ic ) = data%PERM( ic ) + 1
        END DO
      END DO
      nbmx = MAXVAL( data%PERM( : n ) )
      nrl = MAXVAL( data%IWK( : n ) )

!  test for a band matrix. If so, set ibnd to half the bandwidth + 1

      nbnd = ( ijmx + 1 ) * ( 2 * n - ijmx ) / 2
      IF ( nz == nbnd ) data%ibnd = ijmx + 1

!  test whether permutation might improve estimation. If not, set up
!  information about the new structure (identical to the old one) immediately 
!  and branch to group forming

      IF ( n * ( nrl - 1 ) < nz .OR. data%ibnd /= 0 ) THEN
!     IF ( .TRUE. ) THEN
        DO i = 1, nz
          data%ROW_perm( i ) = ROW( i )
          data%OLD( i ) = i
        END DO
        DO i = 1,n
          data%DIAG_perm( i ) = DIAG( i )
          data%IWK( i ) = i
        END DO
        GO TO 200
      END IF

!  sort the rows by decreasing order of unknowns. First set up a linked list 
!  of rows with same number of non-zeros. Header pointers in GROUP, links in 
!  ROW_perm

      data%DIAG_perm( 1 ) = data%IWK( 1 )

!  if n=1 the matrix is banded and we do not execute this code

      DO i = 2, n
        data%DIAG_perm( i ) = data%DIAG_perm( i - 1 ) + data%IWK( i ) - 1
      END DO
      DO i = n, 1, - 1
        iwi = data%PERM( i )
        data%ROW_perm( i ) = data%GROUP( iwi )
        data%GROUP( iwi ) = i
      END DO

!  build the list of rows by decreasing number of unknowns. This list is held 
!  in OLD( i ),i=n+1,2*n with a pointer to the position of the last row with i 
!  non-zeros held in OLD(i). The inverse permutation indicating the position 
!  of row i in the list is held in IV

      ia = n
      DO i = nbmx, 1, - 1
        n1 = data%GROUP( i )
        DO idum = 1, n
          IF ( n1 <= 0 ) EXIT
          ia = ia + 1
          data%OLD( ia ) = n1
          data%IWK( n1 ) = ia
          n1 = data%ROW_perm( n1 )
        END DO
        data%OLD( i ) = ia
        IF ( ia >= 2 * n ) EXIT
      END DO

!  Start the loop to build the permutation

      DO j = 1, n
        IF ( j /= n ) THEN
          i2 = DIAG( j + 1 ) - 1
        ELSE
          i2 = nz
        END IF
        DO ii = DIAG( j ) + 1, i2
          i = ROW( ii )
          ipos = data%DIAG_perm( i ) - 1
          data%ROW_perm( ipos ) = j
          data%DIAG_perm( i ) = ipos
        END DO
      END DO
      nm2 = n - 2

!  Not executed for any matrix with n = 1 or 2: such a matrix is banded or 
!  diagonal

      DO i = 1, nm2

!  choose the row with minimum number of unknowns in the leading (n-i+1)*(n-i+1)
!  submatrix.

        nnp1mi = 2 * n + 1 - i
        minrow = data%OLD( nnp1mi )
        DIAG( minrow ) = - DIAG( minrow )
        minum = data%PERM( minrow )
        data%OLD( minum ) = data%OLD( minum ) - 1

!  scan the elements of the chosen row in order to find the rows whose number 
!  of unknowns is decreasing

        DO ipass = 1, 2
          IF ( ipass == 1 ) THEN
            j1 = ABS( DIAG( minrow ) ) + 1
            IF ( minrow /= n ) THEN
              j2 = ABS( DIAG( minrow + 1 ) ) - 1
            ELSE
              j2 = nz
            END IF
          ELSE
            j1 = data%DIAG_perm( minrow )
            IF ( minrow /= n ) THEN
              j2 = data%DIAG_perm( minrow + 1 ) - 1
            ELSE
              j2 = nz - n
            END IF
          END IF
          DO jj = j1, j2
            IF ( ipass == 1 ) THEN
              j = ROW( jj )
            ELSE
              j = data%ROW_perm( jj )
            END IF
            IF ( DIAG( j ) < 0 ) CYCLE

!  revise the counts of unknowns and the various pointers

            jpos = data%IWK( j )
            numj = data%PERM( j )
            data%PERM( j ) = data%PERM( j ) - 1
            jptr = data%OLD( numj )
            IF ( jptr /= jpos ) THEN
              jsw = data%OLD( jptr )
              data%OLD( jpos ) = jsw
              data%IWK( j ) = jptr
              data%OLD( jptr ) = j
              data%IWK( jsw ) = jpos
            END IF
            data%OLD( numj ) = jptr - 1
            IF ( numj == minum ) data%OLD( numj - 1 ) = nnp1mi - 1
          END DO
        END DO
      END DO

!  end of the permutation building loop: the permutation is now available in IWK

      DO i = 1, n
        data%IWK( i ) = data%IWK( i ) - n
        DIAG( i ) = ABS( DIAG( i ) )
        data%GROUP( i ) = 0
      END DO
!write(6,*) data%IWK
!stop

!   ********************************************************
!   *                                                      *
!   *   set up information about the permuted structure    *
!   *                                                      *
!   ********************************************************
 
!  build the new column numbers in ROW_perm and the counts of elements by 
!  column in GROUP. Also set OLD(i) to - (old row number + nshift * 
!  (new row number))

!  ---------------------------------- IGNORED ----------------------------------
!  |                                                                           |
!  v                                                                           v
!   IF ( .TRUE. ) THEN
    IF ( .FALSE. ) THEN
      ic = 1
      DO i = 1, nz
        IF ( i /= 1 .AND. ic /= n ) THEN
          ic1 = ic + 1
          IF ( i == DIAG( ic1 ) ) ic = ic1
        END IF
        ivir = data%IWK( ROW( i ) )
        ivic = data%IWK( ic )
        data%OLD( i ) = - i - nshift * MAX( ivir, ivic )
        row_permi = MIN( ivir, ivic )
        data%ROW_perm( i ) = row_permi
        data%GROUP( row_permi ) = data%GROUP( row_permi ) + 1
      END DO

!  build the new information on the beginning of each row in DIAG_perm and let
!  data%GROUP(i) be the position of the next element in the i-th column

      data%DIAG_perm( 1 ) = 1
      DO i = 2, n
        data%DIAG_perm( i ) = data%DIAG_perm( i - 1 ) + data%GROUP( i - 1 )
      END DO
      data%GROUP( : n ) = data%DIAG_perm( : n )

!  reorder ROW_perm and OLD to gather the elements of the same column ...
!  start the reordering with the first element in ROW_perm ...

      ia = 1

!  ... and consider the actual element in ROW_perm:

  100 CONTINUE
      icons = data%ROW_perm( ia )

!   if it is in the right position, move one position further in OLD

      IF ( ia == data%GROUP( icons ) ) THEN
        data%OLD( ia ) = - data%OLD( ia )
        data%GROUP( icons ) = IA + 1

!  if not, exchange with the right one ...

      ELSE
        ikicon = data%GROUP( icons )
        ir1 = data%OLD( ikicon )
        data%OLD( ikicon ) = - data%OLD( ia )
        data%OLD( ia ) = ir1
        data%ROW_perm( ia ) = data%ROW_perm( ikicon )
        data%ROW_perm( ikicon ) = ICONS

!   ... and revise the position of the next element

        data%GROUP( icons ) = ikicon + 1
        GO TO 100

      END IF
  110 CONTINUE
      ia = ia + 1

!  once the whole vector has been considered, stop the reordering

      IF ( ia <= nz ) THEN

!  if this position has been considered already, move one position further
!  in OLD. Otherwise, consider it now

        IF ( data%OLD( ia ) > 0 ) THEN
          GO TO 110
        ELSE
          GO TO 100
        END IF

!  end of the reordering to gather the columns

      END IF

!  set ROW_perm(i) to the new row number of the i-th element in ROW

      DO i = 1, nz
        data%ROW_perm( i ) = data%OLD( i ) / nshift
        data%OLD( i ) = data%OLD( i ) - data%ROW_perm( i ) * nshift
      END DO
!  ^                                                                           ^
!  |                                                                           |
!  --------------------------- END OF IGNORED ----------------------------------

    ELSE

!  compute the number of nonzeros in the lower triangular part of each row
!  of the permuted matrix

      DO ic = 1, n
        ivic = data%IWK( ic )
        IF ( ic /= n ) THEN
          i2 = DIAG( ic + 1 ) - 1
        ELSE
          i2 = nz
        END IF
        DO ii = DIAG( ic ), i2
          ir = ROW( ii )
          ivir = data%IWK( ir )
!        data%OLD( i ) = - i - nshift * MAX( ivir, ivic )
          row_permi = MIN( ivir, ivic )
!        data%ROW_perm( i ) = row_permi
          data%GROUP( row_permi ) = data%GROUP( row_permi ) + 1
        END DO
      END DO

!  now set the starting addresses

      i1 = 1
      DO i = 1, n - 1
        i2 = i1 + data%GROUP( i )
        data%GROUP( i ) = i1
        i1 = i2
      END DO  
      i2 = i1 + data%GROUP( n )
      data%GROUP( n ) = i1
!  check - i2 should be nz

      data%DIAG_perm( : n ) = data%GROUP( : n )

!  now set up the permuted matrix

      DO ic = 1, n
        ivic = data%IWK( ic )
        IF ( ic /= n ) THEN
          i2 = DIAG( ic + 1 ) - 1
        ELSE
          i2 = nz
        END IF
        DO ii = DIAG( ic ), i2
          ir = ROW( ii )
          ivir = data%IWK( ir )
!        data%OLD( i ) = - i - nshift * MAX( ivir, ivic )
          row_permi = MIN( ivir, ivic )
          i1 = data%GROUP( row_permi )
          data%ROW_perm( i1 ) = MAX( ivir, ivic )
          data%OLD( i1 ) = ii
          data%GROUP( row_permi ) = data%GROUP( row_permi ) + 1
        END DO
      END DO

    END IF

  200 CONTINUE

!   *************************************************
!   *                                               *
!   *    form the groups for a band matrix          *
!   *                                               *
!   *************************************************

      IF ( data%ibnd /= 0 ) THEN
        DO i = 1, data%ibnd
          data%GROUP( i ) = i
        END DO
        il = data%ibnd + 1
        DO i = il, n
          data%GROUP( i ) = MOD( i, data%ibnd )
          IF ( data%GROUP( i ) == 0 ) data%GROUP( i ) = data%ibnd
        END DO
        DO i = 1, n
          data%PERM( i ) = i
        END DO
        data%np = data%ibnd
        data%DIAG_perm( 1 ) = - data%DIAG_perm( 1 )
        GO TO 900
      END IF

!   **********************************************************
!   *                                                        *
!   *   Curtis-Powell-Reid procedure on the lower triangle   *
!   *                                                        *
!   **********************************************************

!  ---------------------------------- IGNORED ----------------------------------
!  |                                                                           |
!  v                                                                           v
    IF ( .FALSE. ) THEN
        DO i = 1, n - 1
         WRITE(6,*) ' row ', i, ' entries ',                               &
           data%ROW_perm(data%DIAG_perm( i ):data%DIAG_perm(i+1)-1)
       END DO
       WRITE(6,*) ' row ', n, ' entries ',  data%ROW_perm(data%DIAG_perm(n):nz)
       stop
    END IF

      IF ( .FALSE. ) THEN
        data%GROUP( : n ) = 0
        DO i = 1, n - 1
         do j = data%DIAG_perm( i ),data%DIAG_perm(i+1)-1
           if ( data%GROUP( data%ROW_perm( j ) ) /= 0 ) THEN
             write(6,*) i, data%GROUP( data%ROW_perm( j ) ), j
             write(6,*)  data%ROW_perm( data%GROUP( data%ROW_perm( j ) ) ),    &
                         data%ROW_perm( j )
             WRITE(6,*) ' row ', i, ' entries ',                               &
               data%ROW_perm(data%DIAG_perm( i ):data%DIAG_perm(i+1)-1)
             stop
           end if
           data%GROUP( data%ROW_perm( j ) ) = j
         end do
         data%GROUP( data%ROW_perm(data%DIAG_perm( i ):                        &
                                   data%DIAG_perm(i+1)-1) ) = 0
         IF ( COUNT( data%ROW_perm(data%DIAG_perm( i ):                        &
                                   data%DIAG_perm(i+1)-1) < i ) > 0 ) then
           WRITE(6,*) ' row ', i, ' entries ',                                 &
             data%ROW_perm(data%DIAG_perm( i ):data%DIAG_perm(i+1)-1)
           stop
           end if  
        END DO
        IF ( COUNT( data%ROW_perm(data%DIAG_perm( n ):nz) < n ) > 0 ) then
        WRITE(6,*) ' row ', n, ' entries ',                                    &
            data%ROW_perm(data%DIAG_perm( n ): nz )
          stop
        end if
      END IF
!  ^                                                                           ^
!  |                                                                           |
!  --------------------------- END OF IGNORED ----------------------------------

!  ROW_perm is now equivalent to ROW for the permuted stucture, but the
!    row numbers are not reordered in the columns
!  DIAG_perm is equivalent to DIAG
!  OLD(i) gives the position in ROW of the i-th element of ROW_perm

!  consider the columns in the permuted order

      ihgst = n + 1 ; lwst = 0 ; ntst = 0
      data%np = 1
      data%GROUP( : n ) = 0
      data%PERM( : n ) = 0

!  start a new group

  300 CONTINUE
      nga = 0
      iact = lwst + 1

!  start a new column

  310 CONTINUE

!  has this column been included in some previous group already? If so, go to 
!  the next column

      IF ( data%GROUP( iact ) <= 0 ) THEN

!  scan the iact-th column

        IF ( iact < n ) THEN
          iend = data%DIAG_perm( iact + 1 ) - 1
        ELSE
          iend = nz
        END IF
        id2 = data%DIAG_perm( iact )
        DO i = id2, iend
          ir2 = data%ROW_perm( i )

!  has this row already been considered in the current group? If so, abandon 
!  the column

          IF ( data%PERM( ir2 ) > ntst ) THEN
            DO j = id2, i - 1
              data%PERM( data%ROW_perm( j ) ) = 0
            END DO
            GO TO 320
          END IF

!  mark the row temporarily

          data%PERM( ir2 ) = nga + 1 + ntst
        END DO

!  include the iact-th column in the np-th group

        nga = nga + 1
        data%GROUP( iact ) = data%np

!  if necessary, revise the interval lwst, ihgst of possibly unassigned columns

      END IF
      IF ( iact == lwst + 1 ) lwst = iact
      IF ( iact == ihgst - 1 ) ihgst = iact

!  consider the next column

  320 CONTINUE
      iact = iact + 1

!  was it the last column? If so, close the group

      IF ( iact < ihgst ) GO TO 310

!   close the np-th group. Compute the number of non assigned columns, and
!   stop if zero

      ntst = ntst + nga
      IF ( ntst < n ) THEN
        data%np = data%np + 1
        GO TO 300
      END IF

!     ************************************************
!     *                                              *
!     *   invert the permutation IWK(i) in PERM(i)   *
!     *                                              *
!     ************************************************

      DO i = 1, n
        data%PERM( data%IWK( i ) ) = i
      END DO

!  prepare to return

 900  CONTINUE
      IF ( inform%status == GALAHAD_ok ) THEN
        inform%products = data%np

!  allocate real workspace

        array_name = 'fdh: data%X'
        CALL SPACE_resize_array( n, data%X,                                    &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdh: data%G'
        CALL SPACE_resize_array( n, data%G,                                    &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        data%FDH_analyse_called = .TRUE.
        data%branch = 0

!  report error returns if required

      ELSE
        IF ( control%out > 0 .AND. control%print_level > 0 ) THEN
          IF ( LEN( TRIM( control%prefix ) ) > 2 )                             &
            prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
          WRITE( control%out, "( A, ' error in FDH_analyse, status = ',        &
         &  I0 ) " ) prefix, inform%status
          IF ( inform%status == GALAHAD_error_upper_entry ) THEN
            WRITE ( control%out, "( A, ' error in the',                        &
           &   ' data input for row/col ', I0 )" ) prefix, inform%bad_row
          ELSE IF ( inform%status == GALAHAD_error_restrictions ) THEN
            WRITE ( control%out, "( A, ' illegal values',                      &
           &   ' for n or nz = ', I0, 1X, I0 )" ) prefix, n, nz
          ELSE IF ( inform%status == GALAHAD_error_allocate ) THEN
            WRITE( control%out,                                                &
              "( A, ' Allocation error, for ', A, /, A, ' status = ', I0 ) " ) &
              prefix, inform%bad_alloc, inform%alloc_status
          END IF
        END IF
      END IF

      RETURN

!  End of subroutine FDH_analyse

      END SUBROUTINE FDH_analyse

!-*-*-*-  G A L A H A D -  F D H _ e s t i m a t e  S U B R O U T I N E -*-*-*-

      SUBROUTINE FDH_estimate( n, nz, ROW, DIAG, X, G, STEPSIZE, H,            &
                               data, control, inform, userdata, eval_G )

!     **************************************************************
!     *                                                            *
!     *   Estimation of a sparse Hessian matrix by differences     *
!     *                                                            *
!     *   Method of lower triangular substitution                  *
!     *                                                            *
!     *   See M.J.D. Powell and Ph.L. Toint, "On the estimation    *
!     *    of sparse Hessian matrices", SIAM J. Numer. Analysis    *
!     *    Vol. 16, No. 6, (1979) pp. 1060–1074                    *
!     *                                                            *
!     **************************************************************
!
!   The Hessian stucture given by n, nz, ROW and DIAG is described in 
!   FDH_analyse and should not have been changed since the last call to
!   FDH_analyse. Additional arguments are

!       X(i) is the actual value of the i-th variable.(i=1,n)
!       G(i) is the i-th component of the gradient at X (i=1,n)
!       STEPSIZE(i) is the difference step in the i-th variable.(i=1,n)
!       H(i) is the i-th nonzero in the estimated Hessian matrix.(i=1,nz)

!   The analysed and permuted structure and the groups are stored in the
!   derived type data (see preface)
 
!   Action of the subroutine is controlled by components of the derived type
!   control, while information about the progress of the subroutine is reported
!   in inform (again, see preface). Success or failure is flagged by the 
!   component inform%status -
!     0 if no error was detected
!     1 The user should compute the gradient of the objective function 
!       nabla_x f(x) at the point x indicated in data%%X  and then re-enter the 
!       subroutine. The value of the i-th component of the gradient should be 
!       set in data%%G(i), for i = 1, ..., n and data%eval_status should be set
!       to 0. If the user is unable to evaluate a component of nabla_x f(x) 
!       - for instance if a component of the gradient is undefined at x - the
!       user need not set data%G, but should then set data%eval_status to a 
!       non-zero value
!    -3 invalid values input for n or nz
!   -23 if there was an error in the inform%bad_row-th row or column
!   -31 if the call to FDH_estimate was not preceded by a call to FDH_analyse

!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient 
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in 
!   G, and the status variable set to 0. If the evaluation is impossible at X, 
!   status should be set to a nonzero value. If eval_G is not present, 
!   TRB_solve will return to the user with inform%status = 3 each time an 
!   evaluation is required.

!---------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

      INTEGER, INTENT( IN ) :: n, nz
      INTEGER, INTENT( IN ), DIMENSION( n ) :: DIAG
      INTEGER, INTENT( IN ), DIMENSION( nz ) :: ROW
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, G, STEPSIZE
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nz ) :: H
      TYPE ( FDH_control_type ), INTENT( IN ) :: control
      TYPE ( FDH_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( FDH_data_type ), INTENT( INOUT ) :: data
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
      OPTIONAL :: eval_G

!----------------------------------
!   I n t e r f a c e   B l o c k s 
!----------------------------------

      INTERFACE
        SUBROUTINE eval_G( status, X, userdata, G )
        USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        INTEGER, INTENT( OUT ) :: status
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
        TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
        END SUBROUTINE eval_G
      END INTERFACE

!   Programming: F77 version by Philippe Toint (1980) with mods by 
!   Iain Duff (1980) and rewrite in F2003 by Nick Gould (2012)

!---------------------------------
!   L o c a l   V a r i a b l e s
!---------------------------------

      INTEGER :: i, j, ic2, ic21, iej, ii, iopa, ip, ip1, iz, lp1
      INTEGER :: ipath, ipos1, iposs, ir2, ir21, row_permi
      REAL ( KIND = wp ) :: ct
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix

!  branch to different sections of the code depending on input status

      IF ( data%branch /= 0 ) GO TO 100

!  test for an error in the input data

      IF ( nz < n .OR. n <= 0 .OR. nz > ( n * ( n + 1 ) ) / 2 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 900
      ELSE IF ( .NOT. data%FDH_analyse_called ) THEN
        inform%status = GALAHAD_error_call_order ; GO TO 900
      END IF

!   *******************************************************
!   *                                                     *
!   *   loop on the groups to evaluate the differences    *
!   *                                                     *
!   *******************************************************

!  test for band structure

      IF ( data%DIAG_perm( 1 ) <= 0 ) THEN
        data%istop = n / 2
        data%ibnd = DIAG( 2 ) - 1
      ELSE
        data%ibnd = 0
      END IF

!  start the loop on the groups

      data%ng = 0
  10  CONTINUE  
        data%ng = data%ng + 1

!  build the displacement vector in data%X

        DO i = 1, n
          ip = data%PERM( i )
          IF ( data%GROUP( i ) == data%ng ) THEN
            data%X( ip ) = X( ip ) + STEPSIZE( ip )
          ELSE
            data%X( ip ) = X( ip )
          END IF
        END DO

!  evaluate the perturbed gradient in data%G

        IF ( PRESENT( eval_G ) ) THEN
           CALL eval_G( data%eval_status, data%X, userdata, data%G )
        ELSE
          data%branch = 1 ; inform%status = 1 ; RETURN
        END IF

!  evaluate the difference in gradient and store it in H

 100    CONTINUE  
        DO i = 1, n
          IF ( data%ibnd == 0 ) THEN
            IF ( data%GROUP( i ) /= data%ng ) CYCLE
            IF ( i < n ) THEN
              iej = data%DIAG_perm( i + 1 ) - 1
            ELSE
              iej = nz
            END IF
            DO j = data%DIAG_perm( i ), iej
              lp1 = data%PERM( data%ROW_perm( j ) )
              H( data%OLD( j ) ) = data%G( lp1 ) - G( lp1 )
            END DO
          ELSE

!  do the same in the case of a band structure

            IF ( i >= data%istop ) THEN

!  lower half of the structure

              IF ( data%GROUP( i ) /= data%ng ) CYCLE
              IF ( i < n ) THEN
                iej = DIAG( i + 1 ) - 1
              ELSE
                iej = nz
              END IF
              DO j = DIAG( i ), iej
                lp1 = ROW( j )
                H( j ) = data%G( lp1 ) - G( lp1 )
              END DO

!  upper half

            ELSE
              ii = MOD( data%ibnd - MOD( i, data%ibnd ) + data%ng, data%ibnd )
              H( DIAG( i ) + ii ) = data%G( i ) - G( i )
            END IF
          END IF
        END DO

!  end of the loop on the groups

      IF ( data%ng < data%np ) GO TO 10
      inform%status = GALAHAD_ok

!   *********************************************************
!   *                                                       *
!   *   substitution to obtain an estimate of the hessian   *
!   *                                                       *
!   *********************************************************

!  test for band substitution

      IF ( data%ibnd == 0 ) THEN

!  loop to define the path of evaluation in the matrix H(i,j) * H(j)

        ic2 = n
        DO ip1 = 1, nz
          ipath = nz - ip1 + 1
          IF ( ic2 /= 1 ) THEN
            IF ( ipath + 1 == data%DIAG_perm( ic2 ) ) ic2 = ic2 - 1

!  find the group of the column under consideration

            data%ng = data%GROUP( ic2 )
          END IF

!  Find the correction term for the difference by scanning the complementary 
!  row (half-column)

          ir2 = data%ROW_perm( ipath )
          ir21 = ir2 + 1
          iopa = data%OLD( ipath )
          IF ( ic2 /= ir2 .AND. ir2 /= n ) THEN
            IF ( data%DIAG_perm( ir2 ) + 1 /= data%DIAG_perm( ir21 ) ) THEN
              ct = 0.0_wp
              DO i = data%DIAG_perm( ir2 ), data%DIAG_perm( ir21 ) - 1
                row_permi = data%ROW_perm( i )
                IF ( data%GROUP( row_permi ) == data%ng .AND.                  &
                     data%ROW_perm( i ) /= ir2 ) THEN
                  ct = ct                                                      &
                    + H( data%OLD( i ) ) * STEPSIZE( data%PERM( row_permi ) )
                END IF
              END DO

!  obtain the correct term H(i,j) * H(j) by substracting the correction

              H( iopa ) = H( iopa ) - ct
            END IF
          END IF
          H( iopa ) = H( iopa ) / STEPSIZE( data%PERM( ic2 ) )
        END DO

!   *************************************************
!   *                                               *
!   *    special substitution for band structure    *
!   *                                               *
!   *************************************************

!  start the backward loop to define the matrix H(i,j) * H(j)

      ELSE
        ic2 = n
        DO ip1 = 1, nz
          ipath = nz - ip1 + 1
          IF ( ipath + 1 == DIAG( ic2 ) ) ic2 = ic2 - 1
          IF ( ic2 < data%istop ) EXIT

!  find the correction term for the difference by scanning the complementary row

          ir2 = ROW( ipath )

!  no corrections required for diagonal or last row and column

          IF ( ir2 /= n .AND. ic2 /= ir2 ) THEN
            iposs = DIAG( ir2 ) + data%ibnd - ir2 + ic2
            ir21 = ir2 + 1

!  no corrections required for entries H(n-1,(n-ibnd+1)..n-1), ... ,
!  H(n-2,(n-ibnd)..n-2) etc

            IF ( iposs < DIAG( ir21 ) ) THEN
              H( ipath ) = H( ipath ) - H( iposs ) * STEPSIZE( ROW( iposs ) )
            END IF
          END IF
          H( ipath ) = H( ipath ) / STEPSIZE( ic2 )
        END DO

!  start the forward loop to define the same matrix

        ic2 = 1
        DO ipath = 1, nz
          ic21 = ic2 + 1
          IF ( ipath == DIAG( ic21 ) ) ic2 = ic21
          IF ( ic2 >= data%istop ) EXIT

!  find again the correction term for the difference

          ir2 = ROW( ipath )
          IF ( ic2 /= 1 .AND. ic2 /= ir2 ) THEN
            iz = data%ibnd - ir2 + ic2
            IF ( ic2 > iz ) THEN
              ipos1 = ic2 - iz
              iposs = DIAG( ipos1 ) + iz
              H( ipath ) = H( ipath ) - H( iposs ) * STEPSIZE( ipos1 )
            END IF
          END IF
          H( ipath ) = H( ipath ) / STEPSIZE( ir2 )
        END DO
      END IF

!  prepare to return

 900  CONTINUE
      data%branch = 0

!  report error returns if required

      IF ( inform%status /= 0 ) THEN
        IF( control%out > 0 .AND. control%print_level > 0 ) THEN
          IF ( LEN( TRIM( control%prefix ) ) > 2 )                             &
            prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
          WRITE( control%out, "( A, ' error in FDH_estimate, status = ',       &
        &        I0 )" ) prefix, inform%status
          IF ( inform%status == GALAHAD_error_restrictions ) THEN
           WRITE ( control%out, "( A, ' illegal',                              &
         &   ' values for n or nz = ', I0, 1X, I0 )" ) prefix, n, nz
          ELSE IF ( inform%status == GALAHAD_error_call_order ) THEN
           WRITE ( control%out, "( A, ' call to FDH_estimate',                 &
         &  ' must be preceded by a call to FDH_analyse' )" ) prefix
          ELSE IF ( inform%status == GALAHAD_error_allocate ) THEN
            WRITE( control%out,                                                &
              "( A, ' Allocation error, for ', A, /, A, ' status = ', I0 ) " ) &
              prefix, inform%bad_alloc, inform%alloc_status
          END IF
        END IF
      END IF

      RETURN

!  End of subroutine FDH_estimate

      END SUBROUTINE FDH_estimate

!-*-*-  G A L A H A D -  F D H _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE FDH_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( FDH_data_type ), INTENT( INOUT ) :: data
     TYPE ( FDH_control_type ), INTENT( IN ) :: control
     TYPE ( FDH_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'fdh: data%DIAG_perm'
     CALL SPACE_dealloc_array( data%DIAG_perm,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%ROW_perm'
     CALL SPACE_dealloc_array( data%ROW_perm,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%OLD'
     CALL SPACE_dealloc_array( data%OLD,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%PERM'
     CALL SPACE_dealloc_array( data%PERM,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%GROUP'
     CALL SPACE_dealloc_array( data%GROUP,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%IWK'
     CALL SPACE_dealloc_array( data%IWK,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%X'
     CALL SPACE_dealloc_array( data%X,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'fdh: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  re-initial private data

     data%FDH_analyse_called = .FALSE.
     data%branch = 0

     RETURN

!  End of subroutine FDH_terminate

     END SUBROUTINE FDH_terminate

!  End of module GALAHAD_FDH

   END MODULE GALAHAD_FDH_double


