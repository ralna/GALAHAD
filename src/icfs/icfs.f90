! THIS VERSION: GALAHAD 4.1 - 2022-11-29 AT 08:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ I C F S   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  BSD(3) License:

!  Copyright (c) 1998, Chih-Jen Lin and Jorge J. More'.
!  All rights reserved.

!  Redistribution and use in source and binary forms, with or without
!  modification, are permitted provided that the following conditions
!  are met:

!  1. Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.

!  2. Redistributions in binary form must reproduce the above copyright
!     notice, this list of conditions and the following disclaimer in
!     the documentation and/or other materials provided with the
!     distribution.

!  3. Neither the name of the copyright holder nor the names of its
!     & contributors may be used to endorse or promote products derived
!     & from this software without specific prior written permission.

!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
!  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
!  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
!  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
!  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
!  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
!  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
!  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
!  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
!  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

!  Principal authors: Chih-Jen Lin and Jorge J. More'
!  Enhanced for modern fortran: Nick Gould

!  History -
!   Released as part of MINPACK 2, May 1998
!   Incorporated into GALAHAD Version 4.1, by permission, November 29th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_ICFS_double

!      --------------------------------------------------
!     | Given a symmetric matrix A, compute and apply an |
!     | incomplete Cholesky factorization preconditioner |
!      --------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: ICFS_initialize, ICFS_terminate, ICFS_read_specfile,           &
                ICFS_factorize, ICFS_triangular_solve, DICFS, DSTRSOL,         &
                ICFS_full_initialize, ICFS_full_terminate

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: nbmax = 3
      INTEGER, PARAMETER :: insortf = 20
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: alpham = 0.001_wp
      REAL ( KIND = wp ), PARAMETER :: nbfactor = 512.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ICFS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  number of extra vectors of length n required by the incomplete Cholesky 
!  factorization

        INTEGER :: icfs_vectors = 10

!  an initial "guess" of the shift

        REAL ( KIND = wp ) :: shift = zero

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

      END TYPE ICFS_control_type
!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ICFS_time_type

!  total time

       REAL :: total = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0

!  total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time for the factorization phase

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time for the linear solution phase

       REAL ( KIND = wp ) :: clock_solve = 0.0

      END TYPE ICFS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ICFS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)

       INTEGER :: status = 0

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  the actual value of the shift used

        REAL ( KIND = wp ) :: shift

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( ICFS_time_type ) :: time

      END TYPE ICFS_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: ICFS_data_type
        INTEGER :: p, L_ne
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: L_ptr
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: L_row
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IWA
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: L_val
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: L_diag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WA1
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WA2
      END TYPE ICFS_data_type

!  - - - - - - - - - - - - - - - - - - -
!  The ICFS_full_data_type derived type
!  - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: ICFS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( ICFS_data_type ) :: ICFS_data
        TYPE ( ICFS_control_type ) :: ICFS_control
        TYPE ( ICFS_inform_type ) :: ICFS_inform
      END TYPE ICFS_full_data_type

  CONTAINS

!-*-*-*-*-*-   I C F S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE ICFS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for ICFS. This routine should be called before
!  ICFS_form_and_factorize
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are
!           described above
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( ICFS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ICFS_control_type ), INTENT( OUT ) :: control
     TYPE ( ICFS_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

     RETURN

!  End of ICFS_initialize

     END SUBROUTINE ICFS_initialize

!- G A L A H A D -  I C F S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE ICFS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for ICFS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ICFS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( ICFS_control_type ), INTENT( OUT ) :: control
     TYPE ( ICFS_inform_type ), INTENT( OUT ) :: inform

     CALL ICFS_initialize( data%icfs_data, control, inform )

     RETURN

!  End of subroutine ICFS_full_initialize

     END SUBROUTINE ICFS_full_initialize

!-*-*-*-   I C F S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE ICFS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given could (roughly)
!  have been set as:

! BEGIN ICFS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  number-of-icfs-vectors                            10
!  initial-shift                                     0.0
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END ICFS SPECIFICATIONS

!  Dummy arguments

     TYPE ( ICFS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: icfs_vectors = print_level + 1
     INTEGER, PARAMETER :: shift = icfs_vectors + 1
     INTEGER, PARAMETER :: space_critical = shift + 1
     INTEGER, PARAMETER :: deallocate_error_fatal  = space_critical + 1
     INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'ICFS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( icfs_vectors )%keyword = 'number-of-icfs-vectors'

!  Real key-words

     spec( shift )%keyword = 'initial-shift'

!  Logical key-words

     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal  )%keyword = 'deallocate-error-fatal'

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
      CALL SPECFILE_assign_value( spec( icfs_vectors ),                        &
                                  control%icfs_vectors,                        &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( shift ),                               &
                                  control%shift,                               &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal  ),             &
                                  control%deallocate_error_fatal ,             &
                                  control%error )

!  Set character values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

      RETURN

!  End of ICFS_read_specfile

      END SUBROUTINE ICFS_read_specfile

!-*-*-*-*-*-*-   I C F S _ F A C T O R I Z E  S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE ICFS_factorize( n, ROW, PTR, DIAG, VAL, data, control, inform )

!  form an incomplete Cholesky facorization LL^T of A

!  n is an integer variable that gives the order of A (number of rows/columns)
!
!  PTR is an integer array of dimension n + 1, whose j-th component gives the
!   starting address for list of nonzero values and their corresponding row 
!   indices in column j of the strict lower triangular part of A. That is, 
!   the nonzeros in column j of the strict lower triangle of A must be in 
!   positions
!            acol_ptr(j), ... , acol_ptr(j+1) - 1.
!   Note that PTR(n+1) points to the first position beyond that needed to
!   store A
!
!  ROW is an integer array of dimension at least PTR(n+1)-1 that contains the
!   row indices of the strict lower triangular part of A in the compressed 
!   column storage format
!
!  DIAG is a real array of dimension at least n whose j-th component 
!   contains the value of the j-th diagonal of A
!
!  VAL is a real array of dimension at least PTR(n+1)-1 that contains the
!   values of the strict lower triangular part of A in the compressed 
!   column storage format, in the same order as ROW
!
!  data is a structure of type ICFS_data_type which holds private internal data
!
!  control is a structure of type ICFS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to ICFS_initialize. See ICFS_initialize
!   for details
!
!  inform is a structure of type ICFS_inform_type that provides information on
!   exit from ICFS_form_and_factorize. The component status has possible values:
!
!     0 Normal termination.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!        A%n >=  1

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: PTR
      INTEGER, INTENT( IN ), DIMENSION( PTR( n + 1 ) - 1 ) :: ROW
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: DIAG
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( PTR( n + 1 ) - 1 ) :: VAL
      TYPE ( ICFS_data_type ), INTENT( INOUT ) :: data
      TYPE ( ICFS_control_type ), INTENT( IN ) :: control
      TYPE ( ICFS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: ne, icfs_vectors
      CHARACTER ( LEN = 80 ) :: array_name
      REAL :: time_now, time_start
      REAL ( KIND = wp ) :: alpha, clock_now, clock_start

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )
      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' Entered ICFS_factorize' )" ) prefix

      IF ( n <= 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; GO TO 910
      END IF

!  specify the amount of storage needed for the incomplete Cholesky factor L

      ne = PTR( n + 1 ) - 1
      icfs_vectors = MAX( control%icfs_vectors, 0 )
      data%L_ne = ne + n * icfs_vectors

!  set up the storage for the factors

      array_name = 'icfs: data%L_ptr'
      CALL SPACE_resize_array( n + 1, data%L_ptr,                              &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'icfs: data%L_row'
      CALL SPACE_resize_array( data%L_ne, data%L_row,                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'icfs: data%L_diag'
      CALL SPACE_resize_array( n, data%L_diag,                                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'icfs: data%L_val'
      CALL SPACE_resize_array( data%L_ne, data%L_val,                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  set up workspace

      array_name = 'icfs: data%WA1'
      CALL SPACE_resize_array( n, data%WA1,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'icfs: data%WA2'
      CALL SPACE_resize_array( n, data%WA2,                                    &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'icfs: data%IWA'
      CALL SPACE_resize_array( 3 * n, data%IWA,                                &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  form the factors

      alpha = control%shift
      CALL dicfs( n, ne, VAL, DIAG, PTR, ROW, data%L_val, data%L_diag,         &
                  data%L_ptr, data%L_row, icfs_vectors, alpha,                 &
                  data%iwa, data%wa1, data%wa2 )
      inform%shift = alpha

!  Record the time taken to form the incomplete factors

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_start
      inform%time%clock_factorize                                              &
        = inform%time%clock_factorize + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' Leaving ICFS_factorize' )" ) prefix
      inform%status = GALAHAD_ok
      RETURN

!  Allocation error

  910 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' Leaving ICFS_factorize' )" ) prefix
      RETURN

!  End of ICFS_factorize

      END SUBROUTINE ICFS_factorize

!-*-*-*-*-*-*-*-*-   I C F S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE ICFS_triangular_solve( n, R, transpose, data, control, inform )

!  Solve L x = r or L^T x = r involving the incomplete Cholesky factors L

!  n is an integer variable that gives the order of A (number of rows/columns)
!
!  R is a real array of dimension at least n whose j-th component on input 
!   contains the value of the j-th component of the vector r, and that will
!   be replaced on output by that of the j-th component of the solution x
!
!  transpose is a logical variable that should be set .TRUE. if the
!  solution to L^T x = r is sought and .FALSE. if that of L x = r is required

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      LOGICAL, INTENT( IN ) :: transpose
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: r
      TYPE ( ICFS_data_type ), INTENT( IN ) :: data
      TYPE ( ICFS_control_type ), INTENT( IN ) :: control
      TYPE ( ICFS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 60 ) :: task = REPEAT( ' ', 60 )
      REAL :: time_now, time_start
      REAL ( KIND = wp ) :: clock_now, clock_start

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )
      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' Entered ICFS_triangular_solve' )" ) prefix

      IF ( transpose ) THEN
        task( 1 : 1 ) = 'T'
      ELSE
        task( 1 : 1 ) = 'N'
      END IF

      CALL dstrsol( n, data%L_val, data%L_diag, data%L_ptr, data%L_row,        &
                    R, task )

!  Record the time taken to form the incomplete factors

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%solve = inform%time%solve + time_now - time_start
      inform%time%clock_solve                                                  &
        = inform%time%clock_solve + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' Leaving ICFS_triangular_solve' )" ) prefix
      inform%status = GALAHAD_ok

!  End of ICFS_triangular_solve

      END SUBROUTINE ICFS_triangular_solve

!-*-*-*-*-*-   I C F S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE ICFS_terminate( data, control, inform )

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
!   data    see Subroutine ICFS_initialize
!   control see Subroutine ICFS_initialize
!   inform  see Subroutine ICFS_form_and_factorize

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( ICFS_control_type ), INTENT( IN ) :: control
      TYPE ( ICFS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( ICFS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all allocated arrays

      array_name = 'icfs: data%L_ptr'
      CALL SPACE_dealloc_array( data%L_ptr,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%L_row'
      CALL SPACE_dealloc_array( data%L_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%L_val'
      CALL SPACE_dealloc_array( data%L_val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%L_diag'
      CALL SPACE_dealloc_array( data%L_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%IWA'
      CALL SPACE_dealloc_array( data%IWA,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%WA1'
      CALL SPACE_dealloc_array( data%WA1,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'icfs: data%WA2'
      CALL SPACE_dealloc_array( data%WA2,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

!  End of subroutine ICFS_terminate

      END SUBROUTINE ICFS_terminate

! -  G A L A H A D -  I C F S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE ICFS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ICFS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( ICFS_control_type ), INTENT( IN ) :: control
     TYPE ( ICFS_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

     CALL ICFS_terminate( data%icfs_data, control, inform )
     RETURN

!  End of subroutine ICFS_full_terminate

     END SUBROUTINE ICFS_full_terminate

!-*-*-*-*-*-*-  O R I G I N A L   I C F S   S U B R O U T I N E S   -*-*-*-*-*-

      SUBROUTINE dicfs( n, nnz, a, adiag, acol_ptr, arow_ind, l, ldiag,        &
                        lcol_ptr, lrow_ind, p, alpha, iwa, wa1, wa2 )
      INTEGER :: n, nnz, p
      REAL ( KIND = wp ) :: alpha
      INTEGER, DIMENSION( n + 1 ) :: acol_ptr
      INTEGER, DIMENSION( nnz ) :: arow_ind
      INTEGER, DIMENSION( n + 1 ) :: lcol_ptr
      INTEGER, DIMENSION( nnz + n * p ) :: lrow_ind
      INTEGER, DIMENSION( 3 * n ) :: iwa
      REAL ( KIND = wp ), DIMENSION( n ) :: wa1
      REAL ( KIND = wp ), DIMENSION( n ) :: wa2
      REAL ( KIND = wp ), DIMENSION( nnz ) :: a
      REAL ( KIND = wp ), DIMENSION( n ) :: adiag
      REAL ( KIND = wp ), DIMENSION( nnz + n * p ) :: l
      REAL ( KIND = wp ), DIMENSION( n ) :: ldiag

!     *********
!
!     Subroutine dicfs
!
!     Given a symmetric matrix A in compressed column storage, this
!     subroutine computes an incomplete Cholesky factor of A + alpha*D,
!     where alpha is a shift and D is the diagonal matrix with entries
!     set to the l2 norms of the columns of A.
!
!     The subroutine statement is
!
!       subroutine dicfs(n,nnz,a,adiag,acol_ptr,arow_ind,l,ldiag,lcol_ptr,
!                        lrow_ind,p,alpha,iwa,wa1,wa2)
!
!     where
!
!       n is an integer variable.
!         On entry n is the order of A.
!         On exit n is unchanged.
!
!       nnz is an integer variable.
!         On entry nnz is the number of nonzeros in the strict lower
!            triangular part of A.
!         On exit nnz is unchanged.
!
!       a is a real array of dimension nnz.
!         On entry a must contain the strict lower triangular part
!            of A in compressed column storage.
!         On exit a is unchanged.
!
!       adiag is a real array of dimension n.
!         On entry adiag must contain the diagonal elements of A.
!         On exit adiag is unchanged.
!
!       acol_ptr is an integer array of dimension n + 1.
!         On entry acol_ptr must contain pointers to the columns of A.
!            The nonzeros in column j of A must be in positions
!            acol_ptr(j), ... , acol_ptr(j+1) - 1.
!         On exit acol_ptr is unchanged.
!
!       arow_ind is an integer array of dimension nnz.
!         On entry arow_ind must contain row indices for the strict 
!            lower triangular part of A in compressed column storage.
!         On exit arow_ind is unchanged.
!
!       l is a real array of dimension nnz+n*p.
!         On entry l need not be specified.
!         On exit l contains the strict lower triangular part
!            of L in compressed column storage.
!
!       ldiag is a real array of dimension n.
!         On entry ldiag need not be specified.
!         On exit ldiag contains the diagonal elements of L.
!
!       lcol_ptr is an integer array of dimension n + 1.
!         On entry lcol_ptr need not be specified.
!         On exit lcol_ptr contains pointers to the columns of L.
!            The nonzeros in column j of L are in the
!            lcol_ptr(j), ... , lcol_ptr(j+1) - 1 positions of l.
!
!       lrow_ind is an integer array of dimension nnz+n*p.
!         On entry lrow_ind need not be specified.
!         On exit lrow_ind contains row indices for the strict lower
!            triangular part of L in compressed column storage. 
!
!       p is an integer variable.
!         On entry p specifes the amount of memory available for the
!            incomplete Cholesky factorization.
!         On exit p is unchanged.
!
!       alpha is a real variable.
!         On entry alpha is the initial guess of the shift.
!         On exit alpha is final shift
!
!       iwa is an integer work array of dimension 3*n.
!
!       wa1 is a real work array of dimension n.
!
!       wa2 is a real work array of dimension n.
!
!     MINPACK-2 Project. October 1998.
!     Argonne National Laboratory.
!     Chih-Jen Lin and Jorge J. More'.
!
!     **********

      INTEGER :: i, info, j, k, nb
      REAL ( KIND = wp ) :: alphas

!     Compute the l2 norms of the columns of A.

      do i = 1, n
         wa1(i) = adiag(i)**2
      end do
      do j = 1, n
         do i = acol_ptr(j), acol_ptr(j+1)-1
            k = arow_ind(i)
            wa1(j) = wa1(j) + a(i)**2
            wa1(k) = wa1(k) + a(i)**2
         end do
      end do
      do j = 1, n
         wa1(j) = sqrt(wa1(j))
      end do

!     Compute the scaling matrix D.

      do i = 1, n
         if (wa1(i) > zero) then
            wa2(i) = one/sqrt(wa1(i))
         else
            wa2(i) = one
         endif
      end do

!     Determine a lower bound for the step. 

      if (alpha <= zero) then
         alphas = alpham
      else
         alphas = alpha
      end if

!     Compute the initial shift.

      alpha = zero
      do i = 1, n
         if (adiag(i) == zero) then
            alpha = alphas
         else
            alpha = max(alpha,-adiag(i)*(wa2(i)**2))
         end if
      end do
      if (alpha > zero) alpha = max(alpha,alphas)

!     Search for an acceptable shift. During the search we decrease 
!     the lower bound alphas until we determine a lower bound that 
!     is not acceptable. We then increase the shift.
!     The lower bound is decreased by nbfactor at most nbmax times.

      nb = 1
      do while (1 == 1)

!        Copy the sparsity structure of A into L.

         do i = 1, n+1
            lcol_ptr(i) = acol_ptr(i)
         end do
         do i = 1, nnz
            lrow_ind(i) = arow_ind(i)
         end do

!        Scale A and store in the lower triangular matrix L.

         do j = 1, n
            ldiag(j) = adiag(j)*(wa2(j)**2) + alpha
         end do
         do j = 1, n
            do i = acol_ptr(j), acol_ptr(j+1)-1
               l(i) = a(i)*wa2(j)*wa2(arow_ind(i))
            end do
         end do

!        Attempt an incomplete factorization.

         call dicf(n,nnz,l,ldiag,lcol_ptr,lrow_ind,p,info,                     &
                   iwa(1),iwa(n+1),iwa(2*n+1),wa1)

!        If the factorization exists, then test for termination.
!        Otherwise increment the shift.

         if (info >= 0) then

!           If the shift is at the lower bound, reduce the shift.
!           Otherwise undo the scaling of L and exit.

            if (alpha == alphas .and. nb < nbmax) then
               alphas = alphas/nbfactor
               alpha = alphas
               nb = nb + 1
            else
               do i = 1, n
                  ldiag(i) = ldiag(i)/wa2(i)
               end do
               do j = 1, lcol_ptr(n+1)-1
                  l(j) = l(j)/wa2(lrow_ind(j))
               end do
               return
            end if
         else
            alpha = max(two*alpha,alphas)
         end if
      end do

      RETURN

!  end of  subroutine dicfs

      END SUBROUTINE dicfs
 
      SUBROUTINE dstrsol( n, l, ldiag, jptr, indr, r, task )
      INTEGER :: n
      CHARACTER ( LEN = 60 ) :: task
      INTEGER, DIMENSION( n + 1 ) :: jptr
      INTEGER, DIMENSION( * ) :: indr
      REAL ( KIND = wp ), DIMENSION( * ) :: l
      REAL ( KIND = wp ), DIMENSION( n ) :: ldiag
      REAL ( KIND = wp ), DIMENSION( n ) :: r

!     **********
!
!     Subroutine dstrsol
!
!     This subroutine solves the triangular systems L*x = r or L'*x = r.
!
!     The subroutine statement is
!
!       subroutine dstrsol(n,l,ldiag,jptr,indr,r,task)
!
!     where
!
!       n is an integer variable.
!         On entry n is the order of L.
!         On exit n is unchanged.
!
!       l is a real array of dimension *.
!         On entry l must contain the nonzeros in the strict lower
!            triangular part of L in compressed column storage.
!         On exit l is unchanged.
!
!       ldiag is a real array of dimension n.
!         On entry ldiag must contain the diagonal elements of L.
!         On exit ldiag is unchanged.
!
!       jptr is an integer array of dimension n + 1.
!         On entry jptr must contain pointers to the columns of A.
!            The nonzeros in column j of A must be in positions
!            jptr(j), ... , jptr(j+1) - 1.
!         On exit jptr is unchanged.
!
!       indr is an integer array of dimension *.
!         On entry indr must contain row indices for the strict 
!            lower triangular part of L in compressed column storage.
!         On exit indr is unchanged.
!
!       r is a real array of dimension n.
!         On entry r must contain the vector r.
!         On exit r contains the solution vector x.
!
!       task is a character variable of length 60.
!         On entry 
!            task(1:1) = 'N' if we need to solve L*x = r
!            task(1:1) = 'T' if we need to solve L'*x = r
!         On exit task is unchanged.
!
!     MINPACK-2 Project. May 1998.
!     Argonne National Laboratory.
!
!     **********

      INTEGER :: j, k
      REAL ( KIND = wp ) :: temp

!     Solve L*x =r and store the result in r.

      if (task(1:1) == 'N') then

         do j = 1, n
            temp = r(j)/ldiag(j)
            do k = jptr(j), jptr(j+1) - 1
               r(indr(k)) = r(indr(k)) - l(k)*temp
            end do
            r(j) = temp
         end do

         return

      end if

!     Solve L'*x =r and store the result in r.

      if (task(1:1) == 'T') then

         r(n) = r(n)/ldiag(n)
         do j = n - 1, 1, -1
            temp = zero
            do k = jptr(j), jptr(j+1) - 1
               temp = temp + l(k)*r(indr(k))
            end do
            r(j) = (r(j) - temp)/ldiag(j)
         end do

         return

      end if

!  end of subroutine dstrsol 

      END SUBROUTINE dstrsol 
 
      SUBROUTINE dicf( n, nnz, a, diag, col_ptr, row_ind, p, info,             &
                       indr, indf, list, w )
      INTEGER :: n, nnz, p, info
      INTEGER, DIMENSION( n + 1 ) :: col_ptr
      INTEGER, DIMENSION( * ) :: row_ind
      INTEGER, DIMENSION( n ) :: indr
      INTEGER, DIMENSION( n ) :: indf
      INTEGER, DIMENSION( n ) :: list
      REAL ( KIND = wp ), DIMENSION( * ) :: a
      REAL ( KIND = wp ), DIMENSION( n )  :: diag
      REAL ( KIND = wp ), DIMENSION( n )  :: w

!     *********
!     
!     Subroutine dicf
!     
!     Given a sparse symmetric matrix A in compressed row storage,
!     this subroutine computes an incomplete Cholesky factorization.
!
!     Implementation of dicf is based on the Jones-Plassmann code.
!     Arrays indf and list define the data structure.
!     At the beginning of the computation of the j-th column,
!
!       For k < j, indf(k) is the index of a for the first 
!       nonzero l(i,k) in the k-th column with i >= j.
!
!       For k < j, list(i) is a pointer to a linked list of column 
!       indices k with i = row_ind(indf(k)).
!
!     For the computation of the j-th column, the array indr records
!     the row indices. Hence, if nlj is the number of nonzeros in the 
!     j-th column, then indr(1),...,indr(nlj) are the row indices. 
!     Also, for i > j, indf(i) marks the row indices in the j-th  
!     column so that indf(i) = 1 if l(i,j) is not zero.
!
!     The subroutine statement is
!
!       subroutine dicf(n,nnz,a,diag,col_ptr,row_ind,p,info,
!                       indr,indf,list,w)
!
!     where
!
!       n is an integer variable.
!         On entry n is the order of A.
!         On exit n is unchanged.
!
!       nnz is an integer variable.
!         On entry nnz is the number of nonzeros in the strict lower
!            triangular part of A.
!         On exit nnz is unchanged.
!
!       a is a real array of dimension nnz+n*p.
!         On entry the first nnz entries of a must contain the strict 
!            lower triangular part of A in compressed column storage.
!         On exit a contains the strict lower triangular part
!            of L in compressed column storage.
!
!       diag is a real array of dimension n.
!         On entry diag must contain the diagonal elements of A.
!         On exit diag contains the diagonal elements of L.
!
!       col_ptr is an integer array of dimension n + 1.
!         On entry col_ptr must contain pointers to the columns of A.
!            The nonzeros in column j of A must be in positions
!            col_ptr(j), ... , col_ptr(j+1) - 1.
!         On exit col_ptr contains pointers to the columns of L.
!            The nonzeros in column j of L are in the
!            col_ptr(j), ... , col_ptr(j+1) - 1 positions of l.
!
!       row_ind is an integer array of dimension nnz+n*p.
!         On entry row_ind must contain row indices for the strict 
!            lower triangular part of A in compressed column storage.
!         On exit row_ind contains row indices for the strict lower
!            triangular part of L in compressed column storage. 
!
!       p is an integer variable.
!         On entry p specifes the amount of memory available for the
!            incomplete Cholesky factorization.
!         On exit p is unchanged.
!
!       info is an integer variable.
!         On entry info need not be specified.
!         On exit info = 0 if the factorization succeeds, and
!            info < 0 if the -info pivot is not positive.
!
!       indr is an integer work array of dimension n.
!     
!       indf is an integer work array of dimension n.
!
!       list is an integer work array of dimension n.
!
!       w is a real work array of dimension n.
!
!     MINPACK-2 Project. May 1998.
!     Argonne National Laboratory.
!     Chih-Jen Lin and Jorge J. More'.
!
!     **********

      INTEGER :: i, ip, j, k, kth, nlj, newk, np, mlj
      INTEGER :: isj, iej, isk, iek, newisj, newiej
      REAL ( KIND = wp ) :: lval

      info = 0
      do j = 1, n
         indf(j) = 0
         list(j) = 0
      end do

!     Make room for L by moving A to the last n*p positions in a.

      np = n*p
      do  j = 1, n + 1
         col_ptr(j) = col_ptr(j) + np
      end do
      do j = nnz, 1, -1
         row_ind(np+j) = row_ind(j)
         a(np+j) = a(j)
      end do

!     Compute the incomplete Cholesky factorization.

      isj = col_ptr(1)
      col_ptr(1) = 1
      do j = 1, n

!        Load column j into the array w. The first and last elements 
!        of the j-th column of A are a(isj) and a(iej).

         nlj = 0
         iej = col_ptr(j+1) - 1
         do ip = isj, iej
            i = row_ind(ip)
            w(i) = a(ip)
            nlj = nlj + 1
            indr(nlj) = i
            indf(i) = 1
         end do

!        Exit if the current pivot is not positive.

         if (diag(j) <= zero) then
            info = -j
            return
         end if
         diag(j) = sqrt(diag(j))

!        Update column j using the previous columns.

         k = list(j)
         do while (k /= 0) 
            isk = indf(k)
            iek = col_ptr(k+1) - 1

!           Set lval to l(j,k).

            lval = a(isk)

!           Update indf and list.

            newk = list(k)
            isk = isk + 1
            if (isk < iek) then
               indf(k) = isk
               list(k) = list(row_ind(isk))
               list(row_ind(isk)) = k
            endif
            k = newk

!           Compute the update a(i,i) <- a(i,j) - l(i,k)*l(j,k).
!           In this loop we pick up l(i,k) for k < j and i > j.

            do ip = isk, iek
               i = row_ind(ip)
               if (indf(i) /= 0) then
                  w(i) = w(i) - lval*a(ip)
               else
                  indf(i) = 1
                  nlj = nlj + 1
                  indr(nlj) = i
                  w(i) = - lval*a(ip)
               end if
            end do
         end do

!        Compute the j-th column of L.

         do k = 1, nlj
            w(indr(k)) = w(indr(k))/diag(j)
         end do

!        Set mlj to the number of nonzeros to be retained.

         mlj = min(iej-isj+1+p,nlj)
         kth = nlj - mlj + 1

        if (nlj >= 1) then

!           Determine the kth smallest elements in the current 
!           column, and hence, the largest mlj elements.
          
            call dsel2(nlj,w,indr,kth)
          
!           Sort the row indices of the selected elements. Insertion
!           sort is used for small arrays, and heap sort for larger
!           arrays. The sorting of the row indices is required so that
!           we can retrieve l(i,k) with i > k from indf(k).
          
            if (mlj <= insortf) then
               call insort(mlj,indr(kth))
            else
               call ihsort(mlj,indr(kth))
            end if
         end if

!        Store the largest elements in L. The first and last elements 
!        of the j-th column of L are a(newisj) and a(newiej).

         newisj = col_ptr(j)
         newiej = newisj + mlj -1
         do k = newisj, newiej
            a(k) = w(indr(k-newisj+kth))
            row_ind(k) = indr(k-newisj+kth)
         end do

!        Update the diagonal elements.

         do k = kth, nlj
            diag(indr(k)) = diag(indr(k)) - w(indr(k))**2
         end do

!        Update indf and list for the j-th column.

         if (newisj < newiej) then
            indf(j) = newisj
            list(j) = list(row_ind(newisj))
            list(row_ind(newisj)) = j
         endif

!        Clear out elements j+1,...,n of the array indf.

         do k = 1, nlj
            indf(indr(k)) = 0
         end do

!        Update isj and col_ptr.

         isj = col_ptr(j+1)
         col_ptr(j+1) = newiej + 1

      end do

      RETURN

!  end of subroutine dicf

      END SUBROUTINE dicf

      SUBROUTINE insort( n, keys )
      INTEGER :: n
      INTEGER, DIMENSION( n ) :: keys

!     **********
!
!     Subroutine insort
!
!     Given an integer array keys of length n, this subroutine uses
!     an insertion sort to sort the keys in increasing order.
!
!     The subroutine statement is
!
!       subroutine insort(n,keys)
!
!     where
!
!       n is an integer variable.
!         On entry n is the number of keys.
!         On exit n is unchanged.
!
!       keys is an integer array of length n.
!         On entry keys is the array to be sorted.
!         On exit keys is permuted to increasing order.
!
!     MINPACK-2 Project. March 1998.
!     Argonne National Laboratory.
!     Chih-Jen Lin and Jorge J. More'.
!
!     **********

      INTEGER :: i, j, ind

      do j = 2, n
         ind = keys(j)
         i = j - 1
         do while (i > 0 .and. keys(i) > ind)
            keys(i+1) = keys(i)
            i = i - 1
            if ( i <= 0 ) go to 10
         end do
  10     continue
         keys(i+1) = ind
      end do

      RETURN

!  end of subroutine insort

      END SUBROUTINE insort

      SUBROUTINE ihsort( n, keys )
      INTEGER :: n
      INTEGER, DIMENSION( n ) :: keys

!     **********
!
!     Subroutine ihsort
!
!     Given an integer array keys of length n, this subroutine uses
!     a heap sort to sort the keys in increasing order.
!
!     This subroutine is a minor modification of code written by
!     Mark Jones and Paul Plassmann.
!
!     The subroutine statement is
!
!       subroutine ihsort(n,keys)
!
!     where
!
!       n is an integer variable.
!         On entry n is the number of keys.
!         On exit n is unchanged.
!
!       keys is an integer array of length n.
!         On entry keys is the array to be sorted.
!         On exit keys is permuted to increasing order.
!
!     MINPACK-2 Project. March 1998.
!     Argonne National Laboratory.
!     Chih-Jen Lin and Jorge J. More'.
!
!     **********

      INTEGER :: k, m, lheap, rheap, mid, x
     
      if (n <= 1) return
     
!     Build the heap.

      mid = n/2
      do k = mid, 1, -1
         x = keys(k)
         lheap = k
         rheap = n
         m = lheap*2
         do while (m <= rheap)
            if (m < rheap) then
               if (keys(m) < keys(m+1)) m = m + 1
            endif
            if (x >= keys(m)) then
               m = rheap + 1
            else
               keys(lheap) = keys(m)
               lheap = m
               m = 2*lheap
            end if
         end do
      keys(lheap) = x
      end do
     
!     Sort the heap.

      do k = n, 2, -1
         x = keys(k)
         keys(k) = keys(1)
         lheap = 1
         rheap = k-1
         m = 2
         do while (m <= rheap)
            if (m < rheap) then
               if (keys(m) < keys(m+1)) m = m+1
            endif
            if (x >= keys(m)) then
               m = rheap + 1
            else
               keys(lheap) = keys(m)
               lheap = m
               m = 2*lheap
            end if
         end do
      keys(lheap) = x
      end do
     
      RETURN

!  end of subroutine ihsort

      END SUBROUTINE ihsort

      SUBROUTINE dsel2( n, x, keys, k )
      INTEGER :: n, k
      INTEGER, DIMENSION( n ) :: keys
      REAL ( KIND = wp ), DIMENSION( * ) :: x

!     **********
!
!     Subroutine dsel2
!
!     Given an array x of length n, this subroutine permutes
!     the elements of the array keys so that 
!    
!       abs(x(keys(i))) <= abs(x(keys(k))),  1 <= i <= k,
!       abs(x(keys(k))) <= abs(x(keys(i))),  k <= i <= n.
!
!     In other words, the smallest k elements of x in absolute value are
!     x(keys(i)), i = 1,...,k, and x(keys(k)) is the kth smallest element.
!     
!     The subroutine statement is
!
!       subroutine dsel2(n,x,keys,k)
!
!     where
!
!       n is an integer variable.
!         On entry n is the number of keys.
!         On exit n is unchanged.
!
!       x is a real array of length n.
!         On entry x is the array to be sorted.
!         On exit x is unchanged.
!
!       keys is an integer array of length n.
!         On entry keys is the array of indices for x.
!         On exit keys is permuted so that the smallest k elements
!            of x in absolute value are x(keys(i)), i = 1,...,k, and
!            x(keys(k)) is the kth smallest element.
!
!       k is an integer.
!         On entry k specifes the kth largest element.
!         On exit k is unchanged.
!
!     MINPACK-2 Project. March 1998.
!     Argonne National Laboratory.
!     William D. Kastak, Chih-Jen Lin, and Jorge J. More'.
!
!     **********

      INTEGER :: i, l, lc, lp, m, p, p1, p2, p3, u, swap

      if (n <= 1 .or. k <= 0 .or. k > n) return

      u = n
      l = 1
      lc = n
      lp = 2*n

!     Start of iteration loop.

      do while (l < u)

!        Choose the partition as the median of the elements in
!        positions l+s*(u-l) for s = 0, 0.25, 0.5, 0.75, 1.
!        Move the partition element into position l.

         p1 = (u+3*l)/4
         p2 = (u+l)/2
         p3 = (3*u+l)/4

!        Order the elements in positions l and p1.

         if (dabs(x(keys(l))) > dabs(x(keys(p1)))) then
            swap = keys(l)
            keys(l) = keys(p1)
            keys(p1) = swap
            end if

!        Order the elements in positions p2 and p3.

         if (dabs(x(keys(p2))) > dabs(x(keys(p3)))) then
            swap = keys(p2)
            keys(p2) = keys(p3)
            keys(p3) = swap
            end if

!        Swap the larger of the elements in positions p1
!        and p3, with the element in position u, and reorder
!        the first two pairs of elements as necessary.

         if (dabs(x(keys(p3))) > dabs(x(keys(p1)))) then
            swap = keys(p3)
            keys(p3) = keys(u)
            keys(u) = swap
            if (dabs(x(keys(p2))) > dabs(x(keys(p3)))) then
               swap = keys(p2)
               keys(p2) = keys(p3)
               keys(p3) = swap
               end if
         else
            swap = keys(p1)
            keys(p1) = keys(u)
            keys(u) = swap
            if (dabs(x(keys(l))) > dabs(x(keys(p1)))) then
               swap = keys(l)
               keys(l) = keys(p1)
               keys(p1) = swap
               end if
            end if

!        If we define a(i) = abs(x(keys(i)) for i = 1,..., n, we have 
!        permuted keys so that 
!
!          a(l) <= a(p1), a(p2) <= a(p3), max(a(p1),a(p3)) <= a(u).
!
!        Find the third largest element of the four remaining
!        elements (the median), and place in position l.

         if (dabs(x(keys(p1))) > dabs(x(keys(p3)))) then
            if (dabs(x(keys(l))) <= dabs(x(keys(p3)))) then
               swap = keys(l)
               keys(l) = keys(p3)
               keys(p3) = swap
               end if
         else
            if (dabs(x(keys(p2))) <= dabs(x(keys(p1)))) then
               swap = keys(l)
               keys(l) = keys(p1)
               keys(p1) = swap
            else
               swap = keys(l)
               keys(l) = keys(p2)
               keys(p2) = swap
               end if
            end if

!        Partition the array about the element in position l.

         m = l
         do i = l+1, u
            if (dabs(x(keys(i))) < dabs(x(keys(l))))then
               m = m + 1
               swap = keys(m)
               keys(m) = keys(i)
               keys(i) = swap
               end if
         end do

!        Move the partition element into position m.

         swap = keys(l)
         keys(l) = keys(m)
         keys(m) = swap

!        Adjust the values of l and u.

         if (k >= m) l = m + 1
         if (k <= m) u = m - 1

!        Check for multiple medians if the length of the subarray
!        has not decreased by 1/3 after two consecutive iterations.

         if (3*(u-l) > 2*lp .and. k > m) then

!           Partition the remaining elements into those elements
!           equal to x(m), and those greater than x(m). Adjust
!           the values of l and u.

            p = m
            do i = m+1, u
               if (dabs(x(keys(i))) == dabs(x(keys(m)))) then
                  p = p + 1
                  swap = keys(p)
                  keys(p) = keys(i)
                  keys(i) = swap
                  end if
            end do
            l = p + 1
            if (k <= p) u = p - 1
            end if

!        Update the length indicators for the subarray.

         lp = lc
         lc = u-l

      end do

      RETURN

!  end of subroutine dsel2

      END SUBROUTINE dsel2

!  End of module GALAHAD_ICFS_double

   END MODULE GALAHAD_ICFS_double
