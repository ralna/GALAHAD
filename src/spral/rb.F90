! THIS VERSION: GALAHAD 5.3 - 2025-08-22 AT 15:20 GMT.

!#include "galahad_modules.h"
#include "ssids_procedures.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ R B   M O D U L E  *-*-*-*-*-*-*-*-*-*-*-

!   --------------------------------------------------------------------------
!  | Rutherford-Boeing utility package originally spral_ral_boeing from SPRAL |
!   -------------------------------------------------------------------------

!  COPYRIGHT (c) 2000,2010,2013,2016 Science and Technology Facilities Council
!  licence: BSD licence, see LICENCE file for details
!  Authors: Jonathan Hogg and Iain Duff
!  Forked from SPRAL and extended for GALAHAD, Nick Gould, version 3.1, 2016

!  A set of utilities for reading and writing files containing matrices 
!  encoded in the Rutherford-Boeing format

  MODULE GALAHAD_RB_precision

    USE GALAHAD_KINDS_precision
    USE GALAHAD_MU_precision, ONLY: SSIDS_MATRIX_UNSPECIFIED,                  &
                                    SSIDS_MATRIX_REAL_RECT,                    &
                                    SSIDS_MATRIX_REAL_UNSYM,                   &
                                    SSIDS_MATRIX_REAL_SYM_PSDEF,               &
                                    SSIDS_MATRIX_REAL_SYM_INDEF,               &
                                    SSIDS_MATRIX_REAL_SKEW,                    &
                                    MU_half_to_full                       

    USE GALAHAD_RAND_precision, ONLY: RAND_random_state_lcg,                   &
                                      RAND_random_real_lcg
    IMPLICIT none

    PRIVATE
    PUBLIC :: RB_peek, RB_read, RB_write           

!  possible values options%lwr_upr_full

    INTEGER( ip_ ), PARAMETER :: TRI_LWR  = 1 ! Lower triangle
    INTEGER( ip_ ), PARAMETER :: TRI_UPR  = 2 ! Upper triangle
    INTEGER( ip_ ), PARAMETER :: TRI_FULL = 3 ! Both lower and upper triangles

!  possible values of options%values

    INTEGER( ip_ ), PARAMETER :: VALUES_FILE     = 0 ! As per file
    INTEGER( ip_ ), PARAMETER :: VALUES_PATTERN  = 1 ! Pattern only
    INTEGER( ip_ ), PARAMETER :: VALUES_SYM      = 2 ! Random values, symmetric
    INTEGER( ip_ ), PARAMETER :: VALUES_DIAG_DOM = 3 ! Random vals, diag dominnt
    INTEGER( ip_ ), PARAMETER :: VALUES_UNSYM    = 4 ! Random vals, unsymmetric

!  possible error returns

    INTEGER( ip_ ), PARAMETER :: SUCCESS           =  0 ! No errors
    INTEGER( ip_ ), PARAMETER :: ERROR_BAD_FILE    = -1 ! Failed to open file
    INTEGER( ip_ ), PARAMETER :: ERROR_NOT_RB      = -2 ! Header not val for RB
    INTEGER( ip_ ), PARAMETER :: ERROR_IO          = -3 ! Error return from io
    INTEGER( ip_ ), PARAMETER :: ERROR_TYPE        = -4 ! Tried to read bad type
    INTEGER( ip_ ), PARAMETER :: ERROR_ELT_ASM     = -5 ! Read elt as asm or v/v
    INTEGER( ip_ ), PARAMETER :: ERROR_MATRIX_TYPE = -6 ! Bad val of matrix_type
    INTEGER( ip_ ), PARAMETER :: ERROR_EXTRA_SPACE = -10 ! opt%extra_space<1
    INTEGER( ip_ ), PARAMETER :: ERROR_LWR_UPR_FULL= -11 ! opt%lwr_up_full oor
    INTEGER( ip_ ), PARAMETER :: ERROR_VALUES      = -13 ! opt%values oor
    INTEGER( ip_ ), PARAMETER :: ERROR_ALLOC       = -20 ! failed on allocate

!  possible warnings

    INTEGER( ip_ ), PARAMETER :: WARN_AUX_FILE = 1 ! values in auxiliary file

!  options that control what RB_read returns

    TYPE, PUBLIC :: RB_read_options
      LOGICAL  :: add_diagonal = .FALSE. ! Add missing diagonal entries
      REAL     :: extra_space = 1.0      ! Array sizes are mult by this
      INTEGER  :: lwr_upr_full = TRI_LWR ! Ensure entries in lwr/upr tri
      INTEGER  :: values = VALUES_FILE   ! As per file
    END TYPE RB_read_options

!  options that control what RB_write does

    TYPE, PUBLIC :: RB_write_options
      CHARACTER( LEN = 20 ) :: val_format = "( 3E24.16 )"
    END TYPE RB_write_options

!  peeks at the header of a RB file

    INTERFACE RB_peek
      MODULE PROCEDURE RB_peek_file, RB_peek_unit
    END INTERFACE RB_peek

!  reads a RB file

    INTERFACE RB_read
      MODULE PROCEDURE RB_read_int32, RB_read_int64
    END INTERFACE RB_read

!  writes a RB file

    INTERFACE RB_write
      MODULE PROCEDURE RB_write_int32, RB_write_int64
    END INTERFACE RB_write

  CONTAINS

!-*-*-*-*-*-  R B _  S U B R O U T I N E  *-*-*-*-*-*-

    SUBROUTINE RB_peek_file( filename, info, m, n, nelt, nvar, nval,           &
                             matrix_type, type_code, title, identifier )

!  read header information from file (filename version)

    IMPLICIT none
    CHARACTER( LEN = * ), INTENT( IN ) :: filename   ! < File to peek at
    INTEGER( ip_ ), INTENT( OUT ) :: info               ! < Return code
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: m        ! < # rows
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: n        ! < # columns
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nelt !< # elements ( 0 if asm )
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nvar !< # indices in file
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nval !< # values in file
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: matrix_type !< SPRAL matrix type
    CHARACTER( LEN = 3 ), OPTIONAL, INTENT( OUT ) :: type_code !< eg "rsa"
    CHARACTER( LEN = 72 ), OPTIONAL, INTENT( OUT ) :: title! < file title field
    CHARACTER( LEN = 8 ), OPTIONAL, INTENT( OUT ) :: identifier !< file id field

    INTEGER( ip_ ) :: iunit ! unit file is open on
    INTEGER( ip_ ) :: iost ! stat parameter for io calls

    info = SUCCESS

!  find a free unit and open file on it

    OPEN( newunit = iunit, file = filename, status = "old", action = "read",   &
          iostat = iost )
    IF ( iost /= 0 ) THEN
      info = ERROR_BAD_FILE
      RETURN
    END IF

!  call unit version to do hard work, no need to rewind as we will close
!  file immediately

    CALL RB_peek_unit( iunit, info, m = m, n = n, nelt = nelt, nvar = nvar,    &
                       nval = nval, matrix_type = matrix_type,                 &
                       type_code = type_code, title = title,                   &
                       identifier = identifier, no_rewind = .TRUE. )

!  close file

    CLOSE( iunit, iostat=iost )

!  note: we ignore close errors if info indicates a previous error

    IF ( iost /= 0 .AND. info == SUCCESS ) THEN
      info = ERROR_IO
      RETURN
    END IF
    RETURN

    END SUBROUTINE RB_peek_file

!-*-*-*-*-*-*-*-*-*-*-*-  R B _ P E E K  S U B R O U T I N E  *-*-*-*-*-*-*-*-*-

    SUBROUTINE RB_peek_unit( iunit, info, m, n, nelt, nvar, nval, matrix_type, &
                             type_code, title, identifier, no_rewind )

!  read header information from file ( unit version ).

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: iunit             ! unit file is open on
    INTEGER( ip_ ), INTENT( OUT ) :: info             ! return code
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: m      ! # rows
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: n      ! # columns
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nelt ! # elements ( 0 if asm )
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nvar ! # indices in file
    INTEGER( long_ ), OPTIONAL, INTENT( OUT ) :: nval ! # values in file
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: matrix_type !  matrix type code
    CHARACTER( LEN = 3 ), OPTIONAL, INTENT( OUT ) :: type_code ! eg "rsa"
    CHARACTER( LEN = 72 ), OPTIONAL, INTENT( OUT ) :: title ! file title field
    CHARACTER( LEN = 8 ), OPTIONAL, INTENT( OUT ) :: identifier ! file id field

! If present and true, don't backspace unit to start

    LOGICAL, OPTIONAL, INTENT( IN ) :: no_rewind 

!  "shadow" versions of file data - can't rely on arguments being present
!  so data is read into these and copied to arguments if required

    INTEGER( ip_ ) :: r_m, r_n, r_nelt, r_nvar, r_nval
    CHARACTER( LEN = 3 ) :: r_type_code
    CHARACTER( LEN = 72 ) :: r_title
    CHARACTER( LEN = 8 ) :: r_identifier
    LOGICAL :: r_rewind

!  other local variables

    CHARACTER( LEN = 80 ) :: buffer1, buffer2 ! Buffers for reading char data
    INTEGER( ip_ ) :: t1, t2, t3, t4 ! Temporary variables for reading int data
    INTEGER( ip_ ) :: iost ! stat parameter for io ops

    info = SUCCESS

    r_rewind = .TRUE.
    IF ( PRESENT( no_rewind ) ) r_rewind = .NOT. no_rewind

!  Nibble top of file to find desired information, then return to original
!  position if required

    READ( iunit, '( A72, A8 / A80 / A80 )', iostat = iost )                    &
      r_title, r_identifier, buffer1, buffer2
    IF ( iost /= 0 ) THEN
      info = ERROR_IO
      RETURN
    END IF
    IF ( r_rewind ) THEN
      BACKSPACE( iunit ); BACKSPACE( iunit ); BACKSPACE( iunit )
    END IF

    READ( buffer2, '( A3, 11X, 4( 1X, I13 ) )' ) r_type_code, t1, t2, t3, t4

!  validate type_code code, remap data depending on value of type_code(3:3)

    SELECT CASE ( r_type_code( 1 : 1 ) )

!  good, do nothing

    CASE( "r", "c", "i", "p", "q" )

!  not a matrix in RB format

    CASE DEFAULT
      info = ERROR_NOT_RB
      RETURN
    END SELECT

    SELECT CASE ( r_type_code( 2:2 ) )

!  good, do nothing

    CASE( "s", "u", "h", "z", "r" )

!  not a matrix in RB format

    CASE DEFAULT
      info = ERROR_NOT_RB
      RETURN
    END SELECT

    SELECT CASE ( r_type_code( 3 : 3 ) )

!  assembled format

    CASE( "a" )
      r_m = t1
      r_n = t2
      r_nvar = t3
      IF ( t4 /= 0 ) THEN ! RB format requires t4 to be an explicit zero
        info = ERROR_NOT_RB
        RETURN
      END IF
      r_nval = r_nvar ! one-to-one correspondence between INTEGERs and reals
      r_nelt = 0 ! no elemental matrices

!  elemental format

    CASE( "e" )
      r_m = t1
      r_n = r_m ! Elemental matrices are square
      r_nelt = t2
      r_nvar = t3
      r_nval = t4

!  not a valid RB letter code

    CASE DEFAULT
      info = ERROR_NOT_RB
      RETURN
    END SELECT

!  copy out data if requested

    IF ( PRESENT( m ) ) m = r_m
    IF ( PRESENT( n ) ) n = r_n
    IF ( PRESENT( nelt ) ) nelt = r_nelt
    IF ( PRESENT( nvar ) ) nvar = r_nvar
    IF ( PRESENT( nval ) ) nval = r_nval
    IF ( PRESENT( matrix_type ) )                                              &
      matrix_type = sym_to_matrix_type( r_type_code( 2:2 ) )
    IF ( PRESENT( type_code ) ) type_code = r_type_code
    IF ( PRESENT( title ) ) title = r_title
    IF ( PRESENT( identifier ) ) identifier = r_identifier
    RETURN

    END SUBROUTINE RB_peek_unit

!-*-*-*-*-*-*-*-  R B _ r e a d _ i n t 3 2   S U B R O U T I N E  -*-*-*-*-*-*-

    SUBROUTINE RB_read_int32( filename, m, n, ptr, row, val, options, info,    &
                              matrix_type, type_code, title, identifier, state )

!  read a matrix from a Rutherford Boeing file

    IMPLICIT none
    CHARACTER( LEN = * ), INTENT( IN ) :: filename ! File to read
    INTEGER( ip_ ), INTENT( OUT ) :: m
    INTEGER( ip_ ), INTENT( OUT ) :: n
    INTEGER( i4_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: ptr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, target, INTENT( OUT ) :: row
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE, target, INTENT( OUT ) :: val
    TYPE( RB_read_options ), INTENT( IN ) :: options ! control variables
    INTEGER( ip_ ), INTENT( OUT ) :: info ! return code
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: matrix_type ! matrix type code
    CHARACTER( LEN = 3 ), OPTIONAL, INTENT( OUT ) :: type_code ! file data type
    CHARACTER( LEN = 72 ), OPTIONAL, INTENT( OUT ) :: title ! file title
    CHARACTER( LEN = 8 ), OPTIONAL, INTENT( OUT ) :: identifier ! file identfier

!  state to use for

    TYPE( RAND_random_state_lcg ), OPTIONAL, INTENT( INOUT ) :: state

!  random number generation

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ptr64
    INTEGER( ip_ ) :: st

    CALL RB_read_int64( filename, m, n, ptr64, row, val, options, info,        &
                        matrix_type = matrix_type, type_code = type_code,      &
                        title = title, identifier = identifier, state = state )

!  FIXME: Add an error code if ne > maxint

    IF ( ALLOCATED( ptr64 ) ) THEN
      DEALLOCATE( ptr, stat = st )
      ALLOCATE( ptr( n + 1 ), stat = st )
      IF ( st /= 0 ) THEN
        info = ERROR_ALLOC
        RETURN
      END IF

!  forced conversion, FIXME: add guard

      ptr( 1 : n + 1 ) = int( ptr64( 1 : n + 1 ) )
    END IF
    RETURN

    END SUBROUTINE RB_read_int32

!-*-*-*-*-*-*-*-  R B _ r e a d _ i n t 6 4   S U B R O U T I N E  -*-*-*-*-*-*-

    SUBROUTINE RB_read_int64( filename, m, n, ptr, row, val, options, info,    &
                              matrix_type, type_code, title, identifier, state )
    IMPLICIT none
    CHARACTER( LEN = * ), INTENT( IN ) :: filename ! File to read
    INTEGER( ip_ ), INTENT( OUT ) :: m
    INTEGER( ip_ ), INTENT( OUT ) :: n
    INTEGER( i8_ ), DIMENSION( : ), ALLOCATABLE, INTENT( OUT ) :: ptr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, target, INTENT( OUT ) :: row
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE, target, INTENT( OUT ) :: val
    TYPE( RB_read_options ), INTENT( IN ) :: options ! control variables
    INTEGER( ip_ ), INTENT( OUT ) :: info ! return code
    INTEGER( ip_ ), OPTIONAL, INTENT( OUT ) :: matrix_type ! matrix type code
    CHARACTER( LEN = 3 ), OPTIONAL, INTENT( OUT ) :: type_code ! file data type
    CHARACTER( LEN = 72 ), OPTIONAL, INTENT( OUT ) :: title ! file title
    CHARACTER( LEN = 8 ), OPTIONAL, INTENT( OUT ) :: identifier ! file identfier

!  state to use for random number generation

    TYPE( RAND_random_state_lcg ), OPTIONAL, INTENT( INOUT ) :: state

!  variables below are required for calling read_data_real()

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: ival

!  shadow variable for type_code (actual argument is optional)

    CHARACTER( LEN = 3 ) :: r_type_code

!  pointers to simplify which array we are reading in to

    INTEGER( ip_ ), pointer, DIMENSION( : ), contiguous :: rcptr => null(  )
    REAL( rp_ ), pointer, DIMENSION( : ), contiguous :: vptr => null(  )
    REAL( rp_ ), target :: temp( 1 ) ! place holder array
    INTEGER( ip_ ) :: k ! loop indices
    INTEGER( long_ ) :: j ! loop indices
    INTEGER( ip_ ) :: r, c ! loop indices
    INTEGER( long_ ) :: nnz ! number of non-zeroes
    INTEGER( long_ ) :: nelt ! number of elements in file, should be 0
    INTEGER( long_ ) :: len, len2 ! length of arrays to allocate
    INTEGER( ip_ ) :: iunit ! unit we open the file on
    INTEGER( ip_ ) :: st, iost ! error codes from allocate and file operations
    LOGICAL :: symmetric ! .TRUE. if file claims to be ( skew ) symmetric or H
    LOGICAL :: skew ! .TRUE. if file claims to be skew symmetric
    LOGICAL :: read_val ! .TRUE. if we are only reading pattern from file
    LOGICAL :: expanded ! .TRUE. if pattern has been expanded

! random state used if state not present

    TYPE( RAND_random_state_lcg ) :: state2

!  work array used by MU_half_to_full

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: iw_h2f 

! work array in case we need to flip from lwr to upr.

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, target :: col 

! number of entries in row

    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: nzrow

    info = SUCCESS

!  initialize variables to avoid compiler warnings

    symmetric = .FALSE.
    skew = .FALSE.

!  validate options paramters

    IF ( options%extra_space < 1.0 ) THEN
      info = ERROR_EXTRA_SPACE
      RETURN
    END IF
    IF ( options%lwr_upr_full < 1 .OR. options%lwr_upr_full > 3 ) THEN
      info = ERROR_LWR_UPR_FULL
      RETURN
    END IF
    IF ( options%values == - 1 .OR. ABS( options%values ) > 4 ) THEN
      info = ERROR_VALUES
      RETURN
    END IF

!  find a free unit and open file on it

    OPEN( newunit = iunit, file = filename, status = "old", action = "read",   &
          iostat = iost )
    IF ( iost /= 0 ) THEN
      info = ERROR_BAD_FILE
      RETURN
    END IF

!  read top of file (and rewind) to determine space required

    CALL RB_peek_unit( iunit, info, m = m, n = n, nelt = nelt,                 &
                       nval = nnz, matrix_type = matrix_type,                  &
                       type_code = r_type_code, title = title,                 &
                       identifier = identifier )
    IF ( info /= 0 ) GO TO 100

!  attempting to read element file as assembled

    IF ( nelt /= 0 ) THEN
      info = ERROR_ELT_ASM
      GO TO 100
    END IF

!  allocate space for matrix

!  ptr

    len = n + 1
    len = MAX( len, INT( REAL( len,rp_ ) * options%extra_space, long_ ) )
    ALLOCATE( ptr( len ), stat = st )
    IF ( st /= 0 ) GO TO 200

!  row and/or col

    len = nnz
    SELECT CASE ( r_type_code( 2 : 2 ) )

!  symmetric

    CASE( "s", "h", "z" )
      symmetric = .TRUE.
      skew = r_type_code( 2 : 2 ) == "z"

!  do we need to allow for expansion?
!  (a) to get both upper and lower triangles

      IF ( options%lwr_upr_full == TRI_FULL ) len = len * 2

!  (b) to add additional diagonal entries

      IF ( options%add_diagonal .OR. options%values == - VALUES_DIAG_DOM .OR.  &
           ( options%values == VALUES_DIAG_DOM .AND.                           &
             ( r_type_code( 1 : 1 ) == "p" .OR.                                &
               r_type_code( 1 : 1 ) == "q" ) ) ) len = len + n

!  unsymmetric or rectangular, no need to worry about upper/lower, but
!  may need to add diagonal.

    CASE( "u", "r" )
      symmetric = .FALSE.
      IF ( options%add_diagonal ) len = len + n
    END SELECT
    len2 = len
    len = max( len, int( REAL( len,rp_ ) * options%extra_space, long_ ) )
    ALLOCATE( row( len ), stat = st )
    IF ( st /= 0 ) GO TO 200
    rcptr => row

!  We need to read into col then copy into row as we flip from lower to upper

    IF ( symmetric .AND. ( options%lwr_upr_full == TRI_UPR ) ) THEN
      ALLOCATE( col( len2 ), stat = st )
      rcptr => col
    END IF
    IF ( st /= 0 ) GO TO 200

!  allocate val if required

    IF ( ABS( options%values ) >= VALUES_SYM .OR. ( options%values == 0 .AND.  &
         r_type_code( 1 : 1 ) /= "p" .AND. r_type_code( 1 : 1 ) /= "q" ) ) THEN

!  we are actually going to store some values

       ALLOCATE( val( len ), stat = st )
       IF ( st /= 0 ) GO TO 200
       vptr => val

!  use a place holder in call to read_data_real()

    ELSE
       vptr => temp
    END IF

!  read matrix in its native format (real/integer)

    IF ( r_type_code( 1 : 1 ) == "q" ) info = WARN_AUX_FILE

!  determine whether we are reading values from file or not

    read_val = options%values >= 0 .AND. options%values /= VALUES_PATTERN
    read_val = read_val .AND. r_type_code( 1 : 1 ) /= "p"
    read_val = read_val .AND. r_type_code( 1 : 1 ) /= "q"

    SELECT CASE( r_type_code( 1 : 1 ) )
    CASE ( "r" ) ! real
      IF ( read_val ) THEN ! we want the pattern and values
        CALL read_data_real( iunit, n, nnz, ptr, rcptr, iost, val = vptr )
      ELSE !  we want the pattern only
        CALL read_data_real( iunit, n, nnz, ptr, rcptr, iost )
      END IF
    CASE ( "c" ) ! complex
      info = ERROR_TYPE
      GO TO 100
    CASE ( "i" ) ! integer
      IF ( read_val ) THEN
        ALLOCATE( ival( nnz ), stat = st )
        IF ( st /= 0 ) GO TO 200
        CALL read_data_integer( iunit, n, nnz, ptr, rcptr, iost, val = ival )
        IF ( iost /= 0 ) val( 1 : nnz ) = real( ival )
      ELSE
        CALL read_data_integer( iunit, n, nnz, ptr, rcptr, iost )
      END IF
    CASE ( "p", "q" ) ! pattern only
      CALL read_data_real( iunit, n, nnz, ptr, rcptr, iost )
    END SELECT
    IF ( iost /= 0 ) THEN  ! error
      info = ERROR_IO
      GO TO 100
    END IF

!  add any missing diagonal entries

    IF ( options%add_diagonal .OR. ( symmetric .AND. .NOT. read_val .AND.      &
         ABS( options%values ) == 3 ) ) THEN
      IF ( read_val ) THEN
        CALL add_missing_diag( m, n, ptr, rcptr, val = val )
      ELSE
        CALL add_missing_diag( m, n, ptr, rcptr )
      END IF
    END IF

!  expand pattern if we need to generate unsymmetric values for it

    IF ( .NOT. read_val .AND. ABS( options%values ) == VALUES_UNSYM            &
         .AND. symmetric .AND. options%lwr_upr_full == TRI_FULL ) THEN
      ALLOCATE( iw_h2f( n ),stat = st )
      IF ( st /= 0 ) GO TO 200
      CALL MU_half_to_full( n, rcptr, ptr, iw_h2f )
      expanded = .TRUE.
    ELSE
      expanded = .FALSE.
    END IF

!  generate values if required

    IF ( .NOT. read_val .AND.                                                  &
         ( options%values < 0 .OR. options%values >= 2 ) ) THEN
      IF ( ABS( options%values ) == 3 ) THEN
        ALLOCATE( nzrow( n ), stat = st )
        IF ( st /= 0 ) GO TO 200
        nzrow( : ) = 0
      END IF
      DO c = 1, n
        k = int(  ptr( c + 1 ) - ptr( c )  )
        IF ( PRESENT( state ) ) THEN
          DO j = ptr( c ), ptr( c + 1 ) - 1
            val( j ) = RAND_random_real_lcg( state, .FALSE. )
            r = rcptr( j )
            IF ( ( abs( options%values ) == 3 ) .AND. symmetric ) THEN
              nzrow( r ) = nzrow( r ) + 1
              IF ( r == c )                                                    &
                val( j ) = REAL( MAX( 100, 10 * ( k + nzrow( r ) - 1 ) ), rp_ )
            END IF
          END DO
        ELSE
          DO j = ptr( c ), ptr( c + 1 ) - 1
            val( j ) = RAND_random_real_lcg( state2, .FALSE. )
            r = rcptr( j )
            IF ( ( abs( options%values ) == 3 ) .AND. symmetric ) THEN
              nzrow( r ) = nzrow( r ) + 1
              IF ( r == c )                                                    &
                val( j ) = REAL( MAX( 100, 10 * ( k + nzrow( r ) - 1 ) ), rp_ )
            END IF
          END DO
        END IF
      END DO
    END IF

!  expand to full storage or flip lower/upper as required

    IF ( symmetric ) THEN
      SELECT CASE ( options%lwr_upr_full )
      CASE( TRI_LWR ) !  No-op

!  only need to flip from upper to lower if want to end up as CSC

      CASE( TRI_UPR )
        IF ( ALLOCATED( val ) ) THEN
           CALL flip_lower_upper( n, ptr, col, row, st, val = val )
        ELSE
           CALL flip_lower_upper( n, ptr, col, row, st )
        END IF
        IF ( st /= 0 ) GO TO 200
        IF ( skew .AND. associated( vptr, val ) )                              &
          CALL sym_to_skew( n, ptr, row, val )
      CASE( TRI_FULL )
        IF ( .NOT. ALLOCATED( iw_h2f ) ) ALLOCATE( iw_h2f( n ),stat = st )
        IF ( st /= 0 ) GO TO 200
        IF ( .NOT. expanded ) THEN
          IF ( ALLOCATED( val ) ) THEN
            CALL MU_half_to_full( n, rcptr, ptr, iw_h2f, a = val )
          ELSE
            CALL MU_half_to_full( n, rcptr, ptr, iw_h2f )
          END IF
          expanded = .TRUE.

!  MU_half_to_full doesn't cope with skew symmetry, need to flip -ve all
!  entries in the upper triangle

          IF ( skew .AND. ALLOCATED( val ) )                                   &
            CALL sym_to_skew( n, ptr, row, val )
        END IF
      END SELECT
    END IF

100 CONTINUE
    IF ( PRESENT( type_code ) ) type_code = r_type_code
    CLOSE( iunit, iostat = iost )

!  ignore close errors if info indicates a previous error

    IF ( iost /= 0 .AND. info == SUCCESS ) THEN
      info = ERROR_IO
      RETURN
    END IF
    RETURN

!  error handlers
 
200 CONTINUE

!  allocation error

    info = ERROR_ALLOC
    GO TO 100

    END SUBROUTINE RB_read_int64

!-*-*-*-*-*-*-*-  R B _ w r i t e _ i n t 3 2   S U B R O U T I N E  -*-*-*-*-*-

    SUBROUTINE RB_write_int32( filename, matrix_type, m, n, ptr, row, options, &
                               inform, val, title, identifier )

!  Write a CSC matrix to the specified file. Arguments:
!   filename File to write to. If it already exists, it will be
!   overwritten.
!   matrix_type SPRAL matrix type, as defined in matrix_utils.
!   m Number of rows in matrix.
!   n Number of columns in matrix.
!   ptr Column pointers for matrix. Column i has entries corresponding
!   to row( ptr( i ):ptr( i+1 )-1 ) and val( ptr( i ):ptr( i+1 )-1 ).
!   row Row indices for matrix.
!   val Floating point values for matrix.
!   options User-specifyable options.
!   info Status on output, 0 for success.

    IMPLICIT none
    CHARACTER( LEN = * ), INTENT( IN ) :: filename
    INTEGER( ip_ ), INTENT( IN ) :: matrix_type
    INTEGER( ip_ ), INTENT( IN ) :: m
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( i4_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr
    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row
    TYPE( RB_write_options ), INTENT( IN ) :: options
    INTEGER( ip_ ), INTENT( OUT ) :: inform
    REAL( rp_ ), DIMENSION( ptr( n + 1 ) - 1 ), OPTIONAL, INTENT( IN ) :: val
    CHARACTER( LEN = * ), OPTIONAL, INTENT( IN ) :: title
    CHARACTER( LEN = * ), OPTIONAL, INTENT( IN ) :: identifier

    INTEGER( long_ ), DIMENSION( : ), ALLOCATABLE :: ptr64
    INTEGER( ip_ ) :: st

!  copy from 32-bit to 64-bit ptr array and call 64-bit version

    ALLOCATE( ptr64( n + 1 ), stat = st )
    IF ( st /= 0 ) THEN
      inform = ERROR_ALLOC
      RETURN
    END IF
    ptr64( : ) = ptr( : )

    CALL RB_write_int64( filename, matrix_type, m, n, ptr64, row, options,     &
                         inform, val = val, title = title,                     &
                         identifier = identifier )
    RETURN

    END SUBROUTINE RB_write_int32

!-*-*-*-*-*-*-*-  R B _ w r i t e _ i n t 6 4   S U B R O U T I N E  -*-*-*-*-*-

    SUBROUTINE RB_write_int64( filename, matrix_type, m, n, ptr, row, options, &
                               inform, val, title, identifier )

!  Write a CSC matrix to the specified file. Arguments:
!   filename File to write to. If it already exists, it will be overwritten.
!   matrix_type SPRAL matrix type, as defined in matrix_utils.
!   m Number of rows in matrix.
!   n Number of columns in matrix.
!   ptr Column pointers for matrix. Column i has entries corresponding
!   to row( ptr( i ):ptr( i+1 )-1 ) and val( ptr( i ):ptr( i+1 )-1 ).
!   row Row indices for matrix.
!   val Floating point values for matrix.
!   options User-specifyable options.
!   inform Status on output, 0 for success.
!   title Title to use in file, defaults to "Matrix"
!   id Matrix name/identifier to use in file, defaults to "0"

    IMPLICIT none
    CHARACTER( LEN = * ), INTENT( IN ) :: filename
    INTEGER( ip_ ), INTENT( IN ) :: matrix_type
    INTEGER( ip_ ), INTENT( IN ) :: m
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( i8_ ), DIMENSION( n + 1 ), INTENT( IN ) :: ptr
    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row
    TYPE( RB_write_options ), INTENT( IN ) :: options
    INTEGER( ip_ ), INTENT( OUT ) :: inform
    REAL( rp_ ), DIMENSION( ptr( n + 1 ) - 1 ), OPTIONAL, INTENT( IN ) :: val
    CHARACTER( LEN = * ), OPTIONAL, INTENT( IN ) :: title
    CHARACTER( LEN = * ), OPTIONAL, INTENT( IN ) :: identifier

    CHARACTER( LEN = 3 ) :: type
    INTEGER( ip_ ) :: i, iunit, ptr_lines, row_lines, val_lines, total_lines
    INTEGER( ip_ ) :: max_row, ptr_prec, row_prec
    INTEGER( ip_ ) :: ptr_per_line, row_per_line, val_per_line
    INTEGER( i8_ ) :: max_ptr
    CHARACTER( LEN = 16 ) :: ptr_format, row_format
    CHARACTER( LEN = 72 ) :: the_title
    CHARACTER( LEN = 8 ) :: the_id
    INTEGER( ip_ ) :: st

    inform = 0 ! by default, success

!  check arguments

    IF ( matrix_type < 0 .OR. matrix_type > 6 .OR. matrix_type == 5 ) THEN
      inform = ERROR_MATRIX_TYPE
      RETURN
    END IF

!  open file

    OPEN( file = filename, newunit = iunit, status = 'replace', iostat = st )
    IF ( st /= 0 ) THEN
      inform = ERROR_BAD_FILE
      RETURN
    END IF

!  determine formats

    max_ptr = MAXVAL( ptr( 1 : n + 1 ) )
    ptr_prec = INT( LOG10( REAL( max_ptr, rp_ ) ) ) + 2
    ptr_per_line = 80 / ptr_prec ! 80 CHARACTER per line
    ptr_format = create_format( ptr_per_line, ptr_prec )
    max_row = MAXVAL( row( 1 : ptr( n + 1 ) - 1 ) )
    row_prec = INT( LOG10( REAL( max_row, rp_ ) ) ) + 2
    row_per_line = 80 / row_prec ! 80 CHARACTER per line
    row_format = create_format( row_per_line, row_prec )

!  calculate lines: First find val_per_line

    DO i = 2, len( options%val_format )
      IF ( options%val_format( i : i ) == 'e' .OR.                             &
           options%val_format( i : i ) == 'f' ) EXIT
    END DO
    READ( options%val_format( 2 : i - 1 ), * ) val_per_line
    ptr_lines = ( SIZE( ptr ) - 1 ) / ptr_per_line + 1
    row_lines = ( SIZE( row ) - 1 ) / row_per_line + 1
    IF ( PRESENT( val ) ) THEN
      val_lines = ( SIZE( val ) - 1 ) / val_per_line + 1
    ELSE
      val_lines = 0
    END IF
    total_lines = ptr_lines + row_lines + val_lines

!  determine type string

    IF ( PRESENT( val ) ) THEN
      type( 1 : 1 ) = 'r' ! real
    ELSE
      type( 1 : 1 ) = 'p' ! pattern
    END IF
    type( 2:2 ) = matrix_type_to_sym( matrix_type )
    type( 3:3 ) = 'a' ! assembled

!  write header

    the_title = "Matrix"
    IF ( PRESENT( title ) ) the_title = title
    the_id = "0"
    IF ( PRESENT( identifier ) ) the_id = identifier
    WRITE( iunit, "( A72, A8 )" ) the_title, the_id
    WRITE( iunit, "( I14, 1X, I13, 1X, I13, 1X, I13 )" )                       &
      total_lines, ptr_lines, row_lines, val_lines
    WRITE( iunit, "( A3, 11X, I14, 1X, I13, 1X, I13, 1X, I13 )" )              &
      type, m, n, ptr( n + 1 ) - 1, 0 ! last entry is explicitly zero by RB spec
    WRITE( iunit, "( A16, A16, A20 )" )                                        &
      ptr_format, row_format, options%val_format

!  write matrix

    WRITE( iunit, ptr_format ) ptr( : )
    WRITE( iunit, row_format ) row( : )
    IF ( PRESENT( val ) ) &
         WRITE( iunit, options%val_format ) val( : )

!  close file

    CLOSE( iunit )
    RETURN

    END SUBROUTINE RB_write_int64

!-*-*-*-*-*-*-  R B _  c r e a t e _ f o r m a t  F U N C T I O N  *-*-*-*-*-*-

    CHARACTER( LEN = 16 ) FUNCTION create_format( per_line, prec )
    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: per_line
    INTEGER( ip_ ), INTENT( IN ) :: prec

!  we assume inputs are both < 100

    IF ( per_line < 10 ) THEN
      IF ( prec < 10 ) THEN
        WRITE( create_format, "( '( ',i1,'i',i1,' )' )" ) per_line, prec
      ELSE ! prec >= 10
        WRITE( create_format, "( '( ',i1,'i',i2,' )' )" ) per_line, prec
      END IF
    ELSE ! per_line >= 10
      IF ( prec < 10 ) THEN
        WRITE( create_format, "( '( ',i2,'i',i1,' )' )" ) per_line, prec
      ELSE ! prec >= 10
        WRITE( create_format, "( '( ',i2,'i',i2,' )' )" ) per_line, prec
      END IF
    END IF
    RETURN

    END FUNCTION create_format

!-*-*-*-*-*-*-  R B _  s y m _ t o _ s k e w  S U B R O U T I N E  *-*-*-*-*-*-

    SUBROUTINE sym_to_skew( n, ptr, row, val )

!  convert symmetric matrix to skew symmetric

!  sets all entries in the upper triangle to minus their original value
!  (i.e. this is a no-op if matrix is only stored as lower triangle)

    IMPLICIT none
    INTEGER( ip_ ), INTENT( INOUT ) :: n
    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( INOUT ) :: ptr
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE, INTENT( INOUT ) :: row
    REAL( rp_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( INOUT ) :: val

    INTEGER( ip_ ) :: i
    INTEGER( long_ ) :: j

!  CSC format

    DO i = 1, n
      DO j = ptr( i ), ptr( i + 1 ) - 1
        IF ( row( j ) >= i ) cycle ! in lower triangle
        val( j ) =  - val( j )
      END DO
    END DO
    RETURN

    END SUBROUTINE sym_to_skew

!-*-*-*-*-  R B _  f l i p _ l o w e r _ u p p e r  S U B R O U T I N E  *-*-*-

    SUBROUTINE flip_lower_upper( n, ptr, row, col, st, val )

!  transpose a symmetric matrix. To reduce copying we supply the destination 
!  integer matrix distinct from the source. The destination val and ptr arrays 
!  is the same as the source (if required). The matrix must be symmetric.

    IMPLICIT none

!  number of rows/columns in matrix (symmetric)

    INTEGER( ip_ ), INTENT( IN ) :: n

! pointers into rows/columns

    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( INOUT ) :: ptr

!  source index array

    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( IN ) :: row

!  destination index array

    INTEGER( ip_ ), DIMENSION( ptr( n + 1 ) - 1 ), INTENT( OUT ) :: col
    INTEGER( ip_ ), INTENT( OUT ) :: st ! stat parameter for allocates

!  values can be flipped as well, if required (indiciated by presence)

    REAL( rp_ ), DIMENSION( ptr( n + 1 ) - 1 ), OPTIONAL, INTENT( INOUT ) :: val

    INTEGER( long_ ) :: i ! loop indices
    INTEGER( ip_ ) :: r, c ! loop indices
    INTEGER( ip_ ), DIMENSION( : ), ALLOCATABLE :: wptr ! working copy of ptr
    REAL( rp_ ), DIMENSION( : ), ALLOCATABLE :: wval ! working copy of val

!  allocate memory

    ALLOCATE( wptr( n + 2 ), stat = st )
    IF ( st /= 0 ) RETURN
    IF ( PRESENT( val ) ) ALLOCATE( wval( ptr( n + 1 ) - 1 ), stat = st )
    IF ( st /= 0 ) RETURN

!  count number of entries in row r as wptr( r+2 )

    wptr( 2 : n + 2 ) = 0
    DO c = 1, n
      DO i = ptr( c ), ptr( c + 1 ) - 1
        r = row( i )
        wptr( r + 2 ) = wptr( r + 2 )  +  1
      END DO
    END DO

!  determine insert point for row r as wptr( r+1 )

    wptr( 1 : 2 ) = 1
    DO r = 1, n
      wptr( r + 2 ) = wptr( r + 1 )  +  wptr( r + 2 )
    END DO

!  now loop over matrix inserting entries at correct points

    IF ( PRESENT( val ) ) THEN
      DO c = 1, n
        DO i = ptr( c ), ptr( c + 1 ) - 1
          r = row( i )
          col( wptr( r + 1 ) ) = c
          wval( wptr( r + 1 ) ) = val( i )
          wptr( r + 1 ) = wptr( r + 1 ) + 1
        END DO
      END DO
    ELSE
      DO c = 1, n
        DO i = ptr( c ), ptr( c + 1 ) - 1
          r = row( i )
          col( wptr( r + 1 ) ) = c
          wptr( r + 1 ) = wptr( r + 1 ) + 1
        END DO
      END DO
    END IF

!  finally copy data back to where it needs to be

    ptr( 1 : n + 1 ) = wptr( 1 : n + 1 )
    IF ( PRESENT( val ) )                                                      &
      val( 1 : ptr( n + 1 ) - 1 ) = wval( 1 : ptr( n + 1 ) - 1 )
    RETURN

    END SUBROUTINE flip_lower_upper

!-*-*-*-*-  R B _ a d d _ m i s s i n g _ d i a g  S U B R O U T I N E  *-*-*-*-

    SUBROUTINE add_missing_diag( m, n, ptr, row, val )

!  add any missing values to matrix

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: m
    INTEGER( ip_ ), INTENT( IN ) :: n
    INTEGER( long_ ), DIMENSION( n + 1 ), INTENT( INOUT ) :: ptr
    INTEGER( ip_ ), DIMENSION( : ), INTENT( INOUT ) :: row
    REAL( rp_ ), DIMENSION( * ), OPTIONAL, INTENT( INOUT ) :: val

    INTEGER( ip_ ) :: col
    INTEGER( long_ ) :: i
    INTEGER( ip_ ) :: ndiag
    LOGICAL :: found

!  count number of missing diagonal entries

    ndiag = 0
    DO col = 1, min( m, n )
      DO i = ptr( col ), ptr( col + 1 ) - 1
        IF ( row( i ) == col ) ndiag = ndiag + 1
      END DO
    END DO

    ndiag = min( m, n ) - ndiag ! Determine number missing

!  process matrix, adding diagonal entries as first entry in column if
!  not otherwise present

    DO col = n, 1, - 1
      IF ( ndiag == 0 ) RETURN
      found = .FALSE.
      IF ( PRESENT( val ) ) THEN
        DO i = ptr( col + 1 ) - 1, ptr( col ),  - 1
          found = found .OR. row( i ) == col
          row( i + ndiag ) = row( i )
          val( i + ndiag ) = val( i )
        END DO
      ELSE
        DO i = ptr( col + 1 ) - 1, ptr( col ),  - 1
          found = found .OR. row( i ) == col
          row( i + ndiag ) = row( i )
        END DO
      END IF
      ptr( col + 1 ) = ptr( col + 1 ) + ndiag

!  add a diagonal if we're in the square submatrix

       IF ( .NOT. found .AND. col <= m ) THEN
         ndiag = ndiag - 1
         i = ptr( col ) + ndiag
         row( i ) = col
         IF ( PRESENT( val ) ) val( i ) = 0.0_rp_
       END IF
    END DO
    RETURN

    END SUBROUTINE add_missing_diag

!-*-*-*-*-*-  R B _ r e a d _ d a t a _ r e a l  S U B R O U T I N E  *-*-*-*-*-

    SUBROUTINE read_data_real( lunit, n, nnz, ptr, row, iost, val )

!  read data from file: Real-valued version

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: lunit !< unit from which to read data
    INTEGER( ip_ ), INTENT( IN ) :: n !< Number of columns to read
    INTEGER( long_ ), INTENT( IN ) :: nnz ! Number of entries to read
    INTEGER( long_ ), DIMENSION( * ), INTENT( OUT ) :: ptr ! Column pointers
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: row ! Row indices
    INTEGER( ip_ ), INTENT( OUT ) :: iost ! iostat from failed op, or 0
    REAL( rp_ ), DIMENSION( * ), OPTIONAL, INTENT( OUT ) :: val ! If present,
    !  returns the numerical data.

    CHARACTER( LEN = 80 ) :: buffer1, buffer2, buffer3
    CHARACTER( LEN = 16 ) :: ptr_format, row_format
    CHARACTER( LEN = 20 ) :: val_format

!  Skip past header information that isn't formats
    READ( lunit,'( a80/a80/a80 )', iostat=iost ) buffer1, buffer2, buffer3
    IF ( iost /= 0 ) RETURN

!  Read formats
    READ( lunit,'( 2a16,a20 )', iostat=iost ) ptr_format, row_format, val_format
    IF ( iost /= 0 ) RETURN

!  Read column pointers
    READ( lunit,ptr_format, iostat=iost ) ptr( 1 : n + 1 )
    IF ( iost /= 0 ) RETURN

!  Read row indices
    READ( lunit,row_format, iostat=iost ) row( 1 : nnz )
    IF ( iost /= 0 ) RETURN

!  Read values if desired
    IF ( PRESENT( val ) ) THEN
       READ( lunit,val_format, iostat=iost ) val( 1 : nnz )
       IF ( iost /= 0 ) RETURN
    END IF
    RETURN

    END SUBROUTINE read_data_real

!-*-*-*-*-  R B _ r e a d _ d a t a _ i n t e g e r  S U B R O U T I N E  *-*-*-

    SUBROUTINE read_data_integer( lunit, n, nnz, ptr, row, iost, val )

!  read data from file: integer-valued version

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: lunit !< unit from which to read data
    INTEGER( ip_ ), INTENT( IN ) :: n !< Number of columns to read
    INTEGER( long_ ), INTENT( IN ) :: nnz ! Number of entries to read
    INTEGER( long_ ), DIMENSION( * ), INTENT( OUT ) :: ptr ! Column pointers
    INTEGER( ip_ ), DIMENSION( * ), INTENT( OUT ) :: row ! Row indices
    INTEGER( ip_ ), INTENT( OUT ) :: iost ! iostat from failed op, or 0
    INTEGER( ip_ ), DIMENSION( * ), OPTIONAL, INTENT( OUT ) :: val ! If present,

!  returns the numerical data

    CHARACTER( LEN = 80 ) :: buffer1, buffer2, buffer3
    CHARACTER( LEN = 16 ) :: ptr_format, row_format
    CHARACTER( LEN = 20 ) :: val_format

!  skip past header information that isn't formats

    READ( lunit, '( A80 / A80 / A80 )', iostat = iost )                        &
      buffer1, buffer2, buffer3
    IF ( iost /= 0 ) RETURN

!  read formats

    READ( lunit, '( 2A16, A20 )', iostat = iost )                              &
      ptr_format, row_format, val_format
    IF ( iost /= 0 ) RETURN

!  read column pointers

    READ( lunit, ptr_format, iostat=iost ) ptr( 1 : n + 1 )
    IF ( iost /= 0 ) RETURN

!  read row indices

    READ( lunit, row_format, iostat=iost ) row( 1 : nnz )
    IF ( iost /= 0 ) RETURN

!  read values if desired

    IF ( PRESENT( val ) ) THEN
      READ( lunit, val_format, iostat=iost ) val( 1 : nnz )
      IF ( iost /= 0 ) RETURN
    END IF
    RETURN

    END SUBROUTINE read_data_integer

!-*-*-*-*-  R B _  m a t r i x _ t y p e _ t o _ s y m  F U N C T I O N  *-*-*-

    CHARACTER( LEN = 1 ) FUNCTION matrix_type_to_sym( matrix_type )

!  convert SPRAL matrix type code to type_code( 2:2 ) CHARACTER

    IMPLICIT none
    INTEGER( ip_ ), INTENT( IN ) :: matrix_type

    SELECT CASE ( matrix_type )
    CASE( SSIDS_MATRIX_UNSPECIFIED )
      matrix_type_to_sym = "r"
    CASE( SSIDS_MATRIX_REAL_RECT )
      matrix_type_to_sym = "r"
    CASE( SSIDS_MATRIX_REAL_UNSYM )
      matrix_type_to_sym = "u"
    CASE( SSIDS_MATRIX_REAL_SYM_PSDEF )
      matrix_type_to_sym = "s"
    CASE( SSIDS_MATRIX_REAL_SYM_INDEF )
      matrix_type_to_sym = "s"
    CASE( SSIDS_MATRIX_REAL_SKEW )
      matrix_type_to_sym = "z"
    END SELECT
    RETURN

    END FUNCTION matrix_type_to_sym

!-*-*-*-*-  R B _  s y m _ t o _ m a t r i x _ t y p e  F U N C T I O N  *-*-*-

    INTEGER FUNCTION sym_to_matrix_type( sym )

!  convert type_code(2:2) character to SPRAL matrix type code

    IMPLICIT none
    CHARACTER( LEN = 1 ), INTENT( IN ) :: sym

    SELECT CASE ( sym )
    CASE( "r" )
      sym_to_matrix_type = SSIDS_MATRIX_REAL_RECT
    CASE( "s" )
      sym_to_matrix_type = SSIDS_MATRIX_REAL_SYM_INDEF
    CASE( "u" )
      sym_to_matrix_type = SSIDS_MATRIX_REAL_UNSYM
    CASE( "z" )
      sym_to_matrix_type = SSIDS_MATRIX_REAL_SKEW
    CASE DEFAULT !  this should never happen
      sym_to_matrix_type = SSIDS_MATRIX_UNSPECIFIED
    END SELECT
    RETURN

    END FUNCTION sym_to_matrix_type

  END MODULE GALAHAD_RB_precision















