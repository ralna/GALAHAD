! THIS VERSION: GALAHAD 3.3 - 11/08/2021 AT 16:15 GMT.

!-*-*-*-*-*-*-*-*-*-*-*- G A L A H A D   M O D U l E -*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   fortran 77 version originally released as part of CUTE, December 1990
!   Became separate subroutines in SifDec, April 2004
!   Updated fortran 2003 version packaged and released for CUTEst, December 2012
!   Released as standalone GALAHAD module, July 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_HASH

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_SPACE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: HASH_initialize, HASH_read_specfile, HASH_insert,              &
                HASH_search, HASH_remove, HASH_rebuild, HASH_terminate,        &
                HASH_full_initialize, HASH_full_terminate,                     &
                HASH_import, HASH_information

!----------------------
!   I n t e r f a c e s
!----------------------

      INTERFACE HASH_initialize
        MODULE PROCEDURE HASH_initialize, HASH_full_initialize
      END INTERFACE HASH_initialize

      INTERFACE HASH_terminate
        MODULE PROCEDURE HASH_terminate, HASH_full_terminate
      END INTERFACE HASH_terminate

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: nbytes = 8

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: buffer = 75
      INTEGER, PARAMETER :: nbytes_by_2 = nbytes / 2
      INTEGER, PARAMETER :: increase_n = 3
      INTEGER, PARAMETER :: increase_d = 2

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: HASH_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, >= 1 enables debugging

        INTEGER :: print_level = 0

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

      END TYPE HASH_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: HASH_inform_type

!  return status. See DGO_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

     END TYPE HASH_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: HASH_data_type

!  the number of characters permitted in each word in the hash table

       INTEGER :: nchar

!  the maximum number of words that can be held in the dictionary

       INTEGER :: length

!  the number of unfilled entries in the current hash table

       INTEGER :: hash_empty

!  the largest prime that is no larger than the size of current hash table

       REAL ( KIND = dp ) :: hash_prime

!  TABLE(i) gives the status of table entry i
!  if TABLE(i) = - (length+1), the entry is unused
!  if TABLE(i) = - k, the entry was used but has been deleted.
!                k gives the index of the next entry in the chain
!  if TABLE(i) = 0, the entry is used and lies at the end of a chain
!  if TABLE(i) = k, the entry is used. k gives the index of the next
!                entry in the chain

      INTEGER, DIMENSION( : ), ALLOCATABLE :: TABLE

!  KEY is the dictionary of hashed entries. Entry KEY(i,j) gives
!  the ith component of the j-th word when TABLE(j) >= 0

      CHARACTER ( LEN = 1 ), DIMENSION( : , : ), ALLOCATABLE :: KEY

!  temporary workspace that is only used if the dictionary is enlarged

      INTEGER, DIMENSION( : ), ALLOCATABLE :: TABLE_temp
      CHARACTER ( LEN = 1 ), DIMENSION( : ), ALLOCATABLE :: FIELD_temp
      CHARACTER ( LEN = 1 ), DIMENSION( : , : ), ALLOCATABLE :: KEY_temp

      END TYPE HASH_data_type

      TYPE, PUBLIC :: HASH_full_data_type
        TYPE ( HASH_data_type ) :: HASH_data
        TYPE ( HASH_control_type ) :: HASH_control
        TYPE ( HASH_inform_type ) :: HASH_inform
      END TYPE HASH_full_data_type

    CONTAINS

!-*-*-*-*-  H A S H  _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE HASH_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by HASH_initialize could (roughly)
!  have been set as:

! BEGIN HASH SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  space-critical                                  no
!  deallocate-error-fatal                          no
! END HASH SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HASH_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: space_critical = print_level + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'HASH'
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

     spec( alive_file )%keyword = 'alive-filename'
     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ), control%error, control%error )
     CALL SPECFILE_assign_value( spec( out ), control%out, control%error )
     CALL SPECFILE_assign_value( spec( print_level ), control%print_level,     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ), control%prefix, control%error )

     RETURN

!  End of subroutine HASH_read_specfile

     END SUBROUTINE HASH_read_specfile

!-*-  G A L A H A D   H A S H _ i n i t i a l i z e    S U B R O U T I N E  -*-

      SUBROUTINE HASH_initialize( nchar, length, data, control, inform )

!  ------------------------------------------------------------
!  set up initial scatter table (Williams, CACM 2, 21-24, 1959)

!  nchar is an upper bound on the number of characters in each word
!  length gives the number of words that can be held in the dictionary
!  TABLE(i) gives the status of table entry i. If TABLE(i) = - (length+1), 
!     the entry is unused
!  data private internal data
!  data%TABLE(i) gives the status of table entry i
!  if data%TABLE(i) = - (length+1), the entry is unused
!  if data%TABLE(i) = - k, the entry was used but has been deleted.
!                    k gives the index of the next entry in the chain
!  if data%TABLE(i) = 0, the entry is used and lies at the end of a chain
!  if data%TABLE(i) = k, the entry is used. k gives the index of the next
!                    entry in the chain
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  ------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nchar, length
      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER :: prime
      CHARACTER ( LEN = 80 ) :: array_name

      data%nchar = nchar ; data%length = length
      data%hash_empty = length + 1

!  Find an appropriate prime number for the hash function. Compute the largest
!  prime smaller than length

      prime = 2 * ( ( length + 1 ) / 2 ) - 1

!  Is prime prime?

   10 CONTINUE
      IF ( .NOT. HASH_is_prime( prime ) ) THEN
        prime = prime - 2
        GO TO 10
      END IF

!  store prime as a double-precision real

      data%hash_prime = REAL( prime, KIND = dp )

!     ALLOCATE( data%TABLE( length ) )

!  create space for the table and keys

      array_name = 'hash: dataa%TABLE'
      CALL SPACE_resize_array( length, data%TABLE,                             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'hash: dataa%KEY'
      CALL SPACE_resize_array( nchar, length, data%KEY,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  initialize each table entry as unfilled

      data%TABLE( : length ) = - data%hash_empty
      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine HASH_initialize

      END SUBROUTINE HASH_initialize

!- G A L A H A D -  H A S H _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE HASH_full_initialize( nchar, length, data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for HASH controls

!   Arguments:

!   see above

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: nchar, length
     TYPE ( HASH_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( HASH_control_type ), INTENT( OUT ) :: control
     TYPE ( HASH_inform_type ), INTENT( OUT ) :: inform

     CALL HASH_initialize( nchar, length, data%hash_data, control, inform )

     RETURN

!  End of subroutine HASH_full_initialize

     END SUBROUTINE HASH_full_initialize

!-*-*-*- G A L A H A D   H A S H _ i n s e r t    S U B R O U T I N E -*-*-*-*-

      SUBROUTINE HASH_insert( nchar, FIELD, position, data, control, inform )

!  -------------------------------------------------------------------
!  insert in a chained scatter table (Williams, CACM 2, 21-24, 1959)

!  nchar is an upper bound on the number of characters in each word
!  FIELD(i) gives the ith component of the field to be hashed
!  position is the index of the table that data will occupy
!  data     private internal data
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  -------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nchar
      INTEGER, INTENT( OUT ) :: position
      CHARACTER ( LEN = 1 ), INTENT( IN ) :: FIELD( nchar )
      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i

      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( ' entering HASH_insert' )" )

!  find a starting position, position, for the insertion

      position = HASH_field( nchar, data%hash_prime, FIELD )

!  is there a list?

      IF ( data%TABLE( position ) >= 0 ) THEN

!  compare to see if the key has been found

   10   CONTINUE
        DO i = 1, data%nchar
          IF ( FIELD( i ) /= data%KEY( i, position ) ) GO TO 20
        END DO

!  the key already exists and therefore cannot be inserted

        IF ( data%TABLE( position ) >= 0 ) THEN
          position = - position ; GO TO 100
        END IF

!  the key used to exist but has been deleted and must be restored

        GO TO 90

!  advance along the chain to the next entry

   20   CONTINUE
        IF ( data%TABLE( position ) /= 0 ) THEN
          position = IABS( data%TABLE( position ) )
          GO TO 10
        END IF

!  the end of the chain has been reached. Find empty entry in the table

   30   CONTINUE
        data%hash_empty = data%hash_empty - 1
        IF ( data%hash_empty == 0 ) THEN
          position = 0 ; GO TO 100
        END IF
        IF ( data%TABLE( data%hash_empty ) >= - data%length ) GO TO 30
        data%TABLE( position ) = data%hash_empty
        position = data%hash_empty

!  the starting entry for the chain is unused

      ELSE
        IF ( data%TABLE( position ) >= - data%length ) THEN
          data%TABLE( position ) = - data%TABLE( position )
          GO TO 90
        END IF
      END IF

!  there is no link from the newly inserted field

      data%TABLE( position ) = 0

!  insert new key

   90 CONTINUE
      data%KEY( : data%nchar, position ) = FIELD( : data%nchar )

  100 CONTINUE
      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine HASH_insert

      END SUBROUTINE HASH_insert

!-*-*- G A L A H A D   H A S H _ s e a r c h    S U B R O U T I N E -*-*-

      SUBROUTINE HASH_search( nchar, FIELD, position, data, control, inform )

!  -------------------------------------------------------------------
!  search within chained scatter table (Williams, CACM 2, 21-24, 1959)

!  nchar is an upper bound on the number of characters in each word
!  FIELD(i) gives the ith component of the field to be hashed
!  position is the index of the table that data for the field occupies
!  data     private internal data
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  -------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nchar
      INTEGER, INTENT( OUT ) :: position
      CHARACTER ( LEN = 1 ), INTENT( IN ) :: FIELD( nchar )
      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i

      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( ' entering HASH_search' )" )

!  find a starting position, position, for the chain leading to the required
!  location

      position = HASH_field( nchar, data%hash_prime, FIELD )

!  is there a list?

      IF ( data%TABLE( position ) < - data%length ) THEN
        position = 0 ; GO TO 100
      END IF

!  compare to see if the key has been found

   10 CONTINUE

!  advance to next

      DO i = 1, data%nchar
        IF ( FIELD( i ) /= data%KEY( i, position ) ) THEN
          IF ( data%TABLE( position ) == 0 ) THEN
            position = 0 ; GO TO 100
          END IF
          position = IABS( data%TABLE( position ) )
          GO TO 10
        END IF
      END DO

!  check that the table item has not been removed

      IF ( data%TABLE( position ) < 0 ) position = - position

  100 CONTINUE
      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine HASH_search

      END SUBROUTINE HASH_search

!-*-*-*-  G A L A H A D   H A S H _ r e m o v e    S U B R O U T I N E -*-*-*-

      SUBROUTINE HASH_remove( nchar, FIELD, position, data, control, inform )

!  ---------------------------------------------------------------------------
!  remove a field from a chained scatter table (Williams, CACM 2, 21-24, 1959)

!  nchar is an upper bound on the number of characters in each word
!  FIELD(i) gives the ith component of the field to be removed
!  position is the index of the table that data for the field occupied
!  data     private internal data
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  ----------------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: nchar
      INTEGER, INTENT( OUT ) :: position
      CHARACTER ( LEN = 1 ), INTENT( IN ) :: FIELD( nchar )
      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: i, i_delete, i_penultimate

      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( ' entering HASH_remove' )" )

!  find a starting position, position, for the insertion by hashing field

      position = HASH_field( nchar, data%hash_prime, FIELD )

!  initialize the addresses of the penultimate entry in the chain and of the 
!  entry which is to be removed

      i_penultimate = 0 ; i_delete = 0

!  is there a list?

      IF ( data%TABLE( position ) >= - data%length ) THEN

!  compare to see if the key has been found

   10   CONTINUE
        DO i = 1, data%nchar
          IF ( FIELD( I ) /= data%KEY( i, position ) ) GO TO 20
        END DO

!  record the address of the element to be deleted

        i_delete = position

!  advance to next

   20   CONTINUE
        IF ( data%TABLE( position ) /= 0 ) THEN
          i_penultimate = position
          position = ABS( data%TABLE( position ) )
          GO TO 10
        END IF
      END IF

!  check that the field exists in the table

      IF ( i_delete == 0 ) THEN
        position = 0 ; GO TO 900
      END IF

!  check that the chain is not trivial

      IF ( i_penultimate == 0 ) THEN
        data%TABLE( position ) = - data%length - 1 ; GO TO 900
      END IF

!  if the end of the chain does not coincide with the entry to be removed, 
!  move the field key at the chain's end to replace the deleted field key

      IF ( position /= i_delete ) THEN
        DO i = 1, data%nchar
          data%KEY( i, i_delete ) = data%KEY( i, position )
        END DO
      END IF

!  cut the link from the penultimate entry on the chain to the end entry

      data%TABLE( i_penultimate ) = 0
      data%TABLE( position ) = - data%length - 1
      position = i_delete

  900 CONTINUE
      inform%status = GALAHAD_ok
      RETURN

!  end OF subroutine HASH_remove

      END SUBROUTINE HASH_remove

!-*-*- G A L A H A D   H A S H _ r e b u i l d    S U B R O U T I N E -*-*-

      SUBROUTINE HASH_rebuild( length, new_length, MOVED_TO, data, control,    &
                               inform )

!  -------------------------------------------------------------------
!  rebuild the chained scatter table (Williams, CACM 2, 21-24, 1959)
!  to account for an increase in its length

!  length is the current length
!  new_length is the new (increased) length
!  MOVED_TO is an array that gives the position in the new table
!    that the old table entry has been moved to. Any 0 entry was prevously
!    unoccupied
!  data     private internal data
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  -------------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: length, new_length
      INTEGER, DIMENSION( length ), INTENT( OUT ) :: MOVED_TO
      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER :: position, k, old_length
      CHARACTER ( LEN = 80 ) :: array_name

      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( ' entering HASH_rebuild' )" )

!  make a temporary copy of the table data

      array_name = 'hash: dataa%TABLE_temp'
      CALL SPACE_resize_array( data%length, data%TABLE_temp,                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      data%TABLE_temp( : data%length ) = data%TABLE( : data%length )

!  free the current table space

      array_name = 'hash: data%TABLE'
      CALL SPACE_dealloc_array( data%TABLE,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  make a temporary copy of the key data

      array_name = 'hash: data%KEY_temp'
      CALL SPACE_resize_array( data%nchar, data%length, data%KEY_temp,         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      DO k = 1, data%length
        IF ( data%TABLE_temp( k ) >= 0 )                                       &
          data%KEY_temp( : data%nchar, k ) = data%KEY( : data%nchar, k )
      END DO

!  free the current key space

      array_name = 'hash: data%KEY'
      CALL SPACE_dealloc_array( data%KEY,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  record the old length

      old_length = data%length

!  reinitialize the scatter table for the new length

      CALL HASH_initialize( data%nchar, new_length, data, control, inform )

!  allocate temporary space of the key data

      array_name = 'hash: data%FIELD_temp'
      CALL SPACE_resize_array( data%nchar, data%FIELD_temp,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  initialize the "moved to" array

      MOVED_TO = 0

!  run through the entries in the old table seeing if the kth is empty

      DO k = 1, old_length

!  if the kth entry was previously occupied, copy its key into the new table

        IF ( data%TABLE_temp( k ) >= 0 ) THEN
          data%FIELD_temp( : data%nchar ) = data%KEY_temp( : data%nchar, k )
          CALL HASH_insert( data%nchar, data%FIELD_temp, position,             &
                            data, control, inform )

!  record that old entry k has moved to position in the new table

          MOVED_TO( k ) = position

!  check that there is sufficient space

          IF ( position == 0 ) THEN
            inform%status = - 1
            RETURN
          END IF
        END IF
      END DO

      data%length = new_length

!  deallocate temporary storage

      array_name = 'hash: data%TABLE_temp'
      CALL SPACE_dealloc_array( data%TABLE_temp,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'hash: data%KEY_temp'
      CALL SPACE_dealloc_array( data%KEY_temp,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'hash: data%FIELD_temp'
      CALL SPACE_dealloc_array( data%FIELD_temp,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine HASH_rebuild

      END SUBROUTINE HASH_rebuild

!-*-*-*-*-  G A L A H A D   H A S H _ v a l u e    F U N C T I O N  -*-*-*-*-*-

      INTEGER FUNCTION HASH_value( IVALUE, hash_prime )
      INTEGER :: IVALUE( 2 )
      REAL ( KIND = dp ) :: hash_prime

!  -------------------------------------
!  a hash function proposed by John Reid
!  -------------------------------------

      HASH_value = INT( DMOD( DBLE( IVALUE( 1 ) ) + IVALUE( 2 ), hash_prime ) )
      HASH_value = ABS( HASH_value ) + 1

      RETURN

!  end of function HASH_value

      END FUNCTION HASH_value

!-*-*-*-*-   G A L A H A D   H A S H _ f i e l d    F U N C T I O N   -*-*-*-*-

      FUNCTION HASH_field( nchar, hash_prime, FIELD )

!  hash the string FIELD

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER :: HASH_field
      INTEGER, INTENT( IN ) :: nchar
      REAL ( KIND = dp ), INTENT( IN ) :: hash_prime
      CHARACTER ( LEN = 1 ), INTENT( IN ) :: FIELD( nchar )

!  local variables

      INTEGER :: i, j, k
      CHARACTER ( LEN = 1 ) :: BFIELD( nbytes )
      INTEGER :: IVALUE( 2 )

!  perform hashing on 8 characters of FIELD at a time

      HASH_field = 0
      DO j = 1, nchar, nbytes
        DO i = 1, nbytes
          k = j + i - 1
          IF ( k <= nchar ) THEN
            BFIELD( i ) = FIELD( k )
          ELSE
            BFIELD( i ) = ' '
          END IF
        END DO

!  convert the character string into two integer numbers

        IVALUE( 1 ) = ICHAR( BFIELD( 1 ) ) / 2
        IVALUE( 2 ) = ICHAR( BFIELD( nbytes_by_2 + 1 ) ) / 2
        DO i = 2, nbytes_by_2
          IVALUE( 1 ) = 256 * IVALUE( 1 ) + ICHAR( BFIELD( i ) )
          IVALUE( 2 ) = 256 * IVALUE( 2 ) + ICHAR( BFIELD( nbytes_by_2 + i ) )
        END DO

!  hash and add the result to HASH_field

        HASH_field = HASH_field + HASH_value( IVALUE( 1 ), hash_prime )
      END DO

!  ensure that HASH_field lies within the allowed range

      HASH_field = MOD( HASH_field, IDINT( hash_prime ) ) + 1

!  end of FUNCTION HASH_field

   END FUNCTION HASH_field

!-*-*-*- G A L A H A D   H A S H  _ i s _ p r i m e    F U N C T I O N -*-*-*-

      LOGICAL FUNCTION HASH_is_prime( prime )
      INTEGER :: prime

!  -------------------------------------------
!  returns the value .TRUE. if prime is prime
!  -------------------------------------------

!  local variables

      INTEGER :: i

      HASH_is_prime = .FALSE.
      IF ( MOD( prime, 2 ) == 0 ) RETURN
      DO i = 3, INT( DSQRT( DBLE( prime ) ) ), 2
        IF ( MOD( prime, i ) == 0 ) RETURN
      END DO
      HASH_is_prime = .TRUE.

      RETURN

!  end of function HASH_is_prime

      END FUNCTION HASH_is_prime

!-*-  G A L A H A D   H A S H _ i n i t i a l i z e    S U B R O U T I N E  -*-

      SUBROUTINE HASH_terminate( data, control, inform )

!  ------------------------------------------------------------
!  deallocate internal workspace

!  data     private internal data
!  control  a structure containing control parameters. See preamble
!  inform   a structure containing output information. See preamble
!  ------------------------------------------------------------

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( HASH_data_type ), INTENT( INOUT ) :: data
      TYPE ( HASH_control_type ), INTENT( IN ) :: control
      TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'hash: data%TABLE'
      CALL SPACE_dealloc_array( data%TABLE,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'hash: data%KEY'
      CALL SPACE_dealloc_array( data%KEY,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine HASH_terminate

      END SUBROUTINE HASH_terminate

! -  G A L A H A D -  H A S H _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE HASH_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HASH_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( HASH_control_type ), INTENT( IN ) :: control
     TYPE ( HASH_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL HASH_terminate( data%hash_data, control, inform )

!  deallocate any internal problem arrays

     RETURN

!  End of subroutine HASH_full_terminate

     END SUBROUTINE HASH_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  H A S H _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE HASH_import( control, data, status )

!  import problem data into internal storage prior to solution. 
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading 
!   comments to HASH_solve
!
!  data is a scalar variable of type HASH_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    0. The import was succesful
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. An input restriction has been violated.

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HASH_control_type ), INTENT( INOUT ) :: control
     TYPE ( HASH_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: status

!  local variables

     INTEGER :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     data%hash_control = control

     error = data%hash_control%error
     space_critical = data%hash_control%space_critical
     deallocate_error_fatal = data%hash_control%space_critical

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%hash_inform%status
     RETURN

!  End of subroutine HASH_import

     END SUBROUTINE HASH_import

!-  G A L A H A D -  H A S H _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE HASH_information( data, inform, status )

!  return solver information during or after solution by HASH
!  See HASH_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HASH_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( HASH_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%hash_inform
     
!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine HASH_information

     END SUBROUTINE HASH_information

!  end of module GALAHAD_HASH

    END MODULE GALAHAD_HASH
