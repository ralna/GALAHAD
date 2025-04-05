! THIS VERSION: GALAHAD 5.2 - 2025-03-30 AT 13:50 GMT

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-  G A L A H A D _ N O D E N D   C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 5.2. March 13th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!  C interface module to GALAHAD_NODEND types and interfaces

  MODULE GALAHAD_NODEND_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_NODEND_precision, ONLY:                                        &
        f_nodend_control_type   => NODEND_control_type,                        &
        f_nodend_time_type      => NODEND_time_type,                           &
        f_nodend_inform_type    => NODEND_inform_type,                         &
        f_nodend_full_data_type => NODEND_full_data_type,                      &
        f_nodend_initialize     => NODEND_initialize,                          &
        f_nodend_read_specfile  => NODEND_read_specfile,                       &
        f_nodend_order_a        => NODEND_order_a,                             &
        f_nodend_information    => NODEND_information
    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: nodend_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: version
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: no_metis_4_use_5_instead
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      INTEGER ( KIND = ipc_ ) :: metis4_ptype
      INTEGER ( KIND = ipc_ ) :: metis4_ctype
      INTEGER ( KIND = ipc_ ) :: metis4_itype
      INTEGER ( KIND = ipc_ ) :: metis4_rtype
      INTEGER ( KIND = ipc_ ) :: metis4_dbglvl
      INTEGER ( KIND = ipc_ ) :: metis4_oflags
      INTEGER ( KIND = ipc_ ) :: metis4_pfactor
      INTEGER ( KIND = ipc_ ) :: metis4_nseps
      INTEGER ( KIND = ipc_ ) :: metis5_ptype
      INTEGER ( KIND = ipc_ ) :: metis5_objtype
      INTEGER ( KIND = ipc_ ) :: metis5_ctype
      INTEGER ( KIND = ipc_ ) :: metis5_iptype
      INTEGER ( KIND = ipc_ ) :: metis5_rtype
      INTEGER ( KIND = ipc_ ) :: metis5_dbglvl
      INTEGER ( KIND = ipc_ ) :: metis5_niter
      INTEGER ( KIND = ipc_ ) :: metis5_ncuts
      INTEGER ( KIND = ipc_ ) :: metis5_seed
      INTEGER ( KIND = ipc_ ) :: metis5_no2hop
      INTEGER ( KIND = ipc_ ) :: metis5_minconn
      INTEGER ( KIND = ipc_ ) :: metis5_contig
      INTEGER ( KIND = ipc_ ) :: metis5_compress
      INTEGER ( KIND = ipc_ ) :: metis5_ccorder
      INTEGER ( KIND = ipc_ ) :: metis5_pfactor
      INTEGER ( KIND = ipc_ ) :: metis5_nseps
      INTEGER ( KIND = ipc_ ) :: metis5_ufactor
      INTEGER ( KIND = ipc_ ) :: metis5_niparts
      INTEGER ( KIND = ipc_ ) :: metis5_ondisk
      INTEGER ( KIND = ipc_ ) :: metis5_dropedges
      INTEGER ( KIND = ipc_ ) :: metis5_twohop
      INTEGER ( KIND = ipc_ ) :: metis5_fast
    END TYPE nodend_control_type

    TYPE, BIND( C ) :: nodend_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: metis
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_metis
    END TYPE nodend_time_type

    TYPE, BIND( C ) :: nodend_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 4 ) :: version
      TYPE ( nodend_time_type ) :: time
    END TYPE nodend_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( nodend_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_nodend_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

   ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%metis4_ptype = ccontrol%metis4_ptype
    fcontrol%metis4_ctype = ccontrol%metis4_ctype
    fcontrol%metis4_itype = ccontrol%metis4_itype
    fcontrol%metis4_rtype = ccontrol%metis4_rtype
    fcontrol%metis4_dbglvl = ccontrol%metis4_dbglvl
    fcontrol%metis4_oflags = ccontrol%metis4_oflags
    fcontrol%metis4_pfactor = ccontrol%metis4_pfactor
    fcontrol%metis4_nseps = ccontrol%metis4_nseps
    fcontrol%metis5_ptype = ccontrol%metis5_ptype
    fcontrol%metis5_objtype = ccontrol%metis5_objtype
    fcontrol%metis5_ctype = ccontrol%metis5_ctype
    fcontrol%metis5_iptype = ccontrol%metis5_iptype
    fcontrol%metis5_rtype = ccontrol%metis5_rtype
    fcontrol%metis5_dbglvl = ccontrol%metis5_dbglvl
    fcontrol%metis5_niter = ccontrol%metis5_niter
    fcontrol%metis5_ncuts = ccontrol%metis5_ncuts
    fcontrol%metis5_seed = ccontrol%metis5_seed
    fcontrol%metis5_no2hop = ccontrol%metis5_no2hop
    fcontrol%metis5_minconn = ccontrol%metis5_minconn
    fcontrol%metis5_contig = ccontrol%metis5_contig
    fcontrol%metis5_compress = ccontrol%metis5_compress
    fcontrol%metis5_ccorder = ccontrol%metis5_ccorder
    fcontrol%metis5_pfactor = ccontrol%metis5_pfactor
    fcontrol%metis5_nseps = ccontrol%metis5_nseps
    fcontrol%metis5_ufactor = ccontrol%metis5_ufactor
    fcontrol%metis5_niparts = ccontrol%metis5_niparts
    fcontrol%metis5_ondisk = ccontrol%metis5_ondisk
    fcontrol%metis5_dropedges = ccontrol%metis5_dropedges
    fcontrol%metis5_twohop = ccontrol%metis5_twohop
    fcontrol%metis5_fast = ccontrol%metis5_fast

    ! Logicals
    fcontrol%no_metis_4_use_5_instead = ccontrol%no_metis_4_use_5_instead

    ! Strings
    DO i = 1, LEN( fcontrol%version )
      IF ( ccontrol%version( i ) == C_NULL_CHAR ) EXIT
      fcontrol%version( i : i ) = ccontrol%version( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_nodend_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( nodend_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%metis4_ptype = fcontrol%metis4_ptype
    ccontrol%metis4_ctype = fcontrol%metis4_ctype
    ccontrol%metis4_itype = fcontrol%metis4_itype
    ccontrol%metis4_rtype = fcontrol%metis4_rtype
    ccontrol%metis4_dbglvl = fcontrol%metis4_dbglvl
    ccontrol%metis4_oflags = fcontrol%metis4_oflags
    ccontrol%metis4_pfactor = fcontrol%metis4_pfactor
    ccontrol%metis4_nseps = fcontrol%metis4_nseps
    ccontrol%metis5_ptype = fcontrol%metis5_ptype
    ccontrol%metis5_objtype = fcontrol%metis5_objtype
    ccontrol%metis5_ctype = fcontrol%metis5_ctype
    ccontrol%metis5_iptype = fcontrol%metis5_iptype
    ccontrol%metis5_rtype = fcontrol%metis5_rtype
    ccontrol%metis5_dbglvl = fcontrol%metis5_dbglvl
    ccontrol%metis5_niter = fcontrol%metis5_niter
    ccontrol%metis5_ncuts = fcontrol%metis5_ncuts
    ccontrol%metis5_seed = fcontrol%metis5_seed
    ccontrol%metis5_no2hop = fcontrol%metis5_no2hop
    ccontrol%metis5_minconn = fcontrol%metis5_minconn
    ccontrol%metis5_contig = fcontrol%metis5_contig
    ccontrol%metis5_compress = fcontrol%metis5_compress
    ccontrol%metis5_ccorder = fcontrol%metis5_ccorder
    ccontrol%metis5_pfactor = fcontrol%metis5_pfactor
    ccontrol%metis5_nseps = fcontrol%metis5_nseps
    ccontrol%metis5_ufactor = fcontrol%metis5_ufactor
    ccontrol%metis5_niparts = fcontrol%metis5_niparts
    ccontrol%metis5_ondisk = fcontrol%metis5_ondisk
    ccontrol%metis5_dropedges = fcontrol%metis5_dropedges
    ccontrol%metis5_twohop = fcontrol%metis5_twohop
    ccontrol%metis5_fast = fcontrol%metis5_fast

    ! Logicals
    ccontrol%no_metis_4_use_5_instead = fcontrol%no_metis_4_use_5_instead

    ! Strings
    l = LEN( fcontrol%version )
    DO i = 1, l
      ccontrol%version( i ) = fcontrol%version( i : i )
    END DO
    ccontrol%version( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( nodend_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_nodend_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%metis = ctime%metis
    ftime%clock_total = ctime%clock_total
    ftime%clock_metis = ctime%clock_metis
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_nodend_time_type ), INTENT( IN ) :: ftime
    TYPE ( nodend_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%metis = ftime%metis
    ctime%clock_total = ftime%clock_total
    ctime%clock_metis = ftime%clock_metis
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( nodend_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_nodend_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, LEN( finform%version )
      IF ( cinform%version( i ) == C_NULL_CHAR ) EXIT
      finform%version( i : i ) = cinform%version( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran information parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_nodend_inform_type ), INTENT( IN ) :: finform
    TYPE ( nodend_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    DO i = 1, 3
      cinform%version( i ) = finform%version( i : i )
    END DO
    cinform%version( 4 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_NODEND_precision_ciface

!  ----------------------------------------
!  C interface to fortran nodend_initialize
!  ----------------------------------------

  SUBROUTINE nodend_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_NODEND_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( nodend_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_nodend_full_data_type ), POINTER :: fdata
  TYPE ( f_nodend_control_type ) :: fcontrol
  TYPE ( f_nodend_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_nodend_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE nodend_initialize

!  -------------------------------------------
!  C interface to fortran nodend_read_specfile
!  -------------------------------------------

  SUBROUTINE nodend_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_NODEND_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( nodend_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_nodend_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_nodend_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE nodend_read_specfile

!  -----------------------------------
!  C interface to fortran nodend_order
!  -----------------------------------

  SUBROUTINE nodend_order( ccontrol, cdata, status, n, perm,                   &
                           chtype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_NODEND_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( nodend_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: aptr
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: perm
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_nodend_control_type ) :: fcontrol
  TYPE ( f_nodend_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  order the problem data into the required NODEND structure

  CALL f_nodend_order_a( fcontrol, fdata, status, n, perm, fhtype,             &
                         ane, arow, acol, aptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE nodend_order

!  -----------------------------------------
!  C interface to fortran nodend_information
!  -----------------------------------------

  SUBROUTINE nodend_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_NODEND_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( nodend_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_nodend_full_data_type ), pointer :: fdata
  TYPE ( f_nodend_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain NODEND solution information

  CALL f_nodend_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE nodend_information




