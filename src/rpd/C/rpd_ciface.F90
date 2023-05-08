! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  R P D    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_RPD_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_RPD_precision, ONLY:                                           &
        f_rpd_control_type   => RPD_control_type,                              &
        f_rpd_inform_type    => RPD_inform_type,                               &
        f_rpd_full_data_type => RPD_full_data_type,                            &
        f_rpd_initialize     => RPD_initialize,                                &
        f_rpd_get_stats      => RPD_get_stats,                                 &
        f_rpd_get_g          => RPD_get_g,                                     &
        f_rpd_get_f          => RPD_get_f,                                     &
        f_rpd_get_xlu        => RPD_get_xlu,                                   &
        f_rpd_get_clu        => RPD_get_clu,                                   &
        f_rpd_get_H          => RPD_get_H,                                     &
        f_rpd_get_A          => RPD_get_A,                                     &
        f_rpd_get_H_c        => RPD_get_H_c,                                   &
        f_rpd_get_x_type     => RPD_get_x_type,                                &
        f_rpd_get_x          => RPD_get_x,                                     &
        f_rpd_get_y          => RPD_get_y,                                     &
        f_rpd_get_z          => RPD_get_z,                                     &
        f_rpd_information    => RPD_information,                               &
        f_rpd_terminate      => RPD_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: rpd_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: qplib
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
    END TYPE rpd_control_type

    TYPE, BIND( C ) :: rpd_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: io_status
      INTEGER ( KIND = ipc_ ) :: line
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 4 ) :: p_type
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
    END TYPE rpd_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( rpd_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_rpd_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%qplib = ccontrol%qplib
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level

    ! Logicals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_rpd_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( rpd_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%qplib = fcontrol%qplib
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level

    ! Logicals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( rpd_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_rpd_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%io_status = cinform%io_status
    finform%line = cinform%line

    ! Strings
    DO i = 1, LEN( finform%p_type )
      IF ( cinform%p_type( i ) == C_NULL_CHAR ) EXIT
      finform%p_type( i : i ) = cinform%p_type( i )
    END DO
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_rpd_inform_type ), INTENT( IN ) :: finform
    TYPE ( rpd_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%io_status = finform%io_status
    cinform%line = finform%line

    ! Strings
    l = LEN( finform%p_type )
    DO i = 1, l
      cinform%p_type( i ) = finform%p_type( i : i )
    END DO
    cinform%p_type( l + 1 ) = C_NULL_CHAR
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_RPD_precision_ciface

!  -------------------------------------
!  C interface to fortran rpd_initialize
!  -------------------------------------

  SUBROUTINE rpd_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( rpd_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata
  TYPE ( f_rpd_control_type ) :: fcontrol
  TYPE ( f_rpd_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_rpd_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE rpd_initialize

!  ------------------------------------
!  C interface to fortran rpd_get_stats
!  ------------------------------------

  SUBROUTINE rpd_get_stats( qplib_file, qplib_file_len, ccontrol, cdata,       &
                            status, p_type, n, m, h_ne, a_ne, h_c_ne ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  CHARACTER ( KIND = C_CHAR ), DIMENSION( 80 ) :: qplib_file
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: qplib_file_len
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status, n, m, h_ne, a_ne, h_c_ne
  TYPE ( rpd_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  CHARACTER ( KIND = C_CHAR ), DIMENSION( 4 ) :: p_type

!  local variables

  TYPE ( f_rpd_control_type ) :: fcontrol
  TYPE ( f_rpd_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing
  CHARACTER ( LEN = 4 ) :: fp_type
  CHARACTER ( LEN = 1001 ) :: fqplib_file
  INTEGER ( KIND = ip_ ) :: i

!  copy QPLIB filename to a fortran string

  DO i = 1, qplib_file_len
    fqplib_file( i : i ) = qplib_file( i )
  END DO

!  copy control to fortran

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  open the QPLIB file

  OPEN( fcontrol%qplib, file = fqplib_file(:qplib_file_len),                   &
        FORM = 'FORMATTED', STATUS = 'OLD' )

!  get statistics

  CALL f_rpd_get_stats( fcontrol, fdata, status, fp_type,                      &
                        n, m, h_ne, a_ne, h_c_ne )

!  close the QPLIB file after use

  CLOSE( fcontrol%qplib )

!  translate fortran character string to a c one

  p_type( 1 : 3 ) = fp_type( 1 : 3 )
  p_type( 4 ) = C_NULL_CHAR
  RETURN

  END SUBROUTINE rpd_get_stats

!  --------------------------------
!  C interface to fortran rpd_get_g
!  --------------------------------

  SUBROUTINE rpd_get_g( cdata, status, n, g  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get g

  CALL f_rpd_get_g( fdata, status, g )
  RETURN

  END SUBROUTINE rpd_get_g

!  --------------------------------
!  C interface to fortran rpd_get_f
!  --------------------------------

  SUBROUTINE rpd_get_f( cdata, status, f  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get f

  CALL f_rpd_get_f( fdata, status, f )
  RETURN

  END SUBROUTINE rpd_get_f


!  ----------------------------------
!  C interface to fortran rpd_get_xlu
!  ----------------------------------

  SUBROUTINE rpd_get_xlu( cdata, status, n, x_l, x_u  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: x_l, x_u
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get x_l and x_u

  CALL f_rpd_get_xlu( fdata, status, x_l, x_u )
  RETURN

  END SUBROUTINE rpd_get_xlu

!  ----------------------------------
!  C interface to fortran rpd_get_clu
!  ----------------------------------

  SUBROUTINE rpd_get_clu( cdata, status, m, c_l, c_u  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c_l, c_u
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get c_l and c_u

  CALL f_rpd_get_clu( fdata, status, c_l, c_u )
  RETURN

  END SUBROUTINE rpd_get_clu

!  --------------------------------
!  C interface to fortran rpd_get_h
!  --------------------------------

  SUBROUTINE rpd_get_h( cdata, status, h_ne, h_row, h_col, h_val ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: h_ne
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( h_ne ) :: h_row, h_col
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( h_ne ) :: h_val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get h

  CALL f_rpd_get_h( fdata, status, h_row, h_col, h_val )

!  handle C sparse matrix indexing

  IF ( status == 0 .AND. .NOT. fdata%f_indexing ) THEN
    h_row = h_row - 1
    h_col = h_col - 1
  END IF

  RETURN

  END SUBROUTINE rpd_get_h

!  --------------------------------
!  C interface to fortran rpd_get_a
!  --------------------------------

  SUBROUTINE rpd_get_a( cdata, status, a_ne, a_row, a_col, a_val ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: a_ne
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( a_ne ) :: a_row, a_col
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( a_ne ) :: a_val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get a

  CALL f_rpd_get_a( fdata, status, a_row, a_col, a_val )

!  handle C sparse matrix indexing

  IF ( status == 0 .AND. .NOT. fdata%f_indexing ) THEN
    a_row = a_row - 1
    a_col = a_col - 1
  END IF

  RETURN

  END SUBROUTINE rpd_get_a

!  ----------------------------------
!  C interface to fortran rpd_get_h_c
!  ----------------------------------

  SUBROUTINE rpd_get_h_c( cdata, status, h_c_ne,                               &
                          h_c_ptr, h_c_row, h_c_col, h_c_val ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: h_c_ne
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( h_c_ne ) :: h_c_ptr
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( h_c_ne ) :: h_c_row
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( h_c_ne ) :: h_c_col
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( h_c_ne ) :: h_c_val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get h_c

  CALL f_rpd_get_h_c( fdata, status, h_c_ptr, h_c_row, h_c_col, h_c_val )
  IF ( status /= 0 ) RETURN

!  handle C sparse matrix indexing

  IF ( status == 0 .AND. .NOT. fdata%f_indexing ) THEN
    h_c_ptr = h_c_ptr - 1
    h_c_row = h_c_row - 1
    h_c_col = h_c_col - 1
  END IF

  RETURN

  END SUBROUTINE rpd_get_h_c

!  -------------------------------------
!  C interface to fortran rpd_get_x_type
!  -------------------------------------

  SUBROUTINE rpd_get_x_type( cdata, status, n, x_type  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: x_type
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get x_type

  CALL f_rpd_get_x_type( fdata, status, x_type )
  RETURN

  END SUBROUTINE rpd_get_x_type

!  --------------------------------
!  C interface to fortran rpd_get_x
!  --------------------------------

  SUBROUTINE rpd_get_x( cdata, status, n, x  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: x
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get x

  CALL f_rpd_get_g( fdata, status, x )
  RETURN

  END SUBROUTINE rpd_get_x

!  --------------------------------
!  C interface to fortran rpd_get_y
!  --------------------------------

  SUBROUTINE rpd_get_y( cdata, status, m, y  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get y

  CALL f_rpd_get_y( fdata, status, y )
  RETURN

  END SUBROUTINE rpd_get_y

!  --------------------------------
!  C interface to fortran rpd_get_z
!  --------------------------------

  SUBROUTINE rpd_get_z( cdata, status, n, z  ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: z
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  get z

  CALL f_rpd_get_z( fdata, status, z )
  RETURN

  END SUBROUTINE rpd_get_z

!  --------------------------------------
!  C interface to fortran rpd_information
!  --------------------------------------

  SUBROUTINE rpd_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( rpd_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_rpd_full_data_type ), POINTER :: fdata
  TYPE ( f_rpd_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain RPD solution information

  CALL f_rpd_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE rpd_information

!  ------------------------------------
!  C interface to fortran rpd_terminate
!  ------------------------------------

  SUBROUTINE rpd_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_RPD_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( rpd_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( rpd_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_rpd_full_data_type ), pointer :: fdata
  TYPE ( f_rpd_control_type ) :: fcontrol
  TYPE ( f_rpd_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_rpd_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE rpd_terminate
