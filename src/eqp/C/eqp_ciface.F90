! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  E Q P    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 7th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_EQP_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_EQP_precision, ONLY:                                           &
        f_eqp_control_type => EQP_control_type,                                &
        f_eqp_time_type => EQP_time_type,                                      &
        f_eqp_inform_type => EQP_inform_type,                                  &
        f_eqp_full_data_type => EQP_full_data_type,                            &
        f_eqp_initialize => EQP_initialize,                                    &
        f_eqp_read_specfile => EQP_read_specfile,                              &
        f_eqp_import => EQP_import,                                            &
        f_eqp_solve_qp => EQP_solve_qp,                                        &
        f_eqp_solve_sldqp => EQP_solve_sldqp,                                  &
        f_eqp_resolve_qp => EQP_resolve_qp,                                    &
        f_eqp_reset_control => EQP_reset_control,                              &
        f_eqp_information => EQP_information,                                  &
        f_eqp_terminate => EQP_terminate

    USE GALAHAD_FDC_precision_ciface, ONLY:                                    &
        fdc_inform_type,                                                       &
        fdc_control_type,                                                      &
        copy_fdc_inform_in => copy_inform_in,                                  &
        copy_fdc_inform_out => copy_inform_out,                                &
        copy_fdc_control_in => copy_control_in,                                &
        copy_fdc_control_out => copy_control_out

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in => copy_inform_in,                                 &
        copy_sbls_inform_out => copy_inform_out,                               &
        copy_sbls_control_in => copy_control_in,                               &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_GLTR_precision_ciface, ONLY:                                   &
        gltr_inform_type,                                                      &
        gltr_control_type,                                                     &
        copy_gltr_inform_in => copy_inform_in,                                 &
        copy_gltr_inform_out => copy_inform_out,                               &
        copy_gltr_control_in => copy_control_in,                               &
        copy_gltr_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: eqp_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: factorization
      INTEGER ( KIND = ipc_ ) :: max_col
      INTEGER ( KIND = ipc_ ) :: indmin
      INTEGER ( KIND = ipc_ ) :: valmin
      INTEGER ( KIND = ipc_ ) :: len_ulsmin
      INTEGER ( KIND = ipc_ ) :: itref_max
      INTEGER ( KIND = ipc_ ) :: cg_maxit
      INTEGER ( KIND = ipc_ ) :: preconditioner
      INTEGER ( KIND = ipc_ ) :: semi_bandwidth
      INTEGER ( KIND = ipc_ ) :: new_a
      INTEGER ( KIND = ipc_ ) :: new_h
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: pivot_tol_for_basis
      REAL ( KIND = rpc_ ) :: zero_pivot
      REAL ( KIND = rpc_ ) :: inner_fraction_opt
      REAL ( KIND = rpc_ ) :: radius
      REAL ( KIND = rpc_ ) :: min_diagonal
      REAL ( KIND = rpc_ ) :: max_infeasibility_relative
      REAL ( KIND = rpc_ ) :: max_infeasibility_absolute
      REAL ( KIND = rpc_ ) :: inner_stop_relative
      REAL ( KIND = rpc_ ) :: inner_stop_absolute
      REAL ( KIND = rpc_ ) :: inner_stop_inter
      LOGICAL ( KIND = C_BOOL ) :: find_basis_by_transpose
      LOGICAL ( KIND = C_BOOL ) :: remove_dependencies
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( fdc_control_type ) :: fdc_control
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( gltr_control_type ) :: gltr_control
    END TYPE eqp_control_type

    TYPE, BIND( C ) :: eqp_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: find_dependent
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: solve_inter
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_find_dependent
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE eqp_time_type

    TYPE, BIND( C ) :: eqp_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: cg_iter
      INTEGER ( KIND = ipc_ ) :: cg_iter_inter
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      REAL ( KIND = rpc_ ) :: obj
      TYPE ( eqp_time_type ) :: time
      TYPE ( fdc_inform_type ) :: fdc_inform
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( gltr_inform_type ) :: gltr_inform
    END TYPE eqp_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( eqp_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_eqp_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%factorization = ccontrol%factorization
    fcontrol%max_col = ccontrol%max_col
    fcontrol%indmin = ccontrol%indmin
    fcontrol%valmin = ccontrol%valmin
    fcontrol%len_ulsmin = ccontrol%len_ulsmin
    fcontrol%itref_max = ccontrol%itref_max
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%preconditioner = ccontrol%preconditioner
    fcontrol%semi_bandwidth = ccontrol%semi_bandwidth
    fcontrol%new_a = ccontrol%new_a
    fcontrol%new_h = ccontrol%new_h
    fcontrol%sif_file_device = ccontrol%sif_file_device

    ! Reals
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%pivot_tol_for_basis = ccontrol%pivot_tol_for_basis
    fcontrol%zero_pivot = ccontrol%zero_pivot
    fcontrol%inner_fraction_opt = ccontrol%inner_fraction_opt
    fcontrol%radius = ccontrol%radius
    fcontrol%min_diagonal = ccontrol%min_diagonal
    fcontrol%max_infeasibility_relative = ccontrol%max_infeasibility_relative
    fcontrol%max_infeasibility_absolute = ccontrol%max_infeasibility_absolute
    fcontrol%inner_stop_relative = ccontrol%inner_stop_relative
    fcontrol%inner_stop_absolute = ccontrol%inner_stop_absolute
    fcontrol%inner_stop_inter = ccontrol%inner_stop_inter

    ! Logicals
    fcontrol%find_basis_by_transpose = ccontrol%find_basis_by_transpose
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file

    ! Derived types
    CALL copy_fdc_control_in( ccontrol%fdc_control, fcontrol%fdc_control )
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_gltr_control_in( ccontrol%gltr_control, fcontrol%gltr_control )

    ! Strings
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_eqp_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( eqp_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%factorization = fcontrol%factorization
    ccontrol%max_col = fcontrol%max_col
    ccontrol%indmin = fcontrol%indmin
    ccontrol%valmin = fcontrol%valmin
    ccontrol%len_ulsmin = fcontrol%len_ulsmin
    ccontrol%itref_max = fcontrol%itref_max
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%preconditioner = fcontrol%preconditioner
    ccontrol%semi_bandwidth = fcontrol%semi_bandwidth
    ccontrol%new_a = fcontrol%new_a
    ccontrol%new_h = fcontrol%new_h
    ccontrol%sif_file_device = fcontrol%sif_file_device

    ! Reals
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%pivot_tol_for_basis = fcontrol%pivot_tol_for_basis
    ccontrol%zero_pivot = fcontrol%zero_pivot
    ccontrol%inner_fraction_opt = fcontrol%inner_fraction_opt
    ccontrol%radius = fcontrol%radius
    ccontrol%min_diagonal = fcontrol%min_diagonal
    ccontrol%max_infeasibility_relative = fcontrol%max_infeasibility_relative
    ccontrol%max_infeasibility_absolute = fcontrol%max_infeasibility_absolute
    ccontrol%inner_stop_relative = fcontrol%inner_stop_relative
    ccontrol%inner_stop_absolute = fcontrol%inner_stop_absolute
    ccontrol%inner_stop_inter = fcontrol%inner_stop_inter

    ! Logicals
    ccontrol%find_basis_by_transpose = fcontrol%find_basis_by_transpose
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file

    ! Derived types
    CALL copy_fdc_control_out( fcontrol%fdc_control, ccontrol%fdc_control )
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_gltr_control_out( fcontrol%gltr_control, ccontrol%gltr_control )

    ! Strings
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( eqp_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_eqp_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%find_dependent = ctime%find_dependent
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%solve_inter = ctime%solve_inter
    ftime%clock_total = ctime%clock_total
    ftime%clock_find_dependent = ctime%clock_find_dependent
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_eqp_time_type ), INTENT( IN ) :: ftime
    TYPE ( eqp_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%find_dependent = ftime%find_dependent
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%solve_inter = ftime%solve_inter
    ctime%clock_total = ftime%clock_total
    ctime%clock_find_dependent = ftime%clock_find_dependent
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( eqp_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_eqp_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%cg_iter = cinform%cg_iter
    finform%cg_iter_inter = cinform%cg_iter_inter
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real

    ! Reals
    finform%obj = cinform%obj

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_fdc_inform_in( cinform%fdc_inform, finform%fdc_inform )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
    CALL copy_gltr_inform_in( cinform%gltr_inform, finform%gltr_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_eqp_inform_type ), INTENT( IN ) :: finform
    TYPE ( eqp_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%cg_iter = finform%cg_iter
    cinform%cg_iter_inter = finform%cg_iter_inter
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real

    ! Reals
    cinform%obj = finform%obj

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_fdc_inform_out( finform%fdc_inform, cinform%fdc_inform )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
    CALL copy_gltr_inform_out( finform%gltr_inform, cinform%gltr_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_EQP_precision_ciface

!  -------------------------------------
!  C interface to fortran eqp_initialize
!  -------------------------------------

  SUBROUTINE eqp_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( eqp_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  INTEGER ( KIND = ipc_ ) :: alloc_stat
  TYPE ( f_eqp_full_data_type ), POINTER :: fdata
  TYPE ( f_eqp_control_type ) :: fcontrol
  TYPE ( f_eqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata, STAT = alloc_stat ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_eqp_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE eqp_initialize

!  ----------------------------------------
!  C interface to fortran eqp_read_specfile
!  ----------------------------------------

  SUBROUTINE eqp_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( eqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_eqp_control_type ) :: fcontrol
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

  CALL f_eqp_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE eqp_read_specfile

!  ---------------------------------
!  C interface to fortran eqp_inport
!  ---------------------------------

  SUBROUTINE eqp_import( ccontrol, cdata, status, n, m,                        &
                         chtype, hne, hrow, hcol, hptr,                        &
                         catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( eqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_eqp_control_type ) :: fcontrol
  TYPE ( f_eqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required EQP structure

  CALL f_eqp_import( fcontrol, fdata, status, n, m,                            &
                     fhtype, hne, hrow, hcol, hptr,                            &
                     fatype, ane, arow, acol, aptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE eqp_import

!  ----------------------------------------
!  C interface to fortran eqp_reset_control
!  ----------------------------------------

  SUBROUTINE eqp_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( eqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_eqp_control_type ) :: fcontrol
  TYPE ( f_eqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_eqp_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE eqp_reset_control

!  -----------------------------------
!  C interface to fortran eqp_solve_qp
!  -----------------------------------

  SUBROUTINE eqp_solve_qp( cdata, status, n, m, hne, hval, g, f, ane, aval,    &
                           c, x, y ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane, hne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: c
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_eqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_eqp_solve_qp( fdata, status, hval, g, f, aval, c, x, y )
  RETURN

  END SUBROUTINE eqp_solve_qp

!  --------------------------------------
!  C interface to fortran eqp_solve_sldqp
!  --------------------------------------

  SUBROUTINE eqp_solve_sldqp( cdata, status, n, m, w, x0, g, f, ane, aval,     &
                              c, x, y ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, ane
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: w
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: x0
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: c
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_eqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_eqp_solve_sldqp( fdata, status, w, x0, g, f, aval, c, x, y )
  RETURN

  END SUBROUTINE eqp_solve_sldqp

!  -------------------------------------
!  C interface to fortran eqp_resolve_qp
!  -------------------------------------

  SUBROUTINE eqp_resolve_qp( cdata, status, n, m, g, f, c, x, y ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: c
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_eqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_eqp_resolve_qp( fdata, status, g, f, c, x, y )
  RETURN

  END SUBROUTINE eqp_resolve_qp

!  --------------------------------------
!  C interface to fortran eqp_information
!  --------------------------------------

  SUBROUTINE eqp_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( eqp_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_eqp_full_data_type ), pointer :: fdata
  TYPE ( f_eqp_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain EQP solution information

  CALL f_eqp_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE eqp_information

!  ------------------------------------
!  C interface to fortran eqp_terminate
!  ------------------------------------

  SUBROUTINE eqp_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_EQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( eqp_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( eqp_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  INTEGER ( KIND = ipc_ ) :: alloc_stat
  TYPE ( f_eqp_full_data_type ), pointer :: fdata
  TYPE ( f_eqp_control_type ) :: fcontrol
  TYPE ( f_eqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_eqp_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata, STAT = alloc_stat ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE eqp_terminate
