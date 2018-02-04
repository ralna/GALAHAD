#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 26/02/2010 AT 14:30 GMT.

!-*-*-*-  G A L A H A D _ E Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. February 16th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_EQP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to EQP

      USE GALAHAD_MATLAB
      USE GALAHAD_FDC_MATLAB_TYPES
      USE GALAHAD_GLTR_MATLAB_TYPES
      USE GALAHAD_SBLS_MATLAB_TYPES
      USE GALAHAD_EQP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: EQP_matlab_control_set, EQP_matlab_control_get,                &
                EQP_matlab_inform_create, EQP_matlab_inform_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: slen = 30

!--------------------------
!  Derived type definitions
!--------------------------

      TYPE, PUBLIC :: EQP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, find_dependent, factorize, solve
        mwPointer :: clock_total, clock_find_dependent
        mwPointer :: clock_factorize, clock_solve
      END TYPE 

      TYPE, PUBLIC :: EQP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, cg_iter
        mwPointer :: factorization_integer, factorization_real, obj
        TYPE ( EQP_time_pointer_type ) :: time_pointer
        TYPE ( FDC_pointer_type ) :: FDC_pointer
        TYPE ( SBLS_pointer_type ) :: SBLS_pointer
        TYPE ( GLTR_pointer_type ) :: GLTR_pointer
      END TYPE 
    CONTAINS

!-*-  E Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE EQP_matlab_control_set( ps, EQP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to EQP

!  Arguments

!  ps - given pointer to the structure
!  EQP_control - EQP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( EQP_control_type ) :: EQP_control

!  local variables

      INTEGER :: j, nfields
      mwPointer :: pc, mxGetField
      mwSize :: mxGetNumberOfFields
      LOGICAL :: mxIsStruct
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO j = 1, nfields
        name = mxGetFieldNameByNumber( ps, j )
        SELECT CASE ( TRIM( name ) )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, EQP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, EQP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, EQP_control%print_level )
        CASE( 'factorization' )
          CALL MATLAB_get_value( ps, 'factorization',                          &
                                 pc, EQP_control%factorization )
        CASE( 'max_col' )
          CALL MATLAB_get_value( ps, 'max_col',                                &
                                 pc, EQP_control%max_col )
!       CASE( 'indmin' )
!         CALL MATLAB_get_value( ps, 'indmin',                                 &
!                                pc, EQP_control%indmin )
!       CASE( 'valmin' )
!         CALL MATLAB_get_value( ps, 'valmin',                                 &
!                                pc, EQP_control%valmin )
        CASE( 'itref_max' )
          CALL MATLAB_get_value( ps, 'itref_max',                              &
                                 pc, EQP_control%itref_max )
        CASE( 'cg_maxit' )
          CALL MATLAB_get_value( ps, 'cg_maxit',                               &
                                 pc, EQP_control%cg_maxit )
        CASE( 'preconditioner' )
          CALL MATLAB_get_value( ps, 'preconditioner',                         &
                                 pc, EQP_control%preconditioner )
        CASE( 'new_a' )
          CALL MATLAB_get_value( ps, 'new_a',                                  &
                                 pc, EQP_control%new_a )
        CASE( 'new_h' )
          CALL MATLAB_get_value( ps, 'new_h',                                  &
                                 pc, EQP_control%new_h )
        CASE( 'semi_bandwidth' )
          CALL MATLAB_get_value( ps, 'semi_bandwidth',                         &
                                 pc, EQP_control%semi_bandwidth )
        CASE( 'pivot_tol' )
          CALL MATLAB_get_value( ps, 'pivot_tol',                              &
                                 pc, EQP_control%pivot_tol )
        CASE( 'pivot_tol_for_basis' )
          CALL MATLAB_get_value( ps, 'pivot_tol_for_basis',                    &
                                 pc, EQP_control%pivot_tol_for_basis )
        CASE( 'zero_pivot' )
          CALL MATLAB_get_value( ps, 'zero_pivot',                             &
                                 pc, EQP_control%zero_pivot )
        CASE( 'inner_fraction_opt' )
          CALL MATLAB_get_value( ps, 'inner_fraction_opt',                     &
                                 pc, EQP_control%inner_fraction_opt )
        CASE( 'radius' )
          CALL MATLAB_get_value( ps, 'radius',                                 &
                                 pc, EQP_control%radius )
        CASE( 'min_diagonal' )
          CALL MATLAB_get_value( ps, 'min_diagonal',                           &
                                 pc, EQP_control%min_diagonal )
        CASE( 'max_infeasibility_relative' )
          CALL MATLAB_get_value( ps, 'max_infeasibility_relative',             &
                                 pc, EQP_control%max_infeasibility_relative )
        CASE( 'inner_stop_relative' )
          CALL MATLAB_get_value( ps, 'inner_stop_relative',                    &
                                 pc, EQP_control%inner_stop_relative )
        CASE( 'inner_stop_absolute' )
          CALL MATLAB_get_value( ps, 'inner_stop_absolute',                    &
                                 pc, EQP_control%inner_stop_absolute )
        CASE( 'remove_dependencies' )
          CALL MATLAB_get_value( ps, 'remove_dependencies',                    &
                                 pc, EQP_control%remove_dependencies )
        CASE( 'find_basis_by_transpose' )
          CALL MATLAB_get_value( ps, 'find_basis_by_transpose',                &
                                 pc, EQP_control%find_basis_by_transpose )
        CASE( 'space_critical' )
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, EQP_control%space_critical )
        CASE( 'deallocate_error_fatal' )
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, EQP_control%deallocate_error_fatal )
        CASE( 'prefix' )                                           
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, EQP_control%prefix, len )
        CASE( 'FDC_control' )
          pc = mxGetField( ps, 1_mwi_, 'FDC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component FDC_control must be a structure' )
          CALL FDC_matlab_control_set( pc, EQP_control%FDC_control, len )
        CASE( 'SBLS_control' )
          pc = mxGetField( ps, 1_mwi_, 'SBLS_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SBLS_control must be a structure' )
          CALL SBLS_matlab_control_set( pc, EQP_control%SBLS_control, len )
        CASE( 'GLTR_control' )
          pc = mxGetField( ps, 1_mwi_, 'GLTR_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component GLTR_control must be a structure' )
          CALL GLTR_matlab_control_set( pc, EQP_control%GLTR_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine EQP_matlab_control_set

      END SUBROUTINE EQP_matlab_control_set

!-*-  E Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE EQP_matlab_control_get( struct, EQP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to EQP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  EQP_control - EQP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( EQP_control_type ) :: EQP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 28
      CHARACTER ( LEN = 26 ), PARAMETER :: finform( ninform ) = (/             &
           'error                     ', 'out                       ',         &
           'print_level               ', 'factorization             ',         &
           'max_col                   ', 'itref_max                 ',         &
           'cg_maxit                  ', 'preconditioner            ',         &
           'new_a                     ', 'new_h                     ',         &
           'semi_bandwidth            ', 'pivot_tol                 ',         &
           'pivot_tol_for_basis       ', 'zero_pivot                ',         &
           'inner_fraction_opt        ', 'radius                    ',         &
           'min_diagonal              ', 'max_infeasibility_relative',         &
           'inner_stop_relative       ', 'inner_stop_absolute       ',         &
           'remove_dependencies       ', 'find_basis_by_transpose   ',         &
           'space_critical            ', 'deallocate_error_fatal    ',         &
           'prefix                    ', 'FDC_control               ',         &
           'SBLS_control              ', 'GLTR_control              '        /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, pointer,                &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        pointer = struct
      END IF

!  create the components and get the values

      CALL MATLAB_fill_component( pointer, 'error',                            &
                                  EQP_control%error )
      CALL MATLAB_fill_component( pointer, 'error',                            &
                                  EQP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  EQP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  EQP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'factorization',                    &
                                  EQP_control%factorization )
      CALL MATLAB_fill_component( pointer, 'max_col',                          &
                                  EQP_control%max_col )
      CALL MATLAB_fill_component( pointer, 'itref_max',                        &
                                  EQP_control%itref_max )
      CALL MATLAB_fill_component( pointer, 'cg_maxit',                         &
                                  EQP_control%cg_maxit )
      CALL MATLAB_fill_component( pointer, 'preconditioner',                   &
                                  EQP_control%preconditioner )
      CALL MATLAB_fill_component( pointer, 'new_a',                            &
                                  EQP_control%new_a )
      CALL MATLAB_fill_component( pointer, 'new_h',                            &
                                  EQP_control%new_h )
      CALL MATLAB_fill_component( pointer, 'semi_bandwidth',                   &
                                  EQP_control%semi_bandwidth )
      CALL MATLAB_fill_component( pointer, 'pivot_tol',                        &
                                  EQP_control%pivot_tol )
      CALL MATLAB_fill_component( pointer, 'pivot_tol_for_basis',              &
                                  EQP_control%pivot_tol_for_basis )
      CALL MATLAB_fill_component( pointer, 'zero_pivot',                       &
                                  EQP_control%zero_pivot )
      CALL MATLAB_fill_component( pointer, 'inner_fraction_opt',               &
                                  EQP_control%inner_fraction_opt )
      CALL MATLAB_fill_component( pointer, 'radius',                           &
                                  EQP_control%radius )
      CALL MATLAB_fill_component( pointer, 'min_diagonal',                     &
                                  EQP_control%min_diagonal )
      CALL MATLAB_fill_component( pointer, 'max_infeasibility_relative',       &
                                  EQP_control%max_infeasibility_relative )
      CALL MATLAB_fill_component( pointer, 'inner_stop_relative',              &
                                  EQP_control%inner_stop_relative )
      CALL MATLAB_fill_component( pointer, 'inner_stop_absolute',              &
                                  EQP_control%inner_stop_absolute )
      CALL MATLAB_fill_component( pointer, 'remove_dependencies',              &
                                  EQP_control%remove_dependencies )
      CALL MATLAB_fill_component( pointer, 'find_basis_by_transpose',          &
                                  EQP_control%find_basis_by_transpose )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  EQP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  EQP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  EQP_control%prefix )

!  create the components of sub-structure FDC_control

      CALL FDC_matlab_control_get( pointer, EQP_control%FDC_control,           &
                                   'FDC_control' )

!  create the components of sub-structure SBLS_control

      CALL SBLS_matlab_control_get( pointer, EQP_control%SBLS_control,         &
                                   'SBLS_control' )

!  create the components of sub-structure GLTR_control

      CALL GLTR_matlab_control_get( pointer, EQP_control%GLTR_control,         &
                                   'GLTR_control' )

      RETURN

!  End of subroutine EQP_matlab_control_get

      END SUBROUTINE EQP_matlab_control_get

!-*- E Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE EQP_matlab_inform_create( struct, EQP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold EQP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  EQP_pointer - EQP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( EQP_pointer_type ) :: EQP_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 11
      CHARACTER ( LEN = 21 ), PARAMETER :: finform( ninform ) = (/             &
           'status               ', 'alloc_status         ',                   &
           'cg_iter              ', 'factorization_integer',                   &
           'factorization_real   ', 'obj                  ',                   &
           'bad_alloc            ', 'time                 ',                   &
           'FDC_inform           ', 'SBLS_inform          ',                   &
           'GLTR_inform          '      /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'find_dependent       ',                   &
           'factorize            ', 'solve                ',                   &
           'clock_total          ', 'clock_find_dependent ',                   &
           'clock_factorize      ', 'clock_solve          '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, EQP_pointer%pointer,    &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        EQP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( EQP_pointer%pointer,               &
        'status', EQP_pointer%status )
      CALL MATLAB_create_integer_component( EQP_pointer%pointer,               &
         'alloc_status', EQP_pointer%alloc_status )
      CALL MATLAB_create_char_component( EQP_pointer%pointer,                  &
        'bad_alloc', EQP_pointer%bad_alloc )
      CALL MATLAB_create_integer_component( EQP_pointer%pointer,               &
         'cg_iter', EQP_pointer%cg_iter )
      CALL MATLAB_create_integer_component( EQP_pointer%pointer,               &
         'factorization_integer', EQP_pointer%factorization_integer )
      CALL MATLAB_create_integer_component( EQP_pointer%pointer,               &
         'factorization_real', EQP_pointer%factorization_real )
      CALL MATLAB_create_real_component( EQP_pointer%pointer,                  &
         'obj', EQP_pointer%obj )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( EQP_pointer%pointer,                    &
        'time', EQP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'total', EQP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'find_dependent', EQP_pointer%time_pointer%find_dependent )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'factorize', EQP_pointer%time_pointer%factorize )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'solve', EQP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'clock_total', EQP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'clock_find_dependent', EQP_pointer%time_pointer%clock_find_dependent )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'clock_factorize', EQP_pointer%time_pointer%clock_factorize )
      CALL MATLAB_create_real_component( EQP_pointer%time_pointer%pointer,     &
        'clock_solve', EQP_pointer%time_pointer%clock_solve )


!  Define the components of sub-structure FDC_inform

      CALL FDC_matlab_inform_create( EQP_pointer%pointer,                      &
                                     EQP_pointer%FDC_pointer, 'FDC_inform' )

!  Define the components of sub-structure SBLS_inform

      CALL SBLS_matlab_inform_create( EQP_pointer%pointer,                     &
                                      EQP_pointer%SBLS_pointer, 'SBLS_inform' )

!  Define the components of sub-structure GLTR_inform

      CALL GLTR_matlab_inform_create( EQP_pointer%pointer,                     &
                                      EQP_pointer%GLTR_pointer, 'GLTR_inform' )

      RETURN

!  End of subroutine EQP_matlab_inform_create

      END SUBROUTINE EQP_matlab_inform_create

!-*-*-  E Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE EQP_matlab_inform_get( EQP_inform, EQP_pointer )

!  --------------------------------------------------------------

!  Set EQP_inform values from matlab pointers

!  Arguments

!  EQP_inform - EQP inform structure
!  EQP_pointer - EQP pointer structure

!  --------------------------------------------------------------

      TYPE ( EQP_inform_type ) :: EQP_inform
      TYPE ( EQP_pointer_type ) :: EQP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( EQP_inform%status,                              &
                               mxGetPr( EQP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( EQP_inform%alloc_status,                        &
                               mxGetPr( EQP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( EQP_pointer%pointer,                            &
                               'bad_alloc', EQP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( EQP_inform%cg_iter,                             &
                               mxGetPr( EQP_pointer%cg_iter ) )
      CALL MATLAB_copy_to_ptr( EQP_inform%factorization_integer,               &
                               mxGetPr( EQP_pointer%factorization_integer ) )   
      CALL MATLAB_copy_to_ptr( EQP_inform%factorization_real,                  &
                               mxGetPr( EQP_pointer%factorization_real ) )      
      CALL MATLAB_copy_to_ptr( EQP_inform%obj,                                 &
                               mxGetPr( EQP_pointer%obj ) )                     

!  time components

      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%total, wp ),              &
                       mxGetPr( EQP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%find_dependent, wp ),     &
                          mxGetPr( EQP_pointer%time_pointer%find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%factorize, wp ),          &
                       mxGetPr( EQP_pointer%time_pointer%factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%solve, wp ),              &
                      mxGetPr( EQP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%clock_total, wp ),        &
                      mxGetPr( EQP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%clock_find_dependent, wp),&
                      mxGetPr( EQP_pointer%time_pointer%clock_find_dependent ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%clock_factorize, wp ),    &
                      mxGetPr( EQP_pointer%time_pointer%clock_factorize ) )
      CALL MATLAB_copy_to_ptr( REAL( EQP_inform%time%clock_solve, wp ),        &
                      mxGetPr( EQP_pointer%time_pointer%clock_solve ) )

!  dependency components

      CALL FDC_matlab_inform_get( EQP_inform%FDC_inform,                       &
                                  EQP_pointer%FDC_pointer )

!  preconditioner components

      CALL SBLS_matlab_inform_get( EQP_inform%SBLS_inform,                     &
                                   EQP_pointer%SBLS_pointer )

!  step computation components

      CALL GLTR_matlab_inform_get( EQP_inform%GLTR_inform,                     &
                                   EQP_pointer%GLTR_pointer )

      RETURN

!  End of subroutine EQP_matlab_inform_get

      END SUBROUTINE EQP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ E Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_EQP_MATLAB_TYPES

