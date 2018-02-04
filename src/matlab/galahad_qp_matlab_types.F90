#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 31/01/2011 AT 20:00 GMT.

!-*-*-*-  G A L A H A D _ Q P _ M A T L A B _ T Y P E S   M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. January 31st, 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_QP_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to QP

      USE GALAHAD_MATLAB
      USE GALAHAD_SCALE_MATLAB_TYPES
      USE GALAHAD_PRESOLVE_MATLAB_TYPES
      USE GALAHAD_QPA_MATLAB_TYPES
      USE GALAHAD_QPB_MATLAB_TYPES
      USE GALAHAD_QPC_MATLAB_TYPES
      USE GALAHAD_CQP_MATLAB_TYPES
      USE GALAHAD_QP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QP_matlab_control_set, QP_matlab_control_get,                  &
                QP_matlab_inform_create, QP_matlab_inform_get

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

      TYPE, PUBLIC :: QP_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, presolve, scale, solve
        mwPointer :: clock_total, clock_presolve, clock_scale, clock_solve
      END TYPE 

      TYPE, PUBLIC :: QP_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc
        mwPointer :: obj, primal_infeasibility, dual_infeasibility
        mwPointer :: complementary_slackness
        TYPE ( QP_time_pointer_type ) :: time_pointer
        TYPE ( SCALE_pointer_type ) :: SCALE_pointer
        TYPE ( PRESOLVE_pointer_type ) :: PRESOLVE_pointer
        TYPE ( QPA_pointer_type ) :: QPA_pointer
        TYPE ( QPB_pointer_type ) :: QPB_pointer
        TYPE ( QPC_pointer_type ) :: QPC_pointer
        TYPE ( CQP_pointer_type ) :: CQP_pointer
      END TYPE 
    CONTAINS

!-*-  Q P _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   -*-

      SUBROUTINE QP_matlab_control_set( ps, QP_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to QP

!  Arguments

!  ps - given pointer to the structure
!  QP_control - QP control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( QP_control_type ) :: QP_control

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
                                 pc, QP_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, QP_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, QP_control%print_level )
        CASE( 'scale' )                                                  
          CALL MATLAB_get_value( ps, 'scale',                                  &
                                 pc, QP_control%scale )
        CASE( 'infinity' )                                                     
          CALL MATLAB_get_value( ps, 'infinity',                               &
                                 pc, QP_control%infinity )
        CASE( 'presolve' )
          CALL MATLAB_get_value( ps, 'presolve',                               &
                                 pc, QP_control%presolve )
        CASE( 'space_critical' )                                               
          CALL MATLAB_get_value( ps, 'space_critical',                         &
                                 pc, QP_control%space_critical )        
        CASE( 'deallocate_error_fatal' )                                       
          CALL MATLAB_get_value( ps, 'deallocate_error_fatal',                 &
                                 pc, QP_control%deallocate_error_fatal )
        CASE( 'quadratic_programming_solver' )
          CALL galmxGetCharacter( ps, 'quadratic_programming_solver',          &
                                  pc, QP_control%quadratic_programming_solver, &
                                  len )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, QP_control%prefix, len )
        CASE( 'SCALE_control' )
          pc = mxGetField( ps, 1_mwi_, 'SCALE_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component SCALE_control must be a structure' )
          CALL SCALE_matlab_control_set( pc, QP_control%SCALE_control, len )
        CASE( 'PRESOLVE_control' )
          pc = mxGetField( ps, 1_mwi_, 'PRESOLVE_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
           CALL mexErrMsgTxt( ' component PRESOLVE_control must be a structure')
          CALL PRESOLVE_matlab_control_set( pc,                                &
                                            QP_control%PRESOLVE_control, len )
        CASE( 'QPA_control' )
          pc = mxGetField( ps, 1_mwi_, 'QPA_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component QPA_control must be a structure' )
          CALL QPA_matlab_control_set( pc, QP_control%QPA_control, len )
        CASE( 'QPB_control' )
          pc = mxGetField( ps, 1_mwi_, 'QPB_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component QPB_control must be a structure' )
          CALL QPB_matlab_control_set( pc, QP_control%QPB_control, len )
        CASE( 'QPC_control' )
          pc = mxGetField( ps, 1_mwi_, 'QPC_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component QPC_control must be a structure' )
          CALL QPC_matlab_control_set( pc, QP_control%QPC_control, len )
        CASE( 'CQP_control' )
          pc = mxGetField( ps, 1_mwi_, 'CQP_control' )
          IF ( .NOT. mxIsStruct( pc ) )                                        &
            CALL mexErrMsgTxt( ' component CQP_control must be a structure' )
          CALL CQP_matlab_control_set( pc, QP_control%CQP_control, len )
        END SELECT
      END DO

      RETURN

!  End of subroutine QP_matlab_control_set

      END SUBROUTINE QP_matlab_control_set

!-*-  Q P _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   -*-

      SUBROUTINE QP_matlab_control_get( struct, QP_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to QP

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QP_control - QP control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QP_control_type ) :: QP_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 16
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'error                          ', 'out                            ', &
         'print_level                    ', 'scale                          ', &
         'infinity                       ', 'presolve                       ', &
         'space_critical                 ', 'deallocate_error_fatal         ', &
         'quadratic_programming_solver   ', 'prefix                         ', &
         'SCALE_control                  ', 'PRESOLVE_control               ', &
         'QPA_control                    ', 'QPB_control                    ', &
         'QPC_control                    ', 'CQP_control                    ' /)

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
                                  QP_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  QP_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  QP_control%print_level )
      CALL MATLAB_fill_component( pointer, 'scale',                            &
                                  QP_control%scale )
      CALL MATLAB_fill_component( pointer, 'infinity',                         &
                                  QP_control%infinity )
      CALL MATLAB_fill_component( pointer, 'presolve',                         &
                                  QP_control%presolve )
      CALL MATLAB_fill_component( pointer, 'space_critical',                   &
                                  QP_control%space_critical )
      CALL MATLAB_fill_component( pointer, 'deallocate_error_fatal',           &
                                  QP_control%deallocate_error_fatal )
      CALL MATLAB_fill_component( pointer, 'quadratic_programming_solver',     &
                                  QP_control%quadratic_programming_solver )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  QP_control%prefix )

!  create the components of sub-structure SCALE_control

      CALL SCALE_matlab_control_get( pointer, QP_control%SCALE_control,        &
                                     'SCALE_control' )

!  create the components of sub-structure PRESOLVE_control

      CALL PRESOLVE_matlab_control_get( pointer, QP_control%PRESOLVE_control,  &
                                        'PRESOLVE_control' )

!  create the components of sub-structure QPA_control

      CALL QPA_matlab_control_get( pointer, QP_control%QPA_control,            &
                                   'QPA_control' )

!  create the components of sub-structure QPB_control

      CALL QPB_matlab_control_get( pointer, QP_control%QPB_control,            &
                                   'QPB_control' )

!  create the components of sub-structure QPC_control

      CALL QPC_matlab_control_get( pointer, QP_control%QPC_control,            &
                                   'QPC_control' )

!  create the components of sub-structure CQP_control

      CALL CQP_matlab_control_get( pointer, QP_control%CQP_control,            &
                                   'CQP_control' )

      RETURN

!  End of subroutine QP_matlab_control_get

      END SUBROUTINE QP_matlab_control_get

!-*- Q P _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E  -*-

      SUBROUTINE QP_matlab_inform_create( struct, QP_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold QP_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  QP_pointer - QP pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( QP_pointer_type ) :: QP_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 14
      CHARACTER ( LEN = 24 ), PARAMETER :: finform( ninform ) = (/             &
           'status                  ', 'alloc_status            ',             &
           'bad_alloc               ', 'obj                     ',             &
           'primal_infeasibility    ', 'dual_infeasibility      ',             &
           'complementary_slackness ', 'time                    ',             &
           'SCALE_inform            ', 'PRESOLVE_inform         ',             &
           'QPA_inform              ', 'QPB_inform              ',             &
           'QPC_inform              ', 'CQP_inform              ' /)
      INTEGER * 4, PARAMETER :: t_ninform = 8
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'presolve             ',                   &
           'scale                ', 'solve                ',                   &
           'clock_total          ', 'clock_presolve       ',                   &
           'clock_scale          ', 'clock_solve          '         /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, QP_pointer%pointer,     &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        QP_pointer%pointer = struct
      END IF

!  Define the components of the structure

      CALL MATLAB_create_integer_component( QP_pointer%pointer,                &
        'status', QP_pointer%status )
      CALL MATLAB_create_integer_component( QP_pointer%pointer,                &
         'alloc_status', QP_pointer%alloc_status )
      CALL MATLAB_create_char_component( QP_pointer%pointer,                   &
        'bad_alloc', QP_pointer%bad_alloc )
      CALL MATLAB_create_real_component( QP_pointer%pointer,                   &
        'obj', QP_pointer%obj )
      CALL MATLAB_create_real_component( QP_pointer%pointer,                   &
         'primal_infeasibility', QP_pointer%primal_infeasibility )
      CALL MATLAB_create_real_component( QP_pointer%pointer,                   &
         'dual_infeasibility', QP_pointer%dual_infeasibility )
      CALL MATLAB_create_real_component( QP_pointer%pointer,                   &
         'complementary_slackness', QP_pointer%complementary_slackness )

!  Define the components of sub-structure time

      CALL MATLAB_create_substructure( QP_pointer%pointer,                     &
        'time', QP_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'total', QP_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'presolve', QP_pointer%time_pointer%presolve )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'scale', QP_pointer%time_pointer%scale )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'solve', QP_pointer%time_pointer%solve )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'clock_total', QP_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'clock_presolve', QP_pointer%time_pointer%clock_presolve )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'clock_scale', QP_pointer%time_pointer%clock_scale )
      CALL MATLAB_create_real_component( QP_pointer%time_pointer%pointer,      &
        'clock_solve', QP_pointer%time_pointer%clock_solve )

!  Define the components of sub-structure SCALE_inform

      CALL SCALE_matlab_inform_create( QP_pointer%pointer,                     &
                                       QP_pointer%SCALE_pointer, 'SCALE_inform')

!  Define the components of sub-structure PRESOLVE_inform

      CALL PRESOLVE_matlab_inform_create( QP_pointer%pointer,                  &
                                          QP_pointer%PRESOLVE_pointer,         &
                                         'PRESOLVE_inform' )

!  Define the components of sub-structure QPA_inform

      CALL QPA_matlab_inform_create( QP_pointer%pointer,                       &
                                     QP_pointer%QPA_pointer, 'QPA_inform' )

!  Define the components of sub-structure QPB_inform

      CALL QPB_matlab_inform_create( QP_pointer%pointer,                       &
                                     QP_pointer%QPB_pointer, 'QPB_inform' )

!  Define the components of sub-structure QPC_inform

      CALL QPC_matlab_inform_create( QP_pointer%pointer,                       &
                                     QP_pointer%QPC_pointer, 'QPC_inform' )

!  Define the components of sub-structure CQP_inform

      CALL CQP_matlab_inform_create( QP_pointer%pointer,                       &
                                     QP_pointer%CQP_pointer, 'CQP_inform' )

      RETURN

!  End of subroutine QP_matlab_inform_create

      END SUBROUTINE QP_matlab_inform_create

!-*-*-  Q P _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E   -*-*-

      SUBROUTINE QP_matlab_inform_get( QP_inform, QP_pointer )

!  --------------------------------------------------------------

!  Set QP_inform values from matlab pointers

!  Arguments

!  QP_inform - QP inform structure
!  QP_pointer - QP pointer structure

!  --------------------------------------------------------------

      TYPE ( QP_inform_type ) :: QP_inform
      TYPE ( QP_pointer_type ) :: QP_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( QP_inform%status,                              &
                               mxGetPr( QP_pointer%status ) )
      CALL MATLAB_copy_to_ptr( QP_inform%alloc_status,                        &
                               mxGetPr( QP_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( QP_pointer%pointer,                            &
                               'bad_alloc', QP_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( QP_inform%obj,                                 &
                               mxGetPr( QP_pointer%obj ) )                     
      CALL MATLAB_copy_to_ptr( QP_inform%primal_infeasibility,                &
                               mxGetPr( QP_pointer%primal_infeasibility ) )
      CALL MATLAB_copy_to_ptr( QP_inform%dual_infeasibility,                  &
                               mxGetPr( QP_pointer%dual_infeasibility ) )
      CALL MATLAB_copy_to_ptr( QP_inform%complementary_slackness,             &
                               mxGetPr( QP_pointer%complementary_slackness ) )


!  time components

      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%total, wp ),               &
                               mxGetPr( QP_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%presolve, wp ),            &
                               mxGetPr( QP_pointer%time_pointer%presolve ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%scale, wp ),               &
                               mxGetPr( QP_pointer%time_pointer%scale ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%solve, wp ),               &
                               mxGetPr( QP_pointer%time_pointer%solve ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%clock_total, wp ),         &
                           mxGetPr( QP_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%clock_presolve, wp ),      &
                           mxGetPr( QP_pointer%time_pointer%clock_presolve ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%clock_scale, wp ),         &
                           mxGetPr( QP_pointer%time_pointer%clock_scale ) )
      CALL MATLAB_copy_to_ptr( REAL( QP_inform%time%clock_solve, wp ),         &
                           mxGetPr( QP_pointer%time_pointer%clock_solve ) )

!  scale and presolve strategies 

      CALL SCALE_matlab_inform_get( QP_inform%SCALE_inform,                    &
                                  QP_pointer%SCALE_pointer )
      CALL PRESOLVE_matlab_inform_get( QP_inform%PRESOLVE_inform,              &
                                  QP_pointer%PRESOLVE_pointer )

!  quadratic programming solvers

      CALL QPA_matlab_inform_get( QP_inform%QPA_inform,                        &
                                  QP_pointer%QPA_pointer )
      CALL QPB_matlab_inform_get( QP_inform%QPB_inform,                        &
                                  QP_pointer%QPB_pointer )
      CALL QPC_matlab_inform_get( QP_inform%QPC_inform,                        &
                                  QP_pointer%QPC_pointer )
      CALL CQP_matlab_inform_get( QP_inform%CQP_inform,                        &
                                  QP_pointer%CQP_pointer )

      RETURN

!  End of subroutine QP_matlab_inform_get

      END SUBROUTINE QP_matlab_inform_get

!-*-*-*-  E N D  o f  G A L A H A D _ Q P _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_QP_MATLAB_TYPES
