#include <fintrf.h>

!  THIS VERSION: GALAHAD 5.2 - 2025-03-25 AT 16:00 GMT.

!-*-*-  G A L A H A D _ N O D E N D _ M A T L A B _ T Y P E S   M O D U L E  -*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 5.2. March 24th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_NODEND_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to NODEND

      USE GALAHAD_MATLAB
      USE GALAHAD_NODEND_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: NODEND_matlab_control_set, NODEND_matlab_control_get,          &
                NODEND_matlab_inform_create, NODEND_matlab_inform_get

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

      TYPE, PUBLIC :: NODEND_time_pointer_type
        mwPointer :: pointer
        mwPointer :: total, metis
        mwPointer :: clock_total, clock_metis
      END TYPE

      TYPE, PUBLIC :: NODEND_pointer_type
        mwPointer :: pointer
        mwPointer :: status, alloc_status, bad_alloc, version
      END TYPE

    CONTAINS

!-*-  N O D E N D _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E -*-

      SUBROUTINE NODEND_matlab_control_set( ps, NODEND_control, len )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to NODEND

!  Arguments

!  ps - given pointer to the structure
!  NODEND_control - NODEND control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
      mwSize :: len
      TYPE ( NODEND_control_type ) :: NODEND_control

!  local variables

      INTEGER :: i, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO i = 1, nfields
        name = mxGetFieldNameByNumber( ps, i )
        SELECT CASE ( TRIM( name ) )
        CASE( 'version' )
          CALL galmxGetCharacter( ps, 'version',                               &
                                  pc, NODEND_control%version, len )
        CASE( 'error' )
          CALL MATLAB_get_value( ps, 'error',                                  &
                                 pc, NODEND_control%error )
        CASE( 'out' )
          CALL MATLAB_get_value( ps, 'out',                                    &
                                 pc, NODEND_control%out )
        CASE( 'print_level' )
          CALL MATLAB_get_value( ps, 'print_level',                            &
                                 pc, NODEND_control%print_level )
        CASE( 'no_metis_4_use_5_instead' )
          CALL MATLAB_get_value( ps, 'no_metis_4_use_5_instead',               &
                                 pc, NODEND_control%no_metis_4_use_5_instead )
        CASE( 'prefix' )
          CALL galmxGetCharacter( ps, 'prefix',                                &
                                  pc, NODEND_control%prefix, len )
        CASE( 'metis4_ptype' )
          CALL MATLAB_get_value( ps, 'metis4_ptype',                           &
                                 pc, NODEND_control%metis4_ptype )
        CASE( 'metis4_ctype' )
          CALL MATLAB_get_value( ps, 'metis4_ctype',                           &
                                 pc, NODEND_control%metis4_ctype )
        CASE( 'metis4_itype' )
          CALL MATLAB_get_value( ps, 'metis4_itype',                           &
                                 pc, NODEND_control%metis4_itype )
        CASE( 'metis4_rtype' )
          CALL MATLAB_get_value( ps, 'metis4_rtype',                           &
                                 pc, NODEND_control%metis4_rtype )
        CASE( 'metis4_dbglvl' )
          CALL MATLAB_get_value( ps, 'metis4_dbglvl',                          &
                                 pc, NODEND_control%metis4_dbglvl )
        CASE( 'metis4_oflags' )
          CALL MATLAB_get_value( ps, 'metis4_oflags',                          &
                                 pc, NODEND_control%metis4_oflags )
        CASE( 'metis4_pfactor' )
          CALL MATLAB_get_value( ps, 'metis4_pfactor',                         &
                                 pc, NODEND_control%metis4_pfactor )
        CASE( 'metis4_nseps' )
          CALL MATLAB_get_value( ps, 'metis4_nseps',                           &
                                 pc, NODEND_control%metis4_nseps )
        CASE( 'metis5_ptype' )
          CALL MATLAB_get_value( ps, 'metis5_ptype',                           &
                                 pc, NODEND_control%metis5_ptype )
        CASE( 'metis5_objtype' )
          CALL MATLAB_get_value( ps, 'metis5_objtype',                         &
                                 pc, NODEND_control%metis5_objtype )
        CASE( 'metis5_ctype' )
          CALL MATLAB_get_value( ps, 'metis5_ctype',                           &
                                 pc, NODEND_control%metis5_ctype )
        CASE( 'metis5_iptype' )
          CALL MATLAB_get_value( ps, 'metis5_iptype',                          &
                                 pc, NODEND_control%metis5_iptype )
        CASE( 'metis5_rtype' )
          CALL MATLAB_get_value( ps, 'metis5_rtype',                           &
                                 pc, NODEND_control%metis5_rtype )
        CASE( 'metis5_dbglvl' )
          CALL MATLAB_get_value( ps, 'metis5_dbglvl',                          &
                                 pc, NODEND_control%metis5_dbglvl )
        CASE( 'metis5_niter' )
          CALL MATLAB_get_value( ps, 'metis5_niter',                           &
                                 pc, NODEND_control%metis5_niter )
        CASE( 'metis5_ncuts' )
          CALL MATLAB_get_value( ps, 'metis5_ncuts',                           &
                                 pc, NODEND_control%metis5_ncuts )
        CASE( 'metis5_seed' )
          CALL MATLAB_get_value( ps, 'metis5_seed',                            &
                                 pc, NODEND_control%metis5_seed )
        CASE( 'metis5_no2hop' )
          CALL MATLAB_get_value( ps, 'metis5_no2hop',                          &
                                 pc, NODEND_control%metis5_no2hop )
        CASE( 'metis5_minconn' )
          CALL MATLAB_get_value( ps, 'metis5_minconn',                         &
                                 pc, NODEND_control%metis5_minconn )
        CASE( 'metis5_contig' )
          CALL MATLAB_get_value( ps, 'metis5_contig',                          &
                                 pc, NODEND_control%metis5_contig )
        CASE( 'metis5_compress' )
          CALL MATLAB_get_value( ps, 'metis5_compress',                        &
                                 pc, NODEND_control%metis5_compress )
        CASE( 'metis5_ccorder' )
          CALL MATLAB_get_value( ps, 'metis5_ccorder',                         &
                                 pc, NODEND_control%metis5_ccorder )
        CASE( 'metis5_pfactor' )
          CALL MATLAB_get_value( ps, 'metis5_pfactor',                         &
                                 pc, NODEND_control%metis5_pfactor )
        CASE( 'metis5_nseps' )
          CALL MATLAB_get_value( ps, 'metis5_nseps',                           &
                                 pc, NODEND_control%metis5_nseps )
        CASE( 'metis5_ufactor' )
          CALL MATLAB_get_value( ps, 'metis5_ufactor',                         &
                                 pc, NODEND_control%metis5_ufactor )
        CASE( 'metis5_niparts' )
          CALL MATLAB_get_value( ps, 'metis5_niparts',                         &
                                 pc, NODEND_control%metis5_niparts )
        CASE( 'metis5_ondisk' )
          CALL MATLAB_get_value( ps, 'metis5_ondisk',                          &
                                 pc, NODEND_control%metis5_ondisk )
        CASE( 'metis5_dropedges' )
          CALL MATLAB_get_value( ps, 'metis5_dropedges',                       &
                                 pc, NODEND_control%metis5_dropedges )
        CASE( 'metis5_twohop' )
          CALL MATLAB_get_value( ps, 'metis5_twohop',                          &
                                 pc, NODEND_control%metis5_twohop )
        CASE( 'metis5_fast' )
          CALL MATLAB_get_value( ps, 'metis5_fast',                            &
                                 pc, NODEND_control%metis5_fast )
        END SELECT
      END DO

      RETURN

!  End of subroutine NODEND_matlab_control_set

      END SUBROUTINE NODEND_matlab_control_set

!-*-  N O D E N D _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E -*-

      SUBROUTINE NODEND_matlab_control_get( struct, NODEND_control, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to NODEND

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NODEND_control - NODEND control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NODEND_control_type ) :: NODEND_control
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 36
      CHARACTER ( LEN = 31 ), PARAMETER :: finform( ninform ) = (/             &
         'version                        ', 'error                          ', &
         'out                            ', 'print_level                    ', &
         'no_metis_4_use_5_instead       ', 'prefix                         ', &
         'metis4_ptype                   ', 'metis4_ctype                   ', &
         'metis4_itype                   ', 'metis4_rtype                   ', &
         'metis4_dbglvl                  ', 'metis4_oflags                  ', &
         'metis4_pfactor                 ', 'metis4_nseps                   ', &
         'metis5_ptype                   ', 'metis5_objtype                 ', &
         'metis5_ctype                   ', 'metis5_iptype                  ', &
         'metis5_rtype                   ', 'metis5_dbglvl                  ', &
         'metis5_niter                   ', 'metis5_ncuts                   ', &
         'metis5_seed                    ', 'metis5_no2hop                  ', &
         'metis5_minconn                 ', 'metis5_contig                  ', &
         'metis5_compress                ', 'metis5_ccorder                 ', &
         'metis5_pfactor                 ', 'metis5_nseps                   ', &
         'metis5_ufactor                 ', 'metis5_niparts                 ', &
         'metis5_ondisk                  ', 'metis5_dropedges               ', &
         'metis5_twohop                  ', 'metis5_fast                    ' /)
                   
!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, pointer,                &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        pointer = struct
      END IF

!  create the components and get the values

      CALL MATLAB_fill_component( pointer, 'version',                          &
                                  NODEND_control%version )
      CALL MATLAB_fill_component( pointer, 'error',                            &
                                  NODEND_control%error )
      CALL MATLAB_fill_component( pointer, 'out',                              &
                                  NODEND_control%out )
      CALL MATLAB_fill_component( pointer, 'print_level',                      &
                                  NODEND_control%print_level )
      CALL MATLAB_fill_component( pointer, 'no_metis_4_use_5_instead',         &
                                  NODEND_control%no_metis_4_use_5_instead )
      CALL MATLAB_fill_component( pointer, 'prefix',                           &
                                  NODEND_control%prefix )
      CALL MATLAB_fill_component( pointer, 'metis4_ptype',                     &
                                  NODEND_control%metis4_ptype )
      CALL MATLAB_fill_component( pointer, 'metis4_ctype',                     &
                                  NODEND_control%metis4_ctype )
      CALL MATLAB_fill_component( pointer, 'metis4_itype',                     &
                                  NODEND_control%metis4_itype )
      CALL MATLAB_fill_component( pointer, 'metis4_rtype',                     &
                                  NODEND_control%metis4_rtype )
      CALL MATLAB_fill_component( pointer, 'metis4_dbglvl',                    &
                                  NODEND_control%metis4_dbglvl )
      CALL MATLAB_fill_component( pointer, 'metis4_oflags',                    &
                                  NODEND_control%metis4_oflags )
      CALL MATLAB_fill_component( pointer, 'metis4_pfactor',                   &
                                  NODEND_control%metis4_pfactor )
      CALL MATLAB_fill_component( pointer, 'metis4_nseps',                     &
                                  NODEND_control%metis4_nseps )
      CALL MATLAB_fill_component( pointer, 'metis5_ptype',                     &
                                  NODEND_control%metis5_ptype )
      CALL MATLAB_fill_component( pointer, 'metis5_objtype',                   &
                                  NODEND_control%metis5_objtype )
      CALL MATLAB_fill_component( pointer, 'metis5_ctype',                     &
                                  NODEND_control%metis5_ctype )
      CALL MATLAB_fill_component( pointer, 'metis5_iptype',                    &
                                  NODEND_control%metis5_iptype )
      CALL MATLAB_fill_component( pointer, 'metis5_rtype',                     &
                                  NODEND_control%metis5_rtype )
      CALL MATLAB_fill_component( pointer, 'metis5_dbglvl',                    &
                                  NODEND_control%metis5_dbglvl )
      CALL MATLAB_fill_component( pointer, 'metis5_niter',                     &
                                  NODEND_control%metis5_niter )
      CALL MATLAB_fill_component( pointer, 'metis5_ncuts',                     &
                                  NODEND_control%metis5_ncuts )
      CALL MATLAB_fill_component( pointer, 'metis5_seed',                      &
                                  NODEND_control%metis5_seed )
      CALL MATLAB_fill_component( pointer, 'metis5_no2hop',                    &
                                  NODEND_control%metis5_no2hop )
      CALL MATLAB_fill_component( pointer, 'metis5_minconn',                   &
                                  NODEND_control%metis5_minconn )
      CALL MATLAB_fill_component( pointer, 'metis5_contig',                    &
                                  NODEND_control%metis5_contig )
      CALL MATLAB_fill_component( pointer, 'metis5_compress',                  &
                                  NODEND_control%metis5_compress )
      CALL MATLAB_fill_component( pointer, 'metis5_ccorder',                   &
                                  NODEND_control%metis5_ccorder )
      CALL MATLAB_fill_component( pointer, 'metis5_pfactor',                   &
                                  NODEND_control%metis5_pfactor )
      CALL MATLAB_fill_component( pointer, 'metis5_nseps',                     &
                                  NODEND_control%metis5_nseps )
      CALL MATLAB_fill_component( pointer, 'metis5_ufactor',                   &
                                  NODEND_control%metis5_ufactor )
      CALL MATLAB_fill_component( pointer, 'metis5_niparts',                   &
                                  NODEND_control%metis5_niparts )
      CALL MATLAB_fill_component( pointer, 'metis5_ondisk',                    &
                                  NODEND_control%metis5_ondisk )
      CALL MATLAB_fill_component( pointer, 'metis5_dropedges',                 &
                                  NODEND_control%metis5_dropedges )
      CALL MATLAB_fill_component( pointer, 'metis5_twohop',                    &
                                  NODEND_control%metis5_twohop )
      CALL MATLAB_fill_component( pointer, 'metis5_fast',                      &
                                  NODEND_control%metis5_fast )

      RETURN

!  End of subroutine NODEND_matlab_control_get

      END SUBROUTINE NODEND_matlab_control_get

!-  N O D E N D _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E -

      SUBROUTINE NODEND_matlab_inform_create( struct, NODEND_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold NODEND_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  NODEND_pointer - NODEND pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( NODEND_pointer_type ) :: NODEND_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 5
      CHARACTER ( LEN = 28 ), PARAMETER :: finform( ninform ) = (/             &
           'status                      ', 'alloc_status                ',     &
           'bad_alloc                   ', 'version                     ',     &
           'time                        '  /)

      INTEGER * 4, PARAMETER :: t_ninform = 4
      CHARACTER ( LEN = 21 ), PARAMETER :: t_finform( t_ninform ) = (/         &
           'total                ', 'metis               ',                    &
           'clock_total          ', 'clock_metis         ' /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, NODEND_pointer%pointer, &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      END IF

!  create the components

      CALL MATLAB_create_integer_component( NODEND_pointer%pointer,            &
        'status', NODEND_pointer%status )
      CALL MATLAB_create_integer_component( NODEND_pointer%pointer,            &
         'alloc_status', NODEND_pointer%alloc_status )
      CALL MATLAB_create_char_component( NODEND_pointer%pointer,               &
        'bad_alloc', NODEND_pointer%bad_alloc )
      CALL MATLAB_create_char_component( NODEND_pointer%pointer,               &
        'version', NODEND_pointer%version )

!  create the components of sub-structure time

      CALL MATLAB_create_substructure( NODEND_pointer%pointer,                 &
        'time', NODEND_pointer%time_pointer%pointer, t_ninform, t_finform )
      CALL MATLAB_create_real_component( NODEND_pointer%time_pointer%pointer,  &
        'total', NODEND_pointer%time_pointer%total )
      CALL MATLAB_create_real_component( NODEND_pointer%time_pointer%pointer,  &
        'metis', NODEND_pointer%time_pointer%metis )
      CALL MATLAB_create_real_component( NODEND_pointer%time_pointer%pointer,  &
        'clock_total', NODEND_pointer%time_pointer%clock_total )
      CALL MATLAB_create_real_component( NODEND_pointer%time_pointer%pointer,  &
        'clock_metis', NODEND_pointer%time_pointer%clock_metis )

      RETURN

!  End of subroutine NODEND_matlab_inform_create

      END SUBROUTINE NODEND_matlab_inform_create

!-*-  N O D E N D _ M A T L A B _ I N F O R M _ G E T   S U B R O U T I N E  -*-

      SUBROUTINE NODEND_matlab_inform_get( NODEND_inform, NODEND_pointer )

!  --------------------------------------------------------------

!  Set NODEND_inform values from matlab pointers

!  Arguments

!  NODEND_inform - NODEND inform structure
!  NODEND_pointer - NODEND pointer structure

!  --------------------------------------------------------------

      TYPE ( NODEND_inform_type ) :: NODEND_inform
      TYPE ( NODEND_pointer_type ) :: NODEND_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( NODEND_inform%status,                           &
                               mxGetPr( NODEND_pointer%status ) )
      CALL MATLAB_copy_to_ptr( NODEND_inform%alloc_status,                     &
                               mxGetPr( NODEND_pointer%alloc_status ) )
      CALL MATLAB_copy_to_ptr( NODEND_pointer%pointer,                         &
                               'bad_alloc', NODEND_inform%bad_alloc )
      CALL MATLAB_copy_to_ptr( NODEND_pointer%pointer,                         &
                               'version', NODEND_inform%version )
!  time components

      CALL MATLAB_copy_to_ptr( REAL( NODEND_inform%time%total, wp ),           &
                               mxGetPr( NODEND_pointer%time_pointer%total ) )
      CALL MATLAB_copy_to_ptr( REAL( NODEND_inform%time%metis, wp ),           &
                               mxGetPr( NODEND_pointer%time_pointer%metis ) )
      CALL MATLAB_copy_to_ptr( REAL( NODEND_inform%time%clock_total, wp ),     &
                      mxGetPr( NODEND_pointer%time_pointer%clock_total ) )
      CALL MATLAB_copy_to_ptr( REAL( NODEND_inform%time%clock_metis, wp ),     &
                      mxGetPr( NODEND_pointer%time_pointer%clock_metis ) )
      RETURN

!  End of subroutine NODEND_matlab_inform_get

      END SUBROUTINE NODEND_matlab_inform_get

!-*-*-  E N D  o f  G A L A H A D _ N O D E N D _ T Y P E S   M O D U L E  -*-*-

    END MODULE GALAHAD_NODEND_MATLAB_TYPES

