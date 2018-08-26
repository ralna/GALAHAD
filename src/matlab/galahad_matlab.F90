#include <fintrf.h>

!  THIS VERSION: GALAHAD 3.1 - 20/08/2018 AT 16:50 GMT.

!-*-*-*-*-*-*-*-*- G A L A H A D _ M A T L A B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.1. July 25th 2007

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_MATLAB

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: MATLAB_copy_from_ptr,                                          &
                MATLAB_copy_to_ptr,                                            &
                MATLAB_copy_single_to_ptr,                                     &
                MATLAB_create_integer,                                         &
                MATLAB_create_long,                                            &
                MATLAB_create_real,                                            &
                MATLAB_create_substructure,                                    &
                MATLAB_create_integer_component,                               &
                MATLAB_create_long_component,                                  &
                MATLAB_create_real_component,                                  &
                MATLAB_create_char_component,                                  &
                MATLAB_create_logical_component,                               &
                MATLAB_fill_component,                                         &
                MATLAB_fill_integer_component,                                 &
                MATLAB_fill_long_component,                                    &
                MATLAB_fill_real_component,                                    &
                MATLAB_fill_logical_component,                                 &
                MATLAB_fill_char_component,                                    &
                galmxCreateRealArray4Component,                                &
                galmxCreateRealArray8Component,                                &
                galmxGetCharacter,                                             &
                galmxGetReal,                                                  &
                galmxGetLogical,                                               &
                galmxGetInteger,                                               &
                galmxGetLong,                                                  &
                galmxCopyLongToPtr,                                            &
                galmxCopyInteger8ToPtr,                                        &
                galmxCopyLongArrayToPtr,                                       &
!               galmxCopyRealArrayToPtr,                                       &
!               galmxCopyIntegerArrayToPtr,                                    &
                galmxCopyPtrToInteger44,                                       &
                galmxCopyPtrToInteger84,                                       &
                MATLAB_get_value

!----------------------------
!   P r e c i s i o n,  e t c
!----------------------------

!  MATLAB kinds

      mwPointer :: dummy_mwPointer__
      mwSize :: dummy_mwSize__
      mwIndex :: dummy_mwIndex__
      INTEGER, PARAMETER, PUBLIC :: mwp_ = KIND( dummy_mwPointer__ )
      INTEGER, PARAMETER, PUBLIC :: mws_ = KIND( dummy_mwSize__ )
      INTEGER, PARAMETER, PUBLIC :: mwi_ = KIND( dummy_mwIndex__ )

!  integer kinds

      INTEGER * 1 :: dummy_int1__
      INTEGER * 2 :: dummy_int2__
      INTEGER * 4 :: dummy_int4__
      INTEGER * 8 :: dummy_int8__
      INTEGER :: dummy_default_integer__
      INTEGER, PARAMETER, PUBLIC :: int1_ = KIND( dummy_int1__ )
      INTEGER, PARAMETER, PUBLIC :: int2_ = KIND( dummy_int2__ )
      INTEGER, PARAMETER, PUBLIC :: int4_ = KIND( dummy_int4__ )
      INTEGER, PARAMETER, PUBLIC :: int8_ = KIND( dummy_int8__ )
!     INTEGER, PARAMETER, PUBLIC :: di_ = KIND( dummy_default_integer__ )
      INTEGER, PARAMETER, PUBLIC :: di_ = 4
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!  real kinds

      REAL * 4 :: dummy_real4__
      REAL * 8 :: dummy_real8__
      REAL :: dummy_default_real__
      INTEGER, PARAMETER, PUBLIC :: real4_ = KIND( dummy_real4__ )
      INTEGER, PARAMETER, PUBLIC :: real8_ = KIND( dummy_real8__ )
      INTEGER, PARAMETER, PUBLIC :: dr_ = KIND( dummy_default_real__ )
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  array classes

      INTEGER * 4, PARAMETER :: mx_class_real8 = 6, mx_class_real4 = 7,        &
                                mx_class_int1 = 8,  mx_class_int2 = 10,        &
                                mx_class_int4 = 12, mx_class_int8 = 14

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

      INTERFACE MATLAB_create_integer
        MODULE PROCEDURE galmxCreateInteger,                                   &
                         galmxCreateIntegerArray,                              &
                         galmxCreateIntegerMatrix
      END INTERFACE MATLAB_create_integer

      INTERFACE MATLAB_create_long
        MODULE PROCEDURE galmxCreateLong,                                      &
                         galmxCreateLongArray,                                 &
                         galmxCreateLongMatrix
      END INTERFACE MATLAB_create_long

      INTERFACE MATLAB_create_real
        MODULE PROCEDURE galmxCreateReal,                                      &
                         galmxCreateRealArray4,                                &
                         galmxCreateRealArray8,                                &
                         galmxCreateRealMatrix4,                               &
                         galmxCreateRealMatrix8
      END INTERFACE MATLAB_create_real

      INTERFACE MATLAB_create_integer_component
        MODULE PROCEDURE galmxCreateIntegerComponent,                          &
                         galmxCreateIntegerArray4Component,                    &
                         galmxCreateIntegerArray8Component,                    &
                         galmxCreateIntegerMatrix4Component,                   &
                         galmxCreateIntegerMatrix8Component
      END INTERFACE MATLAB_create_integer_component

      INTERFACE MATLAB_create_long_component
        MODULE PROCEDURE galmxCreateLongComponent,                             &
                         galmxCreateLongArrayComponent,                        &
                         galmxCreateLongMatrixComponent
      END INTERFACE MATLAB_create_long_component

      INTERFACE MATLAB_create_real_component
        MODULE PROCEDURE galmxCreateRealComponent,                             &
                         galmxCreateRealArray4Component,                       &
                         galmxCreateRealArray8Component,                       &
                         galmxCreateRealMatrix4Component,                      &
                         galmxCreateRealMatrix8Component
      END INTERFACE MATLAB_create_real_component

      INTERFACE MATLAB_create_char_component
        MODULE PROCEDURE galmxCreateCharComponent,                             &
                         galmxCreateCharArrayComponent
      END INTERFACE MATLAB_create_char_component

      INTERFACE MATLAB_create_substructure
        MODULE PROCEDURE galmxCreateSubstructure4,                             &
                         galmxCreateSubstructure8
      END INTERFACE MATLAB_create_substructure

      INTERFACE MATLAB_get_value
        MODULE PROCEDURE galmxGetInteger,                                      &
                         galmxGetReal,                                         &
                         galmxGetLogical,                                      &
                         galmxGetCharacter
      END INTERFACE MATLAB_get_value

      INTERFACE MATLAB_fill_integer_component
        MODULE PROCEDURE galmxFillInteger4Component,                           &
                         galmxFillInteger8Component
      END INTERFACE MATLAB_fill_integer_component

      INTERFACE MATLAB_fill_long_component
        MODULE PROCEDURE galmxFillLongComponent
      END INTERFACE MATLAB_fill_long_component

      INTERFACE MATLAB_fill_real_component
        MODULE PROCEDURE galmxFillRealComponent
      END INTERFACE MATLAB_fill_real_component

      INTERFACE MATLAB_fill_component
        MODULE PROCEDURE galmxFillInteger4Component
        MODULE PROCEDURE galmxFillInteger8Component
        MODULE PROCEDURE galmxFillRealComponent
        MODULE PROCEDURE MATLAB_fill_logical_component
        MODULE PROCEDURE MATLAB_fill_char_component
      END INTERFACE MATLAB_fill_component

      INTERFACE MATLAB_copy_from_ptr
        MODULE PROCEDURE galmxCopyPtrToInteger4,                               &
                         galmxCopyPtrToInteger8,                               &
                         galmxCopyPtrToInteger44,                              &
                         galmxCopyPtrToInteger48,                              &
                         galmxCopyPtrToInteger84,                              &
                         galmxCopyPtrToInteger88,                              &
                         galmxCopyPtrToReal,                                   &
                         galmxCopyPtrToRealArray4,                             &
                         galmxCopyPtrToRealArray8,                             &
                         galmxCopyPtrToRealMatrix4,                            &
                         galmxCopyPtrToRealMatrix8
      END INTERFACE MATLAB_copy_from_ptr

      INTERFACE MATLAB_copy_to_ptr
        MODULE PROCEDURE galmxCopyInteger4ToPtr,                               &
                         galmxCopyInteger8ToPtr,                               &
                         galmxCopyInteger44ArrayToPtr,                         &
                         galmxCopyInteger48ArrayToPtr,                         &
                         galmxCopyInteger84ArrayToPtr,                         &
                         galmxCopyInteger88ArrayToPtr,                         &
                         galmxCopyRealToPtr,                                   &
                         galmxCopyRealArray4ToPtr,                             &
                         galmxCopyRealArray8ToPtr,                             &
                         galmxCopyRealMatrix4ToPtr,                            &
                         galmxCopyRealMatrix8ToPtr,                            &
                         galmxCopyLogicalToPtr,                                &
                         galmxCopyLogicalArray4ToPtr,                          &
                         galmxCopyLogicalArray8ToPtr,                          &
                         galmxSetCharacterComponent,                           &
                         galmxSetCharacterArrayComponent
      END INTERFACE MATLAB_copy_to_ptr

      INTERFACE MATLAB_copy_single_to_ptr
        MODULE PROCEDURE galmxCopySingleToPtr,                                 &
                         galmxCopySingleArrayToPtr,                            &
                         galmxCopySingleMatrixToPtr
      END INTERFACE MATLAB_copy_single_to_ptr

      INTERFACE galmxCopyLongToPtr
        MODULE PROCEDURE galmxCopyInteger8ToPtr
      END INTERFACE galmxCopyLongToPtr

      INTERFACE galmxCopyLongArrayToPtr
        MODULE PROCEDURE galmxCopyInteger48ArrayToPtr,                         &
                         galmxCopyInteger88ArrayToPtr
      END INTERFACE galmxCopyLongArrayToPtr



    CONTAINS

!  -*-*-*-*-*-  g a l m x  C r e a t e  S u b s t r u c t u r e  4 -*-*-*-*-*-

      SUBROUTINE galmxCreateSubstructure4( struct, name, pr,                   &
                                           ninform, finform )
      mwPointer :: struct, pr
      CHARACTER ( len = * ) :: name
      INTEGER * 4  :: ninform
      CHARACTER ( LEN = * ), DIMENSION( ninform ) :: finform

!  -----------------------------------------------------

!  Create a named sub-structure of a structure

!  Arguments

!  struct - existing pointer to the structure
!  name - name of component of the structure
!  pr - pointer to the sub-tructure
!  ninform - number of components of the substructure
!  finform - names of components of the substructure

!  ------------------------------------------------------

      mwPointer :: mxCreateStructMatrix

      pr = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateSubstructure4

!  -*-*-*-*-*-  g a l m x  C r e a t e  S u b s t r u c t u r e  8 -*-*-*-*-*-

      SUBROUTINE galmxCreateSubstructure8( struct, name, pr,                   &
                                           ninform, finform )
      mwPointer :: struct, pr
      CHARACTER ( len = * ) :: name
      INTEGER * 8  :: ninform
      CHARACTER ( LEN = * ), DIMENSION( ninform ) :: finform

!  -----------------------------------------------------

!  Create a named sub-structure of a structure

!  Arguments

!  struct - existing pointer to the structure
!  name - name of component of the structure
!  pr - pointer to the sub-tructure
!  ninform - number of components of the substructure
!  finform - names of components of the substructure

!  ------------------------------------------------------

      mwPointer :: mxCreateStructMatrix

      pr = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateSubstructure8


!  -*-*-*- g a l m x  C r e a t e  I n t e g e r  C o m p o n e n t -*-*-*-*-

      SUBROUTINE galmxCreateIntegerComponent( struct, name, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwPointer :: pr

!  -----------------------------------------------

!  Create a named INTEGER component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  -----------------------------------------------

      pr = galmxCreateInteger( )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateIntegerComponent


!  -* g a l m x  C r e a t e  I n t e g e r  A r r a y  4  C o m p o n e n t  *-

      SUBROUTINE galmxCreateIntegerArray4Component( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 4 :: n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named INTEGER array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  -----------------------------------------------------

      mwSize :: nn

      nn = n
      pr = galmxCreateIntegerArray( nn )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateIntegerArray4Component


!  -* g a l m x  C r e a t e  I n t e g e r  A r r a y  8  C o m p o n e n t  *-

      SUBROUTINE galmxCreateIntegerArray8Component( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 8 :: n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named INTEGER array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  -----------------------------------------------------

      mwSize :: nn

      nn = n
      pr = galmxCreateIntegerArray( nn )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateIntegerArray8Component


!  - g a l m x  C r e a t e  I n t e g e r  M a t r i x  4  C o m p o n e n t -

      SUBROUTINE galmxCreateIntegerMatrix4Component( struct, name, m, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 4 :: m, n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named INTEGER matrix component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  m - row dimension of matrix
!  n - column dimension of matrix
!  pr - pointer to the structure

!  -----------------------------------------------------

      mwSize :: mm, nn

      mm = m ; nn = n
      pr = galmxCreateIntegerMatrix( mm, nn )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateIntegerMatrix4Component


!  - g a l m x  C r e a t e  I n t e g e r  M a t r i x  8  C o m p o n e n t -

      SUBROUTINE galmxCreateIntegerMatrix8Component( struct, name, m, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 8 :: m, n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named INTEGER matrix component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  m - row dimension of matrix
!  n - column dimension of matrix
!  pr - pointer to the structure

!  -----------------------------------------------------

      mwSize :: mm, nn

      mm = m ; nn = n
      pr = galmxCreateIntegerMatrix( mm, nn )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateIntegerMatrix8Component


!  -*-*-*-*- g a l m x  C r e a t e  L o n g  C o m p o n e n t -*-*-*-*-*-

      SUBROUTINE galmxCreateLongComponent( struct, name, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwPointer :: pr

!  -----------------------------------------------

!  Create a named LONG INTEGER component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  -----------------------------------------------

      pr = galmxCreateLong( )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateLongComponent


!  -*-*- g a l m x  C r e a t e  L o n g  A r r a y  C o m p o n e n t -*-*-

      SUBROUTINE galmxCreateLongArrayComponent( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwSize :: n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named LONG INTEGER array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  -----------------------------------------------------

      pr = galmxCreateLongArray( n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateLongArrayComponent

!  -*-*- g a l m x  C r e a t e  L o n g  M a t r i x  C o m p o n e n t -*-*-

      SUBROUTINE galmxCreateLongMatrixComponent( struct, name, m, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwSize :: m, n
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named LONG INTEGER matrix component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  m - row dimension of matrix
!  n - column dimension of matrix
!  pr - pointer to the structure

!  -----------------------------------------------------

      pr = galmxCreateLongMatrix( m, n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateLongMatrixComponent


!  -*-*-*-*-*- g a l m x  C r e a t e  R e a l  C o m p o n e n t -*-*-*-*-*-*-

      SUBROUTINE galmxCreateRealComponent( struct, name, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwPointer :: pr

!  --------------------------------------------

!  Create a named REAL component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  --------------------------------------------

      pr = galmxCreateReal( )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateRealComponent


!  -*-*- g a l m x  C r e a t e  R e a l  A r r a y  4  C o m p o n e n t -*-*-

      SUBROUTINE galmxCreateRealArray4Component( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 4 :: n
      mwPointer :: pr

!  --------------------------------------------------

!  Create a named REAL array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  --------------------------------------------------

      pr = galmxCreateRealArray4( n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateRealArray4Component

!  -*-*- g a l m x  C r e a t e  R e a l  A r r a y  8  C o m p o n e n t -*-*-

      SUBROUTINE galmxCreateRealArray8Component( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 8 :: n
      mwPointer :: pr

!  --------------------------------------------------

!  Create a named REAL array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  --------------------------------------------------

      pr = galmxCreateRealArray8( n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateRealArray8Component


!  -*-  g a l m x  C r e a t e  R e a l  M a t r i x  4  C o m p o n e n t  -*-

      SUBROUTINE galmxCreateRealMatrix4Component( struct, name, m, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 4 :: m, n
      mwPointer :: pr

!  --------------------------------------------------

!  Create a named REAL matrix component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  m - row dimension of matrix
!  n - column dimension of matrix
!  pr - pointer to the structure

!  --------------------------------------------------

      pr = galmxCreateRealMatrix4( m, n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateRealMatrix4Component


!  -*-  g a l m x  C r e a t e  R e a l  M a t r i x  8  C o m p o n e n t  -*-

      SUBROUTINE galmxCreateRealMatrix8Component( struct, name, m, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 8 :: m, n
      mwPointer :: pr

!  --------------------------------------------------

!  Create a named REAL matrix component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  m - row dimension of matrix
!  n - column dimension of matrix
!  pr - pointer to the structure

!  --------------------------------------------------

      pr = galmxCreateRealMatrix8( m, n )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateRealMatrix8Component


!  -*-*-*-*-*-*- g a l m x  C r e a t e  C h a r  C o m p o n e n t -*-*-*-*-*-

      SUBROUTINE galmxCreateCharComponent( struct, name, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwPointer :: pr

!  -------------------------------------------------

!  Create a named CHARACTER component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  -------------------------------------------------

      INTEGER, PARAMETER :: len_blank = 80
      CHARACTER ( len = len_blank ) :: blank
      mwPointer :: mxCreateCharMatrixFromStrings

      blank = REPEAT( ' ', len_blank )

      pr = mxCreateCharMatrixFromStrings( 1_mws_, blank )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateCharComponent

!  -*-*-*-*-*-*- g a l m x  C r e a t e  C h a r  C o m p o n e n t -*-*-*-*-*-

      SUBROUTINE galmxCreateCharArrayComponent( struct, name, n, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwSize :: n
      mwPointer :: pr

!  -------------------------------------------------

!  Create a named CHARACTER array component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  n - dimension of array
!  pr - pointer to the structure

!  -------------------------------------------------

      INTEGER, PARAMETER :: len_blank = 80
      CHARACTER ( len = len_blank ), dimension( n ) :: blank
      mwPointer :: mxCreateCharMatrixFromStrings

      blank = REPEAT( ' ', len_blank )

      pr = mxCreateCharMatrixFromStrings( n, blank )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE galmxCreateCharArrayComponent


!  -*-*-*-  M A T L A B _ c r e a t e _ c h a r _ c o m p o n e n t  -*-*-*-

!      SUBROUTINE MATLAB_create_char_component( struct, name, pr )
!      mwPointer :: struct
!      CHARACTER ( len = * ) :: name
!      mwPointer :: pr

!  -------------------------------------------------

!  Create a named CHARACTER component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  -------------------------------------------------

!      INTEGER, PARAMETER :: len_blank = 80
!      CHARACTER ( len = len_blank ) :: blank
!      mwPointer :: mxCreateCharMatrixFromStrings

!      blank = REPEAT( ' ', len_blank )

!      pr = mxCreateCharMatrixFromStrings( 1_mws_, blank )
!      CALL mxSetField( struct, 1_mwi_, name, pr )

!      RETURN
!      END SUBROUTINE MATLAB_create_char_component


!  -*-*-  M A T L A B _ c r e a t e _ l o g i c a l _ c o m p o n e n t  -*-*-

      SUBROUTINE MATLAB_create_logical_component( struct, name, pr )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      mwPointer :: pr

!  -----------------------------------------------------

!  Create a named LOGICAL component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

! ** - NB - ** This is a bodge since Mex doesn't appear
!              to handle Fortran logicals ** - NB - **

!  ----------------------------------------------------

      pr = galmxCreateInteger( )
      CALL mxSetField( struct, 1_mwi_, name, pr )

      RETURN
      END SUBROUTINE MATLAB_create_logical_component


!  -*-*-*-*-*-*-*-*-*-*- g a l m x  G e t  I n t e g e r  -*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE galmxGetInteger( ps, name, pc, value )
      mwPointer :: ps, pc
      CHARACTER ( LEN = * ) :: name
      INTEGER * 4 :: value

!  ---------------------------------------------------------

!  Obtain an integer value from the component of a structure

!  Arguments

!  ps - given pointer to the structure
!  name - name of component
!  pc - pointer to the component
!  value - value of the component

!  ---------------------------------------------------------

      mwPointer :: mxGetField
      REAL ( KIND = wp ) :: mxGetScalar

      pc = mxGetField( ps, 1_mwi_, name )
      value = INT( mxGetScalar( pc ) )

      RETURN
      END SUBROUTINE galmxGetInteger


!  -*-*-*-*-*-*-*-*-*-*- g a l m x  G e t  L o n g  -*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE galmxGetLong( ps, name, pc, value )
      mwPointer :: ps, pc
      CHARACTER ( LEN = * ) :: name
      INTEGER ( KIND = long ) :: value

!  ---------------------------------------------------------

!  Obtain a long integer value from the component of a structure

!  Arguments

!  ps - given pointer to the structure
!  name - name of component
!  pc - pointer to the component
!  value - value of the component

!  ---------------------------------------------------------

      mwPointer :: mxGetField
      REAL ( KIND = wp ) :: mxGetScalar

      pc = mxGetField( ps, 1_mwi_, name )
      value = INT( mxGetScalar( pc ) )

      RETURN
      END SUBROUTINE galmxGetLong


!  -*-*-*-*-*-*-*-*-*-*- g a l m x  G e t  R e a l  -*-*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE galmxGetReal( ps, name, pc, value )
      mwPointer :: ps, pc
      CHARACTER ( LEN = * ) :: name
      REAL ( KIND = real8_ ) :: value

!  -----------------------------------------------------

!  Obtain a real value from the component of a structure

!  Arguments

!  ps - given pointer to the structure
!  name - name of component
!  pc - pointer to the component
!  value - value of the component

!  -----------------------------------------------------

      mwPointer :: mxGetField
      REAL ( KIND = wp ) :: mxGetScalar

      pc = mxGetField( ps, 1_mwi_, name )
      value = mxGetScalar( pc )

      RETURN
      END SUBROUTINE galmxGetReal


!  -*-*-*-*-*-*-*-*-*-*- g a l m x  G e t  L o g i c a l  -*-*-*-*-*-*-*-*-*-*-

      SUBROUTINE galmxGetLogical( ps, name, pc, value )
      mwPointer :: ps, pc
      CHARACTER ( LEN = * ) :: name
      LOGICAL * 4 :: value

!  ---------------------------------------------------------

!  Obtain a logical value from the component of a structure

!  Arguments

!  ps - given pointer to the structure
!  name - name of component
!  pc - pointer to the component
!  value - value of the component

! ** - NB - ** This is a bodge since Mex doesn't appear
!              to handle Fortran logicals ** - NB - **

!  ---------------------------------------------------------

      mwPointer :: mxGetField
      REAL ( KIND = wp ) :: mxGetScalar

      pc = mxGetField( ps, 1_mwi_, name )
      IF ( INT( mxGetScalar( pc ) ) == 1 ) THEN
        value = .TRUE.
      ELSE
        value = .FALSE.
      END IF

      RETURN
      END SUBROUTINE galmxGetLogical


!  -*-*-*-*-*-*-*-*-*- g a l m x  G e t  C h a r a c t e r  -*-*-*-*-*-*-*-*-*-

      SUBROUTINE galmxGetCharacter( ps, name, pc, value, len )
      mwPointer :: ps, pc
      CHARACTER ( LEN = * ) :: name
      CHARACTER ( LEN = * ) :: value
      mwSize :: len

!  -----------------------------------------------------------------

!  Obtain a character string value from the component of a structure

!  Arguments

!  ps - given pointer to the structure
!  name - name of component
!  pc - pointer to the component
!  value - value of the character string
!  len - maximum length of the string

!  ------------------------------------------------------------------

      mwSize :: mxGetString
      mwPointer :: mxGetField
      INTEGER( KIND = int4_ ) :: i

      pc = mxGetField( ps, 1_mwi_, name )
      i = mxGetString( pc, value, len )

      RETURN
      END SUBROUTINE galmxGetCharacter

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r  4 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger4( px, y )
      mwPointer :: px
      INTEGER * 4 :: y

!  -----------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran variable

!  Arguments

!  px  - Pointer to variable
!  y   - Integer*4 Fortran variable

!  -----------------------------------------------------------------

      mwSize :: nn
      INTEGER * 4 :: Y_array( 1 )

!     CALL mexWarnMsgTxt( ' 4 ' )

      nn = 1
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyPtrToInteger8( px, Y_array, nn )
      CASE ( int2_ )
        CALL mxCopyPtrToInteger2( px, Y_array, nn )
      CASE ( int1_ )
        CALL mxCopyPtrToInteger1( px, Y_array, nn )
      CASE default
        CALL mxCopyPtrToInteger4( px, Y_array, nn )
      END SELECT
      y = Y_array( 1 )

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger4

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r  8 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger8( px, y )
      mwPointer :: px
      INTEGER * 8 :: y

!  -----------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran variable

!  Arguments

!  px  - Pointer to variable
!  y   - Integer*4 Fortran variable

!  -----------------------------------------------------------------

      mwSize :: nn
      INTEGER * 8 :: Y_array( 1 )

!     CALL mexWarnMsgTxt( ' 4 ' )

      nn = 1
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyPtrToInteger8( px, Y_array, nn )
      CASE ( int2_ )
        CALL mxCopyPtrToInteger2( px, Y_array, nn )
      CASE ( int1_ )
        CALL mxCopyPtrToInteger1( px, Y_array, nn )
      CASE default
        CALL mxCopyPtrToInteger4( px, Y_array, nn )
      END SELECT
      y = Y_array( 1 )

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger8

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r 4 4 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger44( px, Y, n, sparse )
      mwPointer :: px
      INTEGER * 4 :: n
      INTEGER * 4, DIMENSION( n ) :: Y
      LOGICAL, OPTIONAL :: sparse

!  --------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Integer*4 Fortran array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn
      INTEGER :: alloc_stat
      mwIndex, DIMENSION(:), ALLOCATABLE :: temp_mwi

      CHARACTER ( len = 80 ) :: debug = REPEAT( ' ', 80 )
!     CALL mexWarnMsgTxt( ' 44 ' )

      nn = n
      IF ( PRESENT( sparse ) ) THEN
        SELECT CASE( mwi_ )
        CASE( kind( Y ) )
          CALL mxCopyPtrToInteger4( px, Y, nn )
        CASE default
          ALLOCATE( temp_mwi( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
          SELECT CASE ( mwi_ )
          CASE ( int8_ )
            CALL mxCopyPtrToInteger8( px, temp_mwi, nn )
          CASE ( int2_ )
            CALL mxCopyPtrToInteger2( px, temp_mwi, nn )
          CASE ( int1_ )
            CALL mxCopyPtrToInteger1( px, temp_mwi, nn )
          CASE default
            CALL mxCopyPtrToInteger4( px, temp_mwi, nn )
          END SELECT
          Y( : n ) = temp_mwi( : n )
          DEALLOCATE( temp_mwi, STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
        END SELECT
      ELSE
        SELECT CASE ( di_ )
        CASE ( int8_ )
          CALL mxCopyPtrToInteger8( px, Y, nn )
        CASE ( int2_ )
          CALL mxCopyPtrToInteger2( px, Y, nn )
        CASE ( int1_ )
          CALL mxCopyPtrToInteger1( px, Y, nn )
        CASE default
!         CALL mexWarnMsgTxt( ' default ' )
          CALL mxCopyPtrToInteger4( px, Y, nn )
!         write(debug,*) px, nn
!         CALL mexErrMsgTxt( TRIM(debug))
        END SELECT
      END IF

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger44

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r 8 4 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger84( px, Y, n, sparse )
      mwPointer :: px
      INTEGER * 4 :: n
      INTEGER * 8, DIMENSION( n ) :: Y
      LOGICAL, OPTIONAL :: sparse

!  --------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Integer*8 Fortran array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn
      INTEGER :: alloc_stat
      mwIndex, DIMENSION(:), ALLOCATABLE :: temp_mwi

!     CALL mexWarnMsgTxt( ' 84 ' )

      nn = n

      IF ( PRESENT( sparse ) ) THEN
        SELECT CASE( mwi_ )
        CASE( kind( Y ) )
            CALL mxCopyPtrToInteger8( px, Y, nn )
        CASE default
          ALLOCATE( temp_mwi( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
          SELECT CASE ( mwi_ )
          CASE ( int8_ )
            CALL mxCopyPtrToInteger8( px, temp_mwi, nn )
          CASE ( int2_ )
            CALL mxCopyPtrToInteger2( px, temp_mwi, nn )
          CASE ( int1_ )
            CALL mxCopyPtrToInteger1( px, temp_mwi, nn )
          CASE default
            CALL mxCopyPtrToInteger4( px, temp_mwi, nn )
          END SELECT
          Y( : n ) = temp_mwi( : n )
          DEALLOCATE( temp_mwi, STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
        END SELECT
      ELSE
        SELECT CASE ( di_ )
        CASE ( int8_ )
          CALL mxCopyPtrToInteger8( px, Y, n )
        CASE ( int2_ )
          CALL mxCopyPtrToInteger2( px, Y, n )
        CASE ( int1_ )
          CALL mxCopyPtrToInteger1( px, Y, n )
        CASE default
          CALL mxCopyPtrToInteger4( px, Y, n )
        END SELECT
      END IF

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger84

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r 4 8 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger48( px, Y, n, sparse )
      mwPointer :: px
      INTEGER * 8 :: n
      INTEGER * 4, DIMENSION( n ) :: Y
      LOGICAL, OPTIONAL :: sparse

!  --------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Integer*4 Fortran array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn

!     INTEGER * 4 :: mxGetClassID
      INTEGER :: alloc_stat
      mwIndex, DIMENSION(:), ALLOCATABLE :: temp_mwi

!     CALL mexWarnMsgTxt( ' 48 ' )

      nn = n
      IF ( PRESENT( sparse ) ) THEN
        SELECT CASE( mwi_ )
        CASE( kind( Y ) )
            CALL mxCopyPtrToInteger4( px, Y, nn )
        CASE default
          ALLOCATE( temp_mwi( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
          SELECT CASE ( mwi_ )
          CASE ( int8_ )
            CALL mxCopyPtrToInteger8( px, temp_mwi, nn )
          CASE ( int2_ )
            CALL mxCopyPtrToInteger2( px, temp_mwi, nn )
          CASE ( int1_ )
            CALL mxCopyPtrToInteger1( px, temp_mwi, nn )
          CASE default
            CALL mxCopyPtrToInteger4( px, temp_mwi, nn )
          END SELECT
          Y( : n ) = temp_mwi( : n )
          DEALLOCATE( temp_mwi, STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
        END SELECT
      ELSE
        SELECT CASE ( di_ )
        CASE ( int8_ )
          CALL mxCopyPtrToInteger8( px, Y, nn )
        CASE ( int2_ )
          CALL mxCopyPtrToInteger2( px, Y, nn )
        CASE ( int1_ )
          CALL mxCopyPtrToInteger1( px, Y, nn )
        CASE default
          CALL mxCopyPtrToInteger4( px, Y, nn )
        END SELECT
      END IF

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger48

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  I n t e g e r 8 8 -*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToInteger88( px, Y, n, sparse )
      mwPointer :: px
      INTEGER * 8 :: n
      INTEGER * 8, DIMENSION( n ) :: Y
      LOGICAL, OPTIONAL :: sparse

!  --------------------------------------------------------------

!  Copy INTEGER values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Integer*8 Fortran array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn
      INTEGER :: alloc_stat
      mwIndex, DIMENSION(:), ALLOCATABLE :: temp_mwi

!     CALL mexWarnMsgTxt( ' 88 ' )

      nn = n
      IF ( PRESENT( sparse ) ) THEN
        SELECT CASE( mwi_ )
        CASE( kind( Y ) )
            CALL mxCopyPtrToInteger8( px, Y, nn )
        CASE default
          ALLOCATE( temp_mwi( n ), STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' allocation failure ' )
          SELECT CASE ( mwi_ )
          CASE ( int8_ )
            CALL mxCopyPtrToInteger8( px, temp_mwi, nn )
          CASE ( int2_ )
            CALL mxCopyPtrToInteger2( px, temp_mwi, nn )
          CASE ( int1_ )
            CALL mxCopyPtrToInteger1( px, temp_mwi, nn )
          CASE default
            CALL mxCopyPtrToInteger4( px, temp_mwi, nn )
          END SELECT
          Y( : n ) = temp_mwi( : n )
          DEALLOCATE( temp_mwi, STAT = alloc_stat )
          IF ( alloc_stat /= 0 ) CALL mexErrMsgTxt( ' deallocation failure ' )
        END SELECT
      ELSE
        SELECT CASE ( di_ )
        CASE ( int8_ )
          CALL mxCopyPtrToInteger8( px, Y, n )
        CASE ( int2_ )
          CALL mxCopyPtrToInteger2( px, Y, n )
        CASE ( int1_ )
          CALL mxCopyPtrToInteger1( px, Y, n )
        CASE default
          CALL mxCopyPtrToInteger4( px, Y, n )
        END SELECT
      END IF

      RETURN
      END SUBROUTINE galmxCopyPtrToInteger88

!  -*-*-*-*-*-*- g a l m x  C o p y  P t r  T o  R e a l  -*-*-*-*-*-*-

      SUBROUTINE galmxCopyPtrToReal( px, Y )
      mwPointer :: px
      REAL ( KIND = wp ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Real Fortran array

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp

      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyPtrToReal8( px, Y, 1_mws_ )
      CASE default
        CALL mxCopyPtrToReal4( px, Y, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyPtrToReal


!  -*-*-*-*-*- g a l m x  C o p y  P t r  T o  R e a l  A r r a y 4  -*-*-*-*-

      SUBROUTINE galmxCopyPtrToRealArray4( px, Y, n )
      mwPointer :: px
      INTEGER * 4 :: n
      REAL ( KIND = wp ), DIMENSION( n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Real Fortran array
!  n   - number of elements to copy

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: nn

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyPtrToReal8( px, Y, nn )
      CASE default
        CALL mxCopyPtrToReal4( px, Y, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyPtrToRealArray4


!  -*-*-*-*-*- g a l m x  C o p y  P t r  T o  R e a l  A r r a y 8  -*-*-*-*-

      SUBROUTINE galmxCopyPtrToRealArray8( px, Y, n )
      mwPointer :: px
      INTEGER * 8 :: n
      REAL ( KIND = wp ), DIMENSION( n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Real Fortran array
!  n   - number of elements to copy

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: nn

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyPtrToReal8( px, Y, nn )
      CASE default
        CALL mxCopyPtrToReal4( px, Y, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyPtrToRealArray8


!  -*-*-*-*-*- g a l m x  C o p y  P t r  T o  R e a l  M a t r i x 4  -*-*-*-

      SUBROUTINE galmxCopyPtrToRealMatrix4( px, Y, m, n )
      mwPointer :: px
      INTEGER * 4 :: m, n
      REAL ( KIND = wp ), DIMENSION( m, n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Real Fortran array
!  m   - row dimension
!  n   - column dimension
!
!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: mn

      mn = m * n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyPtrToReal8( px, Y, mn )
      CASE default
        CALL mxCopyPtrToReal4( px, Y, mn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyPtrToRealMatrix4

!  -*-*-*-*-*- g a l m x  C o p y  P t r  T o  R e a l  M a t r i x 8  -*-*-*-

      SUBROUTINE galmxCopyPtrToRealMatrix8( px, Y, m, n )
      mwPointer :: px
      INTEGER * 8 :: m, n
      REAL ( KIND = wp ), DIMENSION( m, n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Matlab pointer array to Fortran array

!  Arguments

!  px  - Pointer to ir or jc array
!  Y   - Real Fortran array
!  m   - row dimension
!  n   - column dimension
!
!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: mn

      mn = m * n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyPtrToReal8( px, Y, mn )
      CASE default
        CALL mxCopyPtrToReal4( px, Y, mn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyPtrToRealMatrix8


!  -*-*-*-*-*-  g a l m x  C o p y  I n t e g e r 4  T o  P t r  -*-*-*-*-*-*-*

      SUBROUTINE galmxCopyInteger4ToPtr( Y, px )
      mwPointer :: px
      INTEGER * 4 :: Y

!  ---------------------------------------------------------

!  Copy INTEGER values from Fortran scalar to Matlab pointer

!  Arguments

!  Y   - Integer*4 Fortran scalar
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  ---------------------------------------------------------

!   INTEGER ::  mexPrintf
!   integer*4 out
!   CHARACTER ( LEN = 200 ) :: str
!   WRITE( str, "( ' Y = ', I0, ' di = ', I0  )" ) Y, di_
!   out = mexPrintf( TRIM( str ) // achar(10) )

      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, 1_mws_ )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, 1_mws_ )
      CASE default
        CALL mxCopyInteger4ToPtr( Y, px, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger4ToPtr


!  -*-*-*-*-*-  g a l m x  C o p y  I n t e g e r 8  T o  P t r  -*-*-*-*-*-*-*

      SUBROUTINE galmxCopyInteger8ToPtr( Y, px )
      mwPointer :: px
      INTEGER * 8 :: Y

!  ---------------------------------------------------------

!  Copy INTEGER values from Fortran scalar to Matlab pointer

!  Arguments

!  Y   - Integer*4 Fortran scalar
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  ---------------------------------------------------------

!   INTEGER ::  mexPrintf
!   integer*4 out
!   CHARACTER ( LEN = 200 ) :: str
!   WRITE( str, "( ' Y = ', I0, ' di = ', I0  )" ) Y, di_
!   out = mexPrintf( TRIM( str ) // achar(10) )

      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, 1_mws_ )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, 1_mws_ )
      CASE default
!       CALL mxCopyInteger4ToPtr( Y, px, 1_mws_ )
        CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger8ToPtr


!  -*-*-  g a l m x  C o p y  I n t e g e r  4  4  A r r a y  T o  P t r  -*-*-

      SUBROUTINE galmxCopyInteger44ArrayToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 4 :: n
      INTEGER * 4, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy INTEGER values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Integer*4 Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn

      nn = n
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, nn )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, nn )
      CASE default
        CALL mxCopyInteger4ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger44ArrayToPtr


!  -*-*-  g a l m x  C o p y  I n t e g e r  4  8  A r r a y  T o  P t r  -*-*-

      SUBROUTINE galmxCopyInteger48ArrayToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 4 :: n
      INTEGER * 8, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy INTEGER values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Integer*8 Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn

      nn = n
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, nn )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, nn )
      CASE default
!       CALL mxCopyInteger4ToPtr( Y, px, nn )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger48ArrayToPtr


!  -*-*-  g a l m x  C o p y  I n t e g e r  8  4  A r r a y  T o  P t r  -*-*-

      SUBROUTINE galmxCopyInteger84ArrayToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 8 :: n
      INTEGER * 4, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy INTEGER values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Integer*4 Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn

      nn = n
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, nn )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, nn )
      CASE default
        CALL mxCopyInteger4ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger84ArrayToPtr


!  -*-*-  g a l m x  C o p y  I n t e g e r  8  8  A r r a y  T o  P t r  -*-*-

      SUBROUTINE galmxCopyInteger88ArrayToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 8 :: n
      INTEGER * 8, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy INTEGER values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Integer*8 Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      mwSize :: nn

      nn = n
      SELECT CASE ( di_ )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, nn )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, nn )
      CASE default
!       CALL mxCopyInteger4ToPtr( Y, px, nn )
        CALL mxCopyInteger8ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyInteger88ArrayToPtr


!  -*-*-*-*-  g a l m x  C o p y  R e a l  T o  P t r  -*-*-*-*-*-

      SUBROUTINE galmxCopyRealToPtr( Y, px )
      mwPointer :: px
      REAL ( KIND = wp ) :: Y

!  -----------------------------------------------------

!  Copy REAL value from Fortran scalar to Matlab pointer

!  Arguments

!  Y   - Real Fortran scalar
!  px  - Pointer to ir or jc array

!  -----------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp

      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, 1_mws_ )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyRealToPtr


!  -*-*-*-  g a l m x  C o p y  R e a l  A r r a y  4  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopyRealArray4ToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 4 :: n
      REAL ( KIND = wp ), DIMENSION( n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Real 1-D Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: nn

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, nn )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyRealArray4ToPtr

!  -*-*-*-  g a l m x  C o p y  R e a l  A r r a y  8  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopyRealArray8ToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 8 :: n
      REAL ( KIND = wp ), DIMENSION( n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Real 1-D Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: nn

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, nn )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, nn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyRealArray8ToPtr


!  -*-*-*-  g a l m x  C o p y  R e a l  M a t r i x  4  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopyRealMatrix4ToPtr( Y, px, m, n )
      mwPointer :: px
      INTEGER * 4 :: m, n
      REAL ( KIND = wp ), DIMENSION( m, n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Real 2-D Fortran array
!  px  - Pointer to ir or jc array
!  m   - row dimension
!  n   - column dimension

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: mn

      mn = m * n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, mn )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, mn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyRealMatrix4ToPtr


!  -*-*-*-  g a l m x  C o p y  R e a l  M a t r i x  8  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopyRealMatrix8ToPtr( Y, px, m, n )
      mwPointer :: px
      INTEGER * 8 :: m, n
      REAL ( KIND = wp ), DIMENSION( m, n ) :: Y

!  -----------------------------------------------------------

!  Copy REAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Real 2-D Fortran array
!  px  - Pointer to ir or jc array
!  m   - row dimension
!  n   - column dimension

!  -----------------------------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwSize :: mn

      mn = m * n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, mn )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, mn )
      END SELECT

      RETURN
      END SUBROUTINE galmxCopyRealMatrix8ToPtr


!  -*-*-*-*-  g a l m x  C o p y  S i n g l e  T o  P t r  -*-*-*-*-*-

      SUBROUTINE galmxCopySingleToPtr( Y, px )
      mwPointer :: px
      REAL :: Y

!  -----------------------------------------------------------------

!  Copy SINGLE PRECISION value from Fortran scalar to Matlab pointer

!  Arguments

!  Y   - Real Fortran scalar
!  px  - Pointer to ir or jc array

!  -----------------------------------------------------------------

      CALL mxCopyReal4ToPtr( Y, px, 1_mws_ )

      RETURN
      END SUBROUTINE galmxCopySingleToPtr


!  -*-*-*-  g a l m x  C o p y  S i n g l e  A r r a y  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopySingleArrayToPtr( Y, px, n )
      mwPointer :: px
      mwSize :: n
      REAL, DIMENSION( n ) :: Y

!  ------------------------------------------------------------------------

!  Copy SINGLE PRECISION values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Single precision 1-D Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  ------------------------------------------------------------------------

      CALL mxCopyReal4ToPtr( Y, px, n )

      RETURN
      END SUBROUTINE galmxCopySingleArrayToPtr


!  -*-*-*-  g a l m x  C o p y  S i n g l e  M a t r i x  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopySingleMatrixToPtr( Y, px, m, n )
      mwPointer :: px
      mwSize :: m, n
      REAL, DIMENSION( m, n ) :: Y

!  ------------------------------------------------------------------------

!  Copy SINGLE PRECISION values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Single precision 2-D Fortran array
!  px  - Pointer to ir or jc array
!  m   - row dimension
!  n   - column dimension

!  ------------------------------------------------------------------------

      CALL mxCopyReal4ToPtr( Y, px, m * n )

      RETURN
      END SUBROUTINE galmxCopySingleMatrixToPtr


!  -*-*-*-  g a l m x  C o p y  L o g i c a l  T o  P t r  -*-*-*-*-

      SUBROUTINE galmxCopyLogicalToPtr( Y, px )
      mwPointer :: px
      LOGICAL * 4 :: Y

!  --------------------------------------------------------

!  Copy LOGICAL value from Fortran scalar to Matlab pointer

!  Arguments

!  Y   - Logical Fortran scalar
!  px  - Pointer to ir or jc array

!  --------------------------------------------------------

      INTEGER * 4 :: ly

      IF ( Y ) THEN
        ly = 1
      ELSE
        ly = 0
      END IF
      CALL galmxCopyInteger4ToPtr( ly, px )

      RETURN
      END SUBROUTINE galmxCopyLogicalToPtr


!  -*-*-*-  g a l m x  C o p y  L o g i c a l  A r r a y  4  T o  P t r  -*-*-*-

      SUBROUTINE galmxCopyLogicalArray4ToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 4 :: n
      LOGICAL, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy LOGICAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Logical Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      INTEGER * 4 :: i
      INTEGER * 4, DIMENSION( n ) :: LY

      DO i = 1, n
        IF ( Y( i ) ) THEN
          LY( i ) = 1
        ELSE
          LY( i ) = 0
        END IF
      END DO

      CALL galmxCopyInteger44ArrayToPtr( LY, px, n )

      RETURN
      END SUBROUTINE galmxCopyLogicalArray4ToPtr



!  -*-*-*-  g a l m x  C o p y  L o g i c a l  A r r a y  8  T o  P t r  -*-*-*-

      SUBROUTINE galmxCopyLogicalArray8ToPtr( Y, px, n )
      mwPointer :: px
      INTEGER * 8 :: n
      LOGICAL, DIMENSION( n ) :: Y

!  --------------------------------------------------------------

!  Copy LOGICAL values from Fortran array to Matlab pointer array

!  Arguments

!  Y   - Logical Fortran array
!  px  - Pointer to ir or jc array
!  n   - number of elements to copy

!  --------------------------------------------------------------

      INTEGER * 8 :: i
      INTEGER * 4, DIMENSION( n ) :: LY

      DO i = 1, n
        IF ( Y( i ) ) THEN
          LY( i ) = 1
        ELSE
          LY( i ) = 0
        END IF
      END DO

      CALL galmxCopyInteger84ArrayToPtr( LY, px, n )

      RETURN
      END SUBROUTINE galmxCopyLogicalArray8ToPtr


!  -*-*-*-  g a l m x  S e t  C h a r a c t e r  C o m p o n e n t  -*-*-*-*-

      SUBROUTINE galmxSetCharacterComponent( struct, name, value )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      CHARACTER ( len = * ) :: value

!  ---------------------------------------------------------

!  Set a named CHARACTER component of a structure to a value

!  Arguments

!  struct - structure
!  name - name of component
!  value - character value to be assigned

!  ---------------------------------------------------------

      mwPointer :: mxCreateString

      CALL mxSetField( struct, 1_mwi_, name, mxCreateString( value ) )

      RETURN
      END SUBROUTINE galmxSetCharacterComponent


!  -*-*-*-  g a l m x  S e t  C h a r a c t e r  C o m p o n e n t  -*-*-*-*-

      SUBROUTINE galmxSetCharacterArrayComponent( struct, name, value )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      CHARACTER ( len = * ), DIMENSION( : ) :: value

!  ---------------------------------------------------------

!  Set a named CHARACTER component of a structure to a value

!  Arguments

!  struct - structure
!  name - name of component
!  value - array of character values to be assigned

!  ---------------------------------------------------------

      mwPointer :: mxCreateString
      INTEGER :: i

      DO i = 1, SIZE( value )
        CALL mxSetField( struct, i, name, mxCreateString( value( i ) ) )
      END DO

      RETURN
      END SUBROUTINE galmxSetCharacterArrayComponent


!  -*-*-*-  g a l m x  C r e a t e  I n t e g e r  -*-*-*-*-

      FUNCTION galmxCreateInteger( )
      mwPointer :: galmxCreateInteger

!  -------------------------------------

!  Create an unpopulated INTEGER pointer

!  No arguments

!  -------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericMatrix

      SELECT CASE ( di_ )
      CASE ( int8_ )
        galmxCreateInteger = mxCreateNumericMatrix( 1_mws_, 1_mws_,            &
                               mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateInteger = mxCreateNumericMatrix( 1_mws_, 1_mws_,            &
                               mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateInteger = mxCreateNumericMatrix( 1_mws_, 1_mws_,            &
                               mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateInteger = mxCreateNumericMatrix( 1_mws_, 1_mws_,            &
                               mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateInteger


!  -*-*-*-  g a l m x  C r e a t e  I n t e g e r  A r r a y -*-*-*-*-

      FUNCTION galmxCreateIntegerArray( n )
      mwPointer :: galmxCreateIntegerArray
      mwSize :: n

!  --------------------------------------------

!  Create an unpopulated INTEGER pointer array

!  Arguments

!  n - number of entries

!  --------------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray

      SELECT CASE ( di_ )
      CASE ( int8_ )
         galmxCreateIntegerArray = mxCreateNumericArray( 2_mws_,               &
           (/ n, 1_mws_ /), mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateIntegerArray = mxCreateNumericArray( 2_mws_,                &
           (/ n, 1_mws_ /), mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateIntegerArray = mxCreateNumericArray( 2_mws_,                &
            (/ n, 1_mws_ /), mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateIntegerArray = mxCreateNumericArray( 2_mws_,                &
            (/ n, 1_mws_ /), mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateIntegerArray

!  -*-*-*-  g a l m x  C r e a t e  I n t e g e r  M a t r i x -*-*-*-*-

      FUNCTION galmxCreateIntegerMatrix( m, n )
      mwPointer :: galmxCreateIntegerMatrix
      mwSize :: m, n

!  --------------------------------------------

!  Create an unpopulated INTEGER pointer array

!  Arguments

!  m - number of rows
!  n - number of columns

!  --------------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray

      SELECT CASE ( di_ )
      CASE ( int8_ )
         galmxCreateIntegerMatrix = mxCreateNumericArray( 2_mws_,              &
           (/ m, n /), mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateIntegerMatrix = mxCreateNumericArray( 2_mws_,               &
           (/ m, n /), mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateIntegerMatrix = mxCreateNumericArray( 2_mws_,               &
            (/ m, n /), mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateIntegerMatrix = mxCreateNumericArray( 2_mws_,               &
            (/ m, n /), mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateIntegerMatrix

!  -*-*-*-  g a l m x  C r e a t e  L o n g  -*-*-*-*-

      FUNCTION galmxCreateLong( )
      mwPointer :: galmxCreateLong

!  -------------------------------------

!  Create an unpopulated LONG INTEGER pointer

!  No arguments

!  -------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericMatrix

      SELECT CASE ( long )
      CASE ( int8_ )
        galmxCreateLong = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                               mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateLong = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                               mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateLong = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                               mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateLong = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                               mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateLong

!  -*-*-*-  g a l m x  C r e a t e  L o n g  A r r a y -*-*-*-*-

      FUNCTION galmxCreateLongArray( n )
      mwPointer :: galmxCreateLongArray
      mwSize :: n

!  --------------------------------------------

!  Create an unpopulated LONG INTEGER pointer array

!  Arguments

!  n - number of entries

!  --------------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray

      SELECT CASE ( long )
      CASE ( int8_ )
         galmxCreateLongArray = mxCreateNumericArray( 2_mws_,                  &
           (/ n, 1_mws_ /), mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateLongArray = mxCreateNumericArray( 2_mws_,                   &
           (/ n, 1_mws_ /), mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateLongArray = mxCreateNumericArray( 2_mws_,                   &
            (/ n, 1_mws_ /), mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateLongArray = mxCreateNumericArray( 2_mws_,                   &
            (/ n, 1_mws_ /), mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateLongArray

!  -*-*-*-  g a l m x  C r e a t e  L o n g  M a t r i x -*-*-*-*-

      FUNCTION galmxCreateLongMatrix( m, n )
      mwPointer :: galmxCreateLongMatrix
      mwSize :: m, n

!  --------------------------------------------

!  Create an unpopulated LONG INTEGER pointer array

!  Arguments

!  m - number of rows
!  n - number of columns

!  --------------------------------------------

      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray

      SELECT CASE ( long )
      CASE ( int8_ )
         galmxCreateLongMatrix = mxCreateNumericArray( 2_mws_,                 &
           (/ m, n /), mxClassIDFromClassName('int64'), 0_int4_ )
      CASE ( int2_ )
        galmxCreateLongMatrix = mxCreateNumericArray( 2_mws_,                  &
           (/ m, n /), mxClassIDFromClassName('int16'), 0_int4_ )
      CASE ( int1_ )
        galmxCreateLongMatrix = mxCreateNumericArray( 2_mws_,                  &
            (/ m, n /), mxClassIDFromClassName('int8'),  0_int4_ )
      CASE default
        galmxCreateLongMatrix = mxCreateNumericArray( 2_mws_,                  &
            (/ m, n /), mxClassIDFromClassName('int32'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateLongMatrix


!  -*-*-*-  g a l m x  C r e a t e  R e a l   -*-*-*-*-

      FUNCTION galmxCreateReal( )
      mwPointer :: galmxCreateReal

!  ----------------------------------

!  Create an unpopulated REAL pointer

!  No arguments

!  ----------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericMatrix

      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        galmxCreateReal = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                            mxClassIDFromClassName('double'), 0_int4_ )
      CASE default
        galmxCreateReal = mxCreateNumericMatrix( 1_mws_, 1_mws_,               &
                            mxClassIDFromClassName('single'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateReal


!  -*-*-*-  g a l m x  C r e a t e  R e a l  A r r a y  4  -*-*-*-*-

      FUNCTION galmxCreateRealArray4( n )
      mwPointer :: galmxCreateRealArray4
      INTEGER * 4 :: n

!  -----------------------------------------

!  Create an unpopulated REAL pointer array

!  Arguments

!  n - number of entries

!  -----------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray
      mwSize :: nn

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        galmxCreateRealArray4 = mxCreateNumericArray( 2_mws_, (/ nn, 1_mws_ /),&
                                 mxClassIDFromClassName('double'), 0_int4_ )
      CASE default
        galmxCreateRealArray4 = mxCreateNumericArray( 2_mws_, (/ nn, 1_mws_ /),&
                                 mxClassIDFromClassName('single'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateRealArray4


!  -*-*-*-  g a l m x  C r e a t e  R e a l  A r r a y  8  -*-*-*-*-

      FUNCTION galmxCreateRealArray8( n )
      mwPointer :: galmxCreateRealArray8
      INTEGER * 8 :: n

!  -----------------------------------------

!  Create an unpopulated REAL pointer array

!  Arguments

!  n - number of entries

!  -----------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      INTEGER * 4 :: mxClassIDFromClassName
      mwSize :: nn
      mwPointer :: mxCreateNumericArray

      nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        galmxCreateRealArray8 = mxCreateNumericArray( 2_mws_, (/ nn, 1_mws_ /),&
                                 mxClassIDFromClassName('double'), 0_int4_ )
      CASE default
        galmxCreateRealArray8 = mxCreateNumericArray( 2_mws_, (/ nn, 1_mws_ /),&
                                 mxClassIDFromClassName('single'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateRealArray8


!  -*-*-*-  g a l m x  C r e a t e  R e a l  M a t r i x  4  -*-*-*-*-

      FUNCTION galmxCreateRealMatrix4( m, n )
      mwPointer :: galmxCreateRealMatrix4
      INTEGER * 4 :: m, n

!  -----------------------------------------

!  Create an unpopulated REAL pointer matrix

!  Arguments

!  m - number of rows
!  n - number of columns

!  -----------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      INTEGER * 4 :: mxClassIDFromClassName
      mwSize :: mm, nn
      mwPointer :: mxCreateNumericArray

      mm = m ; nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        galmxCreateRealMatrix4 = mxCreateNumericArray( 2_mws_, (/ mm, nn /),   &
                            mxClassIDFromClassName('double'), 0_int4_ )
      CASE default
        galmxCreateRealMatrix4 = mxCreateNumericArray( 2_mws_, (/ mm, nn /),   &
                            mxClassIDFromClassName('single'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateRealMatrix4

!  -*-*-*-  g a l m x  C r e a t e  R e a l  M a t r i x  8  -*-*-*-*-

      FUNCTION galmxCreateRealMatrix8( m, n )
      mwPointer :: galmxCreateRealMatrix8
      INTEGER * 8 :: m, n

!  -----------------------------------------

!  Create an unpopulated REAL pointer matrix

!  Arguments

!  m - number of rows
!  n - number of columns

!  -----------------------------------------

      REAL ( KIND = wp ) :: ddig = 1.0_wp
      INTEGER * 4 :: mxClassIDFromClassName
      mwPointer :: mxCreateNumericArray
      mwSize :: mm, nn

      mm = m ; nn = n
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        galmxCreateRealMatrix8 = mxCreateNumericArray( 2_mws_, (/ mm, nn /),   &
                            mxClassIDFromClassName('double'), 0_int4_ )
      CASE default
        galmxCreateRealMatrix8 = mxCreateNumericArray( 2_mws_, (/ mm, nn /),   &
                            mxClassIDFromClassName('single'), 0_int4_ )
      END SELECT

      RETURN
      END FUNCTION galmxCreateRealMatrix8


!  -*-*-*-*-*- g a l m x  F i l l  I n t e g e r 4  C o m p o n e n t -*-*-*-*-

      SUBROUTINE galmxFillInteger4Component( struct, name, Y )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 4 :: Y

!  -----------------------------------------------

!  Create a named INTEGER component of a structure
!  and copy a value into the component

!  Arguments

!  struct - structure
!  name - name of component
!  Y - Integer Fortran scalar

!  -----------------------------------------------

      mwPointer :: pr, px
      mwPointer :: mxGetPr

      pr = galmxCreateInteger( )
      CALL mxSetField( struct, 1_mwi_, name, pr )
      px = mxGetPr( pr )
!     SELECT CASE ( di_ )
!     CASE ( int8_ )
!       CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
!     CASE ( int2_ )
!       CALL mxCopyInteger2ToPtr( Y, px, 1_mws_ )
!     CASE ( int1_ )
!       CALL mxCopyInteger1ToPtr( Y, px, 1_mws_ )
!     CASE default
        CALL mxCopyInteger4ToPtr( Y, px, 1_mws_ )
!     END SELECT

      RETURN
      END SUBROUTINE galmxFillInteger4Component

!  -*-*-*-*-*- g a l m x  F i l l  I n t e g e r 8  C o m p o n e n t -*-*-*-*-

      SUBROUTINE galmxFillInteger8Component( struct, name, Y )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER * 8 :: Y

!  -----------------------------------------------

!  Create a named INTEGER component of a structure
!  and copy a value into the component

!  Arguments

!  struct - structure
!  name - name of component
!  Y - Integer Fortran scalar

!  -----------------------------------------------

      mwPointer :: pr, px
      mwPointer :: mxGetPr

      pr = galmxCreateInteger( )
      CALL mxSetField( struct, 1_mwi_, name, pr )
      px = mxGetPr( pr )
 !    SELECT CASE ( di_ )
 !    CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
 !    CASE ( int2_ )
 !      CALL mxCopyInteger2ToPtr( Y, px, 1_mws_ )
 !    CASE ( int1_ )
 !      CALL mxCopyInteger1ToPtr( Y, px, 1_mws_ )
 !    CASE default
 !      CALL mxCopyInteger4ToPtr( Y, px, 1_mws_ )
 !    END SELECT

      RETURN
      END SUBROUTINE galmxFillInteger8Component


!  -*-*-*-*-*- g a l m x  F i l l  L o n g   C o m p o n e n t -*-*-*-*-*-

      SUBROUTINE galmxFillLongComponent( struct, name, Y )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      INTEGER ( KIND = long ) :: Y

!  -----------------------------------------------

!  Create a named LONG INTEGER component of a structure
!  and copy a value into the component

!  Arguments

!  struct - structure
!  name - name of component
!  Y - Long Integer Fortran scalar

!  -----------------------------------------------

      mwPointer :: pr, px
      mwPointer :: mxGetPr

      pr = galmxCreateLong( )
      CALL mxSetField( struct, 1_mwi_, name, pr )
      px = mxGetPr( pr )
      SELECT CASE ( long )
      CASE ( int8_ )
        CALL mxCopyInteger8ToPtr( Y, px, 1_mws_ )
      CASE ( int2_ )
        CALL mxCopyInteger2ToPtr( Y, px, 1_mws_ )
      CASE ( int1_ )
        CALL mxCopyInteger1ToPtr( Y, px, 1_mws_ )
      CASE default
        CALL mxCopyInteger4ToPtr( Y, px, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxFillLongComponent


!  -*-*-*-*-*-*- g a l m x  F i l l  R e a l  C o m p o n e n t -*-*-*-*-*-*-*-

      SUBROUTINE galmxFillRealComponent( struct, name, Y )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      REAL ( KIND = wp ) :: Y

!  --------------------------------------------

!  Create a named REAL component of a structure
!  and copy a value into the component

!  Arguments

!  struct - structure
!  name - name of component
!  Y   - Real Fortran scalar

!  --------------------------------------------

      mwPointer :: pr, px
      REAL ( KIND = wp ) :: ddig = 1.0_wp
      mwPointer :: mxGetPr

      pr = galmxCreateReal( )
      CALL mxSetField( struct, 1_mwi_, name, pr )
      px = mxGetPr( pr )
      SELECT CASE ( digits( ddig ) )
      CASE ( 53 )
        CALL mxCopyReal8ToPtr( Y, px, 1_mws_ )
      CASE default
        CALL mxCopyReal4ToPtr( Y, px, 1_mws_ )
      END SELECT

      RETURN
      END SUBROUTINE galmxFillRealComponent


!  -*-*-  M A T L A B _ f i l l  _ l o g i c a l _ c o m p o n e n t  -*-*-

      SUBROUTINE MATLAB_fill_logical_component( struct, name, Y )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      LOGICAL * 4 :: Y

!  -----------------------------------------------------

!  Create a named LOGICAL component of a structure
!  and copy a value into the component

!  Arguments

!  struct - structure
!  name - name of component
!  Y   - Logical Fortran scalar

! ** - NB - ** This is a bodge since Mex doesn't appear
!              to handle Fortran logicals ** - NB - **

!  ----------------------------------------------------

      mwPointer :: pr, px
      INTEGER * 4 :: ly
      mwPointer :: mxGetPr

      pr = galmxCreateInteger( )
      CALL mxSetField( struct, 1_mwi_, name, pr )
      px = mxGetPr( pr )
      IF ( Y ) THEN
        ly = 1
      ELSE
        ly = 0
      END IF
      CALL galmxCopyInteger4ToPtr( ly, px )

      RETURN
      END SUBROUTINE MATLAB_fill_logical_component

!  -*-*-*-  M A T L A B _ f i l l _ c h a r _ c o m p o n e n t  -*-*-*-

      SUBROUTINE MATLAB_fill_char_component( struct, name, value )
      mwPointer :: struct
      CHARACTER ( len = * ) :: name
      CHARACTER ( len = * ) :: value

!  -------------------------------------------------

!  Create a named CHARACTER component of a structure

!  Arguments

!  struct - structure
!  name - name of component
!  pr - pointer to the structure

!  -------------------------------------------------

      mwPointer :: mxCreateString

      CALL mxSetField( struct, 1_mwi_, name, mxCreateString( value ) )

      RETURN
      END SUBROUTINE MATLAB_fill_char_component


!-*-*-*-*-*-*- E N D  o f  G A L A H A D _ M A T L A B   M O D U L E -*-*-*-*-*-

    END MODULE GALAHAD_MATLAB
