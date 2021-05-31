! THIS VERSION: GALAHAD 2.3 - 15/04/2009 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ D E M O  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  April 15th 2009

   MODULE GALAHAD_USEDEMO_double

!  This is the driver program for running DEMO for a variety of computing 
!  systems. It opens and closes all the files, allocate arrays, reads and 
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_SYMBOLS
     USE GALAHAD_DEMO_double
     USE GALAHAD_SPECFILE_double 
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_DEMO

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ D E M O   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_DEMO( input )

!  Dummy argument - input is the file unit for data output by SifDec

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

!  Problem characteristics

     INTEGER :: n

!  Specfile characteristics

     LOGICAL :: is_specfile
     INTEGER, PARAMETER :: input_specfile = 34
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNDEMO.SPC'

!  demo derived types

     TYPE ( DEMO_control_type ) :: control
     TYPE ( DEMO_inform_type ) :: inform
     TYPE ( DEMO_data_type ) :: data

!  Set up data for problem

     CALL DEMO_initialize( data, control )

!  Read specfile data

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) CALL DEMO_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Solve the problem

     CALL DEMO_main( n, control, inform, data )

!  Remove temporary storage

     CALL DEMO_terminate( data, control, inform )

!  End of subroutine USE_DEMO

     END SUBROUTINE USE_DEMO

!  End of module USEDEMO_double

   END MODULE GALAHAD_USEDEMO_double
