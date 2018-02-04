! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*     check  M O D U L E      *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Daniel Robinson

!  For full documentation, see http://galahad.rl.ac.uk/galahad-www/specs.html

 MODULE GALAHAD_CHECK_double

 !-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
 !  This module checks the derivatives of problem functions.                  !
 !-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

 USE GALAHAD_NLPT_double
 USE GALAHAD_SMT_double
 USE GALAHAD_MOP_double
 USE GALAHAD_SPACE_double
 USE GALAHAD_SPECFILE_double
 USE GALAHAD_SYMBOLS

 IMPLICIT NONE

 PRIVATE
 PUBLIC :: CHECK_initialize, CHECK_verify, CHECK_read_specfile, CHECK_terminate

 ! Define parameters.

 INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
 REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
 REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
 REAL ( KIND = wp ), PARAMETER :: tenm6 = 0.000001_wp
 REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
 REAL ( KIND = wp ), PARAMETER :: sqrteps = epsmch ** 0.5_wp

 ! ==================================
 ! The CHECK_control_type derived type
 ! ==================================

 TYPE, PUBLIC :: CHECK_control_type
    INTEGER :: error, out, print_level, verify_level
    INTEGER :: f_availability, c_availability
    INTEGER :: g_availability, J_availability, H_availability
    LOGICAL :: checkG, checkJ, checkH, deallocate_error_fatal
 END TYPE CHECK_control_type
 
 ! ==================================
 ! The CHECK_inform_type derived type
 ! ==================================
 
 TYPE, PUBLIC :: CHECK_inform_type
    INTEGER :: status, alloc_status, numG_wrong, numJ_wrong, numH_wrong
    CHARACTER( LEN = 80 ) :: bad_alloc
    LOGICAL :: derivative_ok
 END TYPE CHECK_inform_type
 
 ! ============================================
 ! The CHECK_reverse_communication derived type
 ! ===========================================
 
 TYPE, PUBLIC :: CHECK_reverse_communication_type
    REAL( KIND = wp ) :: f
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x, y, c, g, u, v
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Jval, Hval
 END TYPE CHECK_reverse_communication_type
 
 ! ==================================
 ! The CHECK_data_type derived type
 ! ==================================
 
 TYPE, PUBLIC :: CHECK_data_type
    INTEGER :: i, j, branch
    REAL( KIND = wp ) :: f_plus, alpha, normx, scale, tol, temp, fd_len
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_plus, G_plus
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Jval_plus, gradL_plus
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: gradL, X1, Jv, Hv
    REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: s, s_back, ej
    LOGICAL :: f_filled, c_filled, g_filled, J_filled, Jv_filled
    TYPE( CHECK_control_type ) :: control
    TYPE( CHECK_reverse_communication_type ) :: RC
 END TYPE CHECK_data_type
 
 ! Interfaces
 
 INTERFACE IAMAX
    FUNCTION IDAMAX( N, X, INCX )
      INTEGER :: IDAMAX
      INTEGER, INTENT( IN ) :: N, INCX
      DOUBLE PRECISION, INTENT( IN ), DIMENSION( INCX*(N-1)+1  ) :: X
    END FUNCTION IDAMAX
 END INTERFACE
 
 INTERFACE NORM2
    FUNCTION DNRM2( N, X, INCX )
      DOUBLE PRECISION :: DNRM2
      INTEGER, INTENT( IN ) :: N, INCX
      DOUBLE PRECISION, INTENT( IN ), DIMENSION( INCX*(N-1)+1  ) :: X
    END FUNCTION DNRM2
 END INTERFACE
 
 CONTAINS
  
 !******************************************************************************
 !           G A L A H A D -  CHECK_initialize  S U B R O U T I N E            !
 !******************************************************************************
  
 SUBROUTINE CHECK_initialize( control )
 !------------------------------------------------------------------------------
 ! Provide default control values for type CHECK_control_type.
 !------------------------------------------------------------------------------
 implicit none
 !------------------------------------------------------------------------------
 ! Dummy arguments.
 !------------------------------------------------------------------------------
 TYPE ( CHECK_control_type ), INTENT( OUT ) :: control
 !------------------------------------------------------------------------------

 ! Error and ordinary output unit numbers.

 control%error = 6
 control%out   = 6

 ! Level of output required.
 !    <= 0  gives no output;
 !       1  gives summar of derivative checking;
 !       2  add verification details, control parameters, basic matrix info;
 !       3  add full matrix info; and
 !    >= 4  add private data used during verification process (debugging).

 control%print_level = 1

 ! Logicals that determine which derivatives are verified:
 !     checkG  gradients of the objective;
 !     checkJ  Jacobian of constraint function; and
 !     checkH  Hessian of the Lagrangian.

 control%checkG = .TRUE.
 control%checkJ = .TRUE.
 control%checkH = .TRUE.
 
 ! Level of verification required.

 control%verify_level = 2

 ! Method for supplying values of problem functions.

 control%f_availability = 1
 control%c_availability = 1
 control%g_availability = 1
 control%J_availability = 1
 control%H_availability = 1

 ! If deallocate_error_fatal is true, any array/pointer deallocation error
 ! will terminate execution. Otherwise, computation will continue.

 control%deallocate_error_fatal = .FALSE.
 
 RETURN

 END SUBROUTINE CHECK_initialize

 !******************************************************************************
 !             G A L A H A D  -  CHECK_verify  S U B R O U T I N E             !
 !******************************************************************************

 SUBROUTINE CHECK_verify( nlp, data, control, inform, userdata, eval_F, eval_C,&
                          eval_G, eval_J, eval_HL, eval_Jv, eval_Hv )
 !------------------------------------------------------------------------------
 ! Purpose: Check the gradient of the object function, the Jacobian of the
 !          constraint function, and the Hessian of the Lagrangian function.
 !
 ! Arguments:
 !
 !   nlp      derived type NLPT_problem_type.  Holds the problem data.
 !
 !   data     derived type CHECK_data_type.
 !
 !   control  scalar variable of derived type CHECK_control_type, whose
 !            components are given by:
 !
 !            out          is a scalar variable of type integer, that holds
 !                         the stream number for informational messages.
 !                         Default value is out = 6.
 !
 !            error        is a scalar variable of type integer, that holds
 !                         the stream number for error messages. Default
 !                         Default value is error = 6.
 !
 !            print_level  is scalar variable of type integer.  It controls
 !                         the amount of information that is printed to unit
 !                         number given by out.  Values are:
 !
 !                         print_level <= 0    Nothing;
 !                         print_level  = 1    Basic summary of the results;
 !                         print_level  = 2    More detailed output of above,
 !                                             & additionally print the control
 !                                             parameters and data of problem;
 !                         print_level  = 3    Above, plus full details of the 
 !                                             storage components for nlp%J and
 !                                             nlp%H; and
 !                         print_level >= 4    Full details (debugging).
 !
 !            checkG, checkJ, checkH
 !
 !                         scalar variables of type logical.  The user
 !                         should set checkG = .TRUE. if verification
 !                         of the gradient of the objective is desired.
 !                         Otherwise, set checkG = .FALSE.  Similarly,
 !                         checkJ and checkH control verification of the
 !                         Jacobian of the constraints and the Hessian
 !                         of the Lagrangian.  Default values are
 !                         checkG = checkJ = checkH = .TRUE.
 !
 !            deallocate_error_fatal
 !
 !                         is a scalar variable of type default LOGICAL, that
 !                         must be set .TRUE. if the user wishes to terminate
 !                         execution if a deallocation fails, and .FALSE if an
 !                         attempt to continue will be made. The default value is 
 !                         deallocate_error_fatal = .FALSE.
 !
 !            verify_level is a scalar variable of type integer that controls
 !                         the detail of verification performed:
 !
 !                         <= 0   no verification is performed;
 !                            1   cheap verification is performed; and
 !                         >= 2   full verification is performed.
 !
 !                         Default value is verify_level = 2.
 !
 !            f_availability
 !
 !                         is a scalar variable of type integer that gives the
 !                         avalability of the objection function:
 !
 !                         1  f via subroutine eval_F
 !                         2  f via reverse communication
 !
 !            c_availability
 !
 !                         is a scalar variable of type integer that gives the
 !                         avalability of the constraint function:
 !
 !                         1  c via subroutine eval_C
 !                         2  c via reverse communication  
 !
 !            g_availability
 !
 !                         is a scalar variable of type integer that gives the
 !                         avialability of the gradient of the objective function:
 !
 !                         1  g via subroutine eval_G
 !                         2  g via reverse communication 
 !
 !            J_availability
 !
 !                         is a scalar variable of type integer that gives the
 !                         avialability of the Jacobian of the constraint function:
 !
 !                         1  J via subroutine eval_J
 !                         2  J via reverse communication
 !                         3  Jv via subroutine eval_Jv
 !                         4  Jv via reverse communication
 !
 !            H_availability
 !
 !                         is a scalar variable of type integer that gives the
 !                         availability of the Hessian of the Lagrangian function:
 !
 !                         1  H via subroutine eval_HL
 !                         2  H via reverse communication
 !                         3  Hv via subroutine eval_Hv
 !                         4  Hv via reverse communication                         
 !
 !   inform   scalar variable of derived type CHECK_inform_type, whose
 !            components are given by:
 !
 !            status       scalar variable of type integer.  On entry,
 !                         inform%status should be set to 1.  Upon
 !                         successful exit, inform%staus == 0.  If
 !                         control%status < 0, then an error has occured.
 !                         Any other value signifies that the user must
 !                         supply some information via reverse
 !                         communication.  The values and their meaning
 !                         are given by:
 !
 !                         status = -58  a user supplied function returned
 !                                       status /= 0, implying that a
 !                                       function could not be evaluated
 !                                       at the required point.
 !                         status = -57  some component of nlp%X_l or
 !                                       nlp%X_u is inappropriate. 
 !                         status = -56  based on the control parameters
 !                                       f_availability, c_availability,
 !                                       g_availability, J_availability,
 !                                       and H_availability, an optional
 !                                       function subroutine is missing.
 !                         status = -55  invalid value for one of
 !                                       f_availability, c_availability,
 !                                       g_availability, J_availability,
 !                                       or H_availability.
 !                         status = -51  user has called CHECK_verify with
 !                                       inform%status = 0; this should not
 !                                       ever happen since the user should
 !                                       only ever have status = 1 for the
 !                                       initial call to CHECK_verify,
 !                                       status < 0 if they can not provide
 !                                       the required calculation, and left 
 !                                       unchanged when using reverse
 !                                       communication for which values
 !                                       between 2 and 9 are possible.
 !                         status - -50  user has called CHECK_verify with
 !                                       inform%status < 0, implying that some
 !                                       requested function evaluation was not
 !                                       possible.
 !                         status = -3   One of the following has occured:
 !                                       nlp%m < 0, nlp%n <= 0, nlp%J%type
 !                                       value not recognized, or nlp%H%type
 !                                       value not recognized. 
 !                         status = -2   deallocation error occurred.
 !                         status = -1   allocation error occurred.
 !                         status =  0   successful return.
 !                         status =  1   first entry to subroutine CHECK_verify.
 !                         status =  2   user must evaluate f at data%RC%x and
 !                                       place the value in data%RC%f, and
 !                                       recall CHECK_verify.
 !                         status =  3   user must evaluate c at data%RC%x and
 !                                       place the value in data%RC%c, and
 !                                       recall CHECK_verify.
 !                         status =  4   user must evaluate the gradient of f at
 !                                       data%RC%x and place the value in
 !                                       data%RC%g, reset status = 0, and
 !                                       recall CHECK_verify.
 !                         status =  5   user must evaluate the Jacobian of c at
 !                                       data%RC%x and place the value in
 !                                       data%RC%Jval, and recall CHECK_verify.
 !                         status =  6   user must place the value of data%RC%u
 !                                       with the value of data%RC%u plus the
 !                                       product of the Jabobian of c evaluated
 !                                       at data%RC%x with the vector data%RC%v,
 !                                       and then recall CHECK_verify:
 !                                               U <-- U + J(x) * V
 !                         status =  7   user must place the value of data%RC%u
 !                                       with the value of data%RC%u plus the
 !                                       product of the Jabobian (transposed) of
 !                                       c evaluated at data%RC%x with the vector
 !                                       data%RC%v, reset status = 0, and then
 !                                       recall CHECK_verify:
 !                                               U <-- U + J(x)^T * V
 !                         status =  8   user must evaluate the Hessian of the
 !                                       Lagrangian at ( data%RC%x, data%RC%y )
 !                                       and place the values in data%RC%Hval,
 !                                       reset status = 0, and recall CHECK_verify.
 !                         status =  9   user must place the value of data%RC%u
 !                                       with the value of data%RC%u plus the
 !                                       product of the Hessian of the Lagrangian
 !                                       evaluated at ( data%RC%x, data%RC%y )
 !                                       with the vector data%RC%v, and
 !                                       recall CHECK_verify.
 !                                               U <-- U + H(x,y) * V
 !
 !            derivate_ok  upon successful completion (inform%status = 0), the
 !                         value of dervative_ok has the following meaning:
 !                            .TRUE.   derivatives were all correct.
 !                            .FALSE.  at least one derivative appears wrong.
 !
 !            numG_wrong   scalar variable of type integer that holds the current
 !                         number of incorrect gradient entries.
 !
 !            numJ_wrong   scalar variable of type integer that holds the current
 !                         number of incorrect Jacobian entries.
 !
 !            numH_wrong   scalar variable of type integer that holds the current
 !                         number of incorrect Hessian entries.
 !
 !            alloc_status scalar variable of type integer that gives the
 !                         result of the last allocation/deallocation.
 !
 !            bad_alloc    is a scalar variable of type default CHARACTER
 !                         and length 80, that gives the name of the last
 !                         array for which an array allocation/de-allocation
 !                         was unsuccessful.
 !
 !   eval_F   (optional) subroutine that supplies the value of the objective
 !            function.  The subroutine structure of eval_F is given by
 !            eval_F( status, X, userdata, F).
 !
 !   eval_C   (optional) subroutine that supplies the value of the constraints.
 !            The subroutine structure of eval_C is given by
 !            eval_C( status, X, userdata, C).
 !
 !   eval_G   (optional) subroutine that supplies the gradient of the objective
 !            function.  The subroutine structure of eval_G is given by
 !            eval_G( status, X, userdata, G).
 !
 !   eval_J   (optional) subroutine that supplies the Jacobian of the
 !            constraints.  The subroutine structure is given by
 !            eval_J( status, X, userdata, Jval).
 !
 !   eval_Jv  (optional) subroutine that supplies the product of the Jacobian
 !            of the constraint vector with a given vector V.
 !            That is, given a vector V the subroutine returns the product
 !            of the Jacobian matrix with V.  The form is given
 !            by eval_Jv(status, userdata, transpose,U,V,X) where userdata is
 !            of type NLPT_userdata_type.  If transpose is set .true., then
 !            it returns of the product of the transpose of the Jacobian with
 !            the vector V.
 !                 transpose = .false.      U <-- U + J(X) V
 !                 transpose = .true.       U <-- U + J(x)^T V
 !
 !   eval_HL  (optional) subroutine that supplies the Hessian of the
 !            Lagrangian function.  The subroutine structure is given by
 !            eval_H(status, X, Y, userdata, Hval,no_f).
 !
 !   eval_Hv  (optional) subroutine that supplies the product of the Hessian of
 !            the Lagrangian with a given vector V.  That is, give a vector V,
 !            the subroutine returns the product of the Hessian matrix with V.
 !            The structure of the subroutine is given by
 !            eval_Hv( status, userdata, U, V, X, Y )  where userdata is
 !            of type NLPT_userdata_type: U <-- U + H(X,Y) V.
 !
 !   userdata (optional) is a scalar variable of type NLPT_userdata_type.
 !            Intended for the sole use of the "user" if needed.
 !
 !-------------------------------------------------------------------------------
 ! Dummy arguments
 !-------------------------------------------------------------------------------
 TYPE( NLPT_problem_type ), INTENT( INOUT ) :: nlp
 TYPE( CHECK_control_type ), INTENT( INOUT ) :: control
 TYPE( CHECK_inform_type ), INTENT( OUT ) :: inform
 TYPE( CHECK_data_type ), INTENT( INOUT ) :: data
 TYPE( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
 OPTIONAL eval_F, eval_C, eval_G, eval_J, eval_Jv, eval_HL, eval_Hv
 !-------------------------------------------------------------------------------
 ! Interfaces
 !-------------------------------------------------------------------------------
 INTERFACE
    SUBROUTINE eval_F(status, X, userdata, F)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
      REAL ( kind = wp ), INTENT( OUT ) :: F
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_F
    SUBROUTINE eval_C(status, X, userdata, C)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
      REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) :: C
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_C
    SUBROUTINE eval_G(status, X, userdata, G)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_G
   SUBROUTINE eval_J(status, X, userdata, Jval)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jval
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_J
    SUBROUTINE eval_Jv(status, X, userdata, transpose, U, V)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      LOGICAL, INTENT( IN ) :: transpose
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_Jv
    SUBROUTINE eval_HL(status, X, Y, userdata, Hval,no_f)
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
      REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: Y
      REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
      LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_HL
    SUBROUTINE eval_Hv(status, X, Y, userdata, U, V )
      USE GALAHAD_NLPT_double
      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
      REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
      TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
    END SUBROUTINE eval_Hv
 END INTERFACE
 !------------------------------------------------------------------------------
 ! Local variables
 !------------------------------------------------------------------------------
 REAL ( KIND = wp ) :: gts, diff, err, temp, Jij, Hij
 LOGICAL :: checkG, checkJ, checkH, deallocate_error_fatal
 INTEGER :: f_availability, c_availability
 INTEGER :: g_availability, J_availability, H_availability
 INTEGER :: out, error, m, n, i, j, nFeas, print_level, verify_level
 CHARACTER ( LEN = 3 ) :: str
 !-------------------------------------------------------------------------------

 ! Catch some errors in value of status.

 if ( inform%status < 0  ) then ; inform%status = -50 ; go to 999 ; end if
 if ( inform%status == 0 ) then ; inform%status = -51 ; go to 999 ; end if
 
 ! Do initializations for first call to CHECK_verify.

 if ( inform%status == 1 ) then

    ! Set branch to initial entry.

    data%branch = 1

    ! Copy control into data.

    data%control = control

    ! Print header.
    
    if ( control%print_level >= 1 .and. control%out >= 1 ) then
       write( control%out, 4000 )
    end if

    ! Check to make sure dimensions make sense.

    if ( nlp%n <= 0 .or. nlp%m < 0 ) then
       inform%status = GALAHAD_error_restrictions ; go to 999
    end if

    ! Make sure dimensions of J and H are defined.

    nlp%J%m = nlp%m ;  nlp%J%n = nlp%n
    nlp%H%m = nlp%n ;  nlp%H%n = nlp%n

    ! Make sure bounds on variables make sense.

    if ( .not. allocated( nlp%X_l ) .or. .not. allocated( nlp%X_u ) ) then
       inform%status = -57 ; go to 999
    end if

    if ( size( nlp%X_l ) < nlp%n .or. size( nlp%X_u ) < nlp%n ) then
       inform%status = -57 ; go to 999
    end if   

    do i = 1, nlp%n
       if ( nlp%X_l(i) > nlp%X_u(i) ) then
          inform%status = -57 ; go to 999
       end if
    end do

    ! Check some control parameters.

    if ( data%control%verify_level <= 0 ) then
       inform%status = 0 ;  go to 999
    else
       data%control%verify_level = min( 2,data%control%verify_level )
    end if

    if ( nlp%m == 0 .and. control%checkJ ) then
       if ( control%print_level >= 1 .and. control%out >= 1 ) then
          write( control%out, 3001 )
       end if
       data%control%checkJ = .FALSE.
    end if

    if ( control%f_availability < 1 .or. control%f_availability > 2 .or. &
         control%c_availability < 1 .or. control%c_availability > 2 .or. &
         control%g_availability < 1 .or. control%g_availability > 2 .or. &
         control%J_availability < 1 .or. control%J_availability > 4 .or. &
         control%H_availability < 1 .or. control%H_availability > 4 ) then
       inform%status = -55 ; go to 999
    end if

    if ( control%f_availability == 1 .and. .not. present(eval_f) ) then
       if ( data%control%checkG ) then
          inform%status = -56 ; go to 999
       end if
    end if

    if ( control%c_availability == 1 .and. .not. present(eval_c) ) then
       if ( data%control%checkJ ) then
          inform%status = -56 ; go to 999
       end if
    end if

    if ( control%g_availability == 1 .and. .not. present(eval_g) ) then
       if ( data%control%checkG .or. data%control%checkH ) then
          inform%status = -56 ; go to 999
       end if
    end if

    if ( control%J_availability == 1 .and. .not. present(eval_J) ) then
       if ( data%control%checkJ .or. data%control%checkH ) then
          inform%status = -56 ; go to 999
       end if
    end if

    if ( control%J_availability == 3 .and. .not. present(eval_Jv) ) then
       if ( data%control%checkJ .or. data%control%checkH ) then
          inform%status = -56 ; go to 999
       end if
    end if    

    if ( control%H_availability == 1 .and. .not. present(eval_HL) ) then
       if ( data%control%checkH ) then
          inform%status = -56 ; go to 999
       end if
    end if

    if ( control%H_availability == 3 .and. .not. present(eval_Hv) ) then
       if ( data%control%checkH ) then
          inform%status = -56 ; go to 999
       end if
    end if  

 end if

 ! For convenience.

 m = nlp%m ;  n = nlp%n

 error = data%control%error
 
 out = data%control%out ;  print_level = data%control%print_level
 if ( out <= 0 ) then
    print_level = 0
 end if
 
 checkG       = data%control%checkG
 checkJ       = data%control%checkJ
 checkH       = data%control%checkH
 verify_level = data%control%verify_level

 f_availability = data%control%f_availability
 c_availability = data%control%c_availability
 g_availability = data%control%g_availability
 J_availability = data%control%J_availability
 H_availability  = data%control%H_availability

 deallocate_error_fatal = data%control%deallocate_error_fatal

 ! Allocate needed vectors and form initial data.

 if ( data%branch == 1 ) then

    ! Set some scalars in inform.

    inform%bad_alloc     = ''
    inform%alloc_status  = 0
    inform%numG_wrong    = 0
    inform%numJ_wrong    = 0
    inform%numH_wrong    = 0
    inform%derivative_ok = .true.

    ! Make sure nlp%X is feasible with respect to bounds.

    nlp%X = MAX( nlp%X, nlp%X_l )
    nlp%X = MIN( nlp%X, nlp%X_u )

    ! Allocate some things.

    i = max(m,n)

    CALL SPACE_resize_array( i, data%RC%u, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( i, data%RC%v, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%RC%x, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( m, data%RC%y, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( m, data%RC%c, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%RC%g, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%s, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%s_back, inform%status, inform%alloc_status,&
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( m, data%C_plus, inform%status, inform%alloc_status,&
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%G_plus, inform%status, inform%alloc_status,&
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%gradL, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array(n,data%gradL_plus,inform%status,inform%alloc_status,&
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%X1, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( m, data%Jv, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%Hv, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array( n, data%ej, inform%status, inform%alloc_status, &
                             bad_alloc=inform%bad_alloc  )
    IF ( inform%status /= 0 ) GO TO 990

    i = nlp%J%ne

    CALL SPACE_resize_array( i, data%RC%Jval,inform%status,inform%alloc_status,&
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    CALL SPACE_resize_array(i,data%Jval_plus,inform%status,inform%alloc_status,&
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    i = nlp%H%ne

    CALL SPACE_resize_array( i, data%RC%Hval,inform%status,inform%alloc_status,&
                             bad_alloc=inform%bad_alloc )
    IF ( inform%status /= 0 ) GO TO 990

    ! Define step length alpha.

    data%tol    = tenm6
    data%fd_len = sqrteps
    data%normx  = NORM2( n, nlp%X, 1 )
    data%alpha  = data%fd_len * ( one + data%normx )

    ! Defind a step.

    data%scale = one/n

    do i = 1, n
       data%s(i)      = data%scale
       data%s_back(i) = data%scale  ! s_back = s
       data%scale     = -data%scale
    end do

    ! Ensure step is feasible with respect to bounds.  If this results in a
    ! zero step, then resort back to original s and use it.

    call get_feas_step( n, nlp%X_l, nlp%X_u, nlp%X, data%alpha, data%s, nFeas )

    if ( nFeas == 0 ) then
       data%s = data%s_back
    end if

    data%X1 = nlp%X + data%alpha * data%s

 end if

 ! -----------------------------------------------------------------------------
 ! -                   BEGIN : Derivative Checking                             -
 ! -----------------------------------------------------------------------------

 SELECT CASE ( verify_level )

 ! -----------
 ! Cheap check
 ! -----------

 CASE ( 1 )

    ! Go to part of the code based on value in data%branch

    SELECT CASE ( data%branch )
    CASE ( 1 )   ! first time in
       go to 100
    CASE ( 2 )   ! f(x)    - cheap - in Gcheck
       GO TO 200
    CASE ( 3  )  ! f(x1)   - cheap - in Gcheck
       GO TO 210
    CASE ( 4 )   ! g(x)    - cheap - in Gcheck
       GO TO 220
    CASE ( 5 )   ! c(x)    - cheap - in Jcheck
       GO TO 230
    CASE ( 6 )   ! c(x1)   - cheap - in Jcheck
       GO TO 240
    CASE ( 7  )  ! J(x)    - cheap - in Jcheck
       GO TO 250
    CASE ( 8 )   ! J*s     - cheap - in Jcheck
       GO TO 260
    CASE ( 9  )  ! g(x)    - cheap - Hcheck when checkG = .false.
       GO TO 270
    CASE ( 10  ) ! g(x1)   - cheap - Hcheck
       GO TO 280
    CASE ( 11  ) ! J(x)    - cheap - in Hcheck when checkJ = .false.
       GO TO 290
    CASE ( 12  ) ! J(x1)   - cheap - in Hcheck
       GO TO 300
    CASE ( 13  ) ! JtY     - cheap - in Hcheck
       GO TO 310
    CASE ( 14  ) ! J(x1)^Y - cheap - in Hcheck
       GO TO 320
   CASE ( 15 )   ! H(x,y)  - cheap - in Hcheck
       GO TO 330
    CASE ( 16  ) ! H*s     - cheap - in Hcheck
       GO TO 340
    END SELECT

100 continue

    ! BEGIN: Cheap check of G
    ! -----------------------

    if ( checkG ) then
       if ( print_level >= 2 ) then
          write( out, 5000 ) ! title
          write( out, 1010 ) ! column headers
       end if
    else
       go to 110
    end if

    ! evaluate f(x)

    if ( f_availability == 1 ) then
       call eval_F( inform%status, nlp%X, userdata, nlp%f )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%f_filled = .true.
    else
       data%f_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 2
       inform%status = 2
       return
    end if

    ! return from reverse communication for f(x)

200 continue

    if ( .not. data%f_filled ) nlp%f = data%RC%f

    ! evaluate f(x1)

    if ( f_availability == 1 ) then
       call eval_F( inform%status, data%X1, userdata, data%f_plus )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%f_filled = .true.
    else
       data%f_filled = .false.
       data%RC%x     = data%X1
       data%branch   = 3
       inform%status = 2
       return
    end if

    ! return from reverse communication for f(x1)

210 continue

    if ( .not. data%f_filled ) data%f_plus = data%RC%f

    ! evaluate g(x)

    if ( g_availability == 1 ) then
       call eval_G( inform%status, nlp%X, userdata, nlp%G )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%g_filled = .true.
    else
       data%g_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 4
       inform%status = 4
       return
    end if

 ! return from reverse communication for g(x)

220 continue

 if ( .not. data%g_filled ) nlp%G = data%RC%g

 ! Compare g^Ts with the finite differences.

 gts  = DOT_PRODUCT( nlp%G, data%s )
 diff = ( data%f_plus - nlp%f ) / data%alpha
 err  = ABS( diff - gts ) / ( one + ABS(diff) )

 if ( err <= data%tol ) then
    if ( print_level >= 2 ) then
       str = ' OK' ; write( out, 1011 ) str, diff, gts, err
    end if
 else
    inform%numG_wrong = inform%numG_wrong + 1
    if ( print_level >= 2 ) then
       str = 'BAD' ; write( out, 1011 ) str, diff, gts, err
    end if
 end if

110 continue

 ! BEGIN: Cheap check of J
 ! -----------------------

 if ( checkJ ) then
    if ( print_level >= 2 ) then
       write( out, 5001 ) ! title
       write( out, 1010 ) ! column headers
    end if
 else
    go to 120
 end if

 ! evaluate c(x)

 if ( c_availability == 1 ) then
    call eval_C( inform%status, nlp%X, userdata, nlp%C )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%c_filled = .true.
 else
    data%c_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 5
    inform%status = 3
    return
 end if

 ! return from reverse communication for c(x)

230 continue

 if ( .not. data%c_filled ) nlp%C = data%RC%c

 ! evaluate c(x1)

 if ( c_availability == 1 ) then
    call eval_C( inform%status, data%X1, userdata, data%c_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%c_filled = .true.
 else
    data%c_filled = .false.
    data%RC%x     = data%X1
    data%branch   = 6
    inform%status = 3
    return
 end if

 ! return from reverse communication for c(x1)

240 continue

 if ( .not. data%c_filled ) data%c_plus = data%RC%c

 ! Evaluate J(x) if it is available explicitly.

 if ( J_availability == 1 ) then
    call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%J_filled = .true.
 elseif ( J_availability == 2 ) then
    data%J_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 7
    inform%status = 5
    return
 end if

 ! return from reverse communication for J(x)

250 continue

 if ( J_availability == 2 ) nlp%J%val = data%RC%Jval

 ! Compute J*s

 if ( J_availability <= 2 ) then
    data%Jv = zero
    call mop_Ax( one, nlp%J, data%s, one, data%Jv )
    data%Jv_filled = .true.
 else
    if ( J_availability == 3 ) then
       data%Jv = zero
       call eval_Jv( inform%status, nlp%X, userdata, .false., data%Jv, data%s )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%Jv_filled = .true.
    else
       data%Jv_filled = .false.
       data%RC%u(:m)  = zero
       data%RC%v(:n)  = data%s 
       data%RC%x      = nlp%X
       data%branch    = 8
       inform%status  = 6
       return
    end if
 end if

 ! return from reverse communication for Js

260 continue

 if ( .not. data%Jv_filled ) data%Jv = data%RC%u(:m)

 ! Check J*s with the finite differences.

 do i = 1, m
    diff = ( data%C_plus(i) - nlp%C(i) ) / data%alpha
    err  = ABS( diff  - data%Jv(i) ) / ( one + ABS( diff ) )
    if ( err <= data%tol ) then
       if ( print_level >= 2 ) then
          str = ' OK' ; write( out, 1014 ) i, str, diff, data%Jv(i), err
       end if
    else
       inform%numJ_wrong = inform%numJ_wrong + 1
       if ( print_level >= 2 ) then
          str = 'BAD' ; write( out, 1014 ) i, str, diff, data%Jv(i), err
       end if
    end if
 end do

120 continue

 ! BEGIN: Cheap check of H
 ! -----------------------

 if ( checkH ) then
    if ( print_level >= 2 ) then
       write( out, 5002 ) ! title
       write( out, 1010 ) ! column headers
    end if
 else
    go to 900
 end if

 ! evaluate g(x)

 if ( .not. checkG ) then
    if ( g_availability == 1 ) then
       call eval_G( inform%status, nlp%X, userdata, nlp%G )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%g_filled = .true.
    else
       data%g_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 9
       inform%status = 4
       return
    end if
 else
    data%g_filled = .true.
 end if

 ! return from reverse communication for g(x) when checkG = .false.

270 continue

 if ( .not. data%g_filled ) nlp%G = data%RC%g

 ! evaluate g(x1)

 if ( g_availability == 1 ) then
    call eval_G( inform%status, data%X1, userdata, data%G_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%g_filled = .true.
 else
    data%g_filled = .false.
    data%RC%x     = data%X1
    data%branch   = 10
    inform%status = 4
    return
 end if

 ! return from reverse communication for g(x1).

280 continue

 if ( .not. data%g_filled ) data%G_plus = data%RC%g

 ! evaluate J(x) if it is available explicitely.

 if ( J_availability <= 2 ) then
    if ( .not. checkJ ) then
       if ( J_availability == 1 ) then
          call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
          if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
          data%J_filled = .true.
       else
          data%J_filled = .false.
          data%RC%x     = nlp%X
          data%branch   = 11
          inform%status = 5
          return
       end if
    else
       data%J_filled = .true.
    end if
 end if

 ! return from reverse communication for J(x) when checkJ = .false.

290 continue

 if ( J_availability <= 2 .and. .not. data%J_filled ) nlp%J%val = data%RC%Jval

 ! evaluate J(x1)

 if ( J_availability == 1 ) then
    call eval_J( inform%status, data%X1, userdata, data%Jval_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 elseif ( J_availability == 2 ) then
    data%RC%x     = data%X1
    data%branch   = 12
    inform%status = 5
    return
 end if

 ! return from reverse communication for J(x1).

300 continue

 if ( J_availability == 2 ) data%Jval_plus = data%RC%Jval

 ! compute gradL = g(x) - J(x)^T y

 data%gradL = nlp%G

 if ( m > 0 ) then
    if ( J_availability <= 2 ) then
       call mop_Ax( -one, nlp%J, nlp%Y, one, data%gradL, transpose=.true. )
       data%Jv_filled = .true.
    else
       if ( J_availability == 3 ) then
          call eval_Jv( inform%status, nlp%X, userdata, .true., data%gradL, -nlp%Y )
          if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
          data%Jv_filled = .true.
       else
          data%RC%u(:n)  = data%gradL
          data%RC%v(:m)  = -nlp%Y
          data%RC%x      = nlp%X
          data%branch    = 13
          inform%status  = 7
          data%Jv_filled = .false.
          return
       end if
    end if
 end if

 ! return from reverse communication for gradL = g(x) - J(x)^T y

310 continue

 if ( m > 0 .and. .not. data%Jv_filled ) data%gradL = data%RC%u(:n)

 ! compute gradL_plus = g_plus - J_plus^T y

 data%gradL_plus = data%G_plus

 if ( m > 0 ) then
    if ( J_availability <= 2 ) then
       nlp%J%val = data%Jval_plus
       call mop_Ax( -one, nlp%J, nlp%Y, one, data%gradL_plus, transpose=.true. )
       data%Jv_filled = .true.
    else
       if ( J_availability == 3 ) then
          call eval_Jv( inform%status, data%X1, userdata, .true.,              &
                        data%gradL_plus, -nlp%Y )
          if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
          data%Jv_filled = .true.
       else
          data%RC%u(:n)  = data%gradL_plus
          data%RC%v(:m)  = -nlp%Y
          data%RC%x      = data%X1
          data%branch    = 14
          inform%status  = 7
          data%Jv_filled = .false.
          return
       end if
    end if
 end if

 ! return from reverse communication for gradL_plus = g_plus - J_plus^T y

320 continue

 if ( m > 0 .and. .not. data%Jv_filled ) data%gradL_plus = data%RC%u(:n)

 ! Evaluate H(x,y) if it is explicitely available.

 if ( H_availability == 1 ) then
    call eval_HL( inform%status, nlp%X, nlp%Y, userdata, nlp%H%val)
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 elseif ( H_availability == 2 ) then
    data%RC%x     = nlp%X
    data%RC%y     = nlp%Y
    inform%status = 8
    data%branch   = 15
    return
 end if

 ! return from reverse communication for H(x,y)

330 continue

 if ( H_availability == 2 ) nlp%H%val = data%RC%Hval

 ! compute H*s

 if ( H_availability <= 2 ) then
    data%Hv = zero
    call mop_Ax( one, nlp%H, data%s, one, data%Hv, symmetric=.true. )
 elseif ( H_availability == 3 ) then
    data%Hv = zero
    call eval_Hv(inform%status, nlp%X, nlp%Y, userdata, data%Hv, data%s )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 else
    data%RC%u(:n) = zero
    data%RC%v(:n) = data%s
    data%RC%x     = nlp%X
    data%RC%y     = nlp%Y
    inform%status = 9
    data%branch   = 16
    return
 end if

 ! return from reverse communicaton for H*s

340 continue

 if ( H_availability == 4 ) data%Hv = data%RC%u(:n)

 ! Compare H*s to the finite differences.

 do i = 1, n
    diff = ( data%gradL_plus(i) - data%gradL(i) ) / data%alpha
    err  = ABS( diff - data%Hv(i) ) / ( one + ABS( diff ) )
    if ( err <= data%tol ) then
       if ( print_level >= 2 ) then
          str = ' OK' ; write( out, 1015 ) i, str, diff, data%Hv(i), err
       end if
    else
       inform%numH_wrong = inform%numH_wrong + 1
       if ( print_level >= 2 ) then
          str = 'BAD' ; write( out, 1015 ) i, str, diff, data%Hv(i), err
       end if
    end if
 end do

 ! ----------
 ! Full check
 ! ----------

 CASE ( 2 )

 ! Go to part of the code based on value in data%branch

 SELECT CASE ( data%branch )
 CASE ( 1 )   ! first time in
    go to 130
 CASE ( 17 )  ! g(x)              - full - in Gcheck
    GO TO 350
 CASE ( 18 )  ! f(x)              - full - in Gcheck
    GO TO 360
 CASE ( 19 )  ! f(x+alpha *ei)    - full - in Gcheck
    GO TO 370
 CASE ( 20 )  ! c(x)              - full - in Jcheck
    GO TO 380
 CASE ( 21 )  ! J(x)              - full - in Jcheck
    GO TO 390
 CASE ( 22 )  ! J*ej              - full - in Jcheck
    GO TO 400
 CASE ( 23 )  ! c(x+alpha*ej)     - full - in Jcheck
    GO TO 410
 CASE ( 24  ) ! g(x)              - full - Hcheck when checkG = .false.
    GO TO 420
 CASE ( 25  ) ! J(x)              - full - Hcheck when checkJ = .false.
    GO TO 430
 CASE ( 26  ) ! gradL(x)          - full - in Hcheck
    GO TO 440
 CASE ( 27  ) ! g(x+alpha*ej)     - full - in Hcheck
    GO TO 450
 CASE ( 28  ) ! J(x+alpha*ej)     - full - in Hcheck
    GO TO 460
 CASE ( 29  ) ! gradL(x+alpha*ej) - full - in Hcheck
    GO TO 470
 CASE ( 30 )  ! H(x,y)            - full - in Hcheck
    GO TO 480
 CASE ( 31  ) ! H*ej              - full - in Hcheck
    GO TO 490
 END SELECT

130 continue

 ! BEGIN: Full check of G
 ! ----------------------

 if ( checkG ) then
    if ( print_level >= 2 ) then
       write( out, 4005 )
       write( out, 1010 )
    end if
 else
    go to 150
 end if

 ! evaluate g(x)

 if ( g_availability == 1 ) then
    call eval_G( inform%status, nlp%X, userdata, nlp%G )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%g_filled = .true.
 else
    data%g_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 17
    inform%status = 4
    return
 end if

 ! return from reverse communication for g(x)

350 continue

 if ( .not. data%g_filled ) nlp%G = data%RC%g

 ! evaluate f(x)

 if ( f_availability == 1 ) then
    call eval_F( inform%status, nlp%X, userdata, nlp%f )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%f_filled = .true.
 else
    data%f_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 18
    inform%status = 2
    return
 end if

 ! return from reverse communication for f(x)

360 continue

 if ( .not. data%f_filled ) nlp%f = data%RC%f

 ! BEGIN : Imitation do loop using reverse communication.

 data%i = 1

140 continue

 i = data%i

 temp     = nlp%X(i)
 nlp%X(i) = nlp%X(i) + data%alpha

 ! evaluate f(x1)

 if ( f_availability == 1 ) then
    call eval_F( inform%status, nlp%X, userdata, data%f_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%f_filled = .true.
    nlp%X(i) = temp
 else
    data%temp = temp
    data%f_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 19
    inform%status = 2
    return
 end if

 ! return from reverse communication for f(x + alpha*ei)

370 continue

 if ( .not. data%f_filled ) then
    data%f_plus = data%RC%f
    nlp%X(data%i) = data%temp
    i = data%i
 end if

 ! compare G with the finite difference.

 diff = ( data%f_plus - nlp%f ) / data%alpha
 err = ABS( diff - nlp%G(i) ) / ( one + ABS(diff) )

 if ( err <= data%tol ) then
    if ( print_level >= 2 ) then
       str = ' OK' ; write( out, 1007 ) i, str, diff, nlp%G(i), err
    end if
 else
    inform%numG_wrong = inform%numG_wrong + 1
    if ( print_level >= 2 ) then
       str = 'BAD' ; write( out, 1007 ) i, str, diff, nlp%G(i), err
    end if
 end if

 if ( data%i == n ) then
    ! relax .... go onto checking J.
 else
    data%i = data%i + 1
    go to 140
 end if

 ! END : Imitation do loop using reverse communication.

 ! BEGIN: Full check of J
 ! ----------------------

150 continue

 if ( checkJ ) then
    if ( print_level >= 2 ) then
       write( out, 4006 )
       write( out, 1010 )
    end if
 else
    go to 170
 end if

 ! evaluate c(x)

 if ( c_availability == 1 ) then
    call eval_C( inform%status, nlp%X, userdata, nlp%C )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
    data%c_filled = .true.
 else
    data%c_filled = .false.
    data%RC%x     = nlp%X
    data%branch   = 20
    inform%status = 3
    return
 end if

 ! return from reverse communication for c(x)

380 continue

 if ( .not. data%c_filled ) nlp%C = data%RC%c

 ! evaluate J(x) if it is available explicitly.

 if ( J_availability == 1 ) then
    call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 elseif ( J_availability == 2 ) then
    data%RC%x     = nlp%X
    data%branch   = 21
    inform%status = 5
    return
 end if
 
 ! return from reverse communication for J(x)

390 continue

 if ( J_availability == 2 ) nlp%J%val = data%RC%Jval

    ! BEGIN : Imitation do loop for jth column of J.

    data%j  = 1
    data%ej = zero

160 continue

    j = data%j
    data%ej(j) = one

    ! compute J*ej = jth column of J.

    if ( J_availability <= 2 ) then
       data%Jv = zero
       call mop_Ax( one, nlp%J, data%ej, one, data%Jv )
       data%Jv_filled = .true.
    elseif ( J_availability == 3 ) then
       data%Jv = zero
       call eval_Jv( inform%status, nlp%X, userdata, .false., data%Jv, data%ej )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%Jv_filled = .true.
    else
       data%Jv_filled = .false.
       data%RC%u(:m)  = zero
       data%RC%v(:n)  = data%ej
       data%RC%x      = nlp%X
       data%branch    = 22
       inform%status  = 6
       return
    end if
    
    ! return from reverse communication for J*ej = jth column of J

400 continue

    j = data%j
    data%ej(j) = zero

    if ( J_availability > 3 ) data%Jv = data%RC%u(:m)

    ! evaluate c(x+alpha*ej)

    data%temp = nlp%X(j)
    nlp%X(j)  = nlp%X(j) + data%alpha

    if ( c_availability == 1 ) then
       call eval_C( inform%status, nlp%X, userdata, data%c_plus )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%c_filled = .true.
    else
       data%c_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 23
       inform%status = 3
       return
    end if

    ! return from reverse communication for c(x + alpha*ej)

410 continue

    j = data%j
    nlp%X(j) = data%temp

    if ( c_availability /= 1 ) data%c_plus = data%RC%c

    ! Compare jth column of J with the finite difference.

    do i = 1, m

       Jij = data%Jv( i )

       diff = ( data%C_plus(i) - nlp%C(i) ) / data%alpha
       err  = ABS( diff - Jij ) / ( one + ABS( diff ) )

       if ( err <= data%tol ) then
          if ( print_level >= 2 ) then
             str = ' OK' ; write( out, 1008 ) i, j, str, diff, Jij, err
          end if
       else
          inform%numJ_wrong = inform%numJ_wrong + 1
          if ( print_level >= 2 ) then
             str = 'BAD' ; write( out, 1008 ) i, j, str, diff, Jij, err
          end if
       end if

    end do

    if ( data%j == n ) then
       ! relax .... go onto checking H.
    else
       data%j = data%j + 1
       go to 160
    end if

    ! END : imitation do loop for jth column of J.

170 continue

 ! BEGIN: Full check of H
 ! ----------------------

 if ( checkH ) then
    if ( print_level >= 2 ) then
       write( out, 4007 )
       write( out, 1010 )
    end if
 else
    go to 900
 end if

 ! compute g(x)

 if ( checkG ) then
    data%g_filled = .true.
 else
    if ( g_availability == 1 ) then
       call eval_G( inform%status, nlp%X, userdata, nlp%G )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%g_filled = .true.
    else
       data%g_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 24
       inform%status = 4
       return
    end if
 end if

 ! return from reverse communication for g(x) when checkG = .false.

420 continue

 if ( .not. data%g_filled ) nlp%G = data%RC%g

 ! compute J(x) if explicitely availabe and not already computed.

 if ( checkJ ) then
    data%J_filled = .true.
 else
    if ( J_availability == 1 ) then
       call eval_J( inform%status, nlp%X, userdata, nlp%J%val )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%J_filled = .true.
    elseif ( J_availability == 2 ) then 
       data%J_filled = .false.
       data%RC%x     = nlp%X
       data%branch   = 25
       inform%status = 5
       return
    else
       data%J_filled = .true.
    end if
 end if

 ! return from reverse communication for J(x) when checkJ = .false.

430 continue

 if ( .not. data%J_filled ) nlp%J%val = data%RC%Jval

 ! compute gradL = g(x) - J(x)^T y

 data%gradL = nlp%G

 if ( m > 0 ) then
    if ( J_availability <= 2 ) then
       call mop_Ax( -one, nlp%J, nlp%Y, one, data%gradL, transpose=.true. )
       data%Jv_filled = .true.
    elseif ( J_availability == 3 ) then
       call eval_Jv( inform%status, nlp%X, userdata, .true., data%gradL, -nlp%Y )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%Jv_filled = .true.
    else
       data%RC%u(:n)  = data%gradL
       data%RC%v(:m)  = -nlp%Y
       data%RC%x      = nlp%X
       data%branch    = 26
       inform%status  = 7
       data%Jv_filled = .false.
       return
    end if
 end if

 ! return from reverse communication for gradL = g(x) - J(x)^T y

440 continue

 if ( m > 0 .and. .not. data%Jv_filled ) data%gradL = data%RC%u(:n)

 ! BEGIN : Imitation do loop for jth column of H.

 data%j  = 1
 data%ej = zero

180 continue

 j = data%j

 data%temp = nlp%X(j)
 nlp%X(j)  = nlp%X(j) + data%alpha
 
 ! compute g(x+alpha*ej)

 if ( g_availability == 1 ) then
    call eval_G( inform%status, nlp%X, userdata, data%G_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 else
    data%RC%x     = nlp%X
    data%branch   = 27
    inform%status = 4
    return
 end if
 
 ! return from reverse communication for g(x+alpha*ej)

450 continue

 if ( g_availability /= 1 ) data%G_plus = data%RC%g

 ! compute J(x+alpha*ej) if explicitly available.
 
 if ( J_availability == 1 ) then
    call eval_J( inform%status, nlp%X, userdata, data%Jval_plus )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 elseif ( J_availability == 2 ) then
    data%RC%x     = nlp%X
    data%branch   = 28
    inform%status = 5
    return
 end if
    
 ! return from reverse communication for J(x+alpha*ej)

460 continue

 if ( J_availability == 2 ) data%Jval_plus = data%RC%Jval

 ! compute gradL_plus = g(xplus) - J(xplus)^T Y

 data%gradL_plus = data%G_plus

 if ( m > 0 ) then
    if ( J_availability <= 2 ) then
       nlp%J%val = data%Jval_plus
       call mop_Ax( -one, nlp%J, nlp%Y, one, data%gradL_plus, transpose=.true. )
       data%Jv_filled = .true.
    elseif ( J_availability == 3 ) then
       call eval_Jv( inform%status, nlp%X, userdata, .true.,                   &
                     data%gradL_plus, -nlp%Y )
       if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
       data%Jv_filled = .true.
    else
       data%RC%u(:n)  = data%gradL_plus
       data%RC%v(:m)  = -nlp%Y
       data%RC%x      = nlp%X
       data%branch    = 29
       inform%status  = 7
       data%Jv_filled = .false.
       return
    end if
 end if

 ! return from reverse communication for gradL_plus = g_plus - J_plus^T y

470 continue

 if ( m > 0 .and. .not. data%Jv_filled ) data%gradL_plus = data%RC%u(:n)
 
 ! return nlp%X to its original value.

 j = data%j
 nlp%X(j) = data%temp

 ! Evaluate H(x,y) if explicitely available.

 if ( H_availability == 1 ) then
    call eval_HL( inform%status, nlp%X, nlp%Y, userdata, nlp%H%val)
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 elseif ( H_availability == 2 ) then
    data%RC%x     = nlp%X
    data%RC%y     = nlp%Y
    inform%status = 8
    data%branch   = 30
    return
 end if

 ! return from reverse communication for H(x,y)

480 continue

 if ( H_availability == 2 ) nlp%H%val = data%RC%Hval

 ! compute H*ej = jth column of H

 j = data%j
 data%ej(j) = one

 if ( H_availability <= 2 ) then
    data%Hv = zero
    call mop_Ax( one, nlp%H, data%ej, one, data%Hv, symmetric=.true. )
 elseif ( H_availability == 3 ) then
    data%Hv = zero
    call eval_Hv(inform%status, nlp%X, nlp%Y, userdata, data%Hv, data%ej )
    if ( inform%status /= 0 ) then ; inform%status = -58 ; go to 999 ; end if
 else
    data%RC%u(:n) = zero
    data%RC%v(:n) = data%ej
    data%RC%x     = nlp%X
    data%RC%y     = nlp%Y
    inform%status = 9
    data%branch   = 31
    return
 end if
 
 ! return from reverse communicaton for H*ej

490 continue

 if ( H_availability > 3 ) data%Hv = data%RC%u(:n)

 ! return ej back to the zero vector

 j = data%j
 data%ej(j) = zero

 ! Compare jth column of H to the finite differences.

 do i = 1, n

    diff = ( data%gradL_plus(i) - data%gradL(i) ) / data%alpha
    Hij = data%Hv(i)
    err = ABS( diff - Hij ) / ( one + ABS(diff) )

    if ( err <= data%tol ) then
       if ( print_level >= 2 ) then
          str = ' OK' ; write( out, 1009 ) i, j, str, diff, Hij, err
       end if
    else
       inform%numH_wrong = inform%numH_wrong + 1
       if ( print_level >= 2 ) then
          str = 'BAD' ; write( out, 1009 ) i, j, str, diff, Hij, err
       end if
    end if

 end do
 
 if ( data%j == n ) then
    ! relax....we are done with all the columns of H.  Proceed on.
 else
    data%j = data%j + 1
    go to 180
 end if
 
 ! END : Imitation do-loop for jth column of H

 END SELECT

 ! ------------------------------------------------------------------------------
 ! -                        END : Derivative Checking                           -
 ! ------------------------------------------------------------------------------

900 continue

 ! Posssibly print more information.
 ! ---------------------------------

 if ( print_level >= 1 ) then

    ! Print header and summary.

    if (verify_level == 1 ) then

       write( out, 1016 ) 'Cheap'

       if ( checkG ) then
          if ( inform%numG_wrong == 0 ) then
             write( out, 1001 )
          else
             write( out, 1002 )
          end if
       end if

       if ( checkJ ) then
          if ( inform%numJ_wrong == 0 ) then
             write( out, 1003 )
          else
             write( out, 1004)
          end if
       end if

       if ( checkH ) then
          if ( inform%numH_wrong == 0 ) then
             write( out, 1005)
          else
             write( out, 1006)
          end if
       end if

    else

       write( out, 1021 ) 'Expensive'

       if ( checkG ) then
          if ( inform%numG_wrong == 0 ) then
             write( out, 8001 )
          else
             write( out, 8002 ) inform%numG_wrong
          end if
       end if

       if ( checkJ ) then
          if ( inform%numJ_wrong == 0 ) then
             write( out, 8003 )
          else
             write( out, 8004) inform%numJ_wrong
          end if
       end if

       if ( checkH ) then
          if ( inform%numH_wrong == 0 ) then
             write( out, 8005)
          else
             write( out, 8006) inform%numH_wrong
          end if
       end if

    end if

    ! Print control parameters and some matrix data.

    if ( print_level >= 2 ) then

       write( out, 1017 ) ! control parameters header
       write( out, 1018 ) checkG, f_availability, deallocate_error_fatal, &
                          checkJ, c_availability, print_level,            &
                          checkH, g_availability,  verify_level, error,   &
                          J_availability, out, H_availability

       write( out, 1019 ) ! matrix data header
       if ( allocated(nlp%J%type) ) then
          write( out, '(/,15X, "J%type --- ", 30A )') nlp%J%type
       end if
       if ( allocated(nlp%J%id) ) then
          write( out, '(15X, "J%id   --- ", 30A )' ) nlp%J%id
       end if
       if ( allocated(nlp%H%type) ) then
          write( out, '(15X, "H%type --- ", 30A )' ) nlp%H%type
       end if
       if ( allocated(nlp%H%id) ) then
          write( out, '(15X, "H%id   --- ", 30A )' ) nlp%H%id
       end if
       write( out, 1020 ) m, n

       if ( print_level >= 3 ) then

          ! Components for nlp%J

          if ( allocated(nlp%J%type) ) then
             SELECT CASE ( SMT_get( nlp%J%type ) )
             CASE ( 'DENSE' )
                WRITE( out, 6000 ) ( nlp%J%val(i), i = 1, m*n )
             CASE ( 'SPARSE_BY_ROWS' )
                WRITE( out, 6001 ) &
                     ( nlp%J%col(i), nlp%J%val(i), i=1, nlp%J%ptr(m+1)-1 )
                WRITE( out, 6002 ) nlp%J%ptr( 1:m+1 )
             CASE ( 'SPARSE_BY_COLUMNS' )
                WRITE( out, 6003 ) &
                     ( nlp%J%row(i), nlp%J%val(i), i = 1, nlp%J%ptr(n+1)-1 )
                WRITE(  out, 6002 ) nlp%J%ptr( 1: n+1 )
             CASE ( 'COORDINATE' )
                WRITE( out, 6004 ) &
                     ( nlp%J%row(i), nlp%J%col(i), nlp%J%val(i), i = 1,nlp%J%ne )
             CASE ( 'DIAGONAL' )
                WRITE( out, 6000 ) ( nlp%J%val(i), i = 1,m )
             CASE DEFAULT
                inform%status = GALAHAD_error_restrictions ; go to 999
             END SELECT
          end if

          ! Components for nlp%H

          if ( allocated(nlp%H%type) ) then
             SELECT CASE ( SMT_get( nlp%H%type ) )
             CASE ( 'DENSE' )
                WRITE( out, 7000 ) ( nlp%H%val(i), i = 1, m * n )
             CASE ( 'SPARSE_BY_ROWS' )
                WRITE( out, 7001 ) &
                     ( nlp%H%col(i), nlp%H%val(i), i=1, nlp%H%ptr(m+1)-1 )
                WRITE( out, 7002 ) nlp%H%ptr( 1:m+1 )
             CASE ( 'SPARSE_BY_COLUMNS' )
                WRITE( out, 7003 ) &
                     ( nlp%H%row(i), nlp%H%val(i), i = 1, nlp%H%ptr(n+1)-1 )
                WRITE(  out, 7002 ) nlp%H%ptr( 1: n+1 )
             CASE ( 'COORDINATE' )
                WRITE( out, 7004 ) &
                     ( nlp%H%row(i), nlp%H%col(i), nlp%H%val(i), i = 1,nlp%H%ne )
             CASE ( 'DIAGONAL' )
                WRITE( out, 7000 ) ( nlp%H%val(i), i = 1,m )
             CASE DEFAULT
                inform%status = GALAHAD_error_restrictions ; go to 999
             END SELECT
          end if

          if (  print_level >= 4 ) then
             write( out, 1022 ) ! private data header
             write( out, 1023 ) data%normx, data%fd_len, &
                                data%alpha, abs(data%scale), data%tol
          end if

       end if

    end if

 end if

 ! Successful return.
 ! ------------------

 inform%status = 0
 inform%derivative_ok = ( inform%numG_wrong == 0 .and. &
                          inform%numJ_wrong == 0 .and. &
                          inform%numH_wrong == 0 )

 if ( print_level >= 1 .and. out >= 1 ) then
    write( out, 4002 ) inform%status ! exit status 
    write( out, 4001 )               ! footer
 end if

 return

 ! Unsuccessful returns.
 ! ---------------------

 990 continue ! Special printing used for allocation error.
     error = data%control%error
     if ( error >= 1 ) then
        write( error, 2002 ) inform%status, inform%alloc_status, inform%bad_alloc
     end if
     inform%status = GALAHAD_error_allocate

 999 continue ! All other unsuccessful exits.
     print_level = data%control%print_level ; out = data%control%out
     if ( print_level >= 1 .and. out >= 1  ) then
        write( out, 4002 ) inform%status ! exit status 
        write( out, 4001 )               ! footer.
     end if

 return

! Formatting statements.

1001 FORMAT(/, T15,'THE GRADIENT OF THE OBJECTIVE FUNCTION IS ---- [OK]')
1002 FORMAT(/, T15,'THE GRADIENT OF THE OBJECTIVE FUNCTION IS --- [BAD]')
1003 FORMAT(/, T15,'THE JACOBIAN OF THE CONSTRAINT FUNCTION IS --- [OK]')
1004 FORMAT(/, T15,'THE JACOBIAN OF THE CONSTRAINT FUNCTION IS -- [BAD]')
1005 FORMAT(/, T15,'THE HESSIAN OF THE LAGRANGIAN FUNCTION IS ---- [OK]')
1006 FORMAT(/, T15,'THE HESSIAN OF THE LAGRANGIAN FUNCTION IS --- [BAD]')
1007 FORMAT(2X,'G(',I11, ')', 3X, A3, 3X, ES16.9, 3X, ES16.9, 3X, ES16.9)
1008 FORMAT(2X,'J(', I5, ',', I5, ')', 3X, A3, 3X, ES16.9, 2(3X,ES16.9) )
1009 FORMAT(2X,'H(', I5, ',', I5, ')', 3X, A3, 3X, ES16.9, 2(3X,ES16.9) )
1010 FORMAT(/, &
     2X,'   Component      Ok      Difference            Value             Error'    ,/,&
     2X,'--------------   ---   ----------------   ----------------   ----------------' )
1011 FORMAT(2X,'G( : )', 11X, A3, 2X, ES16.9, 2(3X,ES16.9))
1014 FORMAT(2X,'J(', I5, ', : )', 5X, A3, 2X, ES16.9, 2(3X,ES16.9))
1015 FORMAT(2X,'H(', I5, ', : )', 5X, A3, 2X, ES16.9, 2(3X,ES16.9))
1016 FORMAT(2/, &
     T15, '---------------------------------------------------',    /,&
     T15, '|                     SUMMARY                     |',    /,&
     T15, '|               ( Verify : ', A5, ' )                |', / &
     T15, '---------------------------------------------------' )
1017 FORMAT(/, &
     T13, '-------------------------------------------------------',/, &
     T13, '|                CONTROL PARAMETERS                   |',/  &
     T13, '-------------------------------------------------------' )
1018 FORMAT(/, &
     12X, 'checkG = ', L2, 3x, 'f_available = ', I2, 3x, 'deall_error_fatal = ', L2, /, &
     12X, 'checkJ = ', L2, 3x, 'c_available = ', I2, 3x, 'print_level       = ', I2, /, &
     12X, 'checkH = ', L2, 3x, 'g_available = ', I2, 3x, 'verify_level      = ', I2, /, &
     12X, 'error  = ', I2, 3x, 'J_available = ', I2, 3x, 'out               = ', I2, /, &
                          T27, 'H_available = ', I2 )
1019 FORMAT(/, &
     T16, '----------------------------------------------',/, &
     T16, '|                MATRIX DATA                 |',/  &
     T16, '----------------------------------------------' )
1020 FORMAT(/,15X, 'm = ', I6, /, 15X, 'n = ', I6 )
1021 FORMAT(/, &
     T15, '---------------------------------------------------',/,&
     T15, '|                     SUMMARY                     |',/ &
     T15, '|             ( Verify : ', A9, ' )              |',/ &
     T15, '---------------------------------------------------' )
1022 FORMAT(/, &
     T10, '-------------------------------------------------------------',/,&
     T10, '|                      PRIVATE DATA                         |',/ &
     T10, '-------------------------------------------------------------' )
1023 FORMAT(/, &
     9X, 'normx = ', ES10.4, 3X, 'fd_len = ', ES10.4, 3X, &
         'alpha = ', ES10.4, /,                           &
     9X, 'scale = ', ES10.4, 3X, 'tol    = ', ES10.4 )

2002 FORMAT(/,'- ERROR:CHECK_verify:allocation error ', I0, &
              ' alloc_status ', I0, 'bad_alloc', A)

3001 FORMAT(/,'- WARNING:CHECK_verify:unable to check Jacobian b/c m = 0.' )

4000 FORMAT(/, &
     1x, 78('-'), /, &
     1x, 20('-'), '          BEGIN: CHECK_verify         ', 20('-'), /, &
     1x, 78('-') )
4001 FORMAT(/, &
     1x, 78('-'), /, &
     1x, 20('-'), '           END: CHECK_verify          ', 20('-'), /, &
     1x, 78('-') )
4002 FORMAT(/, T33, 'EXIT STATUS : ', I3)
4005 FORMAT(/, T20, 'EXPENSIVE VERIFICATION OF THE GRADIENT G(X)' )
4006 FORMAT(/, T20, 'EXPENSIVE VERIFICATION OF THE JACOBIAN C(X)' )
4007 FORMAT(/, T20, 'EXPENSIVE VERIFICATION OF THE HESSIAN H(X,Y)' )

5000 FORMAT(2/, T22, 'CHEAP VERIFICATION OF THE GRADIENT G(X)' )
5001 FORMAT(2/, T22, 'CHEAP VERIFICATION OF THE JACOBIAN J(X)' )
5002 FORMAT(2/, T22, 'CHEAP VERIFICATION OF THE HESSIAN H(X,Y)' )

6000  FORMAT(/,15X,'      J%val    ',/, &
               15X,'  -------------',/, (5X, ES17.10 ) )
6001  FORMAT(/,15X,'  J%col             J%val    ',/,  &
               15X,'  -----         -------------',/,  &
              (15X, I7, 7X, ES17.10) )
6002  FORMAT(/,15X,'  J%ptr',/, &
               15X,'  -----',/, (5X, I7 ) )
6003  FORMAT(/,15X,'  J%row             J%val    ',/,  &
               15X,'  -----         -------------',/,  &
              (15X, I7, 7X, ES17.10) )
6004  FORMAT(/,15X,'  J%row         J%col             J%val    ',/,  &
               15X,'  -----         -----       -----------------',/,  &
              (15X, I7, 7X, I7, 7X, ES17.10) )

7000  FORMAT(/,15X,'      H%val    ',/, &
               15X,'  -------------',/, (5X, ES17.10 ) )
7001  FORMAT(/,15X,'  H%col             H%val    ',/,  &
               15X,'  -----         -------------',/,  &
              (15X, I7, 7X, ES17.10) )
7002  FORMAT(/,15X,'  H%ptr',/, &
               15X,'  -----',/, (5X, I7 ) )
7003  FORMAT(/,15X,'  H%row             H%val    ',/,  &
               15X,'  -----         -------------',/,  &
              (15X, I7, 7X, ES17.10) )
7004  FORMAT(/,15X,'  H%row         H%col             H%val    ',/,  &
               15X,'  -----         -----        ----------------',/,  &
              (15X, I7, 7X, I7, 7X, ES17.10) )

8001 FORMAT(/, T15,'THE GRADIENT OF THE OBJECTIVE FUNCTION IS ---- [OK]')
8002 FORMAT(/, T15,'THE GRADIENT OF THE OBJECTIVE FUNCTION IS --- [BAD]',/,  &
               T15,'XXX THERE APPEARS TO BE ', I7, ' WRONG ENTRIES!' )
8003 FORMAT(/, T15,'THE JACOBIAN OF THE CONSTRAINT FUNCTION IS --- [OK]')
8004 FORMAT(/, T15,'THE JACOBIAN OF THE CONSTRAINT FUNCTION IS -- [BAD]',/,  &
               T15,'XXX THERE APPEARS TO BE ', I7, ' WRONG ENTRIES!' )
8005 FORMAT(/, T15,'THE HESSIAN OF THE LAGRANGIAN FUNCTION IS ---- [OK]')
8006 FORMAT(/, T15,'THE HESSIAN OF THE LAGRANGIAN FUNCTION IS --- [BAD]',/,  &
               T15,'XXX THERE APPEARS TO BE ', I7, ' WRONG ENTRIES!' )

 END SUBROUTINE CHECK_verify

 !*******************************************************************************
 !           G A L A H A D  -  CHECK_terminate  S U B R O U T I N E             !
 !*******************************************************************************

 SUBROUTINE CHECK_terminate( data, control, inform )
 !-------------------------------------------------------------------------------
 ! Purpose: Deallocate all private storage.
 !-------------------------------------------------------------------------------
 implicit none
 !-------------------------------------------------------------------------------
 ! Dummy arguments.
 !-------------------------------------------------------------------------------
 TYPE ( CHECK_data_type ), INTENT( INOUT ) :: data
 TYPE ( CHECK_control_type ), INTENT( INOUT ) :: control
 TYPE ( CHECK_inform_type ), INTENT( INOUT ) :: inform
 !-------------------------------------------------------------------------------
 ! Local variables.
 !-------------------------------------------------------------------------------
 CHARACTER ( LEN = 80 ) :: array_name
 INTEGER :: error
 !-------------------------------------------------------------------------------

 ! For convenience

 error = control%error

 ! Deallocate all remaining allocated arrays

 array_name = 'CHECK: data%C_plus'
 CALL SPACE_dealloc_array( data%C_plus,                                 &
      inform%status, inform%alloc_status, array_name = array_name,      &
      bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991
 
 array_name = 'CHECK: data%G_plus'
 CALL SPACE_dealloc_array( data%G_plus,                                 &
      inform%status, inform%alloc_status, array_name = array_name,      &
      bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%Jv'
 CALL SPACE_dealloc_array( data%Jv,                                     &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%Hv'
 CALL SPACE_dealloc_array( data%Hv,                                     &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%X1'
 CALL SPACE_dealloc_array( data%X1,                                     &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%s'
 CALL SPACE_dealloc_array( data%s,                                      &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%gradL'
 CALL SPACE_dealloc_array( data%gradL,                                  &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%gradL_plus'
 CALL SPACE_dealloc_array( data%gradL_plus,                             &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%s_back'
 CALL SPACE_dealloc_array( data%s_back,                                 &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%ej'
 CALL SPACE_dealloc_array( data%ej,                                     &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%u'
 CALL SPACE_dealloc_array( data%RC%u,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%v'
 CALL SPACE_dealloc_array( data%RC%v,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%x'
 CALL SPACE_dealloc_array( data%RC%x,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%y'
 CALL SPACE_dealloc_array( data%RC%y,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%c'
 CALL SPACE_dealloc_array( data%RC%c,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%g'
 CALL SPACE_dealloc_array( data%RC%g,                                   &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%Jval'
 CALL SPACE_dealloc_array( data%RC%Jval,                                &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%Jval_plus'
 CALL SPACE_dealloc_array( data%Jval_plus,                              &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 array_name = 'CHECK: data%RC%Hval'
 CALL SPACE_dealloc_array( data%RC%Hval,                                &
    inform%status, inform%alloc_status, array_name = array_name,        &
    bad_alloc = inform%bad_alloc, out = control%error )
 IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) go to 991

 RETURN

991 continue
    if ( error >= 1 ) then
       write( error, 1000 ) inform%status, inform%alloc_status, inform%bad_alloc
    end if
    inform%status = GALAHAD_error_deallocate    

 ! format statements

1000 FORMAT(/,'- ERROR:CHECK_terminate:deallocation error ', I0, &
              ' alloc_status ', I0, 'bad_alloc', A) 

 END SUBROUTINE CHECK_terminate

 !*******************************************************************************
 !          G A L A H A D  -  CHECK_read_specfile S U B R O U T I N E           !
 !*******************************************************************************

 SUBROUTINE CHECK_read_specfile( control, device, alt_specname_CHECK )
 !-------------------------------------------------------------------------------
 ! Purpose: Read the contents of a specification file and performs the assignment
 !          of values associated with given keywords to the corresponding control
 !          parameters.
 !
 ! The default values defined in CHECK_initialize could have been set by:
 !      
 ! BEGIN CHECK SPECIFICATIONS (DEFAULT)
 !   error-printout-device             6
 !   printout-device                   6
 !   print-level                       1
 !   verification-level                2
 !   f-availability                    1
 !   c-availability                    1
 !   g-availability                    1
 !   J-availability                    1 
 !   H-availability                    1
 !   check-gradient                    .TRUE.
 !   check-Jacobian                    .TRUE.
 !   check-Hessian                     .TRUE.
 !   deallocate-error-fatal            .FALSE.
 ! END CHECK SPECIFICATIONS
 !-------------------------------------------------------------------------------
 ! Dummy arguments.
 !-------------------------------------------------------------------------------
 TYPE ( CHECK_control_type ), INTENT( INOUT ) :: control
 INTEGER, INTENT( IN ) :: device
 CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname_CHECK
 !-------------------------------------------------------------------------------
 ! Local variables.
 !-------------------------------------------------------------------------------
 INTEGER, PARAMETER :: lspec = 20
 CHARACTER( LEN = 16 ), PARAMETER :: specname_CHECK = 'CHECK           '
 TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
 !-------------------------------------------------------------------------------

 ! Define the keywords
 ! -------------------

 spec%keyword = ''

 ! Integer key-words

 spec( 1 )%keyword = 'error-printout-device'
 spec( 2 )%keyword = 'printout-device'
 spec( 3 )%keyword = 'verification-level'
 spec( 4 )%keyword = 'print-level'
 spec( 5 )%keyword = 'f-availability'
 spec( 6 )%keyword = 'c-availability'
 spec( 7 )%keyword = 'g-availability'
 spec( 8 )%keyword = 'J-availability'
 spec( 9 )%keyword = 'H-availability'

 ! Logical key-words

 spec( 10 )%keyword = 'deallocate-error-fatal'
 spec( 11 )%keyword = 'check-gradient'
 spec( 12 )%keyword = 'check-Jacobian'
 spec( 13 )%keyword = 'check-Hessian'

 ! Read the specfile

 IF ( PRESENT( alt_specname_CHECK ) ) THEN
    CALL SPECFILE_read( device, alt_specname_CHECK, spec, lspec, control%error )
 ELSE
    CALL SPECFILE_read( device, specname_CHECK, spec, lspec, control%error )
 END IF

 ! Set integer values

 CALL SPECFILE_assign_integer( spec( 1 ), control%error, control%error )
 CALL SPECFILE_assign_integer( spec( 2 ), control%out, control%error )
 CALL SPECFILE_assign_integer( spec( 3 ), control%verify_level, control%error )
 CALL SPECFILE_assign_integer( spec( 4 ), control%print_level, control%error )
 CALL SPECFILE_assign_integer( spec( 5 ), control%f_availability, control%error )
 CALL SPECFILE_assign_integer( spec( 6 ), control%c_availability, control%error )
 CALL SPECFILE_assign_integer( spec( 7 ), control%g_availability, control%error )
 CALL SPECFILE_assign_integer( spec( 8 ), control%J_availability, control%error )
 CALL SPECFILE_assign_integer( spec( 9 ), control%H_availability, control%error )

 ! Set logical values

 CALL SPECFILE_assign_logical( spec( 10 ), &
                               control%deallocate_error_fatal, control%error )
 CALL SPECFILE_assign_logical( spec( 11 ), control%checkG, control%error )
 CALL SPECFILE_assign_logical( spec( 12 ), control%checkJ, control%error )
 CALL SPECFILE_assign_logical( spec( 13 ), control%checkH, control%error )

 RETURN

 END SUBROUTINE CHECK_read_specfile

 !******************************************************************************
 !            G A L A H A D  -  get_feas_step   S U B R O U T I N E            !
 !******************************************************************************

 SUBROUTINE get_feas_step( n, xl, xu, x, alpha, s, nFeas )
 !-------------------------------------------------------------------------------
 ! Purpose: ensures that x + alpha * s is feasible.  First, checks if
 !          x(i) + alpha s(i) is feasible.  If it is not, switch the sign of 
 !          s(i) and try again.  If this also fails then resort to s(i) = zero.
 !          On exit, x + alpha s is feasible.
 !
 ! Arguments:
 !
 !    n      real intent in scalar that holds the length of xl, x, xu, and s.
 !    xl     real intent in vector of lower bounds on x.  Restriction: xl <= x.
 !    xu     real intent in vector of upper bounds on x.  Restriction: x <= xu.
 !    x      real intent in vector holding the current value of x.
 !           Restriction: xl <= x <= xu.
 !    alpha  real intent in scalar holding quantity of type real.  See "Purpose".
 !    s      real intent inout vector that holds the step direction.
 !    nFeas  integer intent out scalar that holds upon exit the number of
 !           components of s that are nonzero.  Equivalently, it is the number of
 !           componenents of x +/- alpha * s (the input value of s) that are
 !           feasible.  If nFeas = 0 on exit, then it must hold that s = 0.
 !           Again, see "Purpose".
 !-------------------------------------------------------------------------------
 implicit none
 !-------------------------------------------------------------------------------
 ! Dummy variables
 !-------------------------------------------------------------------------------
 INTEGER, INTENT( IN ) :: n
 INTEGER, INTENT( OUT ) :: nFeas
 REAL( KIND = wp ), INTENT( IN ) :: alpha
 REAL( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: xl, xu, x
 REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: s
 !-------------------------------------------------------------------------------
 ! local variables
 !-------------------------------------------------------------------------------
 integer :: j
 real( KIND = wp ) :: lo, hi, xj, xnew, sj
 !-------------------------------------------------------------------------------
 
 nFeas = 0

 ! Loop over each components of x(j) + alpha * s(j) one at a time.

 do  j = 1, n
    
    xj = x(j)
    lo = xl(j)
    hi = xu(j)
    
    if ( lo .eq. hi ) then
       s(j) = zero ; cycle
    end if

    ! If xj + alpha * sj is infeasible, switch the direction of sj and
    ! try again.  If all else fails, set sj = zero.
    
    sj   = s(j)
    xnew = xj + alpha*sj
    
    if ( sj .gt. zero ) then
       
       if ( xnew .gt. hi ) then
          sj   = -sj
          xnew =  xj + alpha*sj
          if ( xnew .lt. lo ) sj = zero
       end if
       
    else
       
       if ( xnew .lt. lo ) then
          sj   = -sj
          xnew =  xj + alpha*sj
          if ( xnew .gt. hi ) sj = zero
       end if
       
    end if
    
    s(j) = sj
    
    if (sj .ne. zero)  nFeas = nFeas + 1
    
 end do

 END SUBROUTINE get_feas_step

END MODULE GALAHAD_CHECK_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*   END CHECK  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*                             *-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
