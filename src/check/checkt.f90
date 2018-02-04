! THIS VERSION: GALAHAD 2.4 - 4/02/2008 AT 09:00 GMT.
PROGRAM GALAHAD_check_test

  USE GALAHAD_SMT_double   ! double precision version
  USE GALAHAD_NLPT_double  ! double precision version
  USE GALAHAD_MOP_double   ! double precision version
  USE GALAHAD_CHECK_double ! double precision version
  USE GALAHAD_SPACE_double

  IMPLICIT NONE

  ! Interfaces

  INTERFACE 
     SUBROUTINE funF2( status, X, userdata, F )
       USE GALAHAD_NLPT_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
       REAL ( kind = wp ), INTENT( OUT ) :: F
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE funF2
     SUBROUTINE funC2(status, X, userdata, C)
       USE GALAHAD_NLPT_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
       REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) :: C
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE funC2
     SUBROUTINE funG2(status, X, userdata, G)
       USE GALAHAD_NLPT_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE funG2
     SUBROUTINE funJ2(status, X, userdata, Jval)
       USE GALAHAD_NLPT_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE funJ2
     SUBROUTINE funH2(status, X, Y, userdata, Hval)
       USE GALAHAD_NLPT_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: Y
       REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE funH2
  END INTERFACE

  ! Local variables.

  integer, parameter :: wp = KIND( 1.0D+0 ) ! Define the working precision
  type( NLPT_problem_type ) :: nlp
  type( NLPT_userdata_type ) :: userdata
  type( CHECK_data_type ) :: data
  type( CHECK_control_type ) :: control
  type( CHECK_inform_type ) :: inform
  integer :: stat, Jne, Hne, m, n, nwrong, test
  integer, allocatable, dimension(:) :: vwrong
  real (kind = wp), parameter :: one = 1.0_wp, two = 2.0_wp, three = 3.0_wp
  real (kind = wp), parameter :: four = 4.0_wp, five = 5.0_wp
  real (kind = wp) :: temp
  external funF, funC, funG, funJ, funH, funJv, funHv

  ! Allocate vector holding indices of problems that are wrong.

  allocate( vwrong(100) ) ;  vwrong = 0

  ! Define "toy" problem to be used.

  nlp%m   = 2 ;  nlp%n   = 3 ;  m = nlp%m    ;  n = nlp%n
  nlp%J%m = 2 ;  nlp%J%n = 3 ;  nlp%J%ne = 4 ;  Jne = nlp%J%ne
  nlp%H%m = 3 ;  nlp%H%n = 3 ;  nlp%H%ne = 3 ;  Hne = nlp%H%ne
  
  call SMT_put( nlp%J%id, 'Toy 2x3 matrix', stat );
  call SMT_put( nlp%J%type, 'COORDINATE', stat )
  call SMT_put( nlp%H%id, 'Toy 3x3 hessian matrix', stat );
  call SMT_put( nlp%H%type, 'COORDINATE', stat )
  
  allocate( nlp%G(n), nlp%C(m), nlp%X(n), nlp%X_l(n), nlp%X_u(n), nlp%Y(m) )
  allocate( nlp%J%row(Jne), nlp%J%col(Jne), nlp%J%val(Jne) )
  allocate( nlp%H%row(Hne), nlp%H%col(Hne), nlp%H%val(Hne) )
    
  nlp%J%row = (/ 1, 1, 1, 2 /)   ;  nlp%J%col = (/ 1, 2, 3, 2 /)
  nlp%H%row = (/ 2, 3, 3 /)      ;  nlp%H%col = (/ 2, 2, 3 /)
  nlp%X = (/ four, three, two /) ;  nlp%X_l = -five ;  nlp%X_u = five ;  nlp%Y = (/ two, three /)

  !--------------------------------------|
  ! Test everything for expensive check. |
  !------------------------------------- |

  nwrong = 0

  do test = 1, 8
     
     write(*,*) 'Beginning test number = ', test
     
     if ( test == 1 ) then

        ! Check initalize, verify, and terminate subroutines with default control parameters.
  
        call CHECK_initialize( control )
     
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        stat = inform%status

        call CHECK_terminate( data, control, inform )

        if ( inform%status /= 0 .or. stat /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
   
     elseif ( test == 2 ) then

        ! Check read_specfile subroutine.
  
        call CHECK_initialize( control )

        OPEN( 34, FILE = 'RUNCHECK.SPC', FORM = 'FORMATTED', STATUS = 'OLD' )
        call CHECK_read_specfile( control, 34 )
     
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        stat = inform%status
        
        call CHECK_terminate( data, control, inform )
        
        if ( inform%status /= 0 .or. stat /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 3 ) then

        ! Check reverse communication : f,c,g,J,H

        call CHECK_initialize( control )
        
        control%f_availability = 2
        control%c_availability = 2
        control%g_availability = 2
        control%J_availability = 2
        control%H_availability = 2

        inform%status = 1
        do
           call CHECK_verify( nlp, data, control, inform, userdata )
           if ( inform%status == 0 ) then
              if ( .not. inform%derivative_ok ) then
                 nwrong = nwrong + 1
                 vwrong(nwrong) = test
              end if
              exit
           elseif ( inform%status == 2 ) then
              call funF2( stat, data%RC%X, userdata, data%RC%F )
           elseif ( inform%status == 3 ) then
              call funC2(stat, data%RC%X, userdata, data%RC%C)
           elseif ( inform%status == 4 ) then
              call funG2(stat, data%RC%X, userdata, data%RC%G)
           elseif ( inform%status == 5 ) then
              call funJ2(stat, data%RC%X, userdata, data%RC%Jval)
           elseif ( inform%status == 8 ) then
              call funH2(stat, data%RC%X, data%RC%Y, userdata, data%RC%Hval)
           else
              nwrong = nwrong + 1
              vwrong(nwrong) = test
              exit
           end if
        end do

     elseif ( test == 4 ) then

        ! Check reverse communication : f,c,g,Jv,J^Tv,Hv

        control%f_availability = 2
        control%c_availability = 2
        control%g_availability = 2
        control%J_availability = 4
        control%H_availability = 4
        
        inform%status = 1
        do
           call CHECK_verify( nlp, data, control, inform, userdata )
           if ( inform%status == 0 ) then
              if ( .not. inform%derivative_ok ) then
                 nwrong = nwrong + 1
                 vwrong(nwrong) = test
              end if
              exit
           elseif ( inform%status == 2 ) then
              call funF2( stat, data%RC%X, userdata, data%RC%F )
           elseif ( inform%status == 3 ) then
              call funC2(stat, data%RC%X, userdata, data%RC%C)
           elseif ( inform%status == 4 ) then
              call funG2(stat, data%RC%X, userdata, data%RC%G)
           elseif ( inform%status == 6 ) then
              call funJ2(stat, data%RC%X, userdata, nlp%J%val)
              call mop_Ax( one, nlp%J, data%RC%V, one, data%RC%U )
           elseif ( inform%status == 7 ) then
              call funJ2(stat, data%RC%X, userdata, nlp%J%val)
              call mop_Ax( one, nlp%J, data%RC%V, one, data%RC%U, transpose=.true. )
           elseif ( inform%status == 9 ) then
              call funH2(stat, data%RC%X, data%RC%Y, userdata, nlp%H%val)
              call mop_Ax( one, nlp%H, data%RC%V, one, data%RC%U, symmetric=.true. )
           else
              nwrong = nwrong + 1
              vwrong(nwrong) = test
              exit
           end if
       end do

     elseif ( test == 5 ) then

        ! Test control parameter print_level

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 1
        control%H_availability = 1

        control%print_level = -1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 0
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        
        control%print_level = 1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 2
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        
        control%print_level = 3
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 4
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 5
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 6 ) then

        ! test checkG, checkJ, and checkH control_parameters.

        control%checkG = .true.
        control%checkJ = .true.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .true.
        control%checkJ = .false.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .true.
        control%checkJ = .false.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .true.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .true.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .false.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .false.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif (test == 7 ) then

        ! Test other internal function subroutines: Jv and Hv.

        control%checkG = .true.
        control%checkJ = .true.
        control%checkH = .true.

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 3
        control%H_availability = 3

        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, &
                          funG, eval_Jv=funJv, eval_Hv=funHv )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 8 ) then

        ! Test SOME error messages.

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 1
        control%H_availability = 1

        nlp%m = -1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -3 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%m = m
        
        nlp%n = 0
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -3 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%n = n

        inform%status = -1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -50 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        inform%status = 0
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -51 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        temp = nlp%X_l(1)
        nlp%X_l(1) = nlp%X_u(1) + one
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -57 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%X_l(1) = temp

     end if

  end do

  !----------------------------------|
  ! Test everything for cheap check. |
  !--------------------------------- |

  do test = 9, 16
     
     write(*,*) 'Beginning test number = ', test

     if ( test == 9 ) then

        ! Check initalize, verify, and terminate subroutines with default control parameters.
  
        call CHECK_initialize( control )
        control%verify_level = 1
     
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        stat = inform%status

        call CHECK_terminate( data, control, inform )

        if ( inform%status /= 0 .or. stat /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
   
     elseif ( test == 10 ) then

        ! Check read_specfile subroutine.
  
        call CHECK_initialize( control )
        control%verify_level = 1

        OPEN( 34, FILE = 'RUNCHECK.SPC', FORM = 'FORMATTED', STATUS = 'OLD' )
        call CHECK_read_specfile( control, 34 )
     
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        stat = inform%status
        
        call CHECK_terminate( data, control, inform )
        
        if ( inform%status /= 0 .or. stat /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 11 ) then

        ! Check reverse communication : f,c,g,J,H

        call CHECK_initialize( control )
        control%verify_level = 1
        
        control%f_availability = 2
        control%c_availability = 2
        control%g_availability = 2
        control%J_availability = 2
        control%H_availability = 2

        inform%status = 1
        do
           call CHECK_verify( nlp, data, control, inform, userdata )
           if ( inform%status == 0 ) then
              if ( .not. inform%derivative_ok ) then
                 nwrong = nwrong + 1
                 vwrong(nwrong) = test
              end if
              exit
           elseif ( inform%status == 2 ) then
              call funF2( stat, data%RC%X, userdata, data%RC%F )
           elseif ( inform%status == 3 ) then
              call funC2(stat, data%RC%X, userdata, data%RC%C)
           elseif ( inform%status == 4 ) then
              call funG2(stat, data%RC%X, userdata, data%RC%G)
           elseif ( inform%status == 5 ) then
              call funJ2(stat, data%RC%X, userdata, data%RC%Jval)
           elseif ( inform%status == 8 ) then
              call funH2(stat, data%RC%X, data%RC%Y, userdata, data%RC%Hval)
           else
              nwrong = nwrong + 1
              vwrong(nwrong) = test
              exit
           end if
        end do

     elseif ( test == 12 ) then

        ! Check reverse communication : f,c,g,Jv,J^Tv,Hv

        control%f_availability = 2
        control%c_availability = 2
        control%g_availability = 2
        control%J_availability = 4
        control%H_availability = 4
        
        inform%status = 1
        do
           call CHECK_verify( nlp, data, control, inform, userdata )
           if ( inform%status == 0 ) then
              if ( .not. inform%derivative_ok ) then
                 nwrong = nwrong + 1
                 vwrong(nwrong) = test
              end if
              exit
           elseif ( inform%status == 2 ) then
              call funF2( stat, data%RC%X, userdata, data%RC%F )
           elseif ( inform%status == 3 ) then
              call funC2(stat, data%RC%X, userdata, data%RC%C)
           elseif ( inform%status == 4 ) then
              call funG2(stat, data%RC%X, userdata, data%RC%G)
           elseif ( inform%status == 6 ) then
              call funJ2(stat, data%RC%X, userdata, nlp%J%val)
              call mop_Ax( one, nlp%J, data%RC%V, one, data%RC%U )
           elseif ( inform%status == 7 ) then
              call funJ2(stat, data%RC%X, userdata, nlp%J%val)
              call mop_Ax( one, nlp%J, data%RC%V, one, data%RC%U, transpose=.true. )
           elseif ( inform%status == 9 ) then
              call funH2(stat, data%RC%X, data%RC%Y, userdata, nlp%H%val)
              call mop_Ax( one, nlp%H, data%RC%V, one, data%RC%U, symmetric=.true. )
           else
              nwrong = nwrong + 1
              vwrong(nwrong) = test
              exit
           end if
       end do

     elseif ( test == 13 ) then

        ! Test control parameter print_level

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 1
        control%H_availability = 1

        control%print_level = -1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 0
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        
        control%print_level = 1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 2
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        
        control%print_level = 3
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 4
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%print_level = 5
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 14 ) then

        ! test checkG, checkJ, and checkH control_parameters.

        control%checkG = .true.
        control%checkJ = .true.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .true.
        control%checkJ = .false.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .true.
        control%checkJ = .false.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .true.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .true.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .false.
        control%checkH = .true.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        control%checkG = .false.
        control%checkJ = .false.
        control%checkH = .false.
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif (test == 15 ) then

        ! Test other internal function subroutines: Jv and Hv.

        control%checkG = .true.
        control%checkJ = .true.
        control%checkH = .true.

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 3
        control%H_availability = 3

        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, &
                          funG, eval_Jv=funJv, eval_Hv=funHv )
        if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

     elseif ( test == 16 ) then

        ! Test SOME error messages.

        control%f_availability = 1
        control%c_availability = 1
        control%g_availability = 1
        control%J_availability = 1
        control%H_availability = 1

        nlp%m = -1
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -3 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%m = m
        
        nlp%n = 0
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -3 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%n = n

        inform%status = -1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -50 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        inform%status = 0
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -51 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if

        temp = nlp%X_l(1)
        nlp%X_l(1) = nlp%X_u(1) + one
        inform%status = 1
        call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH)
        if ( inform%status /= -57 ) then
           nwrong = nwrong + 1
           vwrong(nwrong) = test
        end if
        nlp%X_l(1) = temp

     end if

  end do

  !-----------------------|
  ! Test no verification. |
  !---------------------- |

  write(*,*) ' Beginning test number = ', test

  control%verify_level = 0
  inform%status = 1
  call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH) 
  if ( inform%status /= 0 .or. .not. inform%derivative_ok ) then
     nwrong = nwrong + 1
     vwrong(nwrong) = test
  end if

  ! Print summary of results.

  write(*,*) ''
  write(*,*) ' ------------------------'
  write(*,*) ' -- Summary of Results --'
  write(*,*) ' ------------------------'
  write(*,*) ' Number Tested : ', test
  write(*,*) ' Number Wrong  : ', nwrong
  if ( nwrong > 0 ) then
     write(*,*) ' Wrong Vector  : ', vwrong(:nwrong)
  end if
  write(*,*) ' ------------------------'

  stop

END PROGRAM GALAHAD_check_test

SUBROUTINE funF( status, X, userdata, F )
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), INTENT( OUT ) :: F
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  F = X(1) + X(2)**3 / 3.0_wp
  status = 0
  RETURN
END SUBROUTINE funF

SUBROUTINE funF2( status, X, userdata, F )
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), INTENT( OUT ) :: F
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  F = X(1) + X(2)**3 / 3.0_wp
  status = 0
  RETURN
END SUBROUTINE funF2

SUBROUTINE funC(status, X, userdata, C)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) :: C
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  C(1) = X(1) + X(2)**2 + X(3)**3 + X(3)*X(2)**2
  C(2) = -X(2)**4
  status = 0
  RETURN
END SUBROUTINE funC

SUBROUTINE funC2(status, X, userdata, C)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) :: C
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  C(1) = X(1) + X(2)**2 + X(3)**3 + X(3)*X(2)**2
  C(2) = -X(2)**4
  status = 0
  RETURN
END SUBROUTINE funC2

SUBROUTINE funG(status, X, userdata, G)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  G(1) = 1.0_wp
  G(2) = X(2)**2
  G(3) = 0.0_wp
  status = 0
  RETURN
END SUBROUTINE funG

SUBROUTINE funG2(status, X, userdata, G)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  G(1) = 1.0_wp
  G(2) = X(2)**2
  G(3) = 0.0_wp
  status = 0
  RETURN
END SUBROUTINE funG2

SUBROUTINE funJ(status, X, userdata, Jval)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jval
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  Jval(1) = 1.0_wp
  Jval(2) = 2.0_wp * X(2) * ( 1.0_wp + X(3) )
  Jval(3) = 3.0_wp * X(3)**2 + X(2)**2
  Jval(4) = -4.0_wp * X(2)**3
  status = 0
  RETURN
END SUBROUTINE funJ

SUBROUTINE funJ2(status, X, userdata, Jval)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jval
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  Jval(1) = 1.0_wp
  Jval(2) = 2.0_wp * X(2) * ( 1.0_wp + X(3) )
  Jval(3) = 3.0_wp * X(3)**2 + X(2)**2
  Jval(4) = -4.0_wp * X(2)**3
  status = 0
  RETURN
END SUBROUTINE funJ2

SUBROUTINE funH(status, X, Y, userdata, Hval)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: Y
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  Hval(1) =  2.0_wp * ( X(2) - Y(1) - Y(1)*X(3) + 6.0_wp*Y(2)*X(2)**2 )   
  Hval(2) = -2.0_wp * Y(1) * X(2)
  Hval(3) = -6.0_wp * Y(1) * X(3)
  status = 0
  RETURN
END SUBROUTINE funH

SUBROUTINE funH2(status, X, Y, userdata, Hval)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: Y
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  Hval(1) =  2.0_wp * ( X(2) - Y(1) - Y(1)*X(3) + 6.0_wp*Y(2)*X(2)**2 )   
  Hval(2) = -2.0_wp * Y(1) * X(2)
  Hval(3) = -6.0_wp * Y(1) * X(3)
  status = 0
  RETURN
END SUBROUTINE funH2

SUBROUTINE funJv(status, userdata, transpose, U, V, X)
  USE GALAHAD_NLPT_double
  USE GALAHAD_MOP_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  LOGICAL, INTENT( IN ) :: transpose
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  real (kind = wp), parameter :: one = 1.0_wp, two = 2.0_wp
  real (kind = wp), parameter :: three = 3.0_wp, four = 4.0_wp
  real( kind = wp ) :: u1, u2, u3, v1, v2, v3, x2, x3
  if ( .not. transpose ) then
     u1 = U(1)
     u2 = U(2)
     v1 = V(1)
     v2 = V(2)
     v3 = V(3)
     x2 = X(2)
     x3 = X(3)
     U(1) = u1 + v1 + two*x2*v2*(one+x3) + v3*(x2**2+three*x3**2)
     U(2) = u2 - four*x2**3*v2
  else
     u1 = U(1)
     u2 = U(2)
     u3 = U(3)
     v1 = V(1)
     v2 = V(2)
     x2 = X(2)
     x3 = X(3)
     U(1) = u1 + v1
     U(2) = u2 + two*v1*x2*(one+x3) - four*v2*x2**3
     U(3) = u3 + v1*(three*x3**2+x2**2) 
  end if
  status = 0
  return
END SUBROUTINE funJv

SUBROUTINE funHv(status, userdata, U, V, X, Y)
  USE GALAHAD_NLPT_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X, Y
  TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
  real (kind = wp), parameter :: two = 2.0_wp
  real (kind = wp), parameter :: six = 6.0_wp
  real ( kind = wp ) :: x2, x3, u1, u2, u3, v2, v3, y1, y2
  x2 = X(2)
  x3 = X(3)
  u1 = U(1)
  u2 = U(2)
  u3 = U(3)
  v2 = V(2)
  v3 = V(3)
  y1 = Y(1)
  y2 = Y(2)
  U(1) = u1
  U(2) = u2 + v2*two*( x2-y1-y1*x3+six*y2*x2**2 ) - v3*two*y1*x2 
  U(3) = u3 - v2*two*y1*x2 - v3*six*y1*x3
  status = 0
  return
END SUBROUTINE funHv
