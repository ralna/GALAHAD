! THIS VERSION: GALAHAD 3.3 - 04/05/2021 AT 14:45 GMT.
PROGRAM GALAHAD_check_example
  USE GALAHAD_SMT_double   ! double precision version
  USE GALAHAD_USERDATA_double  ! double precision version
  USE GALAHAD_NLPT_double  ! double precision version
  USE GALAHAD_MOP_double   ! double precision version
  USE GALAHAD_CHECK_double ! double precision version
  IMPLICIT NONE
  integer, parameter :: wp = KIND( 1.0D+0 ) ! Define the working precision
  type( NLPT_problem_type ) :: nlp
  type( GALAHAD_userdata_type ) :: userdata
  type( CHECK_data_type ) :: data
  type( CHECK_control_type ) :: control
  type( CHECK_inform_type ) :: inform
  integer :: stat, Jne, Hne, m, n
  real (kind = wp), parameter :: one = 1.0_wp, two = 2.0_wp, three = 3.0_wp
  real (kind = wp), parameter :: four = 4.0_wp, five = 5.0_wp
  external funF, funC, funG, funJ, funH
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
  nlp%J%row = (/ 1, 1, 1, 2 /) ;  nlp%J%col = (/ 1, 2, 3, 2 /)
  nlp%H%row = (/ 2, 3, 3 /)    ;  nlp%H%col = (/ 2, 2, 3 /)
  nlp%X = (/ four, three, two /) ;  nlp%X_l = -five ;  nlp%X_u = five ;  nlp%Y = (/ two, three /)
  call CHECK_initialize( control ) ;   control%print_level = 3
  inform%status = 1
  call CHECK_verify( nlp, data, control, inform, userdata, funF, funC, funG, funJ, funH )
  call CHECK_terminate( data, control, inform )
END PROGRAM GALAHAD_check_example

SUBROUTINE funF( status, X, userdata, F )
  USE GALAHAD_USERDATA_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), INTENT( OUT ) :: F
  TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
  F = X(1) + X(2)**3 / 3.0_wp
  status = 0
  RETURN
END SUBROUTINE funF
SUBROUTINE funC(status, X, userdata, C)
  USE GALAHAD_USERDATA_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), INTENT( IN ), DIMENSION( : ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) :: C
  TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
  C(1) = X(1) + X(2)**2 + X(3)**3 + X(3)*X(2)**2
  C(2) = -X(2)**4
  status = 0
  RETURN
END SUBROUTINE funC
SUBROUTINE funG(status, X, userdata, G)
  USE GALAHAD_USERDATA_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
  TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
  G(1) = 1.0_wp
  G(2) = X(2)**2
  G(3) = 0.0_wp
  status = 0
  RETURN
END SUBROUTINE funG
SUBROUTINE funJ(status, X, userdata, Jval)
  USE GALAHAD_USERDATA_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Jval
  TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
  Jval(1) = 1.0_wp
  Jval(2) = 2.0_wp * X(2) * ( 1.0_wp + X(3) )
  Jval(3) = 3.0_wp * X(3)**2 + X(2)**2
  Jval(4) = -4.0_wp * X(2)**3
  status = 0
  RETURN
END SUBROUTINE funJ
SUBROUTINE funH(status, X, Y, userdata, Hval)
  USE GALAHAD_USERDATA_double
  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
  INTEGER, INTENT( OUT ) :: status
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
  REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: Y
  REAL ( kind = wp ), DIMENSION( : ), INTENT( OUT ) ::Hval
  TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
  Hval(1) =  2.0_wp * ( X(2) - Y(1) - Y(1)*X(3) + 6.0_wp*Y(2)*X(2)**2 )   
  Hval(2) = -2.0_wp * Y(1) * X(2)
  Hval(3) = -6.0_wp * Y(1) * X(3)
  status = 0
  RETURN
END SUBROUTINE funH
