! THIS VERSION: 2022-12-29 AT 10:40:00 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 5   M O D U L E  -*-*-*-

   module hsl_mc65_double

     USE GALAHAD_SYMBOLS
     USE hsl_zd11_double
     IMPLICIT none
     PRIVATE

     INTEGER, PARAMETER :: myint = KIND(1)
     INTEGER (KIND = myint), PUBLIC, PARAMETER :: MC65_ERR_MEMORY_ALLOC = - 1
     INTEGER (KIND = myint), PUBLIC, PARAMETER :: MC65_ERR_MEMORY_DEALLOC = - 2

     PUBLIC MC65_matrix_multiply, MC65_matrix_transpose, MC65_matrix_destruct

  contains

    subroutine MC65_matrix_multiply(matrix1,matrix2,result_matrix,info,stat)
    type (zd11_type), intent (in) :: matrix1,matrix2
    type (zd11_type), intent (inout) :: result_matrix
    integer (kind = myint), intent (out) :: info
    integer (kind = myint), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_multiply

    subroutine MC65_matrix_transpose(MATRIX1,MATRIX2,info,merge,pattern,stat)
    type (zd11_type), intent (in) :: MATRIX1
    type (zd11_type), intent (inout) :: MATRIX2
    integer (kind = myint), intent (out) :: info
    logical, intent (in), optional :: merge
    logical, intent (in), optional :: pattern
    integer (kind = myint), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_transpose

    subroutine MC65_matrix_destruct(matrix,info,stat)
    type (zd11_type), intent (inout) :: matrix
    integer (kind = myint), intent (out) :: info
    integer (kind = myint), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_destruct

   end module hsl_mc65_double
