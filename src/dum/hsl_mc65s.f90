! THIS VERSION: GALAHAD 4.3 - 2024-01-25 AT 09:30 GMT.

#include "galahad_hsl.h"

!-*-*-*-*-  G A L A H A D  -  D U M M Y   M C 6 5   M O D U L E  -*-*-*-

   module hsl_mc65_single

     USE GALAHAD_KINDS
     USE GALAHAD_SYMBOLS
     USE hsl_zd11_single
     IMPLICIT none
     PRIVATE

     INTEGER (KIND = ip_), PUBLIC, PARAMETER :: MC65_ERR_MEMORY_ALLOC = - 1
     INTEGER (KIND = ip_), PUBLIC, PARAMETER :: MC65_ERR_MEMORY_DEALLOC = - 2

     PUBLIC MC65_matrix_multiply, MC65_matrix_transpose, MC65_matrix_destruct

  contains

    subroutine MC65_matrix_multiply(matrix1,matrix2,result_matrix,info,stat)
    type (zd11_type), intent (in) :: matrix1,matrix2
    type (zd11_type), intent (inout) :: result_matrix
    integer (kind = ip_), intent (out) :: info
    integer (kind = ip_), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_multiply

    subroutine MC65_matrix_transpose(MATRIX1,MATRIX2,info,merge,pattern,stat)
    type (zd11_type), intent (in) :: MATRIX1
    type (zd11_type), intent (inout) :: MATRIX2
    integer (kind = ip_), intent (out) :: info
    logical(lp_), intent (in), optional :: merge
    logical(lp_), intent (in), optional :: pattern
    integer (kind = ip_), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_transpose

    subroutine MC65_matrix_destruct(matrix,info,stat)
    type (zd11_type), intent (inout) :: matrix
    integer (kind = ip_), intent (out) :: info
    integer (kind = ip_), optional, intent (out) :: stat
    info = GALAHAD_unavailable_option
    end subroutine MC65_matrix_destruct

   end module hsl_mc65_single
