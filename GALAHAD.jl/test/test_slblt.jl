# test_slblt.jl
# Simple code to test the Julia interface to SLBLT

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_slblt(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  control = Ref{slblt_control_type{T,INT}}()
  inform = Ref{slblt_inform_type{T,INT}}()

  # Initialize derived types
  akeep = Ref{Ptr{Cvoid}}(C_NULL)
  fkeep = Ref{Ptr{Cvoid}}(C_NULL) # Important that these are C_NULL to start with

  slblt_default_control(T, INT, control)
  @reset control[].array_base = INT(1)  # Fortran sparse matrix indexing
  @reset control[].nodend_control.print_level = INT(0)

  @printf(" Fortran sparse matrix indexing\n\n")

  # Data for matrix:
  # ( 2  1         )
  # ( 1  4  1    1 )
  # (    1  3  2   )
  # (       2 -1   )
  # (    1       2 )
  posdef = false
  n = INT(5)
  ptr = Int64[1, 3, 6, 8, 9, 10]
  row = INT[1, 2, 2, 3, 5, 3, 4, 4, 5]
  val = T[2.0, 1.0, 4.0, 1.0, 1.0, 3.0, 2.0, -1.0, 2.0]

  # The right-hand side with solution (1.0, 2.0, 3.0, 4.0, 5.0)
  x = T[4.0, 17.0, 19.0, 2.0, 12.0]

  # perform analysis and factorization with data checking
  check = true
  slblt_analyse(T, INT, check, n, C_NULL, ptr, row, C_NULL, akeep, control, inform)
  if inform[].flag < 0
    slblt_free(T, INT, akeep, fkeep)
    error("[SLBLT] The analysis failed!")
  end
  slblt_factor(T, INT, posdef, C_NULL, C_NULL, val, C_NULL, akeep[], fkeep, control,
                     inform)
  if inform[].flag < 0
    slblt_free(T, INT, akeep, fkeep)
    error("[SLBLT] The factorization failed!")
  end

  # solve
  slblt_solve1(T, INT, INT(0), x, akeep[], fkeep[], control, inform)
  if inform[].flag < 0
    slblt_free(T, INT, akeep, fkeep)
    error("[SLBLT] The solve failed!")
  end

  @printf("The computed solution is:")
  for i in 1:n
    @printf(" %9.2e", x[i])
  end
  @printf("\n")

  # Determine and print the pivot order
  piv_order = zeros(INT, 5)
  slblt_enquire_indef(T, INT, akeep[], fkeep[], control, inform, piv_order, C_NULL)
  @printf("Pivot order:")
  for i in 1:n
    @printf(" %3d", piv_order[i])
  end
  @printf("\n")

  # Delete internal workspace
  flag = slblt_free(T, INT, akeep, fkeep)
  (flag != 0) && error("[SLBLT] Error while calling slblt_free.")

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SLBLT -- $T -- $INT" begin
      @test test_slblt(T, INT) == 0
    end
  end
end
