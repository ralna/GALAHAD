# test_rpd.jl
# Simple code to test the Julia interface to RPD

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_rpd(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{rpd_control_type{INT}}()
  inform = Ref{rpd_inform_type{INT}}()

  # extend the qplib_file string to include the actual position of the
  # provided ALLINIT.qplib example file provided as part GALAHAD
  qplib_file = joinpath(ENV["GALAHAD"], "examples", "ALLINIT.qplib")
  qplib_file_len = length(qplib_file)

  status = Ref{INT}()
  n = Ref{INT}()
  m = Ref{INT}()
  h_ne = Ref{INT}()
  a_ne = Ref{INT}()
  h_c_ne = Ref{INT}()
  p_type = Vector{Cchar}(undef, 4)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  # Initialize RPD */
  rpd_initialize(T, INT, data, control, status)

  # Set user-defined control options */
  @reset control[].f_indexing = true # fortran sparse matrix indexing

  # Recover vital statistics from the QPLIB file
  rpd_get_stats(T, INT, qplib_file, qplib_file_len, control, data, status, p_type, n, m, h_ne, a_ne,
                h_c_ne)
  @printf(" QPLIB file is of type %s\n", mapreduce(x -> Char(x), *, p_type))
  @printf(" n = %i, m = %i, h_ne = %i, a_ne = %i, h_c_ne = %i\n", n[], m[], h_ne[], a_ne[],
          h_c_ne[])

  # Recover g
  g = zeros(T, n[])
  rpd_get_g(T, INT, data, status, n[], g)
  @printf(" g = %.1f %.1f %.1f %.1f %.1f\n", g[1], g[2], g[3], g[4], g[5])

  # Recover f
  f = Ref{T}()
  rpd_get_f(T, INT, data, status, f)
  @printf(" f = %.1f\n", f[])

  # Recover xlu
  x_l = zeros(T, n[])
  x_u = zeros(T, n[])
  rpd_get_xlu(T, INT, data, status, n[], x_l, x_u)
  @printf(" x_l = %.1f %.1f %.1f %.1f %.1f\n", x_l[1], x_l[2], x_l[3], x_l[4], x_l[5])
  @printf(" x_u = %.1f %.1f %.1f %.1f %.1f\n", x_u[1], x_u[2], x_u[3], x_u[4], x_u[5])

  # Recover clu
  c_l = zeros(T, m[])
  c_u = zeros(T, m[])
  rpd_get_clu(T, INT, data, status, m[], c_l, c_u)
  @printf(" c_l = %.1f %.1f\n", c_l[1], c_l[2])
  @printf(" c_u = %.1f %.1f\n", c_u[1], c_u[2])

  # Recover H
  h_row = zeros(INT, h_ne[])
  h_col = zeros(INT, h_ne[])
  h_val = zeros(T, h_ne[])
  rpd_get_h(T, INT, data, status, h_ne[], h_row, h_col, h_val)
  @printf(" h_row, h_col, h_val =\n")
  for i in 1:h_ne[]
    @printf("   %i %i %.1f\n", h_row[i], h_col[i], h_val[i])
  end

  # Recover A
  a_row = zeros(INT, a_ne[])
  a_col = zeros(INT, a_ne[])
  a_val = zeros(T, a_ne[])
  rpd_get_a(T, INT, data, status, a_ne[], a_row, a_col, a_val)
  @printf(" a_row, a_col, a_val =\n")
  for i in 1:a_ne[]
    @printf("   %i %i %.1f\n", a_row[i], a_col[i], a_val[i])
  end

  # Recover H_c
  h_c_ptr = zeros(INT, h_c_ne[])
  h_c_row = zeros(INT, h_c_ne[])
  h_c_col = zeros(INT, h_c_ne[])
  h_c_val = zeros(T, h_c_ne[])
  rpd_get_h_c(T, INT, data, status, h_c_ne[], h_c_ptr, h_c_row, h_c_col, h_c_val)
  @printf(" h_c_row, h_c_col, h_c_val =\n")
  for i in 1:h_c_ne[]
    @printf("   %i %i %i %.1f\n", h_c_ptr[i], h_c_row[i], h_c_col[i], h_c_val[i])
  end

  # Recover x_type
  x_type = zeros(INT, n[])
  rpd_get_x_type(T, INT, data, status, n[], x_type)
  @printf(" x_type = %i %i %i %i %i\n", x_type[1], x_type[2], x_type[3], x_type[4],
          x_type[5])

  # Recover x
  x = zeros(T, n[])
  rpd_get_x(T, INT, data, status, n[], x)
  @printf(" x = %.1f %.1f %.1f %.1f %.1f\n", x[1], x[2], x[3], x[4], x[5])

  # Recover y
  y = zeros(T, m[])
  rpd_get_y(T, INT, data, status, m[], y)
  @printf(" y = %.1f %.1f\n", y[1], y[2])

  # Recover z
  z = zeros(T, n[])
  rpd_get_z(T, INT, data, status, n[], z)
  @printf(" z = %.1f %.1f %.1f %.1f %.1f\n", z[1], z[2], z[3], z[4], z[5])

  # Delete internal workspace
  rpd_terminate(T, INT, data, control, inform)
  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad) && haskey(ENV, "GALAHAD")
    @testset "RPD -- $T -- $INT" begin
      @test test_rpd(T, INT) == 0
    end
  end
end
