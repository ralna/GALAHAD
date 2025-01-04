# test_slls.jl
# Simple code to test the Julia interface to SLLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_slls{T}
  scale::T
end

function test_slls(::Type{T}, ::Type{INT}) where {T,INT}

  # Apply preconditioner
  function prec(n::INT, x::Vector{T}, p::Vector{T}, userdata::userdata_slls)
    scale = userdata.scale
    for i in 1:n
      p[i] = scale * x[i]
    end
    return 0
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{slls_control_type{T,INT}}()
  inform = Ref{slls_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_slls(1.0)

  # Set problem data
  n = INT(10)  # dimension
  o = n + INT(1)  # number of residuals
  Ao_ne = 2 * n # sparse Jacobian elements
  Ao_dense_ne = o * n # dense Jacobian elements
  # row-wise storage
  Ao_row = zeros(INT, Ao_ne) # row indices,
  Ao_col = zeros(INT, Ao_ne) # column indices
  Ao_ptr_ne = o + INT(1)  # row pointer length
  Ao_ptr = zeros(INT, Ao_ptr_ne) # row pointers
  Ao_val = zeros(T, Ao_ne) # values
  Ao_dense = zeros(T, Ao_dense_ne) # dense values
  # column-wise storage
  Ao_by_col_row = zeros(INT, Ao_ne) # row indices,
  Ao_by_col_ptr_ne = n + INT(1)  # column pointer length
  Ao_by_col_ptr = zeros(INT, Ao_by_col_ptr_ne) # column pointers
  Ao_by_col_val = zeros(T, Ao_ne) # values
  Ao_by_col_dense = zeros(T, Ao_dense_ne) # dense values
  b = zeros(T, o)  # linear term in the objective
  x = zeros(T, n) # variables
  z = zeros(T, n) # dual variables
  r = zeros(T, o) # residual
  g = zeros(T, n) # gradient

  # Set output storage
  x_stat = zeros(INT, n) # variable status
  st = ""
  status = Ref{INT}()

  #   A = ( I )  and b = (i * e)
  #       (e^T)          (n + 1)
  for i in 1:n
    b[i] = i
  end
  b[n + 1] = n + 1

  # A by rows
  for i in 1:n
    Ao_ptr[i] = i
    Ao_row[i] = i
    Ao_col[i] = i
    Ao_val[i] = 1.0
  end
  Ao_ptr[n + 1] = n + 1
  for i in 1:n
    Ao_row[n + i] = o
    Ao_col[n + i] = i
    Ao_val[n + i] = 1.0
  end
  Ao_ptr[o + 1] = Ao_ne + 1
  l = 0
  for i in 1:n
    for j in 1:n
      l = l + 1
      if i == j
        Ao_dense[l] = 1.0
      else
        Ao_dense[l] = 0.0
      end
    end
  end
  for j in 1:n
    l = l + 1
    Ao_dense[l] = 1.0
  end

  # # A by columns
  l = 0
  for j in 1:n
    l = l + 1
    Ao_by_col_ptr[j] = l
    Ao_by_col_row[l] = j
    Ao_by_col_val[l] = 1.0
    l = l + 1
    Ao_by_col_row[l] = o
    Ao_by_col_val[l] = 1.0
  end
  Ao_by_col_ptr[n + 1] = Ao_ne
  l = 0
  for j in 1:n
    for i in 1:n
      l = l + 1
      if i == j
        Ao_by_col_dense[l] = 1.0
      else
        Ao_by_col_dense[l] = 0.0
      end
    end
    l = l + 1
    Ao_by_col_dense[l] = 1.0
  end

  @printf(" fortran sparse matrix indexing\n\n")
  @printf(" tests reverse-communication options\n\n")

  # reverse-communication input/output
  on = max(o, n)
  eval_status = Ref{INT}()
  nz_v_start = Ref{INT}()
  nz_v_end = Ref{INT}()
  nz_p_end = Ref{INT}()
  nz_v = zeros(INT, on)
  nz_p = zeros(INT, o)
  mask = zeros(INT, o)
  v = zeros(T, on)
  p = zeros(T, on)
  nz_p_end = INT(1)

  # Initialize SLLS
  slls_initialize(T, INT, data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true  # fortran sparse matrix indexing

  # Start from 0
  for i in 1:n
    x[i] = 0.0
    z[i] = 0.0
  end

  st = "RC"
  for i in 1:o
    mask[i] = 0
  end

  slls_import_without_a(T, INT, control, data, status, n, o)

  terminated = false
  while !terminated # reverse-communication loop
    slls_solve_reverse_a_prod(T, INT, data, status, eval_status, n, o, b,
                              x, z, r, g, x_stat, v, p,
                              nz_v, nz_v_start, nz_v_end,
                              nz_p, nz_p_end)
    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # evaluate p = Av
      p[o] = 0.0
      for i in 1:n
        p[i] = v[i]
        p[o] = p[o] + v[i]
      end
    elseif status[] == 3 # evaluate p = A^Tv
      for i in 1:n
        p[i] = v[i] + v[o]
      end
    elseif status[] == 4 # evaluate p = Av for sparse v
      for i in 1:o
        p[i] = 0.0
      end
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        p[i] = v[i]
        p[o] = p[o] + v[i]
      end
    elseif status[] == 5 # evaluate p = sparse Av for sparse v
      nz_p_end = 0
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        nz_p_end = nz_p_end + 1
        nz_p[nz_p_end] = i
        p[i] = v[i]
        if mask[i] == 0
          mask[i] = 1
          nz_p_end = nz_p_end + 1
          nz_p[nz_p_end] = o
          p[o] = v[i]
        else
          p[o] = p[o] + v[i]
        end
      end
      for l in 1:nz_p_end
        mask[nz_p[l]] = 0
      end
    elseif status[] == 6 # evaluate p = sparse A^Tv
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        p[i] = v[i] + v[o]
      end
    elseif status[] == 7 # evaluate p = P^{-}v
      for i in 1:n
        p[i] = userdata.scale * v[i]
      end
    else
      @printf(" the value %1i of status should not occur\n", status)
    end

    eval_status[] = 0
  end

  # Record solution information
  slls_information(T, INT, data, inform, status)

  # Print solution details
  if inform[].status == 0
    @printf("%s:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
            st, inform[].iter, inform[].obj, inform[].status)
  else
    @printf("%s: SLLS_solve exit status = %1i\n", st, inform[].status)
  end

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end
  # @printf("\n")
  # @printf("gradient: ")
  # for i = 1:n
  #   @printf("%f ", g[i])
  # end
  # @printf("\n")

  # Delete internal workspace
  slls_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SLLS -- $T -- $INT" begin
      @test test_slls(T, INT) == 0
    end
  end
end
