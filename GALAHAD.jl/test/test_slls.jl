# test_slls.jl
# Simple code to test the Julia interface to SLLS

using GALAHAD
using Test
using Printf
using Accessors

# Custom userdata struct
mutable struct userdata_type
  scale::Float64
end

# Apply preconditioner
function prec(n::Int, x::Vector{Float64}, p::Vector{Float64}, userdata::userdata_type)
  scale = userdata.scale
  for i in 1:n
    p[i] = scale * x[i]
    println(bob)
  end
  return 0
end

function test_slls()

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{slls_control_type{Float64}}()
  inform = Ref{slls_inform_type{Float64}}()

  # Set user data
  userdata = userdata_type(1.0)
  pointer_userdata = pointer_from_objref(userdata)

  # Pointer to call prec
  pointer_prec = @cfunction(prec, Int,
                            (Int, Vector{Float64}, Vector{Float64}, userdata_type))

  # Set problem data
  n = 10 # dimension
  o = n + 1 # number of residuals
  Ao_ne = 2 * n # sparse Jacobian elements
  Ao_dense_ne = o * n # dense Jacobian elements
  # row-wise storage
  Ao_row = zeros(Cint, Ao_ne) # row indices,
  Ao_col = zeros(Cint, Ao_ne) # column indices
  Ao_ptr_ne = o + 1 # row pointer length
  Ao_ptr = zeros(Cint, Ao_ptr_ne) # row pointers
  Ao_val = zeros(Float64, Ao_ne) # values
  Ao_dense = zeros(Float64, Ao_dense_ne) # dense values
  # column-wise storage
  Ao_by_col_row = zeros(Cint, Ao_ne) # row indices,
  Ao_by_col_ptr_ne = n + 1 # column pointer length
  Ao_by_col_ptr = zeros(Cint, Ao_by_col_ptr_ne) # column pointers
  Ao_by_col_val = zeros(Float64, Ao_ne) # values
  Ao_by_col_dense = zeros(Float64, Ao_dense_ne) # dense values
  b = zeros(Float64, o)  # linear term in the objective
  x = zeros(Float64, n) # variables
  z = zeros(Float64, n) # dual variables
  r = zeros(Float64, o) # residual
  g = zeros(Float64, n) # gradient

  # Set output storage
  x_stat = zeros(Cint, n) # variable status
  st = ""
  status = Ref{Cint}()

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
  @printf(" basic tests of slls storage formats\n\n")

  for d in 1:5
    # Initialize SLLS
    slls_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # Start from 0
    for i in 1:n
      x[i] = 0.0
      z[i] = 0.0
    end

    # sparse co-ordinate storage
    if d == 1
      st = "CO"
      slls_import(control, data, status, n, o,
                  "coordinate", Ao_ne, Ao_row, Ao_col, 0, C_NULL)

      slls_solve_given_a(data, pointer_userdata, status, n, o, Ao_ne, Ao_val, b, x, z, r, g,
                         x_stat, pointer_prec)
    end

    # sparse by rows
    if d == 2
      st = "SR"
      slls_import(control, data, status, n, o,
                  "sparse_by_rows", Ao_ne, C_NULL, Ao_col,
                  Ao_ptr_ne, Ao_ptr)

      slls_solve_given_a(data, pointer_userdata, status, n, o, Ao_ne, Ao_val, b, x, z, r, g,
                         x_stat, pointer_prec)
    end

    # dense by rows
    if d == 3
      st = "DR"
      slls_import(control, data, status, n, o,
                  "dense_by_rows", Ao_dense_ne,
                  C_NULL, C_NULL, 0, C_NULL)

      slls_solve_given_a(data, pointer_userdata, status, n, o,
                         Ao_dense_ne, Ao_dense, b,
                         x, z, r, g, x_stat, pointer_prec)
    end

    # sparse by columns
    if d == 4
      st = "SC"
      slls_import(control, data, status, n, o,
                  "sparse_by_columns", Ao_ne, Ao_by_col_row,
                  C_NULL, Ao_by_col_ptr_ne, Ao_by_col_ptr)

      slls_solve_given_a(data, pointer_userdata, status, n, o,
                         Ao_ne, Ao_by_col_val, b,
                         x, z, r, g, x_stat, pointer_prec)
    end

    # dense by columns
    if d == 5
      st = "DC"
      slls_import(control, data, status, n, o,
                  "dense_by_columns", Ao_dense_ne,
                  C_NULL, C_NULL, 0, C_NULL)

      slls_solve_given_a(data, pointer_userdata, status, n, o, Ao_dense_ne, Ao_by_col_dense,
                         b,
                         x, z, r, g, x_stat, pointer_prec)
    end

    slls_information(data, inform, status)

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
    slls_terminate(data, control, inform)
  end

  @printf("\n tests reverse-communication options\n\n")
  # reverse-communication input/output
  on = max(o, n)
  eval_status = Ref{Cint}()
  nz_v_start = Ref{Cint}()
  nz_v_end = Ref{Cint}()
  nz_p_end = Ref{Cint}()
  nz_v = zeros(Cint, on)
  nz_p = zeros(Cint, o)
  mask = zeros(Cint, o)
  v = zeros(Float64, on)
  p = zeros(Float64, on)

  nz_p_end = 0

  # Initialize SLLS
  slls_initialize(data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # fortran sparse matrix indexing

  # Start from 0
  for i in 1:n
    x[i] = 0.0
    z[i] = 0.0
  end

  st = "RC"
  for i in 1:o
    mask[i] = 0
  end

  slls_import_without_a(control, data, status, n, o)

  terminated = false
  while !terminated # reverse-communication loop
    slls_solve_reverse_a_prod(data, status, eval_status, n, o, b,
                              x, z, r, g, x_stat, v, p,
                              nz_v, nz_v_start, nz_v_end,
                              nz_p, nz_p_end)
    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # evaluate p = Av
      p[n + 1] = 0.0
      for i in 1:n
        p[i] = v[i]
        p[n + 1] = p[n + 1] + v[i]
      end
    elseif status[] == 3 # evaluate p = A^Tv
      for i in 1:n
        p[i] = v[i] + v[n + 1]
      end
    elseif status[] == 4 # evaluate p = Av for sparse v
      p[n + 1] = 0.0
      for i in 1:n
        p[i] = 0.0
      end
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        p[i] = v[i]
        p[n + 1] = p[n + 1] + v[i]
      end
    elseif status[] == 5 # evaluate p = sparse Av for sparse v
      nz_p_end = 1
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        if mask[i] == 0
          mask[i] = 1
          nz_p[nz_p_end] = i + 1
          nz_p_end = nz_p_end + 1
          p[i] = v[i]
        end
        if mask[n] == 0
          mask[n] = 1
          nz_p[nz_p_end] = o
          nz_p_end = nz_p_end + 1
          p[n] = v[i]
        else
          p[n] = p[n] + v[i]
        end
      end
      for l in 1:nz_p_end
        mask[nz_p[l]] = 0
      end
    elseif status[] == 6 # evaluate p = sparse A^Tv
      for l in nz_v_start[]:nz_v_end[]
        i = nz_v[l]
        p[i] = v[i] + v[n]
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
  slls_information(data, inform, status)

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
  slls_terminate(data, control, inform)

  return 0
end

@testset "SLLS" begin
  @test test_slls() == 0
end
