# test_trs.jl
# Simple code to test the Julia interface to TRS

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = trs_control_type{Float64}()
inform = trs_inform_type{Float64}()

# Set problem data
n = 3  # dimension of H
m = 1  # dimension of A
H_ne = 4  # number of elements of H
M_ne = 3  # number of elements of M
A_ne = 3  # number of elements of A
H_dense_ne = 6  # number of elements of H
M_dense_ne = 6  # number of elements of M
H_row = Cint[1, 2, 3, 3]  # row indices, NB lower triangle
H_col = Cint[1, 2, 3, 1]
H_ptr = Cint[1, 2, 3, 5]
M_row = Cint[1, 2, 3]  # row indices, NB lower triangle
M_col = Cint[1, 2, 3]
M_ptr = Cint[1, 2, 3, 4]
A_row = Cint[1, 1, 1]
A_col = Cint[1, 2, 3]
A_ptr = Cint[1, 4]
H_val = Float64[1.0, 2.0, 3.0, 4.0]
M_val = Float64[1.0, 2.0, 1.0]
A_val = Float64[1.0, 1.0, 1.0]
H_dense = Float64[1.0, 0.0, 2.0, 4.0, 0.0, 3.0]
M_dense = Float64[1.0, 0.0, 2.0, 0.0, 0.0, 1.0]
H_diag = Float64[1.0, 0.0, 2.0]
M_diag = Float64[1.0, 2.0, 1.0]
f = 0.96
radius = 1.0
c = Float64[0.0, 2.0, 0.0]

st = ' '
status = Ref{Cint}()
x = zeros(Float64, n)
ma = NTuple{3, Cchar}((32, 32, 32))

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of storage formats\n\n")

for a_is = 0:1  # add a linear constraint?
  for m_is = 0:1  # include a scaling matrix?

    if (a_is == 1) && (m_is == 1)
      strcpy(ma, "MA")

    elseif (a_is == 1)
      strcpy(ma, "A ")

    elseif (m_is == 1)
      strcpy(ma, "M ")

    else
      strcpy(ma, "  ")
    end

    for storage_type = 1:4
      # Initialize TRS
      trs_initialize( data, control, status )

      # Set user-defined control options
      control.f_indexing = true  # fortran sparse matrix indexing

      # sparse co-ordinate storage
      if d == 1
        global st = 'C'

        # import the control parameters and structural data
        trs_import( control, data, status, n,
                    "coordinate", H_ne, H_row, H_col, Cint[] )

        if m_is == 1
          trs_import_m( data, status, n,
                        "coordinate", M_ne, M_row, M_col, Cint[] )
        end

        if a_is == 1
          trs_import_a( data, status, m,
                        "coordinate", A_ne, A_row, A_col, Cint[] )
        end

        # solve the problem
        if (a_is == 1) && (m_is == 1)
          trs_solve_problem( data, status, n,
                             radius, f, c, H_ne, H_val, x,
                             M_ne, M_val, m, A_ne, A_val, Cint[] )
        elseif a_is == 1
          trs_solve_problem( data, status, n,
                             radius, f, c, H_ne, H_val, x,
                             0, Cint[], m, A_ne, A_val, Cint[] )
        elseif m_is == 1
          trs_solve_problem( data, status, n,
                             radius, f, c, H_ne, H_val, x,
                             M_ne, M_val, 0, 0, Cint[], Cint[] )
        else
          trs_solve_problem( data, status, n,
                             radius, f, c, H_ne, H_val, x,
                             0, Cint[], 0, 0, Cint[], Cint[] )
        end
      end

      # sparse by rows
      if d == 2
        global st = 'R'

        # import the control parameters and structural data
        trs_import( control, data, status, n,
                    "sparse_by_rows", H_ne, Cint[], H_col, H_ptr )

        if m_is == 1
          trs_import_m( data, status, n,
                        "sparse_by_rows", M_ne, Cint[], M_col, M_ptr )
        end

        if a_is == 1
          trs_import_a( data, status, m,
                        "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )
        end

        # solve the problem
      if (a_is == 1) && (m_is == 1)
        trs_solve_problem( data, status, n,
                           radius, f, c, H_ne, H_val, x,
                           M_ne, M_val, m, A_ne, A_val, Cint[] )
      elseif a_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, H_ne, H_val, x,
                           0, Cint[], m, A_ne, A_val, Cint[] )
      elseif m_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, H_ne, H_val, x,
                           M_ne, M_val, 0, 0, Cint[], Cint[] )
      else
        trs_solve_problem( data, status, n,
                           radius, f, c, H_ne, H_val, x,
                           0, Cint[], 0, 0, Cint[], Cint[] )
      end
    end

    # dense
    if d == 3
      global st = 'D'

      # import the control parameters and structural data
      trs_import( control, data, status, n,
                  "dense", H_ne, Cint[], Cint[], Cint[] )

      if m_is == 1
        trs_import_m( data, status, n,
                      "dense", M_ne, Cint[], Cint[], Cint[] )
      end

      if a_is == 1
        trs_import_a( data, status, m,
                      "dense", A_ne, Cint[], Cint[], Cint[] )
      end

      # solve the problem
      if (a_is == 1) && (m_is == 1)
        trs_solve_problem( data, status, n,
                           radius, f, c, H_dense_ne, H_dense, x,
                           M_dense_ne, M_dense, m, A_ne, A_val, Cint[] )
      elseif a_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, H_dense_ne, H_dense, x,
                           0, Cint[], m, A_ne, A_val, Cint[] )
      elseif m_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, H_dense_ne, H_dense, x,
                           M_dense_ne, M_dense, 0, 0, Cint[], Cint[] )
      else
        trs_solve_problem( data, status, n,
                           radius, f, c, H_dense_ne, H_dense, x,
                           0, Cint[], 0, 0, Cint[], Cint[] )
      end
    end

    # diagonal
    if d == 4
      global st = 'L'

      # import the control parameters and structural data
      trs_import( control, data, status, n,
                  "diagonal", H_ne, Cint[], Cint[], Cint[] )

      if m_is == 1
        trs_import_m( data, status, n,
                      "diagonal", M_ne, Cint[], Cint[], Cint[] )
      end

      if a_is == 1
        trs_import_a( data, status, m,
                      "dense", A_ne, Cint[], Cint[], Cint[] )
      end

      # solve the problem
      if (a_is == 1) && (m_is == 1)
        trs_solve_problem( data, status, n,
                           radius, f, c, n, H_diag, x,
                           n, M_diag, m, A_ne, A_val, Cint[] )
      elseif a_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, n, H_diag, x,
                           0, Cint[], m, A_ne, A_val, Cint[] )
      elseif m_is == 1
        trs_solve_problem( data, status, n,
                           radius, f, c, n, H_diag, x,
                           n, M_diag, 0, 0, Cint[], Cint[] )
      else
        trs_solve_problem( data, status, n,
                           radius, f, c, n, H_diag, x,
                           0, Cint[], 0, 0, Cint[], Cint[] )
      end
    end

    trs_information( data, inform, status )

    @printf("format %c%s: TRS_solve_problem exit status = %1i, f = %.2f\n", st, ma, inform.status, inform.obj )
    # @printf("x: ")
    # for i = 1:n+m
    # @printf("%f ", x[i])
    # end

    # Delete internal workspace
    trs_terminate( data, control, inform )
  end
end
