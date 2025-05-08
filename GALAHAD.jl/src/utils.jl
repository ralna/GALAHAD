export galahad_linear_solver, run_sif

function galahad_linear_solver(solver::String)
  nchar = length(solver)
  (nchar > 30) && error("The solver can't have more than 30 characters.")

  result = zeros(Cchar, 31)
  for i in 1:nchar
    result[i] = Cchar(solver[i])
  end
  for i in nchar+1:30
    result[i] = Cchar(' ')
  end

  return ntuple(i -> result[i], 31)
end

"""
    run_sif(solver, precision, path_libsif::String, path_outsdif::String)
    run_sif(Val(solver), Val(precision), path_libsif::String, path_outsdif::String)

In both calls, `solver` and `precision` are symbols.
"""
function run_sif end

run_sif(solver::Symbol, precision::Symbol, path_libsif::String, path_outsdif::String) = run_sif(Val(solver), Val(precision), path_libsif, path_outsdif)

"""
    run_qplib(solver, precision, path_qplib::String)
    run_qplib(Val(solver), Val(precision), path_qplib::String)

In both calls, `solver` and `precision` are symbols.

The supported solvers are: `:bqp`, `:bqpb`, `:ccqp`, `:cdqp`, `:cqp`, `:dlp`, `:dqp`,
`:lpa`, `:lpb`, `:qp`, `:qpa`, `:qpb`, and `:qpc`.

The supported precisions are: `:single`, `:double`, and `:quadruple`.
"""
function run_qplib end

run_qplib(solver::Symbol, precision::Symbol, path_qplib::String) = run_qplib(Val(solver), Val(precision), path_qplib)
