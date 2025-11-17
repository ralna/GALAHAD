export galahad_linear_solver, run_sif, run_qplib

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

function setup_env_lbt(command::Cmd)
  config = LinearAlgebra.BLAS.lbt_get_config()
  lbt_default_libs = get(ENV, "LBT_DEFAULT_LIBS", join([lib.libname for lib in config.loaded_libs], ";"))
  command = addenv(command, "LBT_DEFAULT_LIBS" => lbt_default_libs)
  return command
end

"""
    run_sif(solver::Symbol, precision::Symbol, path_libsif::String, path_outsdif::String)

If `CUTEst.jl` is installed, you can also provide the path to the `SIF` file, and decoding
and compilation will be handled automatically for the user:

    run_sif(solver::Symbol, precision::Symbol, path_sif::String)

The supported solvers are: `:arc`, `:bgo`, `:blls`, `:bllsb`, `:bnls`,
`:bqp`, `:bqpb`, `:ccqp`, `:cdqp`, `:clls`, `:cqp`, `:demo`, `:dgo`, `:dlp`,
`:dps`, `:dqp`, `:eqp`, `:expo`, `:fdh`, `:filtrane`, `:glrt`, `:gltr`, `:l1qp`,
`:l2rt`, `:lancelot`, `:lls`, `:lpa`, `:lpb`, `:lpqp`, `:lqr`, `:lqt`, `:lsrt`,
`:lstr`, `:miqr`, `:nls`, `:nodend`, `:presolve`, `:qp`, `:qpa`, `:qpb`, `:qpc`,
`:rqs`, `:sbls`, `:sha`, `:sils`, `:slls`, `:sls`, `:ssls`, `:trb`, `:trek`,
`:trs`, `:tru`, `:ugo`, `:warm`, and `:wcp`.

The supported precisions are: `:single` and `:double`.
"""
function run_sif end

run_sif(solver::Symbol, precision::Symbol, path_libsif::String, path_outsdif::String) = run_sif(Val(solver), Val(precision), path_libsif, path_outsdif)

"""
    run_qplib(solver::Symbol, precision::Symbol, path_qplib::String)

The supported solvers are: `:bqp`, `:bqpb`, `:ccqp`, `:cdqp`, `:cqp`,
`:dlp`, `:dqp`, `:lpa`, `:lpb`, `:qp`, `:qpa`, `:qpb`, and `:qpc`.

The supported precisions are: `:single` and `:double`
"""
function run_qplib end

run_qplib(solver::Symbol, precision::Symbol, path_qplib::String) = run_qplib(Val(solver), Val(precision), path_qplib)
