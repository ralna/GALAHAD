using Test

global n = 0

function exported_modules(path::String)
  modules = String[]
  files = readdir(path)
  for file in files
    if endswith(file, ".mod")
      push!(modules, file)
    end
  end
  return modules
end

modules_single_int32    = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_single.so.p"))
modules_double_int32    = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_double.so.p"))
modules_quadruple_int32 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_quadruple.so.p"))
modules_single_int64    = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_single_64.so.p"))
modules_double_int64    = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_double_64.so.p"))
modules_quadruple_int64 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_quadruple_64.so.p"))

modules_combinations = [(modules_single_int32, modules_double_int32, 32, 32, "libgalahad_single.so and libgalahad_double.so"),
                        (modules_single_int32, modules_single_int64, 32, 64, "libgalahad_single.so and libgalahad_single_64.so"),
                        (modules_single_int32, modules_double_int64, 32, 64, "libgalahad_single.so and libgalahad_double_64.so"),
                        (modules_double_int32, modules_single_int64, 32, 64, "libgalahad_double.so and libgalahad_single_64.so"),
                        (modules_double_int32, modules_double_int64, 32, 64, "libgalahad_double.so and libgalahad_double_64.so"),
                        (modules_single_int64, modules_double_int64, 64, 64, "libgalahad_single_64.so and libgalahad_double_64.so"),
                        (modules_quadruple_int32, modules_single_int32, 32, 32, "libgalahad_quadruple.so and libgalahad_single.so"),
                        (modules_quadruple_int32, modules_single_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_single_64.so"),
                        (modules_quadruple_int32, modules_double_int32, 32, 32, "libgalahad_quadruple.so and libgalahad_double.so"),
                        (modules_quadruple_int32, modules_double_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_double_64.so"),
                        (modules_quadruple_int32, modules_quadruple_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_quadruple_64.so"),
                        (modules_quadruple_int64, modules_single_int32, 64, 32, "libgalahad_quadruple_64.so and libgalahad_single.so"),
                        (modules_quadruple_int64, modules_single_int64, 64, 64, "libgalahad_quadruple_64.so and libgalahad_single_64.so"),
                        (modules_quadruple_int64, modules_double_int32, 64, 32, "libgalahad_quadruple_64.so and libgalahad_double.so"),
                        (modules_quadruple_int64, modules_double_int64, 64, 64, "libgalahad_quadruple_64.so and libgalahad_double_64.so")]

single_double_quadruple_modules = ["galahad_blas_interface",
                                   "galahad_lapack_interface",
                                   "spral_ssids_blas_iface",
                                   "spral_ssids_lapack_iface",
                                   "galahad_clock",
                                   "galahad_string",
                                   "galahad_symbols",
                                   "galahad_tools",
                                   "galahad_copyright",
                                   "galahad_hash",
                                   "galahad_common_ciface",
                                   "galahad_hash_ciface",
                                   "galahad_version",
                                   "galahad_version_64",
                                   "galahad_version_ciface",
                                   "galahad_version_ciface_64",
                                   "gal_hsl_kinds",
                                   "gal_hsl_kinds_64",
                                   "gal_hsl_kb22_long_integer",
                                   "gal_hsl_mc68_integer",
                                   "gal_hsl_mc78_integer",
                                   "gal_hsl_of01_integer",
                                   "gal_hsl_zb01_integer",
                                   "gal_hsl_mc68_integer_ciface",
                                   "gal_hsl_mc68_integer_64_ciface",
                                   "galahad_kinds",
                                   "galahad_kinds_64",
                                   "spral_kinds",
                                   "mkl_pardiso",
                                   "mkl_pardiso_private",
                                   "lancelot_hsl_routines",
                                   "spmf_enums",
                                   "pastixf_enums",
                                   "spral_core_analyse",
                                   "spral_hw_topology",
                                   "spral_metis_wrapper",
                                   "spral_pgm",
                                   "spral_ssids_profile"]

for (modules1, modules2, int1, int2, name) in modules_combinations
  intersect_modules = intersect(modules1, modules2)
  println("---------------------------------------------------------------------------------------------------------------------------")
  println("The following modules are generated for both libraries $name:")
  for mod in intersect_modules
    flag1 = mapreduce(x -> (x * ".mod") == mod, |, single_double_quadruple_modules)
    flag2 = mapreduce(x -> (x * "_64.mod") == mod, |, single_double_quadruple_modules)
    if !flag1 && !flag2
      println(mod)
      global n = n+1
    end
  end
  println("---------------------------------------------------------------------------------------------------------------------------")
  println()
end

@test n == 0
