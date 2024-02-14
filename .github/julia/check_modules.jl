using Test

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

modules_single_int32 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_single.so.p"))
modules_double_int32 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_double.so.p"))
modules_single_int64 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_single.so.p"))
modules_double_int64 = exported_modules(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_double.so.p"))

modules_combinations = [(modules_single_int32, modules_double_int32, 32, 32, "libgalahad_single.so (Int32) and libgalahad_double.so (Int32)"),
                        (modules_single_int32, modules_single_int64, 32, 64, "libgalahad_single.so (Int32) and libgalahad_single.so (Int64)"),
                        (modules_single_int32, modules_double_int64, 32, 64, "libgalahad_single.so (Int32) and libgalahad_double.so (Int64)"),
                        (modules_double_int32, modules_single_int64, 32, 64, "libgalahad_double.so (Int32) and libgalahad_single.so (Int64)"),
                        (modules_double_int32, modules_double_int64, 32, 64, "libgalahad_double.so (Int32) and libgalahad_double.so (Int64)"),
                        (modules_double_int64, modules_double_int64, 64, 64, "libgalahad_double.so (Int64) and libgalahad_double.so (Int64)")]

single_double_modules = String[]

for (modules1, modules2, int1, int2, name) in modules_combinations
  intersect_modules = intersect(modules1, modules2)
  println("---------------------------------------------------------------------------------------------------------------------------")
  @warn("The following modules are generated for both libraries $name.")
  for mod in intersect_modules
    println(mod)
  end
  println("---------------------------------------------------------------------------------------------------------------------------")
  println()
end
