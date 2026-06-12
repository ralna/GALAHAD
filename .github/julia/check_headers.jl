using Test

global n = 0

folder_headers = joinpath(@__DIR__, "..", "..", "include")
path_galahad_h = joinpath(folder_headers, "galahad.h")
content = read(path_galahad_h, String)

excluded_headers = ["cutest_routines.h", "galahad_blas.h", "galahad_c.h",
                    "galahad_callbacks.h", "galahad_cfunctions.h", "galahad_double.h", "galahad_elgrra.h",
                    "galahad_icfs.h", "galahad_kinds.h", "galahad_lapack.h", "galahad_modules.h",
                    "galahad_precision.h", "galahad_python.h", "galahad_quadruple.h", "galahad_single.h"]

for file in readdir(folder_headers)
  !endswith(file, ".h") && continue
  startswith(file, "galahad_pquad_") && continue
  startswith(file, "galahad_c_") && continue
  startswith(file, "galahad_modules_") && continue
  startswith(file, "galahad_sls_") && continue
  startswith(file, "cutest_routines_") && continue
  startswith(file, "hsl_") && continue
  startswith(file, "ssids_") && continue
  (file in excluded_headers) && continue

  if !occursin(file, content)
    global n = n + 1
    println("The line")
    println("```")
    println("#include \"$file\"")
    println("```")
    println("is missing in GALAHAD/include/galahad.h.")
    println()
  end
end

if n > 0
  println("If some header files should not be included in galahad.h, please update")
  println("the variable `excluded_headers` in GALAHAD/.github/julia/check_headers.jl.")
  println()
end

@test n == 0
