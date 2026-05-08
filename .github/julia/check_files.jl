using Test

function meson_check_headers()
  println("--- Meson -- headers ---")
  n = 0

  folder_headers = joinpath(@__DIR__, "..", "..", "include")
  path_galahad_h = joinpath(folder_headers, "meson.build")
  path_galahad_jl = joinpath(@__DIR__, "..", "..", "GALAHAD.jl")
  content = read(path_galahad_h, String)

  excluded_headers = ["galahad_single.h", "galahad_double.h", "galahad_quadruple.h",
                      "hsl_kinds.h", "hsl_metis.h", "hsl_subset.h", "cutest_routines.h",
                      "galahad_kinds.h", "galahad_blas.h", "galahad_lapack.h",
                      "galahad_elgrra.h", "galahad_icfs.h", "amplinter.h"]

  for file in readdir(folder_headers)
    !endswith(file, ".h") && continue
    startswith(file, "galahad_pquad_") && continue
    startswith(file, "galahad_sls_") && continue
    startswith(file, "cutest_routines_") && continue
    startswith(file, "hsl_subset_") && continue
    startswith(file, "ssids_") && continue
    (file in excluded_headers) && continue

    if !occursin(file, content)
      n = n + 1
      println("Please add '$file' in the variable `galahad_headers` of GALAHAD/include/meson.build.")
      println()
    end
  end

  if n > 0
    println("If some header files should not be installed by Meson, please update")
    println("the variable `excluded_headers` in GALAHAD/.github/julia/check_meson.jl.")
    println()
  end

  return n
end

function meson_check_packages()
  println("--- Meson -- packages ---")
  m = 0

  excluded_packages = ["all_go", "all_qp", "all_rq", "apps", "general", "makedefs", "matlab"]

  root_galahad = joinpath(@__DIR__, "..", "..")
  path_packages = joinpath(@__DIR__, "..", "..", "src")
  path_meson_build = joinpath(root_galahad, "meson.build")
  content = read(path_meson_build, String)

  for package in readdir(path_packages)
    path_package = joinpath(path_packages, package)
    !isdir(path_package) && continue
    (package in excluded_packages) && continue

    if !occursin(package, content)
      m = m + 1
      println("Please add `subdir('src/$package')` in GALAHAD/meson.build.")
      path_meson_build = joinpath(path_package, "meson.build")
      if !isfile(path_meson_build)
        println("Please add a file `meson.build` in the folder GALAHAD/src/$package.")
      end
      println()
    end
  end

  if m > 0
    println("If some subfolders should not be explored by Meson, please update")
    println("the variable `excluded_packages` in GALAHAD/.github/julia/check_meson.jl.")
    println()
  end

  return m
end

function meson_check_files()
  println("--- Meson -- files ---")
  p = 0

  path_packages = joinpath(@__DIR__, "..", "..", "src")

  excluded_files = ["makemaster", "meson.build", "LICENCE", "README",
                    "README.rebuild", "README.external", "update-blas-lapack.sh", "rebuild.F90",
                    "trs_paper.F90", "trs_paper_large.F90", "rqs_paper_large.F90",
                    # I should check the following files with Nick!
                    "cdqp_ciface.F90", "check.f90.ver1", "dummy.f", "dummy_hsl.F90",
                    "dummy_hsl_c.F90", "dummy_spral.F90", "empty", "umfpack.F90",
                    "filter_orig.F90", "filtrane_ciface.F90", "glrt.f90.1", "glrtti.F90",
                    "glssbig.F90", "glssbig1.F90", "glssbig2.F90", "runl1qp_qplib.F90",
                    "lancelot.pointers.F90", "details", "details2", "runlsrb", "lsqr.F90",
                    "mop.d90.vers1", "nlst2.F90", "nodendti.F90", "bits", "orig", "pre",
                    "qp_spec.F90", "qp_test.F90", "qp_ciface.F90", "qpcs.f90.real",
                    "qpcs.f90.test", "qpc_ciface.F90", "shati.F90", "shat.c", "shatf.c",
                    "fa04a.f", "fa04ad.f", "ym01a.f", "ym01ad.f", "runsls_rb.F90"]

  for package in readdir(path_packages)
    path_local_package = joinpath(path_packages, package)
    !isdir(path_local_package) && continue
    local_meson_build = joinpath(path_local_package, "meson.build")
    if isfile(local_meson_build)
      local_content = read(local_meson_build, String)
      for file in readdir(path_local_package)
        path_file = joinpath(path_local_package, file)
        isdir(path_file) && continue
        endswith(file, ".meta") && continue
        endswith(file, ".template") && continue
        endswith(file, ".output") && continue
        endswith(file, ".SPC") && continue
        endswith(file, ".data") && continue
        occursin(".options.", file) && continue
        occursin(".data.", file) && continue
        (file in excluded_files) && continue

        if !occursin(file, local_content)
          p = p + 1
          println("Please add the file `$file` in GALAHAD/src/$package/meson.build.")
          println()
        end
      end

      path_C = joinpath(path_local_package, "C")
      if isdir(path_C)
        for file in readdir(path_C)
          endswith(file, ".SPC") && continue
          (file in excluded_files) && continue
          
          if !occursin(file, local_content)
            p = p + 1
            println("Please add the file `C/$file` in GALAHAD/src/$package/meson.build.")
            println()
          end
        end
      end

    end
  end

  if p > 0
    println("If some files should not be compiled by Meson, please update")
    println("the variable `excluded_files` in GALAHAD/.github/julia/check_meson.jl.")
    println()
  end

  return p
end

function julia_check_wrappers()
  println("--- Julia -- wrappers ---")
  q = 0

  path_packages = joinpath(@__DIR__, "..", "..", "src")
  path_wrappers_jl = joinpath(@__DIR__, "..", "..", "GALAHAD.jl", "src", "wrappers")

  for package in readdir(path_packages)
    (package == "dum") && continue
    (package == "common") && continue
    path_local_package = joinpath(path_packages, package)
    !isdir(path_local_package) && continue
    local_C_folder = joinpath(path_local_package, "C")
    if isdir(local_C_folder)
      path_julia_wrapper = joinpath(path_wrappers_jl, "$package.jl")
      if !isfile(path_julia_wrapper)
        q = q + 1
        println("Please add the file `$package.jl` in GALAHAD/GALAHAD.jl/src/wrappers.")
        println("Please read the file GALAHAD/GALAHAD.jl/gen/README.md for more details.")
        println()
      end
    end
  end

  path_galahad_julia = joinpath(@__DIR__, "..", "..", "GALAHAD.jl", "src", "GALAHAD.jl")
  local_content = read(path_galahad_julia, String)
  for file in readdir(path_wrappers_jl)
    if !occursin("include(\"wrappers/$file\")", local_content)
      q = q + 1
      println("Please add `include(\"wrappers/$file\")` in GALAHAD/GALAHAD.jl/src/GALAHAD.jl.")
      println()
    end
  end

  return q
end

function julia_check_tests()
  println("--- Julia -- tests ---")
  r = 0

  folder_headers = joinpath(@__DIR__, "..", "..", "include")
  path_packages = joinpath(@__DIR__, "..", "..", "src")
  path_tests_jl = joinpath(@__DIR__, "..", "..", "GALAHAD.jl", "test")

  for package in readdir(path_packages)
    (package == "dum") && continue
    (package == "common") && continue
    path_local_package = joinpath(path_packages, package)
    !isdir(path_local_package) && continue
    local_C_folder = joinpath(path_local_package, "C")
    package_header = joinpath(folder_headers, "galahad_$package.h")
    if isdir(local_C_folder) && isfile(package_header)
      path_julia_test = joinpath(path_tests_jl, "test_$package.jl")
      if !isfile(path_julia_test)
        r = r + 1
        println("Please add the file `test_$package.jl` in GALAHAD/GALAHAD.jl/test.")
        println()
      end
    end
  end

  path_runtests_julia = joinpath(@__DIR__, "..", "..", "GALAHAD.jl", "test", "runtests.jl")
  local_content = read(path_runtests_julia, String)
  for file in readdir(path_tests_jl)
    (file == "runtests.jl") && continue
    if !occursin("include(\"$file\")", local_content)
      r = r + 1
      println("Please add `include(\"$file\")` in GALAHAD/GALAHAD.jl/test/runtests.jl.")
      println()
    end
  end

  return r
end

n = meson_check_headers()
m = meson_check_packages()
p = meson_check_files()
q = julia_check_wrappers()
r = julia_check_tests()

@test m == 0
@test n == 0
@test p == 0
@test q == 0
@test r == 0
