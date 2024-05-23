using Test

function exported_symbols(path::String)
  symbols = String[]
  str = read(path, String)
  lines = split(str, '\n', keepempty=false)[2:end]
  for line in lines
    tab = split(line, " ", keepempty=false)
    symbol = tab[1]
    push!(symbols, symbol)
  end
  return symbols
end

symbols_single_int32 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_single.so.p", "libgalahad_single.so.symbols"))
symbols_double_int32 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_double.so.p", "libgalahad_double.so.symbols"))
symbols_single_int64 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_single_64.so.p", "libgalahad_single_64.so.symbols"))
symbols_double_int64 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_double_64.so.p", "libgalahad_double_64.so.symbols"))

symbols_combinations = [(symbols_single_int32, symbols_double_int32, 32, 32, "libgalahad_single.so and libgalahad_double.so"),
                        (symbols_single_int32, symbols_single_int64, 32, 64, "libgalahad_single.so and libgalahad_single_64.so"),
                        (symbols_single_int32, symbols_double_int64, 32, 64, "libgalahad_single.so and libgalahad_double_64.so"),
                        (symbols_double_int32, symbols_single_int64, 32, 64, "libgalahad_double.so and libgalahad_single_64.so"),
                        (symbols_double_int32, symbols_double_int64, 32, 64, "libgalahad_double.so and libgalahad_double_64.so"),
                        (symbols_single_int64, symbols_double_int64, 64, 64, "libgalahad_single_64.so and libgalahad_double_64.so")]

single_double_modules = ["hash_MOD", "string_MOD", "clock_MOD", "copyright_MOD", "symbols_MOD", "tools_MOD",
                         "common_ciface_MOD", "hash_ciface_MOD", "hsl_kb22_long_integer_MOD", "hsl_mc68_integer_ciface_MOD",
                         "hsl_mc68_integer_MOD", "hsl_mc78_integer_MOD", "hsl_zb01_integer_MOD", "galahad_hsl_of01_integer_MOD_",
                         "mkl_pardiso_private_MOD", "pastixf_enums_MOD", "pastixf_interfaces_MOD", "spmf_enums_MOD"]

for (symbols1, symbols2, int1, int2, name) in symbols_combinations
  intersect_symbols = intersect(symbols1, symbols2)
  println("---------------------------------------------------------------------------------------------------------------------------")
  println("The following symbols are exported by both the libraries $name:")
  for symbol in intersect_symbols
    flag1 = (startswith(symbol, "galahad_") || startswith(symbol, "cutest_")) && endswith(symbol, "_") && (int1 == int2 == 32)
    flag2 = (startswith(symbol, "galahad_") || startswith(symbol, "cutest_")) && endswith(symbol, "64_") && (int1 == int2 == 64)
    flag3 = mapreduce(x -> contains(symbol, x), |, single_double_modules) && (int1 == int2)
    if !flag1 && !flag2 && !flag3
      println(symbol)
    end
  end
  println("---------------------------------------------------------------------------------------------------------------------------")
  println()
end
