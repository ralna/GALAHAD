using Test

global n = 0

function exported_symbols(path::String)
  symbols = String[]
  str = read(path, String)
  lines = split(str, '\n', keepempty=false)[2:end]
  for line in lines
    tab = split(line, " ", keepempty=false)
    symbol = tab[1]
    symbol = readchomp(`c++filt -t $symbol`)
    push!(symbols, symbol)
  end
  return symbols
end

symbols_single_int32    = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_single.so.p", "libgalahad_single.so.symbols"))
symbols_double_int32    = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_double.so.p", "libgalahad_double.so.symbols"))
symbols_quadruple_int32 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int32", "libgalahad_quadruple.so.p", "libgalahad_quadruple.so.symbols"))
symbols_single_int64    = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_single_64.so.p", "libgalahad_single_64.so.symbols"))
symbols_double_int64    = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_double_64.so.p", "libgalahad_double_64.so.symbols"))
symbols_quadruple_int64 = exported_symbols(joinpath(@__DIR__, "..", "..", "builddir_int64", "libgalahad_quadruple_64.so.p", "libgalahad_quadruple_64.so.symbols"))

symbols_combinations = [(symbols_single_int32, symbols_double_int32, 32, 32, "libgalahad_single.so and libgalahad_double.so"),
                        (symbols_single_int32, symbols_single_int64, 32, 64, "libgalahad_single.so and libgalahad_single_64.so"),
                        (symbols_single_int32, symbols_double_int64, 32, 64, "libgalahad_single.so and libgalahad_double_64.so"),
                        (symbols_double_int32, symbols_single_int64, 32, 64, "libgalahad_double.so and libgalahad_single_64.so"),
                        (symbols_double_int32, symbols_double_int64, 32, 64, "libgalahad_double.so and libgalahad_double_64.so"),
                        (symbols_single_int64, symbols_double_int64, 64, 64, "libgalahad_single_64.so and libgalahad_double_64.so"),
                        (symbols_quadruple_int32, symbols_single_int32, 32, 32, "libgalahad_quadruple.so and libgalahad_single.so"),
                        (symbols_quadruple_int32, symbols_single_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_single_64.so"),
                        (symbols_quadruple_int32, symbols_double_int32, 32, 32, "libgalahad_quadruple.so and libgalahad_double.so"),
                        (symbols_quadruple_int32, symbols_double_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_double_64.so"),
                        (symbols_quadruple_int32, symbols_quadruple_int64, 32, 64, "libgalahad_quadruple.so and libgalahad_quadruple_64.so"),
                        (symbols_quadruple_int64, symbols_single_int32, 64, 32, "libgalahad_quadruple_64.so and libgalahad_single.so"),
                        (symbols_quadruple_int64, symbols_single_int64, 64, 64, "libgalahad_quadruple_64.so and libgalahad_single_64.so"),
                        (symbols_quadruple_int64, symbols_double_int32, 64, 32, "libgalahad_quadruple_64.so and libgalahad_double.so"),
                        (symbols_quadruple_int64, symbols_double_int64, 64, 64, "libgalahad_quadruple_64.so and libgalahad_double_64.so")]

single_double_quadruple_modules = ["hash_MOD", "string_MOD", "string_64_MOD", "clock_MOD", "copyright_MOD", "symbols_MOD", "tools_MOD",
                                   "common_ciface_MOD", "hash_ciface_MOD", "hash_64_MOD", "hash_ciface_64_MOD", "hsl_kb22_long_integer_MOD",
                                   "hsl_mc68_integer_ciface_MOD", "hsl_mc68_integer_MOD", "hsl_mc78_integer_MOD", "hsl_mc78_integer_64_MOD", "hsl_zb01_integer_MOD",
                                   "hsl_of01_integer_MOD", "hsl_of01_integer_64_MOD", "mkl_pardiso_private_MOD", "pastixf_enums_MOD", "pastixf_interfaces_MOD", "spmf_enums_MOD",
                                   "spral_pgm_64_MOD", "pastixf_enums_64_MOD", "mkl_pardiso_private_64_MOD", "spmf_enums_64_MOD", "spral_hw_topology_64_MOD",
                                   "spral_metis_wrapper_64_MOD", "tools_64_MOD", "galahad_symbols_64_MOD", "hsl_mc68_integer_64_ciface_MOD", "hsl_kb22_long_integer_64_MOD",
                                   "common_ciface_64_MOD", "clock_64_MOD", "hsl_mc68_integer_64_MOD", "hsl_zb01_integer_64_MOD", "copyright_64_MOD",
                                   "version_64_MOD", "spral_hw_topology_MOD", "spral_pgm_MOD", "spral_metis_wrapper_MOD", "spral_core_analyse_64_MOD",
                                   "galahad_version_MOD", "spral_core_analyse_MOD", "spral_ssids_profile_64_MOD"]

metis_adapters = ["galahad_nodend4_adapter", "galahad_nodend51_adapter", "galahad_nodend52_adapter"]

metis_symbols = ["PrintBackTrace", "errexit", "getpathname", "iAllocMatrix", "iFreeMatrix", "iSetMatrix", "iargmax", "iargmax_n",
                 "iargmin", "iarray2csr", "iaxpy", "icopy", "idot", "iincset", "ikvAllocMatrix", "ikvFreeMatrix", "ikvSetMatrix",
                 "ikvcopy", "ikvmalloc", "ikvrealloc", "ikvset", "ikvsmalloc", "ikvsortd", "ikvsorti", "ikvsortii", "imalloc",
                 "imax", "imin", "inorm2", "ipqCheckHeap", "ipqCreate", "ipqDelete", "ipqDestroy", "ipqFree", "ipqGetTop", "ipqInit",
                 "ipqInsert", "ipqLength", "ipqReset", "ipqSeeKey", "ipqSeeTopKey", "ipqSeeTopVal", "ipqUpdate", "irandArrayPermuteFine",
                 "irandArrayPermute", "irandInRange", "irand", "irealloc", "iscale", "iset", "ismalloc", "isortd", "isorti", "isrand",
                 "isum", "rAllocMatrix", "rFreeMatrix", "rSetMatrix", "rargmax", "rargmax_n", "rargmin", "raxpy", "rcopy", "rdot",
                 "rincset", "rkvAllocMatrix", "rkvFreeMatrix", "rkvSetMatrix", "rkvcopy", "rkvmalloc", "rkvrealloc", "rkvset", "rkvsmalloc",
                 "rkvsortd", "rkvsorti", "rmalloc", "rmax", "rmin", "rnorm2", "rpqCheckHeap", "rpqCreate", "rpqDelete", "rpqDestroy",
                 "rpqFree", "rpqGetTop", "rpqInit", "rpqInsert", "rpqLength", "rpqReset", "rpqSeeKey", "rpqSeeTopKey", "rpqSeeTopVal",
                 "rpqUpdate", "rrealloc", "rscale", "rset", "rsmalloc", "rsortd", "rsorti", "rsum", "uvwsorti"]

unknown_symbols = ["errexit", "getpathname", "gkfooo", "main", "PrintBackTrace", "spral_hw_topology_free", "spral_hw_topology_guess",
                   "Test_ND", "VerifyND", "xerbla2_"]

for (symbols1, symbols2, int1, int2, name) in symbols_combinations
  intersect_symbols = intersect(symbols1, symbols2)
  println("---------------------------------------------------------------------------------------------------------------------------")
  println("The following symbols are exported by both the libraries $name:")
  for symbol in intersect_symbols
    flag1 = startswith(symbol, "galahad_") && endswith(symbol, "_") && (int1 == int2 == 32)
    flag2 = startswith(symbol, "galahad_") && endswith(symbol, "64_") && (int1 == int2 == 64)
    flag3 = startswith(symbol, "cutest_") && endswith(symbol, "_")
    flag4 = mapreduce(x -> contains(symbol, x), |, single_double_quadruple_modules) && (int1 == int2)
    flag5 = startswith(symbol, "gk_") || startswith(symbol, "galmetis__") || startswith(symbol, "libmetis__")
    flag6 = mapreduce(x -> startswith(symbol, x), |, ["galahad_pardiso", "galahad_pastix", "galahad_spm", "galahad_metis", "galahad_mpi", "galahad_wsmp", "galahad_mkl_pardiso", "galahad_ws"])
    flag7 = mapreduce(x -> symbol == x, |, ["fun_", "grad_", "hprod_", "jprod_", "hess_"]) || mapreduce(x -> startswith(symbol, x), |, ["elfun", "group", "range"])
    flag8 = mapreduce(x -> symbol == x, |, ["version_galahad", "METIS_Free", "METIS_NodeND", "METIS_SetDefaultOptions", "gal_kb07ai_"]) && (int1 == int2 == 32)
    flag9 = mapreduce(x -> symbol == x, |, ["version_galahad_64", "METIS_Free_64", "METIS_NodeND_64", "METIS_SetDefaultOptions_64", "gal_kb07ai_64_"]) && (int1 == int2 == 64)
    flag10 = mapreduce(x -> symbol == x, |, ["CoarsenGraphNlevels", "ComputeBFSOrdering", "Greedy_KWayEdgeCutOptimize", "Greedy_KWayEdgeStats", "GrowBisectionNode2"])
    flag11 = mapreduce(x -> symbol == x, |, unknown_symbols)
    flag12 = mapreduce(x -> symbol == x || symbol == x * "_64", |, metis_adapters)
    flag13 = mapreduce(x -> symbol == x * "_51" || symbol == x * "_51_64" || symbol == x * "_52" || symbol == x * "_52_64", |, metis_symbols)
    flag14 = mapreduce(x -> contains(symbol, x), |, ["::", "<", ">"]) && (int1 == int2)
    flag15 = contains(symbol, "std::") || contains(symbol, "__gnu_cxx::")
    flag16 = contains(symbol, "operator delete") || contains(symbol, "operator new")
    flag17 = contains(symbol, "spral_ssids_cpu_create_symbolic_subtree") || contains(symbol, "spral_ssids_cpu_destroy_symbolic_subtree")
    flag18 = contains(symbol, "spral_ssids_profile")
    if !flag1 && !flag2 && !flag3 && !flag4 && !flag5 && !flag6 && !flag7 && !flag8 && !flag9 && !flag10
      if !flag11 && !flag12 && !flag13 && !flag14 && !flag15 && !flag16 && !flag17 && !flag18
        println(symbol)
        global n = n+1
      end
    end
  end
  println("---------------------------------------------------------------------------------------------------------------------------")
  println()
end

@test n == 0
