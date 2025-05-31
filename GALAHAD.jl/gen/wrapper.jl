# Script to parse HSL headers and generate Julia wrappers.
using Clang
using Clang.Generators
using JuliaFormatter

include("rewriter.jl")

function run_sif_wrapper(name::String, precision::String)
str = "const run$(name)_sif_$(precision) = joinpath(galahad_bindir, \"run$(name)_sif_$(precision)\$(exeext)\")

function run_sif(::Val{:$name}, ::Val{:$precision}, path_libsif::String, path_outsdif::String)
  run(`\$run$(name)_sif_$precision \$path_libsif \$path_outsdif`)
end
"
return str
end

function run_qplib_wrapper(name::String, precision::String)
str = "const run$(name)_qplib_$(precision) = joinpath(galahad_bindir, \"run$(name)_qplib_$(precision)\$(exeext)\")

function run_qplib(::Val{:$name}, ::Val{:$precision}, path_qplib::String)
  open(path_qplib, \"r\") do io
    process = pipeline(`\$run$(name)_qplib_$precision`, stdin=io)
    run(process)
  end
end
"
return str
end

function wrapper(name::String, headers::Vector{String}, optimized::Bool; targets=headers, run_sif::Bool=false, run_qplib::Bool=false)

  @info "Wrapping $name"
  cd(@__DIR__)

  include_dir = joinpath(ENV["GALAHAD"], "include")
  options = load_options(joinpath(@__DIR__, "galahad.toml"))
  options["general"]["library_name"] = "libgalahad_double"
  # options["general"]["extract_c_comment_style"] = "doxygen"
  options["general"]["output_file_path"] = joinpath("..", "src", "wrappers", "$(name).jl")
  optimized && (options["general"]["output_ignorelist"] = ["real_wp_", "real_sp_", "rpc_", "ipc_"])
  args = get_default_args()
  push!(args, "-I$include_dir")
  push!(args, "-DGALAHAD_DOUBLE")

  if !isempty(headers)
    ctx = create_context(headers, args, options)
    build!(ctx, BUILDSTAGE_NO_PRINTING)

    # Only keep the wrapped headers because the dependencies are already wrapped with other headers.
    replace!(get_nodes(ctx.dag)) do node
        path = Clang.get_filename(node.cursor)
        should_wrap = any(targets) do target
            occursin(target, path)
        end
        if !should_wrap
            return ExprNode(node.id, Generators.Skip(), node.cursor, Expr[], node.adj)
        end
        return node
    end

    build!(ctx, BUILDSTAGE_PRINTING_ONLY)

    path = options["general"]["output_file_path"]
    rewrite!(path, name, optimized)
    format_file(path, YASStyle(), indent=2)
  end

  if run_sif
    path = options["general"]["output_file_path"]
    text = !isempty(headers) ? read(path, String) * "\n" : ""
    text = text * run_sif_wrapper(name, "single") * "\n\n"
    text = text * run_sif_wrapper(name, "double") * "\n\n"
    text = text * run_sif_wrapper(name, "quadruple")
    write(path, text)
    format_file(path, YASStyle(), indent=2)
  end

  if run_qplib
    path = options["general"]["output_file_path"]
    text = read(path, String) * "\n"
    text = text * run_qplib_wrapper(name, "single") * "\n\n"
    text = text * run_qplib_wrapper(name, "double") * "\n\n"
    text = text * run_qplib_wrapper(name, "quadruple")
    write(path, text)
    format_file(path, YASStyle(), indent=2)
  end

  # Generate a symbolic link for the Julia wrappers
  if (name ≠ "hsl") && (name ≠ "ssids")
    current_folder = pwd()
    cd("../../src/$name")
    !isdir("Julia") && mkdir("Julia")
    cd("Julia")
    rm("$name.jl", force=true)
    symlink("../../../GALAHAD.jl/src/wrappers/$name.jl", "$name.jl")
    cd(current_folder)
  end

  return nothing
end

function main(name::String="all"; optimized::Bool=true)
  haskey(ENV, "GALAHAD") || error("The environment variable GALAHAD is not defined.")
  galahad = joinpath(ENV["GALAHAD"], "include")

  # Regenerate test_structures.jl
  (name == "all") && optimized && isfile("../test/test_structures.jl") && rm("../test/test_structures.jl")

  (name == "all" || name == "arc")      && wrapper("arc", ["$galahad/galahad_arc.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "bgo")      && wrapper("bgo", ["$galahad/galahad_bgo.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "blls")     && wrapper("blls", ["$galahad/galahad_blls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "bllsb")    && wrapper("bllsb", ["$galahad/galahad_bllsb.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "bnls")     && wrapper("bnls", ["$galahad/galahad_bnls.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "bqp")      && wrapper("bqp", ["$galahad/galahad_bqp.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "bqpb")     && wrapper("bqpb", ["$galahad/galahad_bqpb.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "bsc")      && wrapper("bsc", ["$galahad/galahad_bsc.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "ccqp")     && wrapper("ccqp", ["$galahad/galahad_ccqp.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "cdqp")     && wrapper("cdqp", String[], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "clls")     && wrapper("clls", ["$galahad/galahad_clls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "convert")  && wrapper("convert", ["$galahad/galahad_convert.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "cqp")      && wrapper("cqp", ["$galahad/galahad_cqp.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "cro")      && wrapper("cro", ["$galahad/galahad_cro.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "demo")     && wrapper("demo", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "dgo")      && wrapper("dgo", ["$galahad/galahad_dgo.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "dlp")      && wrapper("dlp", String[], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "dps")      && wrapper("dps", ["$galahad/galahad_dps.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "dqp")      && wrapper("dqp", ["$galahad/galahad_dqp.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "eqp")      && wrapper("eqp", ["$galahad/galahad_eqp.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "fdc")      && wrapper("fdc", ["$galahad/galahad_fdc.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "fdh")      && wrapper("fdh", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "filtrane") && wrapper("filtrane", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "fit")      && wrapper("fit", ["$galahad/galahad_fit.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "glrt")     && wrapper("glrt", ["$galahad/galahad_glrt.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "gls")      && wrapper("gls", ["$galahad/galahad_gls.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "gltr")     && wrapper("gltr", ["$galahad/galahad_gltr.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "hash")     && wrapper("hash", ["$galahad/galahad_hash.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "hsl")      && wrapper("hsl", ["$galahad/hsl_ma48.h", "$galahad/hsl_ma57.h", "$galahad/hsl_ma77.h", "$galahad/hsl_ma86.h", "$galahad/hsl_ma87.h", "$galahad/hsl_ma97.h", "$galahad/hsl_mc64.h", "$galahad/hsl_mc68.h", "$galahad/hsl_mi20.h", "$galahad/hsl_mi28.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "ir")       && wrapper("ir", ["$galahad/galahad_ir.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "l1qp")     && wrapper("l1qp", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "l2rt")     && wrapper("l2rt", ["$galahad/galahad_l2rt.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lancelot") && wrapper("lancelot", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lhs")      && wrapper("lhs", ["$galahad/galahad_lhs.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "lls")      && wrapper("lls", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "llsr")     && wrapper("llsr", ["$galahad/galahad_llsr.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "llst")     && wrapper("llst", ["$galahad/galahad_llst.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "lms")      && wrapper("lms", ["$galahad/galahad_lms.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "lpa")      && wrapper("lpa", ["$galahad/galahad_lpa.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "lpb")      && wrapper("lpb", ["$galahad/galahad_lpb.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "lpqp")     && wrapper("lpqp", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lqr")      && wrapper("lqr", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lqt")      && wrapper("lqt", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lsqp")     && wrapper("lsqp", ["$galahad/galahad_lsqp.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "lsrt")     && wrapper("lsrt", ["$galahad/galahad_lsrt.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "lstr")     && wrapper("lstr", ["$galahad/galahad_lstr.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "miqr")     && wrapper("miqr", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "nls")      && wrapper("nls", ["$galahad/galahad_nls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "nodend")   && wrapper("nodend", ["$galahad/galahad_nodend.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "presolve") && wrapper("presolve", ["$galahad/galahad_presolve.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "psls")     && wrapper("psls", ["$galahad/galahad_psls.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "qp")       && wrapper("qp", String[], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "qpa")      && wrapper("qpa", ["$galahad/galahad_qpa.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "qpb")      && wrapper("qpb", ["$galahad/galahad_qpb.h"], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "qpc")      && wrapper("qpc", String[], optimized, run_sif=true, run_qplib=true)
  (name == "all" || name == "roots")    && wrapper("roots", ["$galahad/galahad_roots.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "rpd")      && wrapper("rpd", ["$galahad/galahad_rpd.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "rqs")      && wrapper("rqs", ["$galahad/galahad_rqs.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "sbls")     && wrapper("sbls", ["$galahad/galahad_sbls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "scu")      && wrapper("scu", ["$galahad/galahad_scu.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "sec")      && wrapper("sec", ["$galahad/galahad_sec.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "sha")      && wrapper("sha", ["$galahad/galahad_sha.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "sils")     && wrapper("sils", ["$galahad/galahad_sils.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "slls")     && wrapper("slls", ["$galahad/galahad_slls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "sls")      && wrapper("sls", ["$galahad/galahad_sls.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "ssids")    && wrapper("ssids", ["$galahad/spral_ssids.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "trb")      && wrapper("trb", ["$galahad/galahad_trb.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "trs")      && wrapper("trs", ["$galahad/galahad_trs.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "tru")      && wrapper("tru", ["$galahad/galahad_tru.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "ugo")      && wrapper("ugo", ["$galahad/galahad_ugo.h"], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "uls")      && wrapper("uls", ["$galahad/galahad_uls.h"], optimized, run_sif=false, run_qplib=false)
  (name == "all" || name == "warm")     && wrapper("warm", String[], optimized, run_sif=true, run_qplib=false)
  (name == "all" || name == "wcp")      && wrapper("wcp", ["$galahad/galahad_wcp.h"], optimized, run_sif=true, run_qplib=false)
end

# If we want to use the file as a script with `julia wrapper.jl`
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

# galahad = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "..", "include")
# headers = readdir(galahad)
# for header in sort(headers)
#   res = split(header, ['_', '.'])
#   println("(name == \"all\" || name == \"$(res[2])\") && wrapper(\"$(res[2])\", [\"\$galahad/$(header)\"], optimized)")
# end
