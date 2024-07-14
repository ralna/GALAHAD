# Script to parse HSL headers and generate Julia wrappers.
using Clang
using Clang.Generators
using JuliaFormatter

include("rewriter.jl")

function wrapper(name::String, headers::Vector{String}, optimized::Bool; targets=headers)

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

  (name == "all" || name == "arc")      && wrapper("arc", ["$galahad/galahad_arc.h"], optimized)
  (name == "all" || name == "bgo")      && wrapper("bgo", ["$galahad/galahad_bgo.h"], optimized)
  (name == "all" || name == "blls")     && wrapper("blls", ["$galahad/galahad_blls.h"], optimized)
  (name == "all" || name == "bllsb")    && wrapper("bllsb", ["$galahad/galahad_bllsb.h"], optimized)
  (name == "all" || name == "bnls")     && wrapper("bnls", ["$galahad/galahad_bnls.h"], optimized)
  (name == "all" || name == "bqp")      && wrapper("bqp", ["$galahad/galahad_bqp.h"], optimized)
  (name == "all" || name == "bqpb")     && wrapper("bqpb", ["$galahad/galahad_bqpb.h"], optimized)
  (name == "all" || name == "bsc")      && wrapper("bsc", ["$galahad/galahad_bsc.h"], optimized)
  (name == "all" || name == "ccqp")     && wrapper("ccqp", ["$galahad/galahad_ccqp.h"], optimized)
  (name == "all" || name == "clls")     && wrapper("clls", ["$galahad/galahad_clls.h"], optimized)
  (name == "all" || name == "convert")  && wrapper("convert", ["$galahad/galahad_convert.h"], optimized)
  (name == "all" || name == "cqp")      && wrapper("cqp", ["$galahad/galahad_cqp.h"], optimized)
  (name == "all" || name == "cro")      && wrapper("cro", ["$galahad/galahad_cro.h"], optimized)
  (name == "all" || name == "dgo")      && wrapper("dgo", ["$galahad/galahad_dgo.h"], optimized)
  (name == "all" || name == "dps")      && wrapper("dps", ["$galahad/galahad_dps.h"], optimized)
  (name == "all" || name == "dqp")      && wrapper("dqp", ["$galahad/galahad_dqp.h"], optimized)
  (name == "all" || name == "eqp")      && wrapper("eqp", ["$galahad/galahad_eqp.h"], optimized)
  (name == "all" || name == "fdc")      && wrapper("fdc", ["$galahad/galahad_fdc.h"], optimized)
  (name == "all" || name == "fit")      && wrapper("fit", ["$galahad/galahad_fit.h"], optimized)
  (name == "all" || name == "glrt")     && wrapper("glrt", ["$galahad/galahad_glrt.h"], optimized)
  (name == "all" || name == "gls")      && wrapper("gls", ["$galahad/galahad_gls.h"], optimized)
  (name == "all" || name == "gltr")     && wrapper("gltr", ["$galahad/galahad_gltr.h"], optimized)
  (name == "all" || name == "hash")     && wrapper("hash", ["$galahad/galahad_hash.h"], optimized)
  (name == "all" || name == "hsl")      && wrapper("hsl", ["$galahad/hsl_ma48.h", "$galahad/hsl_ma57.h", "$galahad/hsl_ma77.h", "$galahad/hsl_ma86.h", "$galahad/hsl_ma87.h", "$galahad/hsl_ma97.h", "$galahad/hsl_mc64.h", "$galahad/hsl_mc68.h", "$galahad/hsl_mi20.h", "$galahad/hsl_mi28.h"], optimized)
  (name == "all" || name == "ir")       && wrapper("ir", ["$galahad/galahad_ir.h"], optimized)
  (name == "all" || name == "l2rt")     && wrapper("l2rt", ["$galahad/galahad_l2rt.h"], optimized)
  (name == "all" || name == "lhs")      && wrapper("lhs", ["$galahad/galahad_lhs.h"], optimized)
  (name == "all" || name == "llsr")     && wrapper("llsr", ["$galahad/galahad_llsr.h"], optimized)
  (name == "all" || name == "llst")     && wrapper("llst", ["$galahad/galahad_llst.h"], optimized)
  (name == "all" || name == "lms")      && wrapper("lms", ["$galahad/galahad_lms.h"], optimized)
  (name == "all" || name == "lpa")      && wrapper("lpa", ["$galahad/galahad_lpa.h"], optimized)
  (name == "all" || name == "lpb")      && wrapper("lpb", ["$galahad/galahad_lpb.h"], optimized)
  (name == "all" || name == "lsqp")     && wrapper("lsqp", ["$galahad/galahad_lsqp.h"], optimized)
  (name == "all" || name == "lsrt")     && wrapper("lsrt", ["$galahad/galahad_lsrt.h"], optimized)
  (name == "all" || name == "lstr")     && wrapper("lstr", ["$galahad/galahad_lstr.h"], optimized)
  (name == "all" || name == "nls")      && wrapper("nls", ["$galahad/galahad_nls.h"], optimized)
  (name == "all" || name == "presolve") && wrapper("presolve", ["$galahad/galahad_presolve.h"], optimized)
  (name == "all" || name == "psls")     && wrapper("psls", ["$galahad/galahad_psls.h"], optimized)
  (name == "all" || name == "qpa")      && wrapper("qpa", ["$galahad/galahad_qpa.h"], optimized)
  (name == "all" || name == "qpb")      && wrapper("qpb", ["$galahad/galahad_qpb.h"], optimized)
  (name == "all" || name == "roots")    && wrapper("roots", ["$galahad/galahad_roots.h"], optimized)
  (name == "all" || name == "rpd")      && wrapper("rpd", ["$galahad/galahad_rpd.h"], optimized)
  (name == "all" || name == "rqs")      && wrapper("rqs", ["$galahad/galahad_rqs.h"], optimized)
  (name == "all" || name == "sbls")     && wrapper("sbls", ["$galahad/galahad_sbls.h"], optimized)
  (name == "all" || name == "scu")      && wrapper("scu", ["$galahad/galahad_scu.h"], optimized)
  (name == "all" || name == "sec")      && wrapper("sec", ["$galahad/galahad_sec.h"], optimized)
  (name == "all" || name == "sha")      && wrapper("sha", ["$galahad/galahad_sha.h"], optimized)
  (name == "all" || name == "sils")     && wrapper("sils", ["$galahad/galahad_sils.h"], optimized)
  (name == "all" || name == "slls")     && wrapper("slls", ["$galahad/galahad_slls.h"], optimized)
  (name == "all" || name == "sls")      && wrapper("sls", ["$galahad/galahad_sls.h"], optimized)
  (name == "all" || name == "ssids")    && wrapper("ssids", ["$galahad/spral_ssids.h"], optimized)
  (name == "all" || name == "trb")      && wrapper("trb", ["$galahad/galahad_trb.h"], optimized)
  (name == "all" || name == "trs")      && wrapper("trs", ["$galahad/galahad_trs.h"], optimized)
  (name == "all" || name == "tru")      && wrapper("tru", ["$galahad/galahad_tru.h"], optimized)
  (name == "all" || name == "ugo")      && wrapper("ugo", ["$galahad/galahad_ugo.h"], optimized)
  (name == "all" || name == "uls")      && wrapper("uls", ["$galahad/galahad_uls.h"], optimized)
  (name == "all" || name == "wcp")      && wrapper("wcp", ["$galahad/galahad_wcp.h"], optimized)
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
