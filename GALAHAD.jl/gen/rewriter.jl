packages = ("arc", "bgo", "blls", "bllsb", "bqp", "bqpb", "bsc", "ccqp",
            "clls", "convert", "cqp", "cro", "dgo", "dps", "dqp", "eqp",
            "fdc", "fit", "glrt", "gls", "gltr", "hash", "ir", "l2rt", "lhs",
            "llsr", "llst", "lms", "lpa", "lpb", "lsqp", "lsrt", "lstr",
            "nls", "presolve", "psls", "qpa", "qpb", "roots", "rpd", "rqs",
            "sbls", "scu", "sec", "sha", "sils", "slls", "sls", "trb", "trs",
            "tru", "ugo", "uls", "wcp", "bnls")

types = ("control", "time", "inform", "history", "subproblem_control", "subproblem_inform", "ainfo", "finfo", "sinfo")

nonparametric_structures_int = ("arc_time_type", "bgo_time_type", "blls_time_type", "bllsb_time_type",
                                "bnls_time_type", "bqp_time_type", "bqpb_time_type", "ccqp_time_type",
                                "clls_time_type", "convert_time_type", "cqp_time_type", "cro_time_type",
                                "dgo_time_type", "dps_time_type", "dqp_time_type", "eqp_time_type",
                                "fdc_time_type", "llsr_time_type", "llsr_history_type", "llst_time_type",
                                "llst_history_type", "lms_time_type", "lpa_time_type", "lpb_time_type",
                                "lsqp_time_type", "nls_time_type", "psls_time_type", "qpa_time_type",
                                "qpb_time_type", "rqs_time_type", "rqs_history_type", "sbls_time_type",
                                "scu_control_type", "slls_time_type", "sls_time_type", "trb_time_type",
                                "trs_time_type", "trs_history_type", "tru_time_type", "ugo_time_type",
                                "wcp_time_type")

nonparametric_structures_float = ("bqp_time_type", "bsc_control_type", "convert_control_type", "fit_control_type",
                                  "fit_inform_type", "gls_sinfo_type", "hash_control_type", "hash_inform_type",
                                  "lhs_control_type", "lhs_inform_type", "lms_control_type", "ma48_sinfo",
                                  "mc64_control", "mc64_info", "mc68_control", "mc68_info", "presolve_inform_type",
                                  "roots_inform_type", "rpd_control_type", "rpd_inform_type", "scu_control_type",
                                  "scu_inform_type", "sec_inform_type", "sha_control_type", "slls_time_type",
                                  "spral_ssids_inform")

# Structures that don't have a field with rpc_ but have an inner structure with rpc_ as a field.
special_structures_float = ("convert_inform_type", "cro_inform_type", "lms_inform_type", "ugo_inform_type", "uls_inform_type")

function rewrite!(path::String, name::String, optimized::Bool)
  structures = "# Structures for $name\n"
  text = read(path, String)
  if optimized
    text = replace(text, "hsl_longc_" => "Int64")
    text = replace(text, "longc_" => "Int64")
    text = replace(text, "real_sp_" => "Float32")
    text = replace(text, "\n    " => "\n  ")

    # GALAHAD
    for type in types
      for package in packages
        type_name = "$(package)_$(type)_type"
        if (type_name ∉ nonparametric_structures_float) && (type_name ∈ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){T}")
          text = replace(text, ", $(type_name)}" => ", $(type_name){T}}")
        end
        if (type_name ∈ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){INT}")
          text = replace(text, ", $(type_name)}" => ", $(type_name){INT}}")
        end
        if (type_name ∉ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){T,INT}")
          text = replace(text, ", $(type_name)}" => ", $(type_name){T,INT}}")
        end
      end
    end

    # HSL
    for type in ("control", "solve_control", "info", "ainfo", "finfo", "sinfo")
      text = replace(text, "$(type)_d" => "$(type)")
      text = replace(text, "$(type)_i" => "$(type)")
      for hsl in ("ma48", "ma57", "ma77", "ma86", "ma87", "ma97", "mc64", "mc68", "mi20", "mi28")
        type_name = "$(hsl)_$(type)"
        if (type_name ∉ nonparametric_structures_float) && (type_name ∈ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){T}")
        end
        if (type_name ∈ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){INT}")
        end
        if (type_name ∉ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
          text = replace(text, "::$(type_name)" => "::$(type_name){T,INT}")
        end
      end
    end

    # SSIDS
    text = replace(text, "::spral_ssids_options" => "::spral_ssids_options{T,INT}")
    text = replace(text, "::spral_ssids_inform" => "::spral_ssids_inform{INT}")

    blocks = split(text, "end\n")
    text = ""
    for (index, code) in enumerate(blocks)
      if contains(code, "function")
        fname = split(split(code, "function ")[2], "(")[1]

        # Int32
        routine_single_int32 = code * "end\n"
        routine_double_int32 = code * "end\n"
        routine_quadruple_int32 = code * "end\n"

        routine_single_int32 = replace(routine_single_int32, "function $fname(" => "function $fname(::Type{Float32}, ::Type{Int32}, ")
        routine_double_int32 = replace(routine_double_int32, "function $fname(" => "function $fname(::Type{Float64}, ::Type{Int32}, ")
        routine_quadruple_int32 = replace(routine_quadruple_int32, "function $fname(" => "function $fname(::Type{Float128}, ::Type{Int32}, ")

        routine_single_int32 = replace(routine_single_int32, "libgalahad_double" => "libgalahad_single")
        routine_quadruple_int32 = replace(routine_quadruple_int32, "libgalahad_double" => "libgalahad_quadruple")

        # Uncomment if one day we compile GALAHAD with -DMULTIPRECISION
        # routine_single_int32 = replace(routine_single_int32, "libgalahad_double.$fname(" => "libgalahad_single.$(fname)_s(")
        # routine_quadruple_int32 = replace(routine_quadruple_int32, "libgalahad_double.$fname(" => "libgalahad_quadruple.$(fname)_q(")

        routine_single_int32 = replace(routine_single_int32, "ipc_" => "Int32")
        routine_double_int32 = replace(routine_double_int32, "ipc_" => "Int32")
        routine_quadruple_int32 = replace(routine_quadruple_int32, "ipc_" => "Int32")

        routine_single_int32 = replace(routine_single_int32, "rpc_" => "Float32")
        routine_double_int32 = replace(routine_double_int32, "rpc_" => "Float64")
        routine_quadruple_int32 = replace(routine_quadruple_int32, "rpc_" => "Float128")

        routine_single_int32 = replace(routine_single_int32, "spral_ssids_options" => "spral_ssids_options{Float32}")
        routine_double_int32 = replace(routine_double_int32, "spral_ssids_options" => "spral_ssids_options{Float64}")
        routine_quadruple_int32 = replace(routine_quadruple_int32, "spral_ssids_options" => "spral_ssids_options{Float128}")

        # Float128 should be passed by value as a Cfloat128
        routine_quadruple_int32 = replace(routine_quadruple_int32, "::Float128" => "::Cfloat128")

        for type in types
          for package in packages
            type_name = "$(package)_$(type)_type"
            if (type_name ∉ nonparametric_structures_float) && (type_name ∈ nonparametric_structures_int)
              routine_single_int32 = replace(routine_single_int32, type_name => "$(type_name){Float32}")
              routine_double_int32 = replace(routine_double_int32, type_name => "$(type_name){Float64}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, type_name => "$(type_name){Float128}")

              routine_single_int32 = replace(routine_single_int32, "{Float32}{Float32}" => "{Float32}")
              routine_double_int32 = replace(routine_double_int32, "{Float64}{Float64}" => "{Float64}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, "{Float128}{Float128}" => "{Float128}")
            end
            if (type_name ∈ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
              routine_single_int32 = replace(routine_single_int32, type_name => "$(type_name){Int32}")
              routine_double_int32 = replace(routine_double_int32, type_name => "$(type_name){Int32}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, type_name => "$(type_name){Int32}")

              routine_single_int32 = replace(routine_single_int32, "{Int32}{Int32}" => "{Int32}")
              routine_double_int32 = replace(routine_double_int32, "{Int32}{Int32}" => "{Int32}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, "{Int32}{Int32}" => "{Int32}")
            end
            if (type_name ∉ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
              routine_single_int32 = replace(routine_single_int32, type_name => "$(type_name){Float32,Int32}")
              routine_double_int32 = replace(routine_double_int32, type_name => "$(type_name){Float64,Int32}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, type_name => "$(type_name){Float128,Int32}")

              routine_single_int32 = replace(routine_single_int32, "{Float32,Int32}{Float32,Int32}" => "{Float32,Int32}")
              routine_double_int32 = replace(routine_double_int32, "{Float64,Int32}{Float64,Int32}" => "{Float64,Int32}")
              routine_quadruple_int32 = replace(routine_quadruple_int32, "{Float128,Int32}{Float128,Int32}" => "{Float128,Int32}")
            end
          end
        end

        # Int64
        routine_single_int64 = routine_single_int32
        routine_double_int64 = routine_double_int32
        routine_quadruple_int64 = routine_quadruple_int32

        routine_single_int64 = replace(routine_single_int64, "libgalahad_single" => "libgalahad_single_64")
        routine_double_int64 = replace(routine_double_int64, "libgalahad_double" => "libgalahad_double_64")
        routine_quadruple_int64 = replace(routine_quadruple_int64, "libgalahad_quadruple" => "libgalahad_quadruple_64")

        # Uncomment if one day we compile GALAHAD with -DMULTIPRECISION
        # routine_single_int64 = replace(routine_single_int64, "libgalahad_single.$(fname)_s(" => "libgalahad_single_64.$(fname)_s_64(")
        # routine_double_int64 = replace(routine_double_int64, "libgalahad_double.$(fname)(" => "libgalahad_double_64.$(fname)_64(")
        # routine_quadruple_int64 = replace(routine_quadruple_int64, "libgalahad_quadruple.$(fname)_q(" => "libgalahad_quadruple_64.$(fname)_q_64(")

        routine_single_int64 = replace(routine_single_int64, "Int32" => "Int64")
        routine_double_int64 = replace(routine_double_int64, "Int32" => "Int64")
        routine_quadruple_int64 = replace(routine_quadruple_int64, "Int32" => "Int64")

        if (name ≠ "hsl") && (name ≠ "ssids")
          text = text * "\n" * "export " * fname * "\n" * routine_single_int32 * "\n" * routine_single_int64 * "\n" *
                                                          routine_double_int32 * "\n" * routine_double_int64 * "\n" *
                                                          routine_quadruple_int32 * "\n" * routine_quadruple_int64
        end
      elseif contains(code, "struct ")
        structure = code * "end\n"
        structure_name = split(split(code, "struct ")[2], "\n")[1]
        contains(structure, "rpc_") && (structure_name ∈ nonparametric_structures_float) && error("$structure_name should not be in nonparametric_structures_float.")
        !contains(structure, "rpc_") && (structure_name ∉ nonparametric_structures_float) && (structure_name ∉ special_structures_float) && error("$structure_name should be in nonparametric_structures_float.")
        contains(structure, "ipc_") && (structure_name ∈ nonparametric_structures_int) && error("$structure_name should not be in nonparametric_structures_int.")
        !contains(structure, "ipc_") && (structure_name ∉ nonparametric_structures_int) && error("$structure_name should be in nonparametric_structures_int.")
        structure = replace(structure, "rpc_" => "T")
        structure = replace(structure, "ipc_" => "INT")
        if (structure_name ∉ nonparametric_structures_float) && (structure_name ∉ nonparametric_structures_int)
          structure = replace(structure, structure_name => structure_name * "{T,INT}")
          structures = structures * "Ref{$(structure_name){Float32,Int32}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float32,Int64}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float64,Int32}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float64,Int64}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float128,Int32}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float128,Int64}}()[]\n"
        elseif (structure_name ∈ nonparametric_structures_float) && (structure_name ∉ nonparametric_structures_int)
          structure = replace(structure, structure_name => structure_name * "{INT}")
          structures = structures * "Ref{$(structure_name){Int32}}()[]\n"
          structures = structures * "Ref{$(structure_name){Int64}}()[]\n"
        elseif (structure_name ∉ nonparametric_structures_float) && (structure_name ∈ nonparametric_structures_int)
          structure = replace(structure, structure_name => structure_name * "{T}")
          structures = structures * "Ref{$(structure_name){Float32}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float64}}()[]\n"
          structures = structures * "Ref{$(structure_name){Float128}}()[]\n"
        else
          structures = structures * "Ref{$(structure_name)}()[]\n"
        end
        if index == 1
          text = text * "export " * structure_name * "\n\n" * structure
        else
          text = text * "\n" * "export " * structure_name * "\n" * structure
        end
      else
        text = text * code
      end
    end

    isfile("../test/test_structures.jl") || write("../test/test_structures.jl", "using GALAHAD\nusing Quadmath\n\n")
    test = read("../test/test_structures.jl", String)
    structures = structures * "\n"
    structures = replace(structures, "Ref{wcp_inform_type{Float128}}()\n" => "Ref{wcp_inform_type{Float128}}()")
    write("../test/test_structures.jl", test * structures)
  end

  write(path, text)
end
