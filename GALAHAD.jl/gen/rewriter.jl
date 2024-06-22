packages = ("arc", "bgo", "blls", "bllsb", "bqp", "bqpb", "bsc", "ccqp",
            "clls", "convert", "cqp", "cro", "dgo", "dps", "dqp", "eqp",
            "fdc", "fit", "glrt", "gls", "gltr", "hash", "ir", "l2rt", "lhs",
            "llsr", "llst", "lms", "lpa", "lpb", "lsqp", "lsrt", "lstr",
            "nls", "presolve", "psls", "qpa", "qpb", "roots", "rpd", "rqs",
            "sbls", "scu", "sec", "sha", "sils", "slls", "sls", "trb", "trs",
            "tru", "ugo", "uls", "wcp")

types = ("control", "time", "inform", "history", "subproblem_control", "subproblem_inform", "ainfo", "finfo", "sinfo")

nonparametric_structures = ("slls_time_type", "sha_control_type", "sha_inform_type", "sec_inform_type",
                            "scu_control_type", "scu_inform_type", "rpd_control_type",
                            "rpd_inform_type", "roots_inform_type", "presolve_inform_type",
                            "lhs_control_type", "lhs_inform_type", "lms_control_type",
                            "hash_control_type", "hash_inform_type", "gls_sinfo_type",
                            "bqp_time_type", "bsc_control_type", "convert_control_type",
                            "fit_control_type", "fit_inform_type", "spral_ssids_inform", "ma48_sinfo",
                            "mc64_control", "mc64_info", "mc68_control", "mc68_info")

function rewrite!(path::String, name::String, optimized::Bool)
  structures = "# Structures for $name\n"
  text = read(path, String)
  if optimized
    text = replace(text, "ipc_" => "Cint")
    text = replace(text, "hsl_longc_" => "Clong")
    text = replace(text, "longc_" => "Int64")
    text = replace(text, "real_sp_" => "Float32")
    text = replace(text, "\n    " => "\n  ")

    # Special case for gls
    text = replace(text, "gls_control" => "gls_control_type")
    text = replace(text, "gls_ainfo" => "gls_ainfo_type")
    text = replace(text, "gls_sinfo" => "gls_sinfo_type")
    text = replace(text, "gls_finfo" => "gls_finfo_type")

    for type in types
      for package in packages
        if "$(package)_$(type)_type" ∉ nonparametric_structures
          text = replace(text, "::$(package)_$(type)_type" => "::$(package)_$(type)_type{T}")
          text = replace(text, ",$(package)_$(type)_type}" => ",$(package)_$(type)_type{T}}")
        end
      end
    end

    for type in ("control", "solve_control", "info", "ainfo", "finfo", "sinfo")
      text = replace(text, "$(type)_d" => "$(type)")
      text = replace(text, "$(type)_i" => "$(type)")
      for hsl in ("ma48", "ma57", "ma77", "ma86", "ma87", "ma97", "mc64", "mc68", "mi20", "mi28")
        if "$(hsl)_$(type)" ∉ nonparametric_structures
          text = replace(text, "::$(hsl)_$(type)" => "::$(hsl)_$(type){T}")
        end
      end
    end

    for type in ("llst_history_type", "llsr_history_type", "rqs_history_type", "trs_history_type")
      text = replace(text, "$type}" => "$type{T}}")
    end

    blocks = split(text, "end\n")
    text = ""
    for (index, code) in enumerate(blocks)
      if contains(code, "function")
        function_name = split(split(code, "function ")[2], "(")[1]
        routine_single = code * "end\n"
        routine_double = code * "end\n"

        routine_single = replace(routine_single, "(" => "_s(")
        routine_single = replace(routine_single, "libgalahad_double" => "libgalahad_single")
        routine_single = replace(routine_single, ",\n" => ",\n  ")

        routine_single = replace(routine_single, "rpc_" => "Float32")
        routine_double = replace(routine_double, "rpc_" => "Float64")

        routine_single = replace(routine_single, "spral_ssids_options" => "spral_ssids_options{Float32}")
        routine_double = replace(routine_double, "spral_ssids_options" => "spral_ssids_options{Float64}")

        for type in types
          for package in packages
            if "$(package)_$(type)_type" ∉ nonparametric_structures
              routine_single = replace(routine_single, "$(package)_$(type)_type" => "$(package)_$(type)_type{Float32}")
              routine_double = replace(routine_double, "$(package)_$(type)_type" => "$(package)_$(type)_type{Float64}")

              routine_single = replace(routine_single, "{Float32}{Float32}" => "{Float32}")
              routine_double = replace(routine_double, "{Float64}{Float64}" => "{Float64}")
            end
          end
        end

        if (name ≠ "hsl") && (name ≠ "ssids")
          text = text * "\n" * "export " * function_name * "_s\n" * routine_single
          text = text * "\n" * "export " * function_name *   "\n" * routine_double
        end
      elseif contains(code, "struct ")
        structure = code * "end\n"
        structure_name = split(split(code, "struct ")[2], "\n")[1]
        structure = replace(structure, "rpc_" => "T")
        if structure_name ∉ nonparametric_structures
          structure = replace(structure, structure_name => structure_name * "{T}")
          structures = structures * "Ref{$(structure_name){Float32}}()\n"
          structures = structures * "Ref{$(structure_name){Float64}}()\n"
        else
          structures = structures * "Ref{$(structure_name)}()\n"
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

    isfile("../test/test_structures.jl") || write("../test/test_structures.jl", "using GALAHAD\n\n")
    test = read("../test/test_structures.jl", String)
    structures = structures * "\n"
    write("../test/test_structures.jl", test * structures)
  end

  write(path, text)
end
