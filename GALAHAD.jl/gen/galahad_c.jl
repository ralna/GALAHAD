mapping = Dict("Cvoid" => "void", "Cchar" => "char", "Bool" => "bool", "Int32" => "int32_t", "Int64" => "int64_t",
               "Float32" => "float", "Float64" => "double", "Float128" => "__float128", "Cfloat128" => "__float128")

function string_callbacks(ipc_::String, rpc_::String, integer_suffix::String, real_suffix::String)
str = "typedef $ipc_ galahad_f$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ x[], $rpc_ *f, const void *userdata);
typedef $ipc_ galahad_g$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ x[], $rpc_ g[], const void *userdata);
typedef $ipc_ galahad_h$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ ne, const $rpc_ x[], $rpc_ h[], const void *userdata);
typedef $ipc_ galahad_prec$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ x[], $rpc_ u[], const $rpc_ v[], const void *userdata);
typedef $ipc_ galahad_hprod$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ x[], $rpc_ u[], const $rpc_ v[], bool got_h, const void *userdata);
typedef $ipc_ galahad_shprod$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ x[], $ipc_ nnz_v, const $ipc_ index_nz_v[], const $rpc_ v[], $ipc_ *nnz_u, $ipc_ index_nz_u[], $rpc_ u[], bool got_h, const void *userdata);
typedef $ipc_ galahad_constant_prec$(real_suffix)$(integer_suffix)($ipc_ n, const $rpc_ v[], $rpc_ p[], const void *userdata);
typedef $ipc_ galahad_r$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, const $rpc_ x[], $rpc_ r[], const void *userdata);
typedef $ipc_ galahad_jr$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, $ipc_ jne, const $rpc_ x[], $rpc_ jr[], const void *userdata);
typedef $ipc_ galahad_hr$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, $ipc_ hne, const $rpc_ x[], const $rpc_ y[], $rpc_ hr[], const void *userdata);
typedef $ipc_ galahad_jrprod$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, const $rpc_ x[], const bool transpose, $rpc_ u[], const $rpc_ v[], bool got_j, const void *userdata);
typedef $ipc_ galahad_hrprod$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, const $rpc_ x[], const $rpc_ y[], $rpc_ u[], const $rpc_ v[], bool got_h, const void *userdata);
typedef $ipc_ galahad_shrprod$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, $ipc_ pne, const $rpc_ x[], const $rpc_ v[], $rpc_ pval[], bool got_h, const void *userdata);
typedef $ipc_ galahad_fc$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, const $rpc_ x[], $rpc_ *f, $rpc_ c[], const void *userdata);
typedef $ipc_ galahad_gj$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, $ipc_ jne, const $rpc_ x[], $rpc_ g[], $rpc_ j[], const void *userdata);
typedef $ipc_ galahad_hl$(real_suffix)$(integer_suffix)($ipc_ n, $ipc_ m, $ipc_ hne, const $rpc_ x[], const $rpc_ y[], $rpc_ h[], const void *userdata);
typedef $ipc_ galahad_fgh$(real_suffix)$(integer_suffix)($rpc_ x, $rpc_ *f, $rpc_ *g, $rpc_ *h, const void *userdata);\n\n"
return str
end

function finalize_header_c(text::String, variant::String)
  begin_guard = "// include guard\n#ifndef GALAHAD_C_$(uppercase(variant))_H\n#define GALAHAD_C_$(uppercase(variant))_H\n\n"
  end_guard = "\n\n// end include guard\n#endif\n"

  if variant == "single"
    header = begin_guard * "// Callbacks\n" * string_callbacks("int32_t", "float", "", "_s") * string_callbacks("int64_t", "float", "_64", "_s") * text * end_guard
  elseif variant == "double"
    header = begin_guard * "// Callbacks\n" * string_callbacks("int32_t", "double", "", "") * string_callbacks("int64_t", "double", "_64", "") * text * end_guard
  elseif variant == "quadruple"
    header = begin_guard * "// Callbacks\n" * string_callbacks("int32_t", "__float128", "", "_q") * string_callbacks("int64_t", "__float128", "_64", "_q") * text * end_guard
  elseif variant == "common"
    header = begin_guard * text * end_guard
  else
    error("The variant \"$variant\" is not supported.")
  end

  return header
end

function prototype(wrapper::String, suffix::String)
  proto = ""
  lines = split(wrapper, '\n')
  for line in lines
    if contains(line, "@ccall")
      line = strip(line)
      line = replace(line, "@ccall" => "")
      for precision in ("single", "double", "quadruple")
        line = replace(line, "libgalahad_$(precision)." => "")
        line = replace(line, "libgalahad_$(precision)_64." => "")
      end
      for T in ("Float32", "Float64", "Float128")
        for INT in ("Int32", "Int64")
          line = replace(line, "$T,$INT" => "$(T)_$(INT)")
        end
      end
      for type in ("Cvoid", "Int32", "Int64")
        line = endswith(line, ")::$type") ? mapping[type] * replace(line, ")::$type" => ");") : line
      end
      args = split(line, '(')
      start_routine = args[1]
      line = args[2]
      args = split(line, ')')
      end_routine = args[2]
      line = args[1]
      args = split(line, ',')
      nargs = length(args)
      for i = 1:nargs
        args[i] = strip(args[i])
        for type in types
          for package in packages
            type_name = "$(package)_$(type)_type"
            if (type_name ∉ nonparametric_structures_float) && (type_name ∈ nonparametric_structures_int)
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float32}}") ? "struct $(type_name)_s *" * replace(args[i], "::Ptr{$(type_name){Float32}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float64}}") ? "struct $(type_name) *" * replace(args[i], "::Ptr{$(type_name){Float64}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float128}}") ? "struct $(type_name)_q *" * replace(args[i], "::Ptr{$(type_name){Float128}}" => "") : args[i]
            end
            if (type_name ∈ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
              args[i] = endswith(args[i], "::Ptr{$(type_name){Int32}}") ? "struct $(type_name) *" * replace(args[i], "::Ptr{$(type_name){Int32}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Int64}}") ? "struct $(type_name)_64 *" * replace(args[i], "::Ptr{$(type_name){Int64}}" => "") : args[i]
            end
            if (type_name ∉ nonparametric_structures_float) && (type_name ∉ nonparametric_structures_int)
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float32_Int32}}") ? "struct $(type_name)_s *" * replace(args[i], "::Ptr{$(type_name){Float32_Int32}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float64_Int32}}") ? "struct $(type_name) *" * replace(args[i], "::Ptr{$(type_name){Float64_Int32}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float128_Int32}}") ? "struct $(type_name)_q *" * replace(args[i], "::Ptr{$(type_name){Float128_Int32}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float32_Int64}}") ? "struct $(type_name)_s_64 *" * replace(args[i], "::Ptr{$(type_name){Float32_Int64}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float64_Int64}}") ? "struct $(type_name)_64 *" * replace(args[i],  "::Ptr{$(type_name){Float64_Int64}}" => "") : args[i]
              args[i] = endswith(args[i], "::Ptr{$(type_name){Float128_Int64}}") ? "struct $(type_name)_q_64 *" * replace(args[i], "::Ptr{$(type_name){Float128_Int64}}" => "") : args[i]
            end
            args[i] = endswith(args[i], "::Ptr{$(type_name)}") ? "struct $(type_name) *" * replace(args[i], "::Ptr{$(type_name)}" => "") : args[i]
          end
        end

        for (Jtype, Ctype) in mapping
          args[i] = endswith(args[i], "::$Jtype") ? "$Ctype " * replace(args[i], "::$Jtype" => "") : args[i]
          args[i] = endswith(args[i], "::Ptr{$Jtype}") ? "$Ctype *" * replace(args[i], "::Ptr{$Jtype}" => "") : args[i]
          args[i] = endswith(args[i], "::Ptr{Ptr{$Jtype}}") ? "$Ctype **" * replace(args[i], "::Ptr{Ptr{$Jtype}}" => "") : args[i]
        end

        for callback in callbacks
          args[i] = endswith(args[i], "::Ptr{$(callback)}") ? "$callback$suffix *" * replace(args[i], "::Ptr{$(callback)}" => "") : args[i]
        end
      end

      # Build the prototype
      proto = proto * start_routine * "("
      for i = 1:nargs
        comma = (i == 1) ? "" : ", "
        proto = proto * comma * args[i]
      end
      proto = proto * ")" * end_routine
    end
  end
  return proto
end

function structure_mp(structure::String, real::String, integer::String, real_suffix::String, integer_suffix::String)
    structure = replace(structure, "_T_INT" => "$(real_suffix)$(integer_suffix)")
    structure = replace(structure, "_T" => real_suffix)
    structure = replace(structure, "_INT" => integer_suffix)
    structure = replace(structure, "INT " => "$integer ")
    structure = replace(structure, "T " => "$real ")
    return structure
end

function galahad_c(structure::String, mode::String, variant_INT::Bool, variant_T::Bool)
  text = ""
  structure = replace(structure, "{T,INT}" => "_T_INT")
  structure = replace(structure, "{T}" => "_T")
  structure = replace(structure, "{INT}" => "_INT")
  lines = split(structure, "\n")
  nlines = length(lines)
  for i = 1:nlines
    lines[i] = strip(lines[i])
    lines[i] = replace(lines[i], r"(\w+)::NTuple\{(\d+),\s*NTuple\{(\d+),\s*([\w\.]+)\}\}" => s"\4 \1[\2][\3]")
    lines[i] = replace(lines[i], r"(\w+)::NTuple\{(\d+), *([\w\.]+)\}" => s"\3 \1[\2]")
    lines[i] = replace(lines[i], r"(.*)::([A-Za-z0-9_]+)" => s"\2 \1")
    lines[i] = replace(lines[i], "Bool" => "bool")
    lines[i] = replace(lines[i], "Cchar" => "char")
    lines[i] = replace(lines[i], "Float32" => "float")
    lines[i] = replace(lines[i], "Int64" => "int64_t")
    if startswith(lines[i], "struct")
      text = text * lines[i] * " {\n"
    elseif lines[i] == "end"
      text = text * "};\n"
    else
      if !isempty(lines[i])
        if contains(lines[i], "_type") && !contains(lines[i], "indicator_type") && mapreduce(x -> !contains(lines[i], x), &, ["char", "int32_t", "int64_t"])
          text = text * "    " * "struct " * lines[i] * ";\n"
        elseif mapreduce(x -> contains(lines[i], x), |, hsl_structures)
          text = text * "    " * "struct " * lines[i] * ";\n"
        else
          text = text * "    " * lines[i] * ";\n"
        end
      end
    end
  end

  structure_int32_float32  = structure_mp(text, "float"     , "int32_t", "_s", ""   )
  structure_int32_float64  = structure_mp(text, "double"    , "int32_t", ""  , ""   )
  structure_int32_float128 = structure_mp(text, "__float128", "int32_t", "_q", ""   )
  structure_int64_float32  = structure_mp(text, "float"     , "int64_t", "_s", "_64")
  structure_int64_float64  = structure_mp(text, "double"    , "int64_t", ""  , "_64")
  structure_int64_float128 = structure_mp(text, "__float128", "int64_t", "_q", "_64")

  if !variant_INT
    if !variant_T
      text = structure_int32_float64
    else
      if mode == "single"
        text = structure_int32_float32
      elseif mode == "double"
        text = structure_int32_float64
      elseif mode == "quadruple"
        text = structure_int32_float128
      else
        error("The current mode \"$mode\" is not supported.")
      end
    end
  else
    if !variant_T
      text = structure_int32_float64 * "\n" * structure_int64_float64
    else
      if mode == "single"
        text = structure_int32_float32 * "\n" * structure_int64_float32
      elseif mode == "double"
        text = structure_int32_float64 * "\n" * structure_int64_float64
      elseif mode == "quadruple"
        text = structure_int32_float128 * "\n" * structure_int64_float128
      else
        error("The current mode \"$mode\" is not supported.")
      end
    end
  end
  return text
end

function generate_galahad_c()
  ordered_packages = ("bsc", "convert", "fit", "glrt", "gls", "gltr", "hash", "hsl", "ir", "l2rt",
                      "lhs", "lms", "lsrt", "lstr", "nodend", "presolve", "roots", "rpd", "scu", "sec",
                      "sha", "sils", "ugo", "ssids", "sls", "rqs", "dps", "psls", "arc", "trs",
                      "trb", "bgo", "uls", "sbls", "blls", "bqp", "fdc", "cro", "bqpb", "ccqp", "cqp",
                      "clls", "dgo", "dqp", "eqp", "lpa", "lpb", "lsqp", "nls", "qpa", "qpb", "slls",
                      "tru", "wcp", "llsr", "llst", "bllsb", "ssls", "expo", "nrek", "trek", "version")

  for variant in ("common", "single", "double", "quadruple")
    @assert length(galahad_mp[variant]) == length(ordered_packages)
    text_c = ""
    for (index, package) in enumerate(ordered_packages)
      val = galahad_mp[variant][package]
      if index == 1
        text_c = text_c * val
      else
        if occursin("\n", val)
          text_c = text_c * "\n\n" * val
        end
      end
    end
    text_c = finalize_header_c(text_c, variant)
    text_c = replace(text_c, "\n\n\n" => "\n\n")
    isfile("../../include/galahad_c_$variant.h") && rm("../../include/galahad_c_$variant.h")
    write("../../include/galahad_c_$variant.h", text_c)
  end

  return nothing
end

function check_galahad_c()
  include_dir = joinpath(ENV["GALAHAD"], "include")
  options = load_options(joinpath(@__DIR__, "galahad.toml"))
  options["general"]["library_name"] = "libgalahad"
  args = get_default_args()
  push!(args, "-I$include_dir")
  push!(args, "-DREAL_128")

  header_galahad_c = joinpath(include_dir, "galahad_c.h")
  headers = [header_galahad_c]
  ctx = create_context(headers, args, options)
  build!(ctx, BUILDSTAGE_NO_PRINTING)
  return nothing
end
