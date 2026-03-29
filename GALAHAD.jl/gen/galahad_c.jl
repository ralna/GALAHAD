mapping = Dict("Cvoid" => "void", "Cchar" => "char", "Bool" => "bool", "Int32" => "int32_t", "Int64" => "int64_t",
               "Float32" => "float", "Float64" => "double", "Float128" => "__float128", "Cfloat128" => "__float128")

function string_callbacks(ipc_::String, rpc_::String, integer_suffix::String, real_suffix::String)
  path_galahad_callbacks = joinpath(ENV["GALAHAD"], "include", "galahad_callbacks.h")
  header = read(path_galahad_callbacks, String)
  header = replace(header, r"\n\s+" => " ")
  lines = split(header, '\n')
  out = String[]
  pattern = r"typedef\s+(\w+)\s+(\w+)\s*\((.*?)\);"
  list_callbacks = String[]
  for line in lines
    line = strip(line)
    isempty(line) && continue
    startswith(line, "//") && continue
    m = match(pattern, line)
    if m !== nothing
      rettype, name, args = m.captures
      rettype = replace(rettype, "ipc_" => ipc_, "rpc_" => rpc_)
      args    = replace(args,    "ipc_" => ipc_, "rpc_" => rpc_)
      newname = name * real_suffix * integer_suffix
      signature = "typedef $rettype $newname($args);"
      signature = replace(signature, "( " => "(")
      signature = replace(signature, " )" => ")")
      signature = replace(signature, ",  " => ", ")
      push!(out, signature)
      push!(list_callbacks, name)
    end
  end

  # validation
  missing_in_header = setdiff(callbacks, list_callbacks)
  missing_in_callbacks = setdiff(list_callbacks, callbacks)
  if !isempty(missing_in_header) || !isempty(missing_in_callbacks)
    msg = ""
    for cb in missing_in_header
      msg = msg * "Callback `$cb` (from `callbacks` in GALAHAD.jl/gen/rewriter.jl) not found in `galahad_callbacks.h` or not parsed correctly.\n"
    end
    for cb in missing_in_callbacks
      msg = msg * "Callback `$cb` (from `galahad_callbacks.h`) missing in the variable `callbacks` in GALAHAD.jl/gen/rewriter.jl. Please add it.\n"
    end
    error("--- Callback mismatch detected ---\n" * msg)
  end

  return join(out, "\n") * "\n\n"
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
                      "nrek", "trek", "tru", "wcp", "llsr", "llst", "bllsb", "ssls", "sllsb", "snls", 
                      "expo", "version")

  for variant in ("common", "single", "double", "quadruple")
    if length(galahad_mp[variant]) != length(ordered_packages)
      error("The values of the variables \"ordered_packages\" in \"galahad_c.jl\" and \"packages\" in \"rewriter.jl\" are inconsistent.")
    end
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

function check_galahad_headers()
  include_dir = joinpath(ENV["GALAHAD"], "include")
  options = load_options(joinpath(@__DIR__, "galahad.toml"))
  options["general"]["library_name"] = "libgalahad"

  for T in ("Float32", "Float64", "Float128")
    for INT in ("Int32", "Int64")
      args = get_default_args()
      push!(args, "-I$include_dir")
      (T == "Float32") && push!(args, "-DREAL_32")
      (T == "Float128") && push!(args, "-DREAL_128")
      (INT == "Int64") && push!(args, "-DINTEGER_64")

      galahad_headers = joinpath(include_dir, "galahad.h")
      headers = [galahad_headers]
      ctx = create_context(headers, args, options)
      build!(ctx, BUILDSTAGE_NO_PRINTING)
    end
  end
  return nothing
end
