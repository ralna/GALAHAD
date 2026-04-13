using Test

global n = 0

mapping_field_c2f = Dict(
  "nstatic" => "static",       # SILS
  "switch_full" => "switch",   # GLS
  "struct_abort" => "struct",  # GLS
)

# Fields that are only in the Fortran structures
excluded_fortran_fields = Dict(
  "lpa_inform_type" => ["threads"],
  "presolve_control_type" => ["get_q", "get_f", "get_g", "get_H", "get_A", "get_x", "get_x_bounds", "get_z", "get_z_bounds", "get_c", "get_c_bounds", "get_y", "get_y_bounds"],
  "wcp_inform_type" => ["x_status", "c_status"],
  "sbls_inform_type" => ["sils_analyse_status", "sils_factorize_status", "sils_solve_status", "sls_analyse_status", "sls_factorize_status", "sls_solve_status", "uls_analyse_status", "uls_factorize_status", "uls_solve_status"],
  "ssids_control_type" => ["auction", "rb_dump"],
  "ssids_inform_type" => ["auction"],
)

function extract_type_fields(text, typename)
  pattern = Regex("TYPE\\s*,\\s*BIND\\s*\\(\\s*C\\s*\\)\\s*::\\s*$typename(.*?)END TYPE", "si")
  m = match(pattern, text)
  m === nothing && return String[], Dict{String,String}()

  block = m.captures[1]

  fields = String[]
  field_types = Dict{String,String}()

  for line in split(block, '\n')
    line = strip(line)
    isempty(line) && continue
    occursin("::", line) || continue

    lhs, rhs = split(line, "::")
    ftype = strip(lhs)

    for var in split(rhs, ",")
      v = strip(var)
      push!(fields, v)
      field_types[v] = ftype
    end
  end

  return fields, field_types
end

function is_string_field(ftype)
  occursin("CHARACTER", ftype)
end

function is_derived_field(ftype)
  occursin("TYPE", ftype)
end

# 🔥 Merge Fortran continuation lines
function merge_fortran_lines(block)
  lines = split(block, '\n')
  merged = String[]
  buffer = ""

  for line in lines
    l = strip(line)
    isempty(l) && continue

    if endswith(l, "&")
      buffer *= replace(l, "&" => "") * " "
    else
      buffer *= l
      push!(merged, strip(buffer))
      buffer = ""
    end
  end

  if !isempty(buffer)
    push!(merged, strip(buffer))
  end

  return merged
end

function extract_assignments(text, funcname, prefix)
  pattern = Regex("SUBROUTINE\\s+$funcname(.*?)END SUBROUTINE", "si")
  m = match(pattern, text)
  m === nothing && return Dict{String,Int}()

  block = m.captures[1]
  lines = merge_fortran_lines(block)

  assigns = Dict{String,Int}()

  for line in lines
    if occursin(r"^\s*\w+%\w+\s*(\([^=]*\))?\s*=", line)
      lhs = strip(split(line, "=", limit=2)[1])

      if startswith(lhs, prefix * "%")
        field = split(lhs, "%")[end]
        field = replace(field, r"\(.*\)" => "")
        field = strip(field)

        assigns[field] = get(assigns, field, 0) + 1
      end
    end
  end

  return assigns
end

function check_copy_block(text, typename, func_in, func_out, prefix_in, prefix_out)
  global n

  fields, types = extract_type_fields(text, typename)
  isempty(fields) && return

  in_exists  = occursin(Regex("SUBROUTINE\\s+$func_in", "si"), text)
  out_exists = occursin(Regex("SUBROUTINE\\s+$func_out", "si"), text)

  if in_exists && out_exists
    in_assigns  = extract_assignments(text, func_in, prefix_in)
    out_assigns = extract_assignments(text, func_out, prefix_out)

    for f in fields
      ftype = types[f]

      if is_string_field(ftype) || is_derived_field(ftype)
        continue
      end

      f2 = haskey(mapping_field_c2f, f) ? mapping_field_c2f[f] : f

      if !haskey(in_assigns, f2) && (f2 != "f_indexing")
        if !haskey(excluded_fortran_fields, typename) || !in(f2, excluded_fortran_fields[typename])
          println("== $typename ==")
          println("❌ Missing in $func_in: $f2")
          n += 1
        end
      end

      if !haskey(out_assigns, f) && (f != "f_indexing")
        if !haskey(excluded_fortran_fields, typename) || !in(f, excluded_fortran_fields[typename])
          println("== $typename ==")
          println("❌ Missing in $func_out: $f")
          n += 1
        end
      end
    end
  end
end

function check_string_init(text, funcname)
  pattern = Regex("SUBROUTINE\\s+$funcname(.*?)END SUBROUTINE", "si")
  m = match(pattern, text)
  m === nothing && return
  block = m.captures[1]
  if occursin("CHARACTER", block) && occursin("C_NULL_CHAR", block) && !occursin("= ''", block)
    println("⚠️ Possible missing string init in $funcname")
  end
end

function check_file(package, path)
  text = read(path, String)
  pkg = lowercase(package)

  control_type = "$(pkg)_control_type"
  inform_type  = "$(pkg)_inform_type"
  time_type    = "$(pkg)_time_type"

  check_copy_block(text,
    control_type,
    "copy_control_in", "copy_control_out",
    "fcontrol", "ccontrol"
  )

  check_copy_block(text,
    inform_type,
    "copy_inform_in", "copy_inform_out",
    "finform", "cinform"
  )

  check_copy_block(text,
    time_type,
    "copy_time_in", "copy_time_out",
    "ftime", "ctime"
  )

  check_string_init(text, "copy_control_in")
  check_string_init(text, "copy_inform_in")
end

rootdir = joinpath(@__DIR__, "..", "..", "src")

for (root, _, files) in walkdir(rootdir)
  for file in files
    if endswith(file, "_ciface.F90")
      path = joinpath(root, file) |> normpath
      m = match(r"(\w+).*_ciface", lowercase(file))
      m === nothing && continue
      package = m.captures[1]
      check_file(package, path)
    end
  end
end

if n > 0
  println("\nMaybe a manual update of the file GALAHAD/.github/julia/check_copies.jl is needed.\n")
end
@test n == 0
