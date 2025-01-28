using Test

global n = 0

# Definition of a Fortran structure with an alias
lazy_definitions = Dict(
  "bllsb_control_type" => "clls_control_type",
  "bllsb_inform_type" => "clls_inform_type",
  "bllsb_time_type" => "clls_time_type",
  "bqpb_control_type" => "cqp_control_type",
  "bqpb_inform_type" => "cqp_inform_type",
  "bqpb_time_type" => "cqp_time_type",
  "gls_ainfo_type" => "gls_ainfo",
  "gls_control_type" => "gls_control",
  "gls_finfo_type" => "gls_finfo",
  "gls_sinfo_type" => "gls_sinfo",
  "lms_control_type" => "lmt_control_type",
  "lms_inform_type" => "lmt_control_type",
  "lms_time_type" => "lmt_time_type",
  "sils_ainfo_type" => "sils_ainfo",
  "sils_control_type" => "sils_control",
  "sils_finfo_type" => "sils_finfo",
  "sils_sinfo_type" => "sils_sinfo",
  "spral_ssids_inform" => "ssids_inform",
  "spral_ssids_options" => "ssids_options",
)

mapping_types_c2f = Dict(
  "[100]" => "[history_max]",
  "ipc_" => "ip_",
  "spc_" => "sp_",
  "rpc_" => "rp_",
  "int64_t" => "long_",
  "bool" => "logical",
  "char[2]" => "char[1]",
  "char[4]" => "char[3]",
  "char[13]" => "char[12]",
  "char[21]" => "char[20]",
  "char[31]" => "char[30]",
  "char[81]" => "char[80]",
  "char[401]" => "char[400]",
  "char[501]" => "char[500]",
) ∪ lazy_definitions

function F_structures()
  f_types = Dict{String,Vector{String}}()
  f_structures = Dict{String,Vector{String}}()
  f_extend = Dict{String, String}()

  # Extract the Fortran structures from the Fortran files *.F90
  f_struct = ""
  for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "src"))
    for file in files
      path = joinpath(root, file) |> normpath
      folders = split(root, '/')
      (folders[end-1] != "src") && continue
      package = folders[end]
      if (file == "$package.F90") || (package == "ssids" && file == "inform.F90") || (package == "ssids" && file == "datatypes.F90")
        code = read(path, String)
        lines = split(code, '\n')
        f_contains = false
        for (i, line) in enumerate(lines)
          startswith(line |> strip, "!") && continue
          startswith(line, "#") && continue
          isempty(line) && continue
          startswith(line |> strip, "PRIVATE") && continue
          startswith(line |> strip, "contains") && continue
          if f_struct == ""
            if contains(line |> uppercase, "TYPE, PUBLIC")
              f_struct = split(line, "::")[2] |> strip
              f_struct = lowercase(f_struct)
              f_types[f_struct] = String[]
              f_structures[f_struct] = String[]

              # Check if some structures are extended
              if contains(line |> uppercase, "EXTENDS")
                extension = split(line, "EXTENDS(")[2]
                extension = split(extension, ")")[1]
                f_extend[f_struct] = lowercase(extension) |> strip
              end 
            end
            if contains(line |> uppercase, "TYPE SSIDS_INFORM") || contains(line |> uppercase, "TYPE SSIDS_OPTIONS")
              f_struct = split(line, "type")[2] |> strip
              f_types[f_struct] = String[]
              f_structures[f_struct] = String[]
            end
          else
            if startswith(line |> strip, "contains")
              f_contains = true
            elseif contains(line |> uppercase, "END TYPE")
              f_struct = ""
            else
              f_contains && continue
              endswith(line, "&") && continue
              newline = line
              j = 1
              while (i-j ≥ 1) && endswith(lines[i-j], "&")
                newline = lines[i-j][1:end-1] * newline
                j = j + 1
              end
              type = split(newline, "::")[1] |> strip
              field = split(newline, "::")[2] |> strip
              field = split(field, "!")[1] |> strip
              field = split(field, "=")[1] |> strip
              for syntax in ("TYPE", "REAL", "INTEGER", "KIND", "ALLOCATABLE", "=", "(", ",", ")")
                type = replace(type, syntax => "")
              end
              type = replace(type, "CHARACTER" => "Char")
              for dim in ("DIMENSION", "dimension")
                if contains(type, dim)
                  type = replace(type, dim => "[")
                  type = type * "]"
                end
              end
              type = replace(type, "len_solver" => "20")
              for len in ("LEN", "len")
                if contains(type, len)
                  type = replace(type, len => "[")
                  type = type * "]"
                end
              end
              field = lowercase(field)
              type = lowercase(type)
              type = replace(type, " " => "")

              if !(field ∈ f_structures[f_struct])
                push!(f_types[f_struct], type)
                push!(f_structures[f_struct], field)
              end
            end
          end
        end
      end
    end
  end

  # Update the structures created with "EXTENDS"
  for f_struct in keys(f_extend)
    f_struct_extended = f_extend[f_struct]
    f_structures[f_struct] = vcat(f_structures[f_struct_extended], f_structures[f_struct])
    f_types[f_struct] = vcat(f_types[f_struct_extended], f_types[f_struct])    
  end

  return f_types, f_structures
end

function C_structures()
  c_types = Dict{String,Vector{String}}()
  c_structures = Dict{String,Vector{String}}()

  # Extract the C structures from the Fortran files *_ciface.F90
  c_struct = ""
  for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "src"))
    for file in files
      path = joinpath(root, file) |> normpath
      if endswith(file, "_ciface.F90")
        code = read(path, String)
        lines = split(code, '\n')
        for (i, line) in enumerate(lines)
          startswith(line, "!") && continue
          if c_struct == ""
            if contains(line |> uppercase, "TYPE, BIND( C )")
              c_struct = split(line, "::")[2] |> strip
              c_struct = lowercase(c_struct)
              c_types[c_struct] = String[]
              c_structures[c_struct] = String[]
            end
          else
            if contains(line |> uppercase, "END TYPE")
              c_struct = ""
            else
              endswith(line, "&") && continue
              if endswith(lines[i-1], "&")
                type = split(lines[i-1], "::")[1] |> strip
                field = strip(line)
              else
                type = split(line, "::")[1] |> strip
                field = split(line, "::")[2] |> strip
              end

              field = split(field, "!")[1] |> strip
              for syntax in ("TYPE", "REAL", "INTEGER", "CHARACTER", "=")
                type = split(type, syntax)[end]
              end
              type = replace(type,  "DIMENSION( 3, 81 )" => "[3][81]")
              type = replace(type, "longc_" => "int64_t")
              type = replace(type, "C_CHAR" => "char")
              type = replace(type, "C_BOOL" => "bool")
              type = replace(type, "(" => "")
              type = replace(type, "," => "")
              type = replace(type, ")" => "")
              for dim in ("DIMENSION", "dimension")
                if contains(type, dim)
                  type = replace(type, dim => "[")
                  type = type * "]"
                end
              end
              field = lowercase(field)
              type = lowercase(type)
              type = replace(type, " " => "")

              push!(c_types[c_struct], type)
              push!(c_structures[c_struct], field)
            end
          end
        end
      end
    end
  end

  return c_types, c_structures
end

function H_structures()
  h_types = Dict{String,Vector{String}}()
  h_structures = Dict{String,Vector{String}}()

  # Extract the structures from the header files *.h
  h_struct = ""
  for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "include"))
    for file in files
      path = joinpath(root, file) |> normpath
      if endswith(file, ".h")
        (file == "ssids_gpu_kernels_datatypes.h") && continue
        (file == "ssids_gpu_kernels_dtrsv.h") && continue
        (file == "galahad_icfs.h") && continue
        code = read(path, String)
        lines = split(code, '\n')
        for (i, line) in enumerate(lines)
          line2 = line |> strip
          length(line2) == 0 && continue
          startswith(line2, "/") && continue
          startswith(line2, "#") && continue
          startswith(line2, "*") && continue
          startswith(line2, "extern") && continue
          if h_struct == ""
            if startswith(line, "struct") && endswith(line, "{")
              h_struct = split(line, "struct")[2]
              h_struct = split(h_struct, "{")[1] |> strip
              h_struct = lowercase(h_struct)
              h_types[h_struct] = String[]
              h_structures[h_struct] = String[]
            end
          else
            if startswith(line, "};")
              h_struct = ""
            else
              line = split(line, '/')[1]
              type = split(line)[end-1]
              field = split(line)[end][1:end-1]  # remove ";" at the end
              if contains(field, "[") && contains(field, "]")
                split_field = split(field, "[")
                if length(split_field) == 2
                  dimension = split(field, "[")[2]
                  type = type * "[$dimension"
                else
                  @assert length(split_field) == 3
                  dimension1 = split(field, "[")[2]
                  dimension2 = split(field, "[")[3]
                  type = type * "[$dimension1[$dimension2"
                end
                field = split(field, "[")[1]
              end
              type = replace(type, "real_sp_" => "spc_")
              field = lowercase(field)
              type = lowercase(type)

              push!(h_types[h_struct], type)
              push!(h_structures[h_struct], field)
            end
          end
        end
      end
    end
  end

  return h_types, h_structures
end

function diff_structures(char1::Char, structure1::Vector{String}, char2::Char, structure2::Vector{String})
  common_fields = intersect(structure1, structure2)
  for field in unique(structure1)
    if !(field in common_fields)
      println("• The field `$field` is in the `$char1` structure but not the `$char2` structure.")
    end
  end
  for field in structure2
    if !(field in common_fields)
      if (field != "f_indexing") || !(char1 == 'F' && char2 == 'C')
        println("• The field `$field` is in the `$char2` structure but not the `$char1` structure.")
      end
    end
  end
  println()
end

f_types, f_structures = F_structures()
c_types, c_structures = C_structures()
h_types, h_structures = H_structures()

f_list = keys(f_structures)
c_list = keys(c_structures)
h_list = keys(h_structures)

println("-------------------------------------------------------------")
println("The following structures are only defined in the header files")
println("-------------------------------------------------------------")
for val in h_list
  if !(val in c_list)
    # Check if it's a structure of an HSL package
    if !startswith(val, "ma") && !startswith(val, "mi") && !startswith(val, "mc")
      println(val)
      global n += 1
    end
  end
end

println("-------------------------------------------------------------")
println("The following structures are only defined in the C interfaces")
println("-------------------------------------------------------------")
for val in c_list
  if val in keys(lazy_definitions)
    val = lazy_definitions[val]
  end
  if !(val in f_list)
    println(val)
    global n += 1
  end
end

println("-------------------------------------------------------------")
println("--------------Check errors in H / C structures---------------")
println("-------------------------------------------------------------")
for structure in c_list
  if !(structure in h_list)
    println("The structure `$structure` can't be find in a header file.")
    global n += 1
  else
    package = split(structure, '_')[1] |> uppercase
    c_nfields = length(c_structures[structure])
    h_nfields = length(h_structures[structure])
    if c_nfields != h_nfields
      println("[$package] -- The structure `$structure` has missing attributes (H:$h_nfields / C:$c_nfields).")
      diff_structures('H', h_structures[structure], 'C', c_structures[structure])
      global n += 1
    else
      for i = 1:h_nfields
        h_field = h_structures[structure][i]
        c_field = c_structures[structure][i]
        if h_field != c_field
          println("[$package] -- The field $i of the structure `$structure` is not consistent (H:$h_field / C:$c_field).")
          global n += 1
        else
          h_type = h_types[structure][i]
          c_type = c_types[structure][i]
          if h_type != c_type
            println("[$package] -- The type of field `$(h_field)` of the structure `$structure` is not consistent (H:$h_type / C:$c_type).")
            global n += 1
          end
        end
      end
    end
  end
end

println("-------------------------------------------------------------")
println("--------------Check errors in F / C structures---------------")
println("-------------------------------------------------------------")
for structure in c_list
  haskey(lazy_definitions, structure)
  structure2 = haskey(lazy_definitions, structure) ? lazy_definitions[structure] : structure
  if !(structure2 in f_list)
    println("The structure `$(structure2)` can't be find in a Fortran file.")
    global n += 1
  else
    package = split(structure, '_')[1] |> uppercase
    f_nfields = length(f_structures[structure2])
    c_nfields = length(c_structures[structure])
    c_nfields = ("f_indexing" in c_structures[structure]) ? c_nfields-1 : c_nfields

    if f_nfields != c_nfields
      println("[$package] -- The structure `$structure` has missing attributes (F:$f_nfields / C:$c_nfields).")
      diff_structures('F', f_structures[structure2], 'C', c_structures[structure])
      global n += 1
    else
      for (i, c_field) in enumerate(c_structures[structure])
        (c_field == "f_indexing") && continue
        if !(c_field in f_structures[structure2])
          println("[$package] -- The field `$(c_field)` of the C structure `$structure` can't be found in the Fortran structure `$structure2`.")
          global n += 1
        else
          j = findfirst(str -> str == c_field, f_structures[structure2])
          c_type = c_types[structure][i]
          f_type = f_types[structure2][j]
          c_type2 = c_type
          for (key, val) in mapping_types_c2f
            c_type2 = replace(c_type2, key => val)
          end
          if (c_type != f_type) && (c_type2 != f_type)
            println("[$package] -- The type of field `$(c_field)` of the structures `$structure2` (Fortran) and `$structure` (C) is not consistent (F:$f_type / C:$c_type).")
            global n += 1
          end
        end
      end
    end
  end
end

@test n == 0
