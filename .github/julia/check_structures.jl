using Test

global n = 0

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
      (package == "bnls") && continue
      if file == "$package.F90"
        code = read(path, String)
        lines = split(code, '\n')
        for (i, line) in enumerate(lines)
          startswith(line |> strip, "!") && continue
          startswith(line, "#") && continue
          isempty(line) && continue
          startswith(line |> strip, "PRIVATE") && continue
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
          else
            if contains(line |> uppercase, "END TYPE")
              f_struct = ""
            else
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
              for len in ("LEN", "len")
                if contains(type, len)
                  type = replace(type, len => "[")
                  type = type * "]"
                end
              end
              field = lowercase(field)
              type = lowercase(type)
              type = replace(type, " " => "")
              println(type)

              push!(f_types[f_struct], type)
              push!(f_structures[f_struct], field)
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
      println("• [$char1] -- The field $field is missing.")
    end
    ncount = mapreduce(x -> x == field, +, structure1, init=0)
    if ncount > 1
      println("• [$char1] -- The field $field appears $ncount times in the structure.")
    end
  end
  for field in structure2
    if !(field in common_fields)
      if (field != "f_indexing") || !(char1 == 'F' && char2 == 'C')
        println("• [$char2] -- The field $field is missing.")
      end
      ncount = mapreduce(x -> x == field, +, structure2, init=0)
      if ncount > 1
        println("• [$char2] -- The field $field appears $ncount times in the structure.")
      end
    end
  end
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
    println("The structure $structure can't be find in a header file.")
    global n += 1
  else
    c_nfields = length(c_structures[structure])
    h_nfields = length(h_structures[structure])
    if c_nfields != h_nfields
      println("The structure $structure has missing attributes (H:$h_nfields / C:$c_nfields).")
      diff_structures('H', h_structures[structure], 'C', c_structures[structure])
      global n += 1
    else
      for i = 1:h_nfields
        h_field = h_structures[structure][i]
        c_field = c_structures[structure][i]
        if h_field != c_field
          println("The field $i of the structure $structure is not consistent (H:$h_field / C:$c_field).")
          global n += 1
        else
          h_type = h_types[structure][i]
          c_type = c_types[structure][i]
          if h_type != c_type
            println("The type of field $(h_field) of the structure $structure is not consistent (H:$h_type / C:$c_type).")
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
  if structure in f_list
    f_nfields = length(f_structures[structure])
    c_nfields = length(c_structures[structure])
    c_nfields = ("f_indexing" in c_structures[structure]) ? c_nfields-1 : c_nfields

    if f_nfields != c_nfields
      println("The structure $structure has missing attributes (F:$f_nfields / C:$c_nfields).")
      diff_structures('F', f_structures[structure], 'C', c_structures[structure])
      global n += 1
    # else
    #   for i = 1:f_nfields
    #     f_field = f_structures[structure][i]
    #     c_field = c_structures[structure][i]
    #     if f_field != c_field
    #       println("The field $i of the structure $structure is not consistent (F:$f_field / C:$c_field).")
    #       global n += 1
    #     else
    #       f_type = f_types[structure][i]
    #       c_type = c_types[structure][i]
    #       if f_type != c_type
    #         println("The type of field $(h_field) of the structure $structure is not consistent (F:$f_type / C:$c_type).")
    #         global n += 1
    #       end
    #     end
    #  end
    end
  end
end

@test n == 0
