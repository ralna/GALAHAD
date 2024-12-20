using Test

global n = 0

function FORTRAN_structures()
  f_types = Dict{String,Vector{String}}()
  f_structures = Dict{String,Vector{String}}()

  # Extract the Fortran structures from the Frotran files *_ciface.F90
  f_struct = ""
  for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "src"))
    for file in files
      path = joinpath(root, file) |> normpath
      if endswith(file, "_ciface.F90")
        code = read(path, String)
        lines = split(code, '\n')
        for (i, line) in enumerate(lines)
          startswith(line, "!") && continue
          if f_struct == ""
            if contains(line |> uppercase, "TYPE, BIND( C )")
              f_struct = split(line, "::")[2] |> strip
              f_struct = lowercase(f_struct)
              f_types[f_struct] = String[]
              f_structures[f_struct] = String[]
            end
          else
            if contains(line |> uppercase, "END TYPE")
              f_struct = ""
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

              push!(f_types[f_struct], type)
              push!(f_structures[f_struct], field)
            end
          end
        end
      end
    end
  end

  return f_types, f_structures
end

function C_structures()
  c_types = Dict{String,Vector{String}}()
  c_structures = Dict{String,Vector{String}}()

  # Extract the C structures from the header files *.h
  c_struct = ""
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
          if c_struct == ""
            if startswith(line, "struct") && endswith(line, "{")
              c_struct = split(line, "struct")[2]
              c_struct = split(c_struct, "{")[1] |> strip
              c_struct = lowercase(c_struct)
              c_types[c_struct] = String[]
              c_structures[c_struct] = String[]
            end
          else
            if startswith(line, "};")
              c_struct = ""
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

f_types, f_structures = FORTRAN_structures()
c_types, c_structures = C_structures()

f_list = keys(f_structures)
c_list = keys(c_structures)

println("-------------------------------------------------------------")
println("The following structures are only defined in the header files")
println("-------------------------------------------------------------")
for val in c_list
  if !(val in f_list)
    # Check if it's a structure of an HSL package
    if !startswith(val, "ma") && !startswith(val, "mi") && !startswith(val, "mc")
      println(val)
      global n += 1
    end
  end
end
println("-------------------------------------------------------------")
println()
println("-------------------------------------------------------------")
println("-----------------Check errors in structures------------------")
println("-------------------------------------------------------------")
for structure in f_list
  if !(structure in c_list)
    println("The structure $structure can't be find in a header file.")
    global n += 1
  else
    f_nfields = length(f_structures[structure])
    c_nfields = length(c_structures[structure])
    if f_nfields != c_nfields
      println("The structure $structure has missing attributes ($c_nfields / $f_nfields).")
      global n += 1
    else
      for i = 1:c_nfields
        c_field = c_structures[structure][i]
        f_field = f_structures[structure][i]
        if c_field != f_field
          println("The field $i of the structure $structure is not consistent ($c_field / $f_field).")
          global n += 1
        else
          c_type = c_types[structure][i]
          f_type = f_types[structure][i]
          if c_type != f_type
            println("The type of field $(c_field) of the structure $structure is not consistent ($c_type / $f_type).")
            global n += 1
          end
        end
      end
    end
  end
end
println("-------------------------------------------------------------")

@test n == 0
