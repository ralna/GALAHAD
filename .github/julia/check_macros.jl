using Test

global n = 0

# Function to easily get the extension of a file
function file_extension(file::String)
  pos_dot = findfirst(==('.'), file)
  extension = pos_dot == nothing ? "" : file[pos_dot+1:end]
  return extension
end

function append_macros!(macros::Dict{String, String}, path::String)
  str = read(path, String)
  lines = split(str, '\n')
  for line in lines
    if startswith(line, "#define")
      tab = split(line, " ", keepempty=false)
      symbol = tab[2]
      pp_symbol = tab[3]
      if !haskey(macros, symbol)
        macros[symbol] = pp_symbol
      else
        if length(macros[symbol]) < length(pp_symbol)
          macros[symbol] = pp_symbol
        end
      end
    end
  end
  return macros
end

macros = Dict{String, String}()
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_modules_quadruple.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_blas.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_lapack.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "cutest_routines_quadruple.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "hsl_subset.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "hsl_subset_quadruple.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "hsl_subset_ciface.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "hsl_subset_ciface_quadruple.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_cfunctions.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_kinds.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "spral_procedures.h"))

# Check the number of characters
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "src"))
  for file in files
    file_extension(file) ∈ ("f", "f90", "F90") || continue
    path = joinpath(root, file)
    code = read(path, String)
    lines = split(code, '\n')
    for (i, line) in enumerate(lines)
      startswith(line |> strip, "!") && continue
      mapreduce(x -> startswith(line |> strip, x), |, ["C", "&", "*"]) && (file_extension(file) == "f") && continue
      if (file_extension(file) == "f") && (length(line) > 72)
        println("Line $i in the file $file has more than 72 characters.")
        global n = n+1
      end
      if (file_extension(file) ∈ ["f90", "F90"]) && (length(line) > 132)
        println("Line $i in the file $file has more than 132 characters.")
        global n = n+1
      end
      for symbol in keys(macros)
        pp_symbol = macros[symbol]
        if occursin(symbol, line) && (file_extension(file) == "f")
          line2 = replace(line, symbol => pp_symbol)
          if length(line2) > 72
            println("Line $i in the file $file has more than 72 characters if $symbol is replaced by $(pp_symbol).")
            global n = n+1
          end
        end
        if occursin(symbol, line) && (file_extension(file) ∈ ["f90", "F90"])
          line2 = replace(line, symbol => pp_symbol)
          if length(line2) > 132
            println("Line $i in the file $file has more than 132 characters if $symbol is replaced by $(pp_symbol).")
            global n = n+1
          end
        end
      end
    end
  end
end

@test n == 0
