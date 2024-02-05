using Test

global m = 0
global n = 0
excluded_files = ["blas_original.f", "blas_original.f90",
                  "lapack_original.f", "lapack_original.f90",
                  "ieeeck_original.f", "noieeeck_original.f"]

# Function to easily get the extension of a file
function file_extension(file::String)
  pos_dot = findfirst(==('.'), file)
  extension = pos_dot == nothing ? "" : file[pos_dot+1:end]
  return extension
end

function append_macros!(macros::Vector{Tuple{String,String}}, path::String)
  str = read(path, String)
  lines = split(str, '\n')
  for line in lines
    if startswith(line, "#define")
      tab = split(line, " ", keepempty=false)
      symbol = tab[2]
      pp_symbol = tab[3]
      push!(macros, (symbol, pp_symbol))
      # if length(pp_symbol) > 31
      #   println("The symbol $(pp_symbol) has more than 31 characters.")
      #   global m = m+1
      # end
    end
  end
  return macros
end

macros = Tuple{String,String}[]
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_modules.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_blas.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_lapack.h"))
# append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "cutest_routines_single.h"))
# append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "cutest_routines_double.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_cfunctions.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "galahad_kinds.h"))
append_macros!(macros, joinpath(@__DIR__, "..", "..", "include", "spral_procedures.h"))

# Check the number of characters
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "..", "..", "src"))
  for file in files
    file_extension(file) ∈ ("f", "f90", "F90") || continue
    (file ∈ excluded_files) && continue
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
      for (symbol, pp_symbol) in macros
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
