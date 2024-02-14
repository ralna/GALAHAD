export galahad_linear_solver

function galahad_linear_solver(solver::String)
  nchar = length(solver)
  (nchar > 30) && error("The solver can't have more than 30 characters.")

  result = zeros(Int8, 31)
  for i in 1:nchar
    result[i] = Cchar(solver[i])
  end
  for i in nchar+1:30
    result[i] = Cchar(' ')
  end

  return ntuple(i -> result[i], 31)
end
