type_modifications = Dict("real_wp_" => "Float64",
                          "real_sp_" => "Float32")

function rewrite!(path::String, name::String, optimized::Bool)
  text = read(path, String)
  text = replace(text, "struct " => "mutable struct ")
  if optimized
    for (keys, vals) in type_modifications
      text = replace(text, keys => vals)
    end
  end
  write(path, text)
end
