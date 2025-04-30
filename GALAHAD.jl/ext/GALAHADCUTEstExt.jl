module GALAHADCUTEstExt

import GALAHAD
import CUTEst

function GALAHAD.run_sif(solver::Symbol, precision::Symbol, sif::String, args...; decode::Bool=true,
                         verbose::Bool=false, libsif_folder::String=CUTEst.libsif_path)
  standalone = true
  if decode
    CUTEst.sifdecoder(sif, args...; verbose, precision, libsif_folder)
    CUTEst.build_libsif(sif; precision, standalone, libsif_folder)
  end
  name_libsif = CUTEst._name_libsif(sif, precision; standalone)
  path_libsif = joinpath(libsif_folder, name_libsif)
  name_outsdif = CUTEst._name_outsdif(sif, precision)
  path_outsdif = joinpath(libsif_folder, name_outsdif)
  if isfile(path_libsif) && isfile(path_outsdif)
    GALAHAD.run_sif(Val(solver), Val(precision), path_libsif, path_outsdif)
  else
    error("The SIF problem $sif was not decoded.")
  end
end

end  # module GALAHADCUTEstExt
