module GALAHADCUTEstExt

import GALAHAD
import CUTEst
using Libdl

function GALAHAD.run_sif(solver::Symbol, precision::Symbol, sif::String, args...; decode::Bool=true, verbose::Bool=false)
	if decode
		CUTEst.sifdecoder(sif, args...; verbose, precision)
		CUTEst.build_libsif(sif; precision)
	end
	path_libsif = joinpath(CUTEst.libsif_path, "lib$(sif)_$(precision).$dlext")
	path_outsdif = joinpath(CUTEst.libsif_path, CUTEst._name_outsdif(sif, precision))
	if isfile(path_libsif) && isfile(path_outsdif)
		GALAHAD.run_sif(Val(solver), Val(precision), path_libsif, path_outsdif)
	else
		error("The SIF problem $sif was not decoded.")
	end
end

end  # module GALAHADCUTEstExt
