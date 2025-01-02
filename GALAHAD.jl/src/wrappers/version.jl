export version_galahad

function version_galahad()
  major = Ref{Int32}(0)
  minor = Ref{Int32}(0)
  patch = Ref{Int32}(0)
  @ccall libgalahad_double.version_galahad(major::Ref{Int32}, minor::Ref{Int32}, patch::Ref{Int32})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

export version_galahad_64

function version_galahad_64()
  major = Ref{Int64}(0)
  minor = Ref{Int64}(0)
  patch = Ref{Int64}(0)
  @ccall libgalahad_double_64.version_galahad_64(major::Ref{Int64}, minor::Ref{Int64}, patch::Ref{Int64})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end
