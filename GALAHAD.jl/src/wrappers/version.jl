export galahad_version

function galahad_version()
  major = Ref{Cint}(0)
  minor = Ref{Cint}(0)
  patch = Ref{Cint}(0)
  @ccall libgalahad_double.version_galahad(major::Ref{Cint}, minor::Ref{Cint}, patch::Ref{Cint})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end
