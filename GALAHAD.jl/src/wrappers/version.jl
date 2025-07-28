export version_galahad

function version_galahad(::Type{Float32}, ::Type{Int32})
  major = Ref{Int32}(0)
  minor = Ref{Int32}(0)
  patch = Ref{Int32}(0)
  @ccall libgalahad_single.version_galahad(major::Ref{Int32}, minor::Ref{Int32}, patch::Ref{Int32})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

function version_galahad(::Type{Float64}, ::Type{Int32})
  major = Ref{Int32}(0)
  minor = Ref{Int32}(0)
  patch = Ref{Int32}(0)
  @ccall libgalahad_double.version_galahad(major::Ref{Int32}, minor::Ref{Int32}, patch::Ref{Int32})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

function version_galahad(::Type{Float128}, ::Type{Int32})
  major = Ref{Int32}(0)
  minor = Ref{Int32}(0)
  patch = Ref{Int32}(0)
  @ccall libgalahad_quadruple.version_galahad(major::Ref{Int32}, minor::Ref{Int32}, patch::Ref{Int32})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

function version_galahad(::Type{Float32}, ::Type{Int64})
  major = Ref{Int64}(0)
  minor = Ref{Int64}(0)
  patch = Ref{Int64}(0)
  @ccall libgalahad_single_64.version_galahad(major::Ref{Int64}, minor::Ref{Int64}, patch::Ref{Int64})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

function version_galahad(::Type{Float64}, ::Type{Int64})
  major = Ref{Int64}(0)
  minor = Ref{Int64}(0)
  patch = Ref{Int64}(0)
  @ccall libgalahad_double_64.version_galahad(major::Ref{Int64}, minor::Ref{Int64}, patch::Ref{Int64})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end

function version_galahad(::Type{Float128}, ::Type{Int64})
  major = Ref{Int64}(0)
  minor = Ref{Int64}(0)
  patch = Ref{Int64}(0)
  @ccall libgalahad_quadruple_64.version_galahad(major::Ref{Int64}, minor::Ref{Int64}, patch::Ref{Int64})::Cvoid
  return VersionNumber(major[], minor[], patch[])
end
