export galahad_f, galahad_g, galahad_h
export galahad_hprod, galahad_shprod
export galahad_prec, galahad_constant_prec
export galahad_fgh, galahad_fc, galahad_gj, galahad_hl

for (T, INT, suffix) in ((:Float32 , :Int32, "_s"   ),
                         (:Float32 , :Int64, "_s_64"),
                         (:Float64 , :Int32, "_d"   ),
                         (:Float64 , :Int64, "_d_64"),
                         (:Float128, :Int32, "_q"   ),
                         (:Float128, :Int64, "_q_64"))

  julia_galahad_f_r = Symbol(:julia_galahad_f, suffix)
  julia_galahad_g_r = Symbol(:julia_galahad_g, suffix)
  julia_galahad_h_r = Symbol(:julia_galahad_h, suffix)
  julia_galahad_hprod_r = Symbol(:julia_galahad_hprod, suffix)
  julia_galahad_shprod_r = Symbol(:julia_galahad_shprod, suffix)
  julia_galahad_fgh_r = Symbol(:julia_galahad_fgh, suffix)
  julia_galahad_fc_r = Symbol(:julia_galahad_fc, suffix)
  julia_galahad_gj_r = Symbol(:julia_galahad_gj, suffix)
  julia_galahad_hl_r = Symbol(:julia_galahad_hl, suffix)
  julia_galahad_prec_r = Symbol(:julia_galahad_prec, suffix)
  julia_galahad_constant_prec_r = Symbol(:julia_galahad_constant_prec, suffix)

  @eval begin
    function $julia_galahad_f_r(n::$INT, x::Ptr{$T}, f::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_f(_x, _f, _userdata)
      else
        _userdata.eval_f(_x, _f)
      end
      return $INT(0)
    end

    galahad_f(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_f_r, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_g_r(n::$INT, x::Ptr{$T}, g::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _g = unsafe_wrap(Vector{$T}, g, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_g(_x, _g, _userdata)
      else
        _userdata.eval_g(_x, _g)
      end
      return $INT(0)
    end

    galahad_g(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_g_r, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_h_r(n::$INT, ne::$INT, x::Ptr{$T}, hval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _hval = unsafe_wrap(Vector{$T}, hval, ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_h(_x, _hval, _userdata)
      else
        _userdata.eval_h(_x, _hval)
      end
      return $INT(0)
    end

    galahad_h(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_h_r, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_hprod_r(n::$INT, x::Ptr{$T}, u::Ptr{$T}, v::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_hprod(_x, _u, _v, got_h, _userdata)
      else
        _userdata.eval_hprod(_x, _u, _v, got_h)
      end
      return $INT(0)
    end

    galahad_hprod(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_hprod_r, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_shprod_r(n::$INT, x::Ptr{$T}, nnz_v::$INT, index_nz_v::Ptr{$INT},
                                     v::Ptr{$T}, nnz_u::Ptr{$INT}, index_nz_u::Ptr{$INT},
                                     u::Ptr{$T}, got_h::Bool, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _index_nz_v = unsafe_wrap(Vector{$INT}, index_nz_v, nnz_v)
      _nnz_u = unsafe_wrap(Vector{$INT}, nnz_u, 1)
      _index_nz_u = unsafe_wrap(Vector{$INT}, index_nz_u, n)  # Is it right?
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_shprod(_x, nnz_v, _index_nz_v, _v, _nnz_u, _index_nz_u, _u, got_h, _userdata)
      else
        _userdata.eval_shprod(_x, nnz_v, _index_nz_v, _v, _nnz_u, _index_nz_u, _u, got_h)
      end
      return $INT(0)
    end

    galahad_shprod(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_shprod_r, $INT, ($INT, Ptr{$T}, $INT, Ptr{$INT}, Ptr{$T}, Ptr{$INT}, Ptr{$INT}, Ptr{$T}, Bool, Ptr{Cvoid}))

    function $julia_galahad_fgh_r(x::$T, f::Ptr{$T}, g::Ptr{$T}, h::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _g = unsafe_wrap(Vector{$T}, g, 1)
      _h = unsafe_wrap(Vector{$T}, h, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _f[1] = _userdata.eval_f(x, _userdata)
        _g[1] = _userdata.eval_g(x, _userdata)
        _h[1] = _userdata.eval_h(x, _userdata)
      else
        _f[1] = _userdata.eval_f(x)
        _g[1] = _userdata.eval_g(x)
        _h[1] = _userdata.eval_h(x)
      end
      return $INT(0)
    end

    galahad_fgh(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_fgh_r, $INT, ($T, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_fc_r(n::$INT, m::$INT, x::Ptr{$T}, f::Ptr{$T}, c::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _c = unsafe_wrap(Vector{$T}, c, m)
      _f = unsafe_wrap(Vector{$T}, f, 1)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_fc(_x, _f, _c, _userdata)
      else
        _userdata.eval_fc(_x, _f, _c)
      end
      return $INT(0)
    end

    galahad_fc(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_fc_r, $INT, ($INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_gj_r(n::$INT, m::$INT, J_ne::$INT, x::Ptr{$T}, g::Ptr{$T}, jval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _g = unsafe_wrap(Vector{$T}, g, n)
      _jval = unsafe_wrap(Vector{$T}, jval, J_ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_gj(_x, _g, _jval, _userdata)
      else
        _userdata.eval_gj(_x, _g, _jval)
      end
      return $INT(0)
    end

    galahad_gj(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_gj_r, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_hl_r(n::$INT, m::$INT, H_ne::$INT, x::Ptr{$T}, y::Ptr{$T}, hval::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _y = unsafe_wrap(Vector{$T}, y, m)
      _hval = unsafe_wrap(Vector{$T}, hval, H_ne)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_hl(_x, _y, _hval, _userdata)
      else
        _userdata.eval_hl(_x, _y, _hval)
      end
      return $INT(0)
    end

    galahad_hl(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_hl_r, $INT, ($INT, $INT, $INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_prec_r(n::$INT, x::Ptr{$T}, u::Ptr{$T}, v::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _x = unsafe_wrap(Vector{$T}, x, n)
      _u = unsafe_wrap(Vector{$T}, u, n)
      _v = unsafe_wrap(Vector{$T}, v, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_prec(_x, _u, _v, _userdata)
      else
        _userdata.eval_prec(_x, _u, _v)
      end
      return $INT(0)
    end

    galahad_prec(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_prec_r, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

    function $julia_galahad_constant_prec_r(n::$INT, v::Ptr{$T}, p::Ptr{$T}, userdata::Ptr{Cvoid})::$INT
      _v = unsafe_wrap(Vector{$T}, v, n)
      _p = unsafe_wrap(Vector{$T}, p, n)
      _userdata = unsafe_pointer_to_objref(userdata)
      if _userdata.pass_userdata
        _userdata.eval_prec(_v, _p, _userdata)
      else
        _userdata.eval_prec(_v, _p)
      end
      return $INT(0)
    end

    galahad_constant_prec(::Type{$T}, ::Type{$INT}) = @cfunction($julia_galahad_constant_prec_r, $INT, ($INT, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))
  end
end
