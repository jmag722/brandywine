from enum import IntEnum, unique
import brandywine.spatial_schemes.lax_friedrich as lf
import brandywine.spatial_schemes.lax_wendroff as lw

@unique
class InviscidFluxId(IntEnum):
    LAX_FRIEDRICH = 0
    LAX_WENDROFF = 1

_inviscid_flux_str2id = {
    "lax": InviscidFluxId.LAX_FRIEDRICH,
    "lax-friedrich": InviscidFluxId.LAX_FRIEDRICH,
    "lf": InviscidFluxId.LAX_FRIEDRICH,

    "lax-wendroff": InviscidFluxId.LAX_WENDROFF,
    "lw": InviscidFluxId.LAX_WENDROFF
}
_inviscid_flux_fmap = {
    InviscidFluxId.LAX_FRIEDRICH: lf.spatial_derivative,
    InviscidFluxId.LAX_WENDROFF: lw.spatial_derivative
}

def get_inviscid_scheme(flux_name:str):
    if flux_name.lower() not in _inviscid_flux_str2id:
        raise KeyError(f"Inviscid flux `{flux_name}` is not supported.")
    flux_id = _inviscid_flux_str2id[flux_name.lower()]
    return _inviscid_flux_fmap[flux_id]