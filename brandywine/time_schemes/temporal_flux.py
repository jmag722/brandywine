from enum import IntEnum, unique
import brandywine.time_schemes.runge_kutta as rk

class TemporalFluxMapper:
    @unique
    class TemporalFluxId(IntEnum):
        RK1 = 0

    _time_flux_str2id = {
        "rk1": TemporalFluxId.RK1,
        "forward-euler": TemporalFluxId.RK1,
        "fe": TemporalFluxId.RK1
    }
    _time_flux_fmap = {
        TemporalFluxId.RK1: rk.runge_kutta1,
    }

    @classmethod
    def get_time_scheme(cls, flux_name:str):
        if flux_name.lower() not in cls._time_flux_str2id:
            raise KeyError(f"Temporal flux type `{flux_name}` is not supported.")
        flux_id = cls._time_flux_str2id[flux_name.lower()]
        return cls._time_flux_fmap[flux_id]