"""DSP stages, one module per stage. Each works on (n_samples, channels) float64 arrays at 48 kHz.

====  ===============================  ============================  ======================
Step  Stage                            Module                        Used in
====  ===============================  ============================  ======================
0     declip                           :mod:`.declip`                v1, v3
1     engine orders                    :mod:`.engine`                v1; v3 if engine_notch
2     auto high-pass                   :mod:`.filters`               v1, v3
2b    persistent tonal lines           :mod:`.tonal`                 v1; v3 (horn-safe)
2c    auto low-pass                    :mod:`.filters`               v1, v3
3     click / rattle repair            :mod:`.clicks`                v1
3w    wind (two microphones)           :mod:`.wind`                  v3
4     noise floor + Wiener gain        :mod:`.noise`                 v1, v3 (own settings)
====  ===============================  ============================  ======================

The order is composed in :func:`carnoise.pipeline.denoise` (v1) and ``_denoise_v3``.
"""
from .channels import as_channels, select_channels
from .clicks import repair_clicks
from .declip import declip, declip_channels
from .engine import remove_engine_orders
from .filters import auto_highpass, auto_lowpass, highpass, lowpass
from .noise import noise_psd, wiener_gain
from .spectral import istft, stft
from .tonal import remove_tonal_lines

__all__ = [
    "as_channels", "select_channels", "repair_clicks", "declip", "declip_channels",
    "remove_engine_orders", "auto_highpass", "auto_lowpass", "highpass", "lowpass",
    "noise_psd", "wiener_gain", "istft", "stft", "remove_tonal_lines",
]
