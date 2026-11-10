"""DSP stages, one module per stage. Each works on (n_samples, channels) float64 arrays at 48 kHz.

====  ===============================  ============================
Step  Stage                            Module
====  ===============================  ============================
0     declip                           :mod:`.declip`
1     engine orders                    :mod:`.engine`
2     auto high-pass                   :mod:`.filters`
2b    persistent tonal lines           :mod:`.tonal`
2c    auto low-pass                    :mod:`.filters`
3     click / rattle repair            :mod:`.clicks`
4     noise floor + Wiener gain        :mod:`.noise`
====  ===============================  ============================

The order is composed in :func:`carnoise.pipeline.denoise`.
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
