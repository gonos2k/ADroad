"""Net radiation (Karsisto 2024, eq 2 & 9).

    R_net = (1 - albedo)*SW*c_SW + emiss*LW*c_LW - emiss*sigma*T_sK^4

Backend-neutral (works on float or NumPy arrays). The expression is written to
match RoadSurf-Python `BalanceModel.CalcRNet` bit-for-bit (T^4 via two squarings),
so python_compat G0a parity holds exactly.
"""

from __future__ import annotations


def calc_rnet(emiss, sb_const, tsurf_ave, albedo, sw, lw,
              sw_radcof=1.0, lw_radcof=1.0):
    tsurf_k = tsurf_ave + 273.15
    tsurf_k2 = tsurf_k * tsurf_k
    rbb = emiss * sb_const * (tsurf_k2 * tsurf_k2)          # black-body emission
    return (1.0 - albedo) * sw * sw_radcof + emiss * lw * lw_radcof - rbb
