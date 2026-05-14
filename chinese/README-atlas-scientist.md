<div align="center">

<img src="https://img.shields.io/badge/SAR-Sentinel--1%20%7C%20ALOS--2%20%7C%20NISAR-0a7c6e?style=for-the-badge&logo=satellite&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/astropy-Astronomy%20Core-ff6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Physics-Validated-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AI%20Agent-Multi--Hazard-blueviolet?style=for-the-badge"/>

# 🛰️ ATLAS\_SCIENTIST

### *基于物理验证的多灾害InSAR分析AI智能体*

**实时地表形变智能 — 将卫星雷达、大气物理学、海洋学和启发式风险评分整合到单一推理层中。**

*地面沉降 · 海上SAR搜救 · 野火周界 · 火山活动 · 滑坡预测*

[概述](#概述) · [物理引擎](#物理引擎) · [多灾害模块](#多灾害模块) · [ATLAS评分](#atlas评分) · [实时融合](#实时数据融合) · [技术栈](#科学技术栈)

</div>

---

## 概述

ATLAS_SCIENTIST是一个**专业AI推理智能体**，旨在作为任何使用干涉合成孔径雷达（InSAR）检测、量化和预测地表形变事件系统的物理同行评审员。

它不是通用助手。它是一个**领域专家** — 专门训练用于以下领域的思考：

- SAR干涉测量数学
- 大气和电离层物理学
- 三维地球大地测量学（WGS84椭球模型）
- 多信号启发式风险量化
- 来自卫星、气象和海洋API的实时传感器融合

### 核心问题

InSAR可以从**700公里高度检测到小至3毫米的地面运动** — 但原始相位测量与可操作情报之间的物理学是一个充满无声错误的雷区：

```
φ_measured ≠ φ_deformation

卫星测量的是相位混合物。将信号从噪声中分离
需要经过验证的物理学 — 而非假设。
```

ATLAS_SCIENTIST在每一步都强制执行该验证。

### 设计哲学

```
                    ┌──────────────────────────────────┐
                    │        AI推理层        │
                    │         ATLAS_SCIENTIST           │
                    │                                  │
                    │  "这个方程是否尊重     │
                    │   地球的物理学？"         │
                    └──────────────┬───────────────────┘
                                   │ 验证
                    ┌──────────────▼───────────────────┐
                    │       实现层        │
                    │  （引擎代码、管道、API）   │
                    └──────────────────────────────────┘

Principle: Separate who implements from who 验证 physics.
如同科学同行评审 — 应用于软件。
```

---

## 物理引擎

### 1. InSAR相位分解

每个干涉相位测量都是物理贡献的叠加。忽略任何项都会产生虚假的形变信号。

```
φ_interferogram = φ_deformation + φ_topography + φ_troposphere 
                + φ_ionosphere + φ_orbit + φ_noise
```

**ATLAS_SCIENTIST的职责：**验证每个项都已被校正、限定，或明确声明为残余不确定性。

---

### 2. 相位 → 位移转换

基本InSAR观测量 → 物理量映射：

```
d_LOS = (φ · λ) / (4π)
```

**传感器特定波长：**

| 卫星 | 波段 | λ (cm) | 1条纹 = |
|---|---|---|---|
| Sentinel-1 A/B | C | 5.6 cm | 2.8 cm LOS |
| ALOS-2 | L | 23.6 cm | 11.8 cm LOS |
| NISAR (2024+) | L + S | 23.6 / 9.3 cm | dual-freq |
| TerraSAR-X | X | 3.1 cm | 1.55 cm LOS |
| COSMO-SkyMed | X | 3.1 cm | 1.55 cm LOS |

```python
def phase_to_displacement_mm(phase_rad: float, wavelength_m: float = 0.056) -> float:
    """
    Convert InSAR phase to Line-of-Sight displacement in millimeters.
    
    Sign convention: positive = moving away from satellite.
    For descending Sentinel-1: positive ≈ subsidence in most geometries.
    
    Args:
        phase_rad: Unwrapped interferometric phase in radians
        wavelength_m: Sensor wavelength (default: Sentinel-1 C-band = 0.056 m)
    
    Returns:
        Displacement in millimeters (LOS direction)
    """
    return (phase_rad * wavelength_m / (4 * math.pi)) * 1000.0

# Verification — always run this:
assert abs(phase_to_displacement_mm(2 * math.pi, 0.056) - 28.0) < 0.01, \
    "Sentinel-1: one full fringe must equal 28.0 mm"
```

---

### 3. LOS → 三维位移分解

卫星几何将LOS测量转换为真实世界分量：

```
d_LOS = −sin(θ)·cos(α)·d_E + sin(θ)·sin(α)·d_N − cos(θ)·d_U
```

Where:
- `θ` = incidence angle (Sentinel-1: 29°–46° depending on swath)
- `α` = heading angle (ascending ≈ −10°, descending ≈ −170° from North)
- `d_E, d_N, d_U` = East, North, Up displacement components

**使用升轨 + 降轨对** (Sentinel-1 tracks A and D over same area):

```python
import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u

def decompose_los_to_3d(
    d_asc_mm: float, d_desc_mm: float,
    theta_asc_deg: float, theta_desc_deg: float,
    alpha_asc_deg: float, alpha_desc_deg: float
) -> tuple[float, float, float]:
    """
    Decompose ascending + descending LOS into East, North, Up.
    Returns (d_east_mm, d_north_mm, d_up_mm).
    Vertical sensitivity ~80% of LOS for typical Sentinel-1 geometry.
    """
    theta_a = np.radians(theta_asc_deg)
    theta_d = np.radians(theta_desc_deg)
    alpha_a = np.radians(alpha_asc_deg)
    alpha_d = np.radians(alpha_desc_deg)

    A = np.array([
        [-np.sin(theta_a)*np.cos(alpha_a),  np.sin(theta_a)*np.sin(alpha_a), -np.cos(theta_a)],
        [-np.sin(theta_d)*np.cos(alpha_d),  np.sin(theta_d)*np.sin(alpha_d), -np.cos(theta_d)],
    ])
    b = np.array([d_asc_mm, d_desc_mm])
    # Least-squares solve (overdetermined system)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1]), float(result[2])
```

---

### 4. 三维地球模型 — WGS84椭球体

**关键维度校正：**地球不是球体。所有坐标转换必须考虑椭球体：

```python
from astropy.coordinates import EarthLocation, ITRS, GCRS
import astropy.units as u
from astropy.time import Time

# WGS84 parameters (ATLAS_SCIENTIST enforces these constants)
WGS84_A = 6_378_137.0        # Semi-major axis (m) — equatorial radius
WGS84_B = 6_356_752.314245  # Semi-minor axis (m) — polar radius
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # First eccentricity squared

def meters_per_degree_lat(lat_deg: float) -> float:
    """Returns meters per degree of latitude at given latitude."""
    lat_rad = math.radians(lat_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(lat_rad)**2)**1.5
    return math.radians(1) * M  # ~110,574 m/deg at equator, ~111,694 at poles

def meters_per_degree_lon(lat_deg: float) -> float:
    """Returns meters per degree of longitude at given latitude.
    
    ⚠️ ATLAS_SCIENTIST INVARIANT: cos(lat) is MANDATORY here.
    Omitting it causes 8% error at lat=−23° (Atacama) and 50% error at lat=−60°.
    """
    lat_rad = math.radians(lat_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    return math.radians(1) * N * math.cos(lat_rad)

# Quick reference:
# At lat=0°  (equator):  m/deg_lon = 111,319 m
# At lat=−23° (Atacama): m/deg_lon = 102,470 m  ← cos(23°) = 0.921
# At lat=−33° (Santiago): m/deg_lon = 93,274 m  ← cos(33°) = 0.839
# At lat=−55° (Patagonia): m/deg_lon = 63,849 m ← cos(55°) = 0.574
```

---

### 5. 时间相干性

测量两次SAR获取之间的相关质量：

```
γ = |⟨s₁ · s₂*⟩| / √(|⟨s₁ · s₁*⟩| · |⟨s₂ · s₂*⟩|)     γ ∈ [0, 1]
```

**按地表覆盖的物理解释：**

| 地表类型 | C-band 12-day | L-band 46-day | Usability |
|---|---|---|---|
| Bare rock / desert | 0.60 – 0.80 | 0.75 – 0.95 | Excellent |
| Urban / infrastructure | 0.55 – 0.85 | 0.70 – 0.90 | Excellent |
| Mining tailings (active) | 0.35 – 0.60 | 0.50 – 0.75 | Good |
| Agricultural fields | 0.15 – 0.45 | 0.40 – 0.70 | Seasonal |
| Dense forest | 0.05 – 0.25 | 0.20 – 0.55 | Poor (C) / OK (L) |
| Ocean surface | < 0.05 | < 0.05 | Not applicable |

**最低阈值：**
- Arid/desert zones: γ ≥ 0.35
- Humid tropical: γ ≥ 0.20 (L-band preferred)

---

### 6. 大气校正 — Smith-Weintraub折射率

对流层使雷达信号弯曲，产生表观形变。 The Smith-Weintraub equation models the refractivity N of the atmosphere:

```
N = k₁·(P/T) + k₂·(e/T) + k₃·(e/T²)
```

- k₁ = 77.6 K/hPa (dry term — pressure dominant)
- k₂ = 70.4 K/hPa (wet term — partial water vapor pressure)
- k₃ = 3.75 × 10⁵ K²/hPa (liquid water term)
- P = total pressure (hPa), e = partial water vapor pressure (hPa), T = temperature (K)

**柱积分产生的相位延迟：**

```
φ_tropo = (4π / λ) · (10⁻⁶ / cos θ) · ∫ N dz
```

**校正数据源（免费）：**

| 来源 | 延迟 | 分辨率 | 用途 |
|---|---|---|---|
| ERA5 (Copernicus CDS) | 5 days | 0.25° | Operational reanalysis |
| GACOS | 1-2 days | ~90m | InSAR-specific, recommended |
| HRES-ECMWF | 6 hours | 0.1° | Near real-time |
| Open-Meteo API | 1 hour | 0.25° | Free, no API key needed |

```python
import openmeteo_requests

def fetch_tropospheric_params(lat: float, lon: float, time_utc: str) -> dict:
    """
    Fetch atmospheric parameters for tropospheric correction.
    Open-Meteo: free, no API key, 0.25° resolution.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["surface_pressure", "temperature_2m",
                   "relativehumidity_2m", "dewpoint_2m"],
        "start_date": time_utc[:10], "end_date": time_utc[:10]
    }
    # Returns P, T, e for Smith-Weintraub computation
    ...
```

---

### 7. 电离层校正（L波段关键）

For L-band sensors (ALOS-2, NISAR), ionospheric delay is significant:

```
φ_iono = −(4π / λ) · (40.28 / f²) · ΔTEC
```

- f = carrier frequency (Hz)
- ΔTEC = differential Total Electron Content (TECU = 10¹⁶ electrons/m²)
- Negative sign: ionosphere advances phase (opposite to troposphere)

**校正方法：**分频谱法 (available in ISCE3, MintPy) or GPS-derived TEC maps (CODE, JPL).

---

## 多灾害模块

ATLAS_SCIENTIST 验证 physics across five operational hazard domains. Each module uses InSAR as its primary data source, combined with domain-specific physical models.

### 模块1：地面沉降与尾矿坍

**Observable:** LOS velocity (mm/day) and acceleration (mm/day²)

**Critical thresholds (empirical):**

```python
# Subsidence rate classification
VELOCITY_THRESHOLDS = {
    "stable":   (0.0,  2.0),   # mm/day — seasonal/noise
    "moderate": (2.0,  5.0),   # mm/day — monitor closely
    "active":   (5.0,  15.0),  # mm/day — heightened concern
    "critical": (15.0, float("inf")),  # mm/day — emergency response
}

# Acceleration bypass: if d²u/dt² ≥ 3.0 mm/day², immediate alert
# regardless of current velocity (tertiary creep signature)
ACCELERATION_BYPASS_MM_DAY2 = 3.0
```

**Physical model:** InSAR SBAS time series → velocity field → acceleration → ATLAS Score

---

### 模块2：海上SAR搜救 — 漂移物理学

When a person or vessel is lost at sea, InSAR is not the sensor — but the **same agent architecture** handles the drift physics validation.

**Drift equation:**

```
dX/dt = u_current(x,t) + α_L · u_wind(x,t) + u_stokes(x,t) + ε(t)
```

**Stokes drift (wave-induced transport):**

```
V_stokes = (π² · Hs² · f_p) / (2 · g · T_p²)
```

**Turbulent diffusion:**

```
ε(t) ~ N(0, σ_d),    σ_d = √(2 · K · Δt),    K = 1.0 m²/s
```

**Coordinate update (WGS84 — ellipsoid correction mandatory):**

```python
def update_particle_position(lat: float, lon: float,
                              u_ms: float, v_ms: float,
                              dt_s: float) -> tuple[float, float]:
    """
    Move a particle by velocity (u, v) in m/s over dt_s seconds.
    
    ⚠️ cos(lat) is MANDATORY in dlon. Without it:
       error = 8% at lat=−23°, 50% at lat=−60°
    """
    dlat = (v_ms * dt_s) / (WGS84_A * (1 - WGS84_E2 * math.sin(math.radians(lat))**2)**0.5)
    dlon = (u_ms * dt_s) / (WGS84_A * math.cos(math.radians(lat)) *
                             (1 - WGS84_E2 * math.sin(math.radians(lat))**2)**0.5)
    return lat + math.degrees(dlat), lon + math.degrees(dlon)
```

**Monte Carlo search zones:** 1,000+ particles → PCA → 1σ/2σ/3σ probability ellipses for SAR rescue teams.

---

### 模块3：野火周界追踪

**SAR advantage:** C-band and L-band penetrate smoke. Optical satellites cannot.

**Physical model — Rothermel fire spread:**

```
R = (I_R · ξ) / (ρ_b · ε · Q_ig)

Where:
  I_R  = reaction intensity (BTU/ft/min)
  ξ    = propagating flux ratio
  ρ_b  = bulk density of fuel bed
  ε    = effective heating number
  Q_ig = heat of pre-ignition
```

**SAR application:** Sentinel-1 backscatter change detection + InSAR coherence loss → burned area perimeter (coherence drops from ~0.6 to <0.1 in burned zones over 6-12 day interval).

**Combination with weather:**

```python
# Wind-driven fire spread direction from ERA5 / Open-Meteo
wind_speed_ms = weather["wind_speed_10m"]   # m/s
wind_dir_deg  = weather["wind_direction_10m"] # meteorological
spread_rate   = rothermel_R * (1 + k_wind * wind_speed_ms**2)
```

---

### 模块4：火山活动

**InSAR is the primary tool for pre-eruptive deformation.**

**Mogi point source model** (spherical magma chamber):

```
u_r = (ΔP · a³) / (μ · r³) · r_hat
u_z = (ΔP · a³) / (μ · r³) · d/r

Where:
  ΔP = pressure change in magma chamber
  a  = chamber radius
  μ  = crustal shear modulus (~30 GPa)
  d  = chamber depth
  r  = distance from epicenter
```

**LOS projection of Mogi displacement:**

```python
from astropy.coordinates import SphericalRepresentation
import astropy.units as u

def mogi_los_displacement(
    lat_obs: float, lon_obs: float, depth_m: float,
    delta_pressure_pa: float, radius_m: float,
    lat_source: float, lon_source: float,
    incidence_deg: float, heading_deg: float
) -> float:
    """Returns predicted LOS displacement in mm for Mogi source."""
    mu = 30e9  # Pa — crustal shear modulus
    dx, dy, dz = _mogi_3d(lat_obs, lon_obs, depth_m,
                           delta_pressure_pa, radius_m,
                           lat_source, lon_source, mu)
    # Project onto LOS unit vector
    los_e = -math.sin(math.radians(incidence_deg)) * math.cos(math.radians(heading_deg))
    los_n =  math.sin(math.radians(incidence_deg)) * math.sin(math.radians(heading_deg))
    los_u = -math.cos(math.radians(incidence_deg))
    return (los_e*dx + los_n*dy + los_u*dz) * 1000.0  # → mm
```

---

### 模块5：滑坡运动学

**InSAR resolves displacement rates but not failure planes.** Combining InSAR with slope geometry:

```
# Factor of Safety — infinite slope approximation
FS = (c + (γ_s - γ_w · r_u) · z · cos²β · tan φ') / (γ_s · z · sin β · cos β)

Where:
  c    = cohesion (kPa)
  γ_s  = soil unit weight
  γ_w  = water unit weight
  r_u  = pore pressure ratio (from InSAR proxy or piezometer)
  z    = slope depth
  β    = slope angle
  φ'   = effective friction angle
```

**InSAR contribution:** LOS velocity field → slope-parallel velocity → kinematic classification (slow creep / seasonal / accelerating toward failure).

---

## ATLAS评分 — 启发式框架

### 二值阈值的问题

Traditional monitoring systems use binary thresholds: "velocity > 5 mm/day → alert". This:

- Ignores signal age (an old sensor reading shouldn't have the same weight as today's)
- Ignores acceleration (the slope of velocity matters more than its current value)
- Cannot combine heterogeneous signals (InSAR + seismic + rain + pore pressure)
- Produces false negatives when each individual signal is below threshold but combined risk is high

### 解决方案：多信号时间启发式

```
R = Σᵢ [wᵢ · φᵢ(sᵢ) · exp(−λᵢ · Δtᵢ)] / Σᵢ [wᵢ · exp(−λᵢ · Δtᵢ)]
```

**变量：**

| 符号 | 含义 | 约束 |
|---|---|---|
| `wᵢ` | Weight of signal i | Σwᵢ = 1.0 (enforced) |
| `φᵢ(sᵢ)` | Activation function of signal i | φᵢ ∈ [0, 1] |
| `λᵢ` | Temporal decay rate | λᵢ > 0 (units: 1/day) |
| `Δtᵢ` | Age of measurement i | Δtᵢ ≥ 0 (days) |
| `R` | Risk score | R ∈ [0, 1] always |

**五个数学不变量** — verified on every algorithm change:

```python
class ATLASInvariants:
    """Property tests that must pass before any production deployment."""

    @staticmethod
    def P1_bounded(score_fn, n=10_000) -> bool:
        """R ∈ [0, 1] for all random inputs."""
        return all(0.0 <= score_fn(random_signals()) <= 1.0
                   for _ in range(n))

    @staticmethod
    def P2_monotone(score_fn) -> bool:
        """Higher deformation magnitude → higher score."""
        low  = score_fn(signals(velocity=1.0))
        high = score_fn(signals(velocity=10.0))
        return high > low

    @staticmethod
    def P3_acceleration_bypass(score_fn) -> bool:
        """Critical acceleration (≥3 mm/day²) forces R ≥ 0.80."""
        return score_fn(signals(acceleration=3.5)) >= 0.80

    @staticmethod
    def P4_temporal_decay(score_fn) -> bool:
        """A 30-day-old signal contributes less than a fresh one."""
        fresh = score_fn(signals(velocity=5.0, age_days=0))
        old   = score_fn(signals(velocity=5.0, age_days=30))
        return old < fresh * 0.60

    @staticmethod
    def P5_weights_sum(weights: dict) -> bool:
        """Weights must sum exactly to 1.0."""
        return abs(sum(weights.values()) - 1.0) < 1e-9
```

### Activation Functions φᵢ

```python
def phi_velocity(v_mm_per_day: float) -> float:
    """
    Normalizes LOS velocity to [0, 1].
    Piecewise linear: stable → active → critical.
    """
    v = abs(v_mm_per_day)
    if v >= 15.0: return 1.00
    if v >= 5.0:  return 0.60 + 0.40 * (v - 5.0) / 10.0
    if v >= 2.0:  return 0.30 + 0.30 * (v - 2.0) / 3.0
    return v / 2.0 * 0.30

def phi_acceleration(a_mm_per_day2: float) -> float:
    """
    Normalizes deformation acceleration to [0, 1].
    Tertiary creep signature (a ≥ 3 mm/day²) → full bypass.
    """
    a = abs(a_mm_per_day2)
    if a >= 3.0: return 1.00   # ← acceleration bypass trigger
    if a >= 1.0: return 0.80
    return a / 1.0 * 0.80
```

### 时间衰减率 — 物理依据

| Signal | λ (1/day) | Half-life | Reasoning |
|---|---|---|---|
| InSAR velocity | 1/14 | 14 days | Sentinel-1 repeat cycle |
| InSAR acceleration | 1/7 | 7 days | Acceleration is more time-sensitive |
| Precipitation | 1/3 | 3 days | Infiltration dissipates in ~3 days |
| Seismic energy | 1/5 | 5 days | Aftershock decay (Omori law) |
| Pore pressure | 1/7 | 7 days | Pressure diffusion timescale |
| Ocean wave height | 1/2 | 2 days | Sea state changes rapidly |
| Fire radiative power | 1/1 | 1 day | Fires evolve in hours |

---

## 实时数据融合

ATLAS_SCIENTIST defines the architecture for combining heterogeneous physical measurements into a unified risk signal.

### 数据源 — 大多数无需API密钥

| 层 | 来源 | 变量 | 延迟 | 费用 |
|---|---|---|---|---|
| SAR | Sentinel-1 (ESA Copernicus) | Phase, coherence, backscatter | 6–12 days | **Free** |
| SAR archive | ASF (Alaska SAF) | All Sentinel-1/ALOS-2 | Historical | **Free** |
| Weather | Open-Meteo | Wind, rain, pressure, humidity | 1h | **Free** |
| Ocean currents | CMEMS Global Analysis | u, v surface velocity | 24h | **Free** |
| Ocean waves | CMEMS WAV | Hs, Tp, wave direction | 6h | **Free** |
| Atmosphere | ERA5 (Copernicus CDS) | Full tropospheric column | 5 days | **Free** |
| Ionosphere | CODE/JPL TEC maps | ΔTEC | 24h | **Free** |
| Seismicity | USGS ComCat API | Magnitude, depth, location | Real-time | **Free** |
| DEM / Topography | Copernicus DEM 30m | Elevation, slope | Static | **Free** |

### 融合管道

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class PhysicalSignal:
    name: str
    value: float          # normalized [0, 1] by φᵢ
    raw_value: float      # original physical quantity with unit
    unit: str
    timestamp_utc: float  # Unix timestamp
    weight: float
    decay_lambda: float   # 1/days
    source: str           # "sentinel-1" | "open-meteo" | "cmems" | ...
    uncertainty: float    # 1σ uncertainty on raw_value

class HazardModule(Protocol):
    """
    Abstract interface for any hazard domain module.
    Modules never import from each other — only from core.
    """
    @property
    def hazard_type(self) -> str: ...
    @property
    def required_signals(self) -> list[str]: ...
    def analyze(self, signals: list[PhysicalSignal]) -> float: ...  # → R

def compute_atlas_score(signals: list[PhysicalSignal],
                        t_now: float) -> float:
    """
    Multi-signal temporal ATLAS Score.
    Guaranteed: result ∈ [0, 1] for any input.
    """
    if not signals:
        return 0.0
    
    numerator, denominator = 0.0, 0.0
    max_acceleration = 0.0
    
    for sig in signals:
        delta_t_days = max(0.0, (t_now - sig.timestamp_utc) / 86_400.0)
        decay = math.exp(-sig.decay_lambda * delta_t_days)
        numerator   += sig.weight * sig.value * decay
        denominator += sig.weight * decay
        
        if "acceleration" in sig.name:
            max_acceleration = max(max_acceleration, abs(sig.raw_value))
    
    raw_score = numerator / max(denominator, 1e-9)
    
    # Acceleration bypass: tertiary creep signature
    if max_acceleration >= 3.0:
        raw_score = max(raw_score, 0.80)
    
    return min(1.0, max(0.0, raw_score))
```

---

## Astropy集成

For precise satellite geometry, time systems, and coordinate transforms:

```python
from astropy.time import Time
from astropy.coordinates import EarthLocation, ITRS, AltAz
import astropy.units as u

def sentinel1_acquisition_to_utc(sensing_time_str: str) -> Time:
    """
    Convert Sentinel-1 annotated sensing time to astropy Time.
    Format: "2024-03-15T06:34:22.123456"
    Uses UTC scale; IERS-A corrections applied automatically.
    """
    return Time(sensing_time_str, format="isot", scale="utc")

def compute_local_incidence_angle(
    lat_deg: float, lon_deg: float, alt_m: float,
    sat_position_ecef: np.ndarray  # [x, y, z] in meters
) -> float:
    """
    Compute exact local incidence angle at a ground point,
    accounting for Earth's curvature (WGS84 ellipsoid).
    
    Returns angle in degrees between radar look direction
    and local vertical at the ground point.
    """
    ground = EarthLocation(lat=lat_deg*u.deg,
                           lon=lon_deg*u.deg,
                           height=alt_m*u.m)
    
    ground_ecef = np.array([
        ground.x.to(u.m).value,
        ground.y.to(u.m).value,
        ground.z.to(u.m).value
    ])
    
    # Look vector: satellite → ground
    look_vec = ground_ecef - sat_position_ecef
    look_vec /= np.linalg.norm(look_vec)
    
    # Local vertical (normal to ellipsoid at ground point)
    # Approximation: ECEF position vector normalized
    local_up = ground_ecef / np.linalg.norm(ground_ecef)
    
    cos_angle = abs(np.dot(look_vec, local_up))
    return math.degrees(math.acos(cos_angle))

# Baseline computation between two SAR acquisitions:
def perpendicular_baseline(pos1_ecef: np.ndarray,
                           pos2_ecef: np.ndarray,
                           look_dir: np.ndarray) -> float:
    """
    Perpendicular baseline in meters.
    Critical for: DEM error estimation, coherence prediction.
    """
    baseline_vec = pos2_ecef - pos1_ecef
    b_par = np.dot(baseline_vec, look_dir)
    b_perp = np.sqrt(np.dot(baseline_vec, baseline_vec) - b_par**2)
    return float(b_perp)
```

---

## 科学技术栈

```
遥感
  asf_search          Sentinel-1 / ALOS-2 scene discovery and download
  pyroSAR + SNAP 10   TOPSAR coregistration, interferogram generation
  MintPy              SBAS / PS time series analysis (conda-forge)
  ISCE3               JPL InSAR processor (NISAR mission standard)

天文学与大地测量
  astropy             Time systems (UTC/TAI/GPS), coordinate transforms, IERS
  pyproj              PROJ-based geodetic transformations (WGS84 ↔ UTM)
  scipy               Signal processing, orbital mechanics, linear algebra

大气
  openmeteo-requests  Weather API client (free, no key)
  copernicusmarine    CMEMS ocean data (currents, waves)
  cdsapi              ERA5 / Copernicus CDS (tropospheric correction)

地理空间
  rasterio / gdal     GeoTIFF I/O, reprojection, raster math
  shapely             Vector geometry (AOI, ellipses, footprints)
  geopandas           地理空间 DataFrames

数值计算 / 机器学习
  numpy               N-dimensional arrays, FFT, linear algebra
  scipy               Optimization (for model inversion), statistics
  xarray              Labeled N-D arrays for SAR time series cubes
  netCDF4 / h5py      Scientific data I/O (SNAP output, MintPy HDF5)
  pydantic v2         Runtime validation of all physical data contracts
  scikit-learn        Phase error clustering, outlier detection (Etapa D+)
  torch               Deep learning for displacement anomaly detection (future)

测试
  pytest + hypothesis Property-based testing for physics invariants
  mypy                Static type checking (strict mode enforced)
```

---

## 验证协议

在接受任何数学更改之前：

```
ATLAS_SCIENTIST REVIEW CHECKLIST
─────────────────────────────────────────────────────────
PHASE PHYSICS
[ ] Wavelength correct for the specific sensor/band
[ ] Phase-to-displacement conversion verified (assert fringe = λ/2)
[ ] LOS vector construction uses correct incidence + heading angles
[ ] cos(lat) present in ALL longitude-to-meter conversions
[ ] WGS84 ellipsoid used (not spherical Earth approximation)
[ ] Temporal baseline within coherence limits for that sensor+surface

ATMOSPHERIC
[ ] Tropospheric correction applied or explicitly documented as residual
[ ] Ionospheric correction applied for L-band (skip for C-band short baselines)
[ ] ERA5 or GACOS data used (not interpolated screen values)

ATLAS SCORE
[ ] All weights sum exactly to 1.0 (automated test in CI)
[ ] R ∈ [0, 1] for 10,000 random input combinations (property test)
[ ] Acceleration bypass fires at correct threshold
[ ] Temporal decay rates physically justified in code comments
[ ] No magic numbers — all constants named with unit and source

MULTI-HAZARD
[ ] Correct leeway coefficient for maritime object type (IAMSAR source)
[ ] cos(lat) in all particle drift position updates
[ ] Mogi source model parameters within physical bounds
[ ] Rothermel spread rate inputs in correct units (ft-lb-min system)

SIGN-OFF: _________________________ DATE: ____________
```

---

## 测试用例

```python
# tests/test_atlas_physics.py

def test_sentinel1_fringe_is_28mm():
    assert abs(phase_to_displacement_mm(2*math.pi, 0.056) - 28.0) < 0.01

def test_alos2_fringe_is_118mm():
    assert abs(phase_to_displacement_mm(2*math.pi, 0.236) - 118.0) < 0.1

def test_cos_lat_correction_at_minus23():
    """At Atacama latitude, longitude meter-correction must be ~0.921."""
    ratio = meters_per_degree_lon(-23.0) / meters_per_degree_lat(-23.0)
    assert abs(ratio - math.cos(math.radians(23.0))) < 0.005

def test_atlas_score_bounded():
    for _ in range(10_000):
        assert 0.0 <= compute_atlas_score(random_signals(), time.time()) <= 1.0

def test_acceleration_bypass():
    score = compute_atlas_score(signals(acceleration=3.5), time.time())
    assert score >= 0.80

def test_temporal_decay_is_exponential():
    fresh = compute_atlas_score(signals(velocity=5.0, age_days=0), time.time())
    old   = compute_atlas_score(signals(velocity=5.0, age_days=30), time.time())
    assert old < fresh * 0.60

def test_weights_must_sum_to_one():
    with pytest.raises(AssertionError):
        validate_weights({"a": 0.5, "b": 0.4})  # sums to 0.9 → fail
```

---

## 如何使用此智能体

### 1. 作为代码审查步骤

```
Prompt pattern:
"Act as ATLAS_SCIENTIST.
Review the following InSAR processing code for physics correctness.
Check: phase equation, LOS conversion, coordinate system, units.
Do NOT suggest code style changes. Only validate physics.
Flag every violation with: [VIOLATION], the specific error,
and the corrected equation."
```

### 2. 作为方程验证器

```
Prompt pattern:
"Act as ATLAS_SCIENTIST.
I am about to implement the following equation in Python:
[paste equation]
Sensor: Sentinel-1 C-band, descending orbit.
Study area: lat −23°, lon −70°.
Verify: dimensional consistency, sign conventions,
expected numerical output, and known edge cases."
```

### 3. 作为测试用例生成器

```
Prompt pattern:
"Act as ATLAS_SCIENTIST.
For the function phase_to_displacement_mm(phase_rad, wavelength_m),
generate 5 pytest test cases covering:
- Sentinel-1 (λ = 5.6 cm)
- ALOS-2 (λ = 23.6 cm)
- Zero phase → zero displacement
- 4π phase → full two-fringe displacement
- Negative phase (opposite motion direction)"
```

---

## 西班牙语版本

### ¿Qué es ATLAS_SCIENTIST?

ATLAS_SCIENTIST es un agente de IA especializado en ser el **revisor de física** para cualquier sistema que use InSAR para detectar, cuantificar y predecir deformación superficial.

No es un asistente general. Es un especialista en:

- Matemáticas de interferometría SAR
- Física atmosférica e ionosférica
- Geodesia 3D con elipsoide WGS84
- Cuantificación heurística de riesgo multi-señal
- Fusión de datos en tiempo real: satélites + clima + oceanografía

### El Principio Fundamental

```
φ_medido ≠ φ_deformación

El satélite mide una mezcla de fases. Separar señal de ruido
requiere física validada — no suposiciones.
```

### Aplicaciones Multi-Hazard

| Módulo | InSAR | Señales adicionales | Output |
|---|---|---|---|
| Subsidencia / Relaves | SBAS velocidad LOS | Lluvia, sismicidad, presión de poros | ATLAS Score [0-1] |
| Rescate marino | No (modelo de deriva) | CMEMS, oleaje, viento | Elipses de búsqueda 1σ/2σ/3σ |
| Incendios forestales | Coherencia + backscatter | ERA5, viento, humedad | Perímetro + vector de propagación |
| Volcanes | Modelo Mogi LOS | GPS, sismicidad, SO₂ | Inflación magmática estimada |
| Deslizamientos | Velocidad campo vectorial | DEM, pendiente, lluvia | Factor de seguridad + FS temporal |

### Invariantes Físicos que Siempre Verifica

```
1. cos(latitud) en TODA conversión de longitud a metros
2. Longitud de onda correcta para el sensor específico
3. Una franja de fase = λ/2 de desplazamiento (no λ)
4. Corrección troposférica aplicada o declarada como residual
5. Pesos del ATLAS Score suman exactamente 1.0
6. Score R ∈ [0, 1] para toda combinación de entradas posible
```

### Stack Científico Clave

```
astropy         → Sistemas de tiempo (UTC/TAI/GPS), transforms ECEF
pyroSAR + SNAP  → Procesamiento InSAR (coregistro, interferograma)
MintPy          → Series temporales SBAS/PSI
Open-Meteo      → Clima en tiempo real (gratuito, sin API key)
CMEMS           → Corrientes y oleaje oceánico (gratuito)
ERA5            → Corrección troposférica (reanalysis ECMWF)
numpy/scipy     → Álgebra lineal, FFT, estadística
pydantic v2     → Validación de contratos de datos en runtime
```

---

<div align="center">

---

*ATLAS_SCIENTIST — 多灾害InSAR分析物理验证智能体*

*方程高于假设。物理高于启发式。验证高于信任。*

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>
