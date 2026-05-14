<div align="center">

<img src="https://img.shields.io/badge/SAR-Sentinel--1%20%7C%20ALOS--2%20%7C%20NISAR-0a7c6e?style=for-the-badge&logo=satellite&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/astropy-Astronomy%20Core-ff6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Physics-Validated-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AI%20Agent-Multi--Hazard-blueviolet?style=for-the-badge"/>

# 🛰️ ATLAS\_SCIENTIST

### *Un Agente de IA Validado Físicamente para el Análisis InSAR Multi-Riesgo*

**Inteligencia de deformación terrestre en tiempo real — combinando radar satelital, física atmosférica, oceanografía y puntuación heurística de riesgo en una única capa de razonamiento.**

*Subsidencia del terreno · Rescate marítimo SAR · Perímetro de incendios forestales · Actividad volcánica · Predicción de deslizamientos de tierra*

[Descripción General](#descripción-general) · [Motor de Física](#motor-de-física) · [Módulos Multi-Riesgo](#módulos-multi-riesgo) · [ATLAS Score](#atlas-score-marco-heurístico) · [Fusión en Tiempo Real](#fusión-de-datos-en-tiempo-real) · [Stack Científico](#stack-científico)

</div>

---

## Descripción General

ATLAS_SCIENTIST es un **agente de razonamiento de IA especializado** diseñado para ser el revisor experto en física para cualquier sistema que utilice Radar de Apertura Sintética Interferométrica (InSAR) para detectar, cuantificar y predecir eventos de deformación superficial.

No es un asistente de propósito general. Es un **especialista de dominio** — entrenado para pensar exclusivamente en:

- Matemáticas de interferometría SAR
- Física atmosférica e ionosférica
- Geodesia terrestre 3D (modelo elipsoidal WGS84)
- Cuantificación heurística de riesgo multi-señal
- Fusión de sensores en tiempo real desde APIs satelitales, meteorológicas y oceánicas

### El Problema Central

InSAR puede detectar movimientos del terreno tan pequeños como **3 mm desde 700 km de altitud** — pero la física entre las mediciones de fase en bruto y la inteligencia accionable es un campo minado de errores silenciosos:

```
φ_medido ≠ φ_deformación

El satélite mide una mezcla de fases. Separar la señal del ruido
requiere física validada — no suposiciones.
```

ATLAS_SCIENTIST hace cumplir esa validación en cada paso.

### Filosofía de Diseño

```
                    ┌──────────────────────────────────┐
                    │      Capa de Razonamiento IA     │
                    │         ATLAS_SCIENTIST          │
                    │                                  │
                    │  "¿Esta ecuación respeta         │
                    │   la física de la Tierra?"       │
                    └──────────────┬───────────────────┘
                                   │ valida
                    ┌──────────────▼───────────────────┐
                    │     Capa de Implementación       │
                    │(Código del motor, pipeline, APIs)│
                    └──────────────────────────────────┘

Principio: Separar quién implementa de quién valida la física.
Igual que la revisión por pares científica — aplicada al software.
```

---

## Motor de Física

### 1. Descomposición de Fase InSAR

Toda medición de fase interferométrica es una superposición de contribuciones físicas. Ignorar cualquier término produce una señal de deformación falsa.

```
φ_interferograma = φ_deformación + φ_topografía + φ_troposfera 
                 + φ_ionosfera + φ_órbita + φ_ruido
```

**Rol de ATLAS_SCIENTIST:** verificar que cada término sea corregido, delimitado, o explícitamente declarado como incertidumbre residual.

---

### 2. Conversión Fase → Desplazamiento

El mapeo fundamental de observable InSAR → cantidad física:

```
d_LOS = (φ · λ) / (4π)
```

**Longitudes de onda específicas por sensor:**

| Satélite | Banda | λ (cm) | 1 franja = |
|---|---|---|---|
| Sentinel-1 A/B | C | 5.6 cm | 2.8 cm LOS |
| ALOS-2 | L | 23.6 cm | 11.8 cm LOS |
| NISAR (2024+) | L + S | 23.6 / 9.3 cm | doble-frec |
| TerraSAR-X | X | 3.1 cm | 1.55 cm LOS |
| COSMO-SkyMed | X | 3.1 cm | 1.55 cm LOS |

```python
def phase_to_displacement_mm(phase_rad: float, wavelength_m: float = 0.056) -> float:
    """
    Convierte la fase InSAR a desplazamiento en la Línea de Visión (LOS) en milímetros.
    
    Convención de signos: positivo = alejándose del satélite.
    Para Sentinel-1 descendente: positivo ≈ subsidencia en la mayoría de geometrías.
    
    Argumentos:
        phase_rad: Fase interferométrica desenrollada en radianes
        wavelength_m: Longitud de onda del sensor (por defecto: Sentinel-1 banda-C = 0.056 m)
    
    Retorna:
        Desplazamiento en milímetros (dirección LOS)
    """
    return (phase_rad * wavelength_m / (4 * math.pi)) * 1000.0

# Verificación — ejecutar siempre esto:
assert abs(phase_to_displacement_mm(2 * math.pi, 0.056) - 28.0) < 0.01, \
    "Sentinel-1: una franja completa debe ser igual a 28.0 mm"
```

---

### 3. Descomposición LOS → Desplazamiento 3D

La geometría satelital transforma las mediciones LOS en componentes del mundo real:

```
d_LOS = −sin(θ)·cos(α)·d_E + sin(θ)·sin(α)·d_N − cos(θ)·d_U
```

Donde:
- `θ` = ángulo de incidencia (Sentinel-1: 29°–46° dependiendo del swath)
- `α` = ángulo de rumbo (ascendente ≈ −10°, descendente ≈ −170° desde el Norte)
- `d_E, d_N, d_U` = componentes de desplazamiento Este, Norte, Arriba

**Con par ascendente + descendente** (Sentinel-1 trayectorias A y D sobre la misma área):

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
    Descompone LOS ascendente + descendente en Este, Norte, Arriba.
    Retorna (d_east_mm, d_north_mm, d_up_mm).
    Sensibilidad vertical ~80% de LOS para geometría típica de Sentinel-1.
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
    # Resolución por mínimos cuadrados (sistema sobredeterminado)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1]), float(result[2])
```

---

### 4. Modelo Terrestre 3D — Elipsoide WGS84

**Corrección dimensional crítica:** la Tierra no es una esfera. Todas las conversiones de coordenadas deben tener en cuenta el elipsoide:

```python
from astropy.coordinates import EarthLocation, ITRS, GCRS
import astropy.units as u
from astropy.time import Time

# Parámetros WGS84 (ATLAS_SCIENTIST hace cumplir estas constantes)
WGS84_A = 6_378_137.0        # Semieje mayor (m) — radio ecuatorial
WGS84_B = 6_356_752.314245  # Semieje menor (m) — radio polar
WGS84_F = 1 / 298.257223563  # Achatamiento
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # Primera excentricidad al cuadrado

def meters_per_degree_lat(lat_deg: float) -> float:
    """Retorna metros por grado de latitud a una latitud dada."""
    lat_rad = math.radians(lat_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(lat_rad)**2)**1.5
    return math.radians(1) * M  # ~110,574 m/deg en ecuador, ~111,694 en polos

def meters_per_degree_lon(lat_deg: float) -> float:
    """Retorna metros por grado de longitud a una latitud dada.
    
    ⚠️ INVARIANTE DE ATLAS_SCIENTIST: cos(lat) es OBLIGATORIO aquí.
    Omitirlo causa un 8% de error a lat=−23° (Atacama) y 50% de error a lat=−60°.
    """
    lat_rad = math.radians(lat_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad)**2)
    return math.radians(1) * N * math.cos(lat_rad)

# Referencia rápida:
# En lat=0°  (ecuador):   m/deg_lon = 111,319 m
# En lat=−23° (Atacama):  m/deg_lon = 102,470 m  ← cos(23°) = 0.921
# En lat=−33° (Santiago): m/deg_lon = 93,274 m   ← cos(33°) = 0.839
# En lat=−55° (Patagonia):m/deg_lon = 63,849 m   ← cos(55°) = 0.574
```

---

### 5. Coherencia Temporal

Mide la calidad de correlación entre dos adquisiciones SAR:

```
γ = |⟨s₁ · s₂*⟩| / √(|⟨s₁ · s₁*⟩| · |⟨s₂ · s₂*⟩|)     γ ∈ [0, 1]
```

**Interpretación física por cobertura del suelo:**

| Tipo de superficie | Banda-C 12-días | Banda-L 46-días | Usabilidad |
|---|---|---|---|
| Roca expuesta / desierto | 0.60 – 0.80 | 0.75 – 0.95 | Excelente |
| Urbano / infraestructura | 0.55 – 0.85 | 0.70 – 0.90 | Excelente |
| Relaves mineros (activos) | 0.35 – 0.60 | 0.50 – 0.75 | Buena |
| Campos agrícolas | 0.15 – 0.45 | 0.40 – 0.70 | Estacional |
| Bosque denso | 0.05 – 0.25 | 0.20 – 0.55 | Pobre (C) / OK (L) |
| Superficie del océano | < 0.05 | < 0.05 | No aplicable |

**Umbrales mínimos:**
- Zonas áridas/desérticas: γ ≥ 0.35
- Tropical húmedo: γ ≥ 0.20 (Banda-L preferida)

---

### 6. Corrección Atmosférica — Refractividad Smith-Weintraub

La troposfera desvía las señales de radar, creando deformación aparente. La ecuación de Smith-Weintraub modela la refractividad N de la atmósfera:

```
N = k₁·(P/T) + k₂·(e/T) + k₃·(e/T²)
```

- k₁ = 77.6 K/hPa (término seco — presión dominante)
- k₂ = 70.4 K/hPa (término húmedo — presión parcial de vapor de agua)
- k₃ = 3.75 × 10⁵ K²/hPa (término de agua líquida)
- P = presión total (hPa), e = presión parcial de vapor de agua (hPa), T = temperatura (K)

**Retraso de fase por integración de columna:**

```
φ_tropo = (4π / λ) · (10⁻⁶ / cos θ) · ∫ N dz
```

**Fuentes de datos de corrección (gratuitas):**

| Fuente | Latencia | Resolución | Caso de Uso |
|---|---|---|---|
| ERA5 (Copernicus CDS) | 5 días | 0.25° | Reanálisis operativo |
| GACOS | 1-2 días | ~90m | Específico InSAR, recomendado |
| HRES-ECMWF | 6 horas | 0.1° | Casi tiempo real |
| Open-Meteo API | 1 hora | 0.25° | Gratis, sin API key |

```python
import openmeteo_requests

def fetch_tropospheric_params(lat: float, lon: float, time_utc: str) -> dict:
    """
    Obtiene parámetros atmosféricos para corrección troposférica.
    Open-Meteo: gratis, sin API key, resolución de 0.25°.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["surface_pressure", "temperature_2m",
                   "relativehumidity_2m", "dewpoint_2m"],
        "start_date": time_utc[:10], "end_date": time_utc[:10]
    }
    # Retorna P, T, e para el cálculo de Smith-Weintraub
    ...
```

---

### 7. Corrección Ionosférica (crítica para Banda-L)

Para sensores de banda L (ALOS-2, NISAR), el retraso ionosférico es significativo:

```
φ_iono = −(4π / λ) · (40.28 / f²) · ΔTEC
```

- f = frecuencia portadora (Hz)
- ΔTEC = Contenido Total de Electrones diferencial (TECU = 10¹⁶ electrones/m²)
- Signo negativo: la ionosfera adelanta la fase (opuesto a la troposfera)

**Corrección:** Método de espectro dividido (disponible en ISCE3, MintPy) o mapas TEC derivados de GPS (CODE, JPL).

---

## Módulos Multi-Riesgo

ATLAS_SCIENTIST valida la física a través de cinco dominios de riesgo operativo. Cada módulo utiliza InSAR como su fuente de datos principal, combinado con modelos físicos específicos del dominio.

### Módulo 1: Subsidencia del Terreno & Tranques de Relaves

**Observable:** Velocidad LOS (mm/día) y aceleración (mm/día²)

**Umbrales críticos (empíricos):**

```python
# Clasificación de tasa de subsidencia
VELOCITY_THRESHOLDS = {
    "stable":   (0.0,  2.0),   # mm/día — estacional/ruido
    "moderate": (2.0,  5.0),   # mm/día — monitorear de cerca
    "active":   (5.0,  15.0),  # mm/día — preocupación elevada
    "critical": (15.0, float("inf")),  # mm/día — respuesta de emergencia
}

# Bypass por aceleración: si d²u/dt² ≥ 3.0 mm/día², alerta inmediata
# sin importar la velocidad actual (firma de fluencia terciaria)
ACCELERATION_BYPASS_MM_DAY2 = 3.0
```

**Modelo físico:** Serie temporal SBAS InSAR → campo de velocidad → aceleración → ATLAS Score

---

### Módulo 2: Rescate Marítimo SAR — Física de Deriva

Cuando una persona o embarcación se pierde en el mar, InSAR no es el sensor — pero la **misma arquitectura del agente** maneja la validación de la física de deriva.

**Ecuación de deriva:**

```
dX/dt = u_corriente(x,t) + α_L · u_viento(x,t) + u_stokes(x,t) + ε(t)
```

**Deriva de Stokes (transporte inducido por oleaje):**

```
V_stokes = (π² · Hs² · f_p) / (2 · g · T_p²)
```

**Difusión turbulenta:**

```
ε(t) ~ N(0, σ_d),    σ_d = √(2 · K · Δt),    K = 1.0 m²/s
```

**Actualización de coordenadas (WGS84 — corrección de elipsoide obligatoria):**

```python
def update_particle_position(lat: float, lon: float,
                              u_ms: float, v_ms: float,
                              dt_s: float) -> tuple[float, float]:
    """
    Mueve una partícula por velocidad (u, v) en m/s sobre dt_s segundos.
    
    ⚠️ cos(lat) es OBLIGATORIO en dlon. Sin él:
       error = 8% a lat=−23°, 50% a lat=−60°
    """
    dlat = (v_ms * dt_s) / (WGS84_A * (1 - WGS84_E2 * math.sin(math.radians(lat))**2)**0.5)
    dlon = (u_ms * dt_s) / (WGS84_A * math.cos(math.radians(lat)) *
                             (1 - WGS84_E2 * math.sin(math.radians(lat))**2)**0.5)
    return lat + math.degrees(dlat), lon + math.degrees(dlon)
```

**Zonas de búsqueda Monte Carlo:** 1,000+ partículas → PCA → Elipses de probabilidad 1σ/2σ/3σ para equipos de rescate SAR.

---

### Módulo 3: Seguimiento de Perímetro de Incendios Forestales

**Ventaja SAR:** Las bandas C y L penetran el humo. Los satélites ópticos no pueden.

**Modelo físico — Propagación de fuego de Rothermel:**

```
R = (I_R · ξ) / (ρ_b · ε · Q_ig)

Donde:
  I_R  = intensidad de reacción (BTU/ft/min)
  ξ    = relación de flujo de propagación
  ρ_b  = densidad aparente del lecho de combustible
  ε    = número de calentamiento efectivo
  Q_ig = calor de pre-ignición
```

**Aplicación SAR:** Detección de cambio de retrodispersión (backscatter) Sentinel-1 + pérdida de coherencia InSAR → perímetro de área quemada (la coherencia cae de ~0.6 a <0.1 en zonas quemadas en un intervalo de 6-12 días).

**Combinación con clima:**

```python
# Dirección de propagación del fuego impulsada por viento desde ERA5 / Open-Meteo
wind_speed_ms = weather["wind_speed_10m"]   # m/s
wind_dir_deg  = weather["wind_direction_10m"] # meteorológico
spread_rate   = rothermel_R * (1 + k_wind * wind_speed_ms**2)
```

---

### Módulo 4: Actividad Volcánica

**InSAR es la herramienta principal para la deformación pre-eruptiva.**

**Modelo de fuente puntual de Mogi** (cámara magmática esférica):

```
u_r = (ΔP · a³) / (μ · r³) · r_hat
u_z = (ΔP · a³) / (μ · r³) · d/r

Donde:
  ΔP = cambio de presión en cámara magmática
  a  = radio de la cámara
  μ  = módulo de corte de la corteza (~30 GPa)
  d  = profundidad de la cámara
  r  = distancia desde el epicentro
```

**Proyección LOS de desplazamiento Mogi:**

```python
from astropy.coordinates import SphericalRepresentation
import astropy.units as u

def mogi_los_displacement(
    lat_obs: float, lon_obs: float, depth_m: float,
    delta_pressure_pa: float, radius_m: float,
    lat_source: float, lon_source: float,
    incidence_deg: float, heading_deg: float
) -> float:
    """Retorna el desplazamiento LOS predicho en mm para fuente Mogi."""
    mu = 30e9  # Pa — módulo de corte de la corteza
    dx, dy, dz = _mogi_3d(lat_obs, lon_obs, depth_m,
                           delta_pressure_pa, radius_m,
                           lat_source, lon_source, mu)
    # Proyección sobre vector unitario LOS
    los_e = -math.sin(math.radians(incidence_deg)) * math.cos(math.radians(heading_deg))
    los_n =  math.sin(math.radians(incidence_deg)) * math.sin(math.radians(heading_deg))
    los_u = -math.cos(math.radians(incidence_deg))
    return (los_e*dx + los_n*dy + los_u*dz) * 1000.0  # → mm
```

---

### Módulo 5: Cinemática de Deslizamientos de Tierra

**InSAR resuelve tasas de desplazamiento pero no planos de falla.** Combinando InSAR con geometría de ladera:

```
# Factor de Seguridad — aproximación de ladera infinita
FS = (c + (γ_s - γ_w · r_u) · z · cos²β · tan φ') / (γ_s · z · sin β · cos β)

Donde:
  c    = cohesión (kPa)
  γ_s  = peso unitario del suelo
  γ_w  = peso unitario del agua
  r_u  = relación de presión de poros (desde proxy InSAR o piezómetro)
  z    = profundidad de la ladera
  β    = ángulo de la ladera
  φ'   = ángulo de fricción efectiva
```

**Contribución InSAR:** Campo de velocidad LOS → velocidad paralela a la ladera → clasificación cinemática (fluencia lenta / estacional / aceleración hacia el fallo).

---

## ATLAS Score — Marco Heurístico

### El Problema Con los Umbrales Binarios

Los sistemas de monitoreo tradicionales usan umbrales binarios: "velocidad > 5 mm/día → alerta". Esto:

- Ignora la antigüedad de la señal (una lectura de sensor antigua no debería tener el mismo peso que la de hoy)
- Ignora la aceleración (la pendiente de la velocidad importa más que su valor actual)
- No puede combinar señales heterogéneas (InSAR + sísmica + lluvia + presión de poros)
- Produce falsos negativos cuando cada señal individual está por debajo del umbral pero el riesgo combinado es alto

### La Solución: Heurística Temporal Multi-Señal

```
R = Σᵢ [wᵢ · φᵢ(sᵢ) · exp(−λᵢ · Δtᵢ)] / Σᵢ [wᵢ · exp(−λᵢ · Δtᵢ)]
```

**Variables:**

| Símbolo | Significado | Restricción |
|---|---|---|
| `wᵢ` | Peso de la señal i | Σwᵢ = 1.0 (forzado) |
| `φᵢ(sᵢ)` | Función de activación de la señal i | φᵢ ∈ [0, 1] |
| `λᵢ` | Tasa de decaimiento temporal | λᵢ > 0 (unidades: 1/día) |
| `Δtᵢ` | Antigüedad de la medición i | Δtᵢ ≥ 0 (días) |
| `R` | Puntuación de riesgo | R ∈ [0, 1] siempre |

**Cinco invariantes matemáticas** — verificadas en cada cambio de algoritmo:

```python
class ATLASInvariants:
    """Pruebas de propiedad que deben pasar antes de cualquier despliegue en producción."""

    @staticmethod
    def P1_bounded(score_fn, n=10_000) -> bool:
        """R ∈ [0, 1] para todas las entradas aleatorias."""
        return all(0.0 <= score_fn(random_signals()) <= 1.0
                   for _ in range(n))

    @staticmethod
    def P2_monotone(score_fn) -> bool:
        """Mayor magnitud de deformación → mayor score."""
        low  = score_fn(signals(velocity=1.0))
        high = score_fn(signals(velocity=10.0))
        return high > low

    @staticmethod
    def P3_acceleration_bypass(score_fn) -> bool:
        """La aceleración crítica (≥3 mm/día²) fuerza R ≥ 0.80."""
        return score_fn(signals(acceleration=3.5)) >= 0.80

    @staticmethod
    def P4_temporal_decay(score_fn) -> bool:
        """Una señal de 30 días aporta menos que una fresca."""
        fresh = score_fn(signals(velocity=5.0, age_days=0))
        old   = score_fn(signals(velocity=5.0, age_days=30))
        return old < fresh * 0.60

    @staticmethod
    def P5_weights_sum(weights: dict) -> bool:
        """Los pesos deben sumar exactamente 1.0."""
        return abs(sum(weights.values()) - 1.0) < 1e-9
```

### Funciones de Activación φᵢ

```python
def phi_velocity(v_mm_per_day: float) -> float:
    """
    Normaliza la velocidad LOS a [0, 1].
    Lineal a trozos: estable → activa → crítica.
    """
    v = abs(v_mm_per_day)
    if v >= 15.0: return 1.00
    if v >= 5.0:  return 0.60 + 0.40 * (v - 5.0) / 10.0
    if v >= 2.0:  return 0.30 + 0.30 * (v - 2.0) / 3.0
    return v / 2.0 * 0.30

def phi_acceleration(a_mm_per_day2: float) -> float:
    """
    Normaliza la aceleración de la deformación a [0, 1].
    Firma de fluencia terciaria (a ≥ 3 mm/día²) → bypass total.
    """
    a = abs(a_mm_per_day2)
    if a >= 3.0: return 1.00   # ← gatillo de bypass por aceleración
    if a >= 1.0: return 0.80
    return a / 1.0 * 0.80
```

### Tasas de Decaimiento Temporal — Justificación Física

| Señal | λ (1/día) | Vida media | Razonamiento |
|---|---|---|---|
| Velocidad InSAR | 1/14 | 14 días | Ciclo de repetición Sentinel-1 |
| Aceleración InSAR | 1/7 | 7 días | La aceleración es más sensible al tiempo |
| Precipitación | 1/3 | 3 días | La infiltración se disipa en ~3 días |
| Energía sísmica | 1/5 | 5 días | Decaimiento de réplicas (ley de Omori) |
| Presión de poros | 1/7 | 7 días | Escala temporal de difusión de presión |
| Altura de olas oceánicas | 1/2 | 2 días | El estado del mar cambia rápidamente |
| Potencia radiativa de incendios | 1/1 | 1 día | Los incendios evolucionan en horas |

---

## Fusión de Datos en Tiempo Real

ATLAS_SCIENTIST define la arquitectura para combinar mediciones físicas heterogéneas en una señal de riesgo unificada.

### Fuentes de Datos — Sin API Key Requerida (La mayoría)

| Capa | Fuente | Variable | Latencia | Costo |
|---|---|---|---|---|
| SAR | Sentinel-1 (ESA Copernicus) | Fase, coherencia, backscatter | 6–12 días | **Gratis** |
| Archivo SAR | ASF (Alaska SAF) | Todo Sentinel-1/ALOS-2 | Histórico | **Gratis** |
| Clima | Open-Meteo | Viento, lluvia, presión, humedad | 1h | **Gratis** |
| Corrientes oceánicas | Análisis Global CMEMS | Velocidad superficial u, v | 24h | **Gratis** |
| Olas oceánicas | CMEMS WAV | Hs, Tp, dirección de las olas | 6h | **Gratis** |
| Atmósfera | ERA5 (Copernicus CDS) | Columna troposférica completa | 5 días | **Gratis** |
| Ionosfera | Mapas TEC CODE/JPL | ΔTEC | 24h | **Gratis** |
| Sismicidad | USGS ComCat API | Magnitud, profundidad, ubicación | Tiempo real | **Gratis** |
| DEM / Topografía | Copernicus DEM 30m | Elevación, pendiente | Estático | **Gratis** |

### Pipeline de Fusión

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class PhysicalSignal:
    name: str
    value: float          # normalizado [0, 1] por φᵢ
    raw_value: float      # cantidad física original con unidad
    unit: str
    timestamp_utc: float  # Marca de tiempo Unix
    weight: float
    decay_lambda: float   # 1/días
    source: str           # "sentinel-1" | "open-meteo" | "cmems" | ...
    uncertainty: float    # 1σ de incertidumbre en raw_value

class HazardModule(Protocol):
    """
    Interfaz abstracta para cualquier módulo de dominio de riesgo.
    Los módulos nunca importan entre sí — solo desde el núcleo.
    """
    @property
    def hazard_type(self) -> str: ...
    @property
    def required_signals(self) -> list[str]: ...
    def analyze(self, signals: list[PhysicalSignal]) -> float: ...  # → R

def compute_atlas_score(signals: list[PhysicalSignal],
                        t_now: float) -> float:
    """
    ATLAS Score temporal multi-señal.
    Garantizado: resultado ∈ [0, 1] para cualquier entrada.
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
    
    # Bypass por aceleración: firma de fluencia terciaria
    if max_acceleration >= 3.0:
        raw_score = max(raw_score, 0.80)
    
    return min(1.0, max(0.0, raw_score))
```

---

## Integración con Astropy

Para geometría satelital precisa, sistemas de tiempo y transformaciones de coordenadas:

```python
from astropy.time import Time
from astropy.coordinates import EarthLocation, ITRS, AltAz
import astropy.units as u

def sentinel1_acquisition_to_utc(sensing_time_str: str) -> Time:
    """
    Convierte el tiempo de adquisición anotado de Sentinel-1 a Time de astropy.
    Formato: "2024-03-15T06:34:22.123456"
    Utiliza la escala UTC; correcciones IERS-A aplicadas automáticamente.
    """
    return Time(sensing_time_str, format="isot", scale="utc")

def compute_local_incidence_angle(
    lat_deg: float, lon_deg: float, alt_m: float,
    sat_position_ecef: np.ndarray  # [x, y, z] en metros
) -> float:
    """
    Calcula el ángulo de incidencia local exacto en un punto terrestre,
    teniendo en cuenta la curvatura de la Tierra (elipsoide WGS84).
    
    Retorna el ángulo en grados entre la dirección de visión del radar
    y la vertical local en el punto terrestre.
    """
    ground = EarthLocation(lat=lat_deg*u.deg,
                           lon=lon_deg*u.deg,
                           height=alt_m*u.m)
    
    ground_ecef = np.array([
        ground.x.to(u.m).value,
        ground.y.to(u.m).value,
        ground.z.to(u.m).value
    ])
    
    # Vector de visión: satélite → tierra
    look_vec = ground_ecef - sat_position_ecef
    look_vec /= np.linalg.norm(look_vec)
    
    # Vertical local (normal al elipsoide en el punto terrestre)
    # Aproximación: vector de posición ECEF normalizado
    local_up = ground_ecef / np.linalg.norm(ground_ecef)
    
    cos_angle = abs(np.dot(look_vec, local_up))
    return math.degrees(math.acos(cos_angle))

# Cálculo de línea base entre dos adquisiciones SAR:
def perpendicular_baseline(pos1_ecef: np.ndarray,
                           pos2_ecef: np.ndarray,
                           look_dir: np.ndarray) -> float:
    """
    Línea base perpendicular en metros.
    Crítica para: estimación de error DEM, predicción de coherencia.
    """
    baseline_vec = pos2_ecef - pos1_ecef
    b_par = np.dot(baseline_vec, look_dir)
    b_perp = np.sqrt(np.dot(baseline_vec, baseline_vec) - b_par**2)
    return float(b_perp)
```

---

## Stack Científico

```
Teledetección
  asf_search          Búsqueda y descarga de escenas Sentinel-1 / ALOS-2
  pyroSAR + SNAP 10   Corregistro TOPSAR, generación de interferogramas
  MintPy              Análisis de series temporales SBAS / PS (conda-forge)
  ISCE3               Procesador InSAR de JPL (estándar de la misión NISAR)

Astronomía y Geodesia
  astropy             Sistemas de tiempo (UTC/TAI/GPS), transformaciones de coordenadas, IERS
  pyproj              Transformaciones geodésicas basadas en PROJ (WGS84 ↔ UTM)
  scipy               Procesamiento de señales, mecánica orbital, álgebra lineal

Atmosférico
  openmeteo-requests  Cliente de API meteorológica (gratuito, sin clave)
  copernicusmarine    Datos oceánicos CMEMS (corrientes, olas)
  cdsapi              ERA5 / Copernicus CDS (corrección troposférica)

Geoespacial
  rasterio / gdal     E/S GeoTIFF, reproyección, matemáticas raster
  shapely             Geometría vectorial (AOI, elipses, huellas)
  geopandas           DataFrames Geoespaciales

Numérico / ML
  numpy               Arrays N-dimensionales, FFT, álgebra lineal
  scipy               Optimización (para inversión de modelos), estadística
  xarray              Arrays N-D etiquetados para cubos de series temporales SAR
  netCDF4 / h5py      E/S de datos científicos (salida SNAP, MintPy HDF5)
  pydantic v2         Validación en tiempo de ejecución de todos los contratos de datos físicos
  scikit-learn        Agrupación de errores de fase, detección de valores atípicos (Etapa D+)
  torch               Aprendizaje profundo para detección de anomalías de desplazamiento (futuro)

Testing
  pytest + hypothesis Pruebas basadas en propiedades para invariantes físicos
  mypy                Verificación de tipos estáticos (modo estricto forzado)
```

---

## Protocolo de Validación

Antes de aceptar cualquier cambio matemático:

```
LISTA DE VERIFICACIÓN DE REVISIÓN DE ATLAS_SCIENTIST
─────────────────────────────────────────────────────────
FÍSICA DE FASE
[ ] Longitud de onda correcta para el sensor/banda específico
[ ] Conversión fase a desplazamiento verificada (assert franja = λ/2)
[ ] La construcción del vector LOS utiliza ángulos de incidencia + rumbo correctos
[ ] cos(lat) presente en TODAS las conversiones de longitud a metros
[ ] Se usa elipsoide WGS84 (no aproximación de Tierra esférica)
[ ] Línea base temporal dentro de los límites de coherencia para ese sensor+superficie

ATMOSFÉRICA
[ ] Corrección troposférica aplicada o explícitamente documentada como residual
[ ] Corrección ionosférica aplicada para Banda-L (omitir para bases cortas Banda-C)
[ ] Datos ERA5 o GACOS utilizados (no valores interpolados de pantalla)

ATLAS SCORE
[ ] Todos los pesos suman exactamente 1.0 (prueba automatizada en CI)
[ ] R ∈ [0, 1] para 10,000 combinaciones de entrada aleatorias (property test)
[ ] El bypass de aceleración se activa en el umbral correcto
[ ] Tasas de decaimiento temporal justificadas físicamente en los comentarios del código
[ ] Sin números mágicos — todas las constantes nombradas con unidad y fuente

MULTI-RIESGO
[ ] Coeficiente de deriva (leeway) correcto para el tipo de objeto marítimo (fuente IAMSAR)
[ ] cos(lat) en todas las actualizaciones de posición de deriva de partículas
[ ] Parámetros del modelo de fuente Mogi dentro de límites físicos
[ ] Entradas de la tasa de propagación de Rothermel en unidades correctas (sistema ft-lb-min)

FIRMA: _________________________ FECHA: ____________
```

---

## Casos de Prueba

```python
# tests/test_atlas_physics.py

def test_sentinel1_fringe_is_28mm():
    assert abs(phase_to_displacement_mm(2*math.pi, 0.056) - 28.0) < 0.01

def test_alos2_fringe_is_118mm():
    assert abs(phase_to_displacement_mm(2*math.pi, 0.236) - 118.0) < 0.1

def test_cos_lat_correction_at_minus23():
    """A la latitud de Atacama, la corrección de longitud en metros debe ser ~0.921."""
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
        validate_weights({"a": 0.5, "b": 0.4})  # suma 0.9 → falla
```

---

## Cómo Usar Este Agente

### 1. Como un Paso de Revisión de Código

```
Patrón de prompt:
"Actúa como ATLAS_SCIENTIST.
Revisa el siguiente código de procesamiento InSAR en busca de corrección física.
Verifica: ecuación de fase, conversión LOS, sistema de coordenadas, unidades.
NO sugieras cambios de estilo de código. Solo valida la física.
Marca cada violación con: [VIOLATION], el error específico,
y la ecuación corregida."
```

### 2. Como Validador de Ecuaciones

```
Patrón de prompt:
"Actúa como ATLAS_SCIENTIST.
Estoy a punto de implementar la siguiente ecuación en Python:
[pegar ecuación]
Sensor: Sentinel-1 Banda-C, órbita descendente.
Área de estudio: lat −23°, lon −70°.
Verificar: consistencia dimensional, convenciones de signos,
salida numérica esperada y casos extremos conocidos."
```

### 3. Como Generador de Casos de Prueba

```
Patrón de prompt:
"Actúa como ATLAS_SCIENTIST.
Para la función phase_to_displacement_mm(phase_rad, wavelength_m),
genera 5 casos de prueba pytest que cubran:
- Sentinel-1 (λ = 5.6 cm)
- ALOS-2 (λ = 23.6 cm)
- Fase cero → desplazamiento cero
- Fase 4π → desplazamiento completo de dos franjas
- Fase negativa (dirección de movimiento opuesta)"
```

---

## Español

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

*ATLAS_SCIENTIST — Agente de Validación Física para Análisis InSAR Multi-Riesgo*

*Ecuaciones sobre suposiciones. Física sobre heurística. Validación sobre confianza.*

**Enfocado para Astrofísicos y Expertos de Dominio Técnico**

---

### Autor y Licencia
**Creador:** Daniel Nuñez Rojas  
**Email:** [contacto@laticces.cl](mailto:contacto@laticces.cl)  
**Web:** [likancenti.cl](https://likancenti.cl)  
**LinkedIn:** [linkedin.com/in/delnr91](https://www.linkedin.com/in/delnr91)  

© 2026 Daniel Nuñez Rojas / Lattice / Likan-Centi. Todos los derechos reservados.  
*Este agente, su arquitectura, heurísticas y marcos matemáticos son propietarios y pertenecen estrictamente al autor. La distribución, modificación o uso comercial no autorizado sin permiso explícito está estrictamente prohibido.*

</div>
