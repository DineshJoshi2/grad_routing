
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly as px

# Constants
DEBRIS = tf.constant(7.0, dtype=tf.float32)
BARE =  tf.constant(8.0, dtype=tf.float32)

raster_resolution = 200.069 # meter
ConvConst = tf.constant((raster_resolution ** 2) / (1000 * 86400), dtype=tf.float32)

# Initial state: 6 tensors (Snow, W_rch, W_rch_dp, Qb_sh, Qb_dp, SurfaceFlow0,Qtot0)

def initial_state(nrow,ncol):
    initial_state = (
        tf.zeros((nrow,ncol)),  # Snow
        tf.zeros((nrow,ncol)),  # W_rch
        tf.zeros((nrow,ncol)),  # W_rch_dp
        tf.zeros((nrow,ncol)),  # Qb_sh
        tf.zeros((nrow,ncol)),  # Qb_dp
        tf.zeros((nrow,ncol)),  # SurfaceFlow0
        tf.zeros((nrow,ncol))   # Qtot0
    )
    return initial_state

# ---------------------------
# 1. Physical ranges
# ---------------------------

runoff_ranges = tf.constant([
    [0, 0],
    [0.2, 0.7],   # class 1: forest
    [0.4, 0.8],   # class 2: crops
    [0.5, 0.8],   # class 3: Grassland
    [0.3, 0.7],   # class 4: bareland/clouds
    [0.9, 1],     # class 5: Water/flooded vegetation
    [0.9, 1],     # class 6: Settlement
    [0.95, 1],    # class 7: Debris Covered Glacier
    [0.95, 1]     # class 8: Clean Glacier
], dtype=tf.float32) 

param_ranges = {
    "ks": (7, 11),
    "kb": (7, 11),
    "interception": (0, 0),
    "delta_gwsh": (10.0, 30.0),
    "delta_gwdp": (10.0, 300.0),
    "alpha_gwsh": (0, 1),
    "alpha_gwdp": (0, 1),
    "beta_dp": (0, 1),
    "kx": (0.0, 1.0),
    "Tc": (0, 2),
    "Kd": (2.9, 3.1),
}

# ---------------------------
# 1. Define raw parameters
# ---------------------------

ks_raw = tf.Variable(tf.zeros((12, 1)), dtype=tf.float32, trainable=True)
kb_raw = tf.Variable(tf.zeros((12, 1)), dtype=tf.float32, trainable=True)

runoff_coeff_raw = tf.Variable(tf.zeros((12, runoff_ranges.shape[0])), dtype=tf.float32, trainable=True)

interception_raw = tf.Variable(tf.zeros((1,)), dtype=tf.float32, trainable=True)
delta_gwsh_raw = tf.Variable(tf.zeros((590, 486)), dtype=tf.float32, trainable=True)
delta_gwdp_raw = tf.Variable(tf.zeros((590, 486)), dtype=tf.float32, trainable=True)
alpha_gwsh_raw = tf.Variable(tf.zeros((590, 486)), dtype=tf.float32, trainable=True)
alpha_gwdp_raw = tf.Variable(tf.zeros((590, 486)), dtype=tf.float32, trainable=True)
beta_dp_raw = tf.Variable(tf.zeros((590, 486)), dtype=tf.float32, trainable=True)

kx_raw = tf.Variable(tf.zeros((12,590, 486)), dtype=tf.float32, trainable=True)
Tc_raw = tf.Variable(tf.zeros((12,1)), dtype=tf.float32, trainable=True)
Kd_raw = tf.Variable(tf.zeros((12, 1)), dtype=tf.float32, trainable=True)

def month_idx(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="D")
    return np.column_stack([
        np.arange(len(dates)),
        dates.month,
        dates.strftime("%Y-%m-%d")
    ])

def scale_param(raw, name, month_idx):
    lo, hi = param_ranges[name]
    p01 = tf.sigmoid(raw[month_idx])
    return lo + p01 * (hi - lo)

def scale_param_baseflow(raw, name):
    lo, hi = param_ranges[name]
    p01 = tf.sigmoid(raw)
    return lo + p01 * (hi - lo)

def scale_param_class_specific(raw, ranges, month_idx):
    p01 = tf.sigmoid(raw[month_idx])
    lo, hi = ranges[:, 0], ranges[:, 1]
    return lo + p01 * (hi - lo)

def get_runoff_coeff(month_idx, lulc_tf):
    coeff_per_class = scale_param_class_specific(runoff_coeff_raw, runoff_ranges, month_idx)
    return tf.gather(coeff_per_class, tf.cast(lulc_tf, tf.int32))

def GDM(states, tmax_t, tmin_t, prec_t, pet_t,lulc_tf,accuflux,tiers_down_tf,tiers_up_tf,month_idx):
    Snow, W_rch, W_rch_dp, Qb_sh, Qb_dp, SurfaceFlow0, Qtot0 = states

    ks = scale_param(ks_raw, "ks", month_idx)
    kb = scale_param(kb_raw, "kb", month_idx)
    kx = scale_param(kx_raw, "kx", month_idx)
    Tc = scale_param(Tc_raw, "Tc", month_idx)
    Kd = scale_param(Kd_raw, "Kd", month_idx)

    runoff_coeff = get_runoff_coeff(month_idx,lulc_tf)

    interception = scale_param_baseflow(interception_raw, "interception")
    delta_gwsh = scale_param_baseflow(delta_gwsh_raw, "delta_gwsh")
    delta_gwdp = scale_param_baseflow(delta_gwdp_raw, "delta_gwdp")
    alpha_gwsh = scale_param_baseflow(alpha_gwsh_raw, "alpha_gwsh")
    alpha_gwdp = scale_param_baseflow(alpha_gwdp_raw, "alpha_gwdp")
    beta_dp = scale_param_baseflow(beta_dp_raw, "beta_dp")


    Temperature = (tmax_t + tmin_t) / 2.0
    interception_eff = tf.minimum(prec_t, interception)
    NetPrecipitation = tf.maximum(prec_t - interception_eff, 0.0)

    Rain0 = tf.where(Temperature > Tc, NetPrecipitation, 0.0)
    Snow  = Snow + tf.where(Temperature <= Tc, NetPrecipitation, 0.0)

    SnowMelt0 = tf.where(Snow > 0, tf.where(Temperature > 0, ks * Temperature, 0.0), 0.0)
    IceMelt0  = tf.where(tf.equal(lulc_tf, BARE),   tf.where(Temperature > 0, kb * Temperature, 0.0), 0.0)
    IceMelt0 += tf.where(tf.equal(lulc_tf, DEBRIS), tf.where(Temperature > 0, Kd * Temperature, 0.0), 0.0)

    Snow = tf.maximum(Snow - SnowMelt0, 0.0)

    TotalWater   = tf.maximum(SnowMelt0 + IceMelt0 + Rain0 - pet_t, 0.0)
    SurfaceRunoff = tf.maximum(TotalWater * runoff_coeff, 0.0)
    W_seep        = tf.maximum(TotalWater - SurfaceRunoff, 0.0)

    W_rch = (1.0 - tf.exp(-1.0 / delta_gwsh)) * W_seep + tf.exp(-1.0 / delta_gwsh) * W_rch
    W_rch = tf.maximum(W_rch, 0.0)

    W_seep_dp = beta_dp * W_rch
    W_rch_sh  = W_rch - W_seep_dp

    Qb_sh = tf.maximum(Qb_sh * tf.exp(-alpha_gwsh) + W_rch_sh * (1.0 - tf.exp(-alpha_gwsh)), 0.0)
    W_rch_dp = (1.0 - tf.exp(-1.0 / delta_gwdp)) * W_seep_dp + tf.exp(-1.0 / delta_gwdp) * W_rch_dp
    W_rch_dp = tf.maximum(W_rch_dp, 0.0)
    Qb_dp = tf.maximum(Qb_dp * tf.exp(-alpha_gwdp) + W_rch_dp * (1.0 - tf.exp(-alpha_gwdp)), 0.0)

    Qb = Qb_sh + Qb_dp

    BaseFlow    = accuflux(Qb,tiers_down_tf, tiers_up_tf) * ConvConst
    SurfaceFlow    =accuflux(SurfaceRunoff, tiers_down_tf, tiers_up_tf) * ConvConst

    SurfaceFlowContribution = (1.0 - kx) * SurfaceFlow + kx * SurfaceFlow0
    Qtot = BaseFlow + SurfaceFlowContribution

    new_states = (Snow, W_rch, W_rch_dp, Qb_sh, Qb_dp, SurfaceFlowContribution, Qtot)

    return new_states

def loss_fn(Qsim, Qpseudo):
    return tf.reduce_mean(tf.square(Qsim - Qpseudo))


trainable_params = [
    ks_raw, kb_raw, runoff_coeff_raw, interception_raw,
    delta_gwsh_raw, delta_gwdp_raw, alpha_gwsh_raw, alpha_gwdp_raw,
    beta_dp_raw, kx_raw, Tc_raw, Kd_raw
]

@tf.function(jit_compile=False)
def train_step(Tmax, Tmin, Prec, Pet, Qobs, state, params,
               lulc, accuflux, tiers_down, tiers_up, month_idx):

    with tf.GradientTape() as tape:
        state = GDM(state, Tmax, Tmin, Prec, Pet, lulc,
                    accuflux, tiers_down, tiers_up, month_idx)

        Qsim_grid = state[-1]
        Qsim_outlet = Qsim_grid[567, 328]

        scale_factor = Qobs / (Qsim_outlet + 1e-6)
        Qpseudo = Qsim_grid * scale_factor

        loss = loss_fn(Qsim_grid, Qpseudo)

    grads = tape.gradient(loss, params)

    return loss, Qsim_outlet, state, grads

def extract_spatial_parameters(trainable_params, lulc_tf):
    """
    Extract monthly and spatial parameters from raw variables
    using the scaling functions defined in model.py.
    """

    param_info = [
        {"name": "ks",             "raw": trainable_params[0],  "type": "monthly"},
        {"name": "kb",             "raw": trainable_params[1],  "type": "monthly"},
        {"name": "runoff_coeff",   "raw": trainable_params[2],  "type": "monthly_class"},
        {"name": "interception",   "raw": trainable_params[3],  "type": "baseflow"},
        {"name": "delta_gwsh",     "raw": trainable_params[4],  "type": "baseflow"},
        {"name": "delta_gwdp",     "raw": trainable_params[5],  "type": "baseflow"},
        {"name": "alpha_gwsh",     "raw": trainable_params[6],  "type": "baseflow"},
        {"name": "alpha_gwdp",     "raw": trainable_params[7],  "type": "baseflow"},
        {"name": "beta_dp",        "raw": trainable_params[8],  "type": "baseflow"},
        {"name": "kx",             "raw": trainable_params[9],  "type": "monthly"},
        {"name": "Tc",             "raw": trainable_params[10], "type": "monthly"},
        {"name": "Kd",             "raw": trainable_params[11], "type": "monthly"}
    ]

    extracted = {info["name"]: [] for info in param_info}

    # --- Loop through months ---
    for m in range(12):
        for info in param_info:
            name = info["name"]
            raw  = info["raw"]

            if info["type"] == "monthly":
                extracted[name].append(scale_param(raw, name, m).numpy())

            elif info["type"] == "monthly_class":
                extracted[name].append(get_runoff_coeff(m, lulc_tf).numpy())

            elif info["type"] == "baseflow":
                # Baseflow parameters are NOT monthly → overwrite only once
                if isinstance(extracted[name], list):
                    extracted[name] = scale_param_baseflow(raw, name).numpy()

    # Convert selected monthly params to array
    for name in ["ks", "kb", "kx", "Tc", "Kd", "runoff_coeff"]:
        extracted[name] = np.array(extracted[name]).squeeze()

    return extracted
    
import plotly.express as px

def plot_spatial_parameters(extracted, mask):
    """
    Plot spatial parameters and mean monthly kx.
    """

    # single-layer spatial parameters
    spatial_params = ["delta_gwsh", "delta_gwdp",
                      "alpha_gwsh", "alpha_gwdp", "beta_dp"]

    def plot_param(arr, title):
        masked = np.where(mask, arr, np.nan)
        fig = px.imshow(masked,
                        color_continuous_scale='viridis',
                        title=title,
                        origin='upper')
        fig.update_layout(
            width=700, height=600,
            xaxis_title="Column Index",
            yaxis_title="Row Index",
            hovermode="closest"
        )
        fig.show()

    # Plot all baseflow spatial layers
    for name in spatial_params:
        plot_param(extracted[name], f"Spatial Distribution of {name}")

    # Plot monthly-average kx
    kx_avg = extracted["kx"].mean(axis=0)
    plot_param(kx_avg, "Average Spatial Distribution of kx")
