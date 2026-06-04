import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    DotProduct,
    ExpSineSquared,
    Matern,
    RBF,
    RationalQuadratic,
)

from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Global settings
# =========================
SEED = 43
X_MIN = 0.0
X_MAX = 5.0
GRID_POINTS = 250
INITIAL_SAMPLE_COUNT = 3
DEFAULT_MAX_POINTS = 15
DEFAULT_NOISE_VARIANCE = 1e-9

np.random.seed(SEED)


# =========================
# Ground truth models
# =========================
def ground_truth_complex_sin(x):
    return np.sin(x) + np.sin(3 * x) + 0.5 * np.sin(6 * x) + 2


def ground_truth_simple_sin(x):
    return np.sin(2 * x) + 2


def ground_truth_quadratic(x):
    return -0.5 * (x - 2.5) ** 2 + 3


def ground_truth_step(x):
    return np.where(x < 2.5, 1.5, 3.0)


def get_ground_truth(name):
    models = {
        "complex_sin": ground_truth_complex_sin,
        "simple_sin": ground_truth_simple_sin,
        "quadratic": ground_truth_quadratic,
        "step": ground_truth_step,
    }
    if name not in models:
        raise ValueError(f"Unknown ground truth: {name}")
    return models[name]


# =========================
# Kernel helpers
# =========================
def create_kernel(name, params):
    amplitude = params["amplitude"]
    length_scale = params["length_scale"]
    period = params["period"]
    matern_nu = params["matern_nu"]
    rq_alpha = params["rq_alpha"]
    linear_sigma_0 = params["linear_sigma_0"]
    poly_c = params["poly_c"]

    base = C(amplitude)
    if name == "rbf":
        return base * RBF(length_scale=length_scale)
    if name == "matern":
        return base * Matern(length_scale=length_scale, nu=matern_nu)
    if name == "periodic":
        return base * ExpSineSquared(length_scale=length_scale, periodicity=period)
    if name in ("rq", "rational_quadratic"):
        return base * RationalQuadratic(length_scale=length_scale, alpha=rq_alpha)
    if name == "linear":
        return base * DotProduct(sigma_0=linear_sigma_0)
    if name in ("poly", "polynomial"):
        # Sklearn has no direct polynomial kernel in this subset, so we approximate via DotProduct.
        return base * DotProduct(sigma_0=poly_c)
    raise ValueError(f"Unknown kernel: {name}")


def describe_manual_kernel(strategy, kernel_type, combined_mode, combined_kernels):
    if strategy == "manual_single":
        return f"{kernel_type.upper()} Kernel"
    if strategy == "manual_combined":
        joiner = "+" if combined_mode == "sum" else "*"
        return f"Combined ({combined_mode}): {joiner.join(combined_kernels)}"
    return "Auto-select kernel"


def build_manual_kernel(strategy, kernel_type, combined_mode, combined_kernels, params):
    if strategy == "manual_single":
        return create_kernel(kernel_type, params), describe_manual_kernel(strategy, kernel_type, combined_mode, combined_kernels)

    if strategy == "manual_combined":
        kernels = combined_kernels if combined_kernels else [kernel_type]
        kernel = create_kernel(kernels[0], params)
        for name in kernels[1:]:
            other = create_kernel(name, params)
            if combined_mode == "sum":
                kernel = kernel + other
            elif combined_mode == "product":
                kernel = kernel * other
            else:
                raise ValueError(f"Unknown combination mode: {combined_mode}")
        return kernel, describe_manual_kernel(strategy, kernel_type, combined_mode, kernels)

    return create_kernel(kernel_type, params), "Auto-select kernel"


def candidate_kernels(params):
    rbf = create_kernel("rbf", params)
    matern = create_kernel("matern", params)
    periodic = create_kernel("periodic", params)
    rq = create_kernel("rq", params)
    return [
        ("RBF", rbf),
        ("Matern", matern),
        ("Periodic", periodic),
        ("RQ", rq),
        ("RBF + Periodic", rbf + periodic),
        ("RBF * Periodic", rbf * periodic),
    ]


def select_best_kernel(X, y, params, noise_variance):
    best_lml = -np.inf
    best_kernel = None
    best_label = None
    best_gp = None

    for label, kernel in candidate_kernels(params):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(noise_variance, 1e-9),
            n_restarts_optimizer=3,
            optimizer="fmin_l_bfgs_b",
            random_state=42,
            normalize_y=False,
        )
        try:
            gp.fit(X, y)
            lml = gp.log_marginal_likelihood_value_
            if np.isfinite(lml) and lml > best_lml:
                best_lml = lml
                best_kernel = kernel
                best_label = label
                best_gp = gp
        except Exception:
            continue

    if best_gp is None:
        fallback_kernel = create_kernel("rbf", params)
        best_gp = GaussianProcessRegressor(
            kernel=fallback_kernel,
            alpha=max(noise_variance, 1e-9),
            n_restarts_optimizer=3,
            optimizer="fmin_l_bfgs_b",
            random_state=42,
            normalize_y=False,
        )
        best_gp.fit(X, y)
        best_kernel = fallback_kernel
        best_label = "RBF (fallback)"

    return best_gp, best_kernel, best_label


# =========================
# Acquisition functions
# =========================
def expected_improvement(x, gp, y_best, xi):
    mu, sigma = gp.predict(x, return_std=True)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - y_best - xi
        z = np.zeros_like(mu)
        valid = sigma > 0.0
        z[valid] = imp[valid] / sigma[valid]
        ei = np.zeros_like(mu)
        ei[valid] = imp[valid] * norm.cdf(z[valid]) + sigma[valid] * norm.pdf(z[valid])
    return ei


def probability_of_improvement(x, gp, y_best, xi):
    mu, sigma = gp.predict(x, return_std=True)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.zeros_like(mu)
        valid = sigma > 0.0
        z[valid] = (mu[valid] - y_best - xi) / sigma[valid]
        pi = np.zeros_like(mu)
        pi[valid] = norm.cdf(z[valid])
    return pi


def upper_confidence_bound(x, gp, kappa):
    mu, sigma = gp.predict(x, return_std=True)
    return mu.reshape(-1) + kappa * sigma.reshape(-1)


def compute_acquisition(x, gp, y_best, method, xi, kappa):
    method = method.lower()
    if method == "ei":
        return expected_improvement(x, gp, y_best, xi), f"EI (xi={xi})"
    if method == "pi":
        return probability_of_improvement(x, gp, y_best, xi), f"PI (xi={xi})"
    if method == "ucb":
        return upper_confidence_bound(x, gp, kappa), f"UCB (kappa={kappa})"
    raise ValueError(f"Unknown acquisition method: {method}")


# =========================
# GP helpers
# =========================
def make_training_arrays(points):
    xs = np.array(points.get("xs", []), dtype=float)
    ys = np.array(points.get("ys", []), dtype=float)
    if xs.size == 0:
        return np.empty((0, 1)), np.empty((0,))
    return xs.reshape(-1, 1), ys.reshape(-1)


def suggest_next_point(acquisition, x_grid, x_used):
    score = np.asarray(acquisition).copy()
    for x in x_used:
        idx = int(np.argmin(np.abs(x_grid[:, 0] - x)))
        score[idx] = -np.inf
    best_idx = int(np.argmax(score))
    if not np.isfinite(score[best_idx]):
        return None
    return float(x_grid[best_idx, 0])


class PriorPredictor:
    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu).reshape(-1)
        self.sigma = np.asarray(sigma).reshape(-1)

    def predict(self, x, return_std=True):
        if return_std:
            return self.mu, self.sigma
        return self.mu


def fit_model(points, controls, x_grid):
    params = {
        "amplitude": controls["kernel_amplitude"],
        "length_scale": controls["length_scale"],
        "period": controls["period"],
        "matern_nu": controls["matern_nu"],
        "rq_alpha": controls["rq_alpha"],
        "linear_sigma_0": controls["linear_sigma_0"],
        "poly_c": controls["poly_c"],
    }
    X_train, y_train = make_training_arrays(points)
    strategy = controls["kernel_strategy"]
    noise_variance = controls["noise_variance"]
    kernel_type = controls["kernel_type"]
    combined_mode = controls["combined_mode"]
    combined_kernels = controls["combined_kernels"]
    auto_select = strategy == "auto_select"

    kernel, kernel_label = build_manual_kernel(strategy, kernel_type, combined_mode, combined_kernels, params)

    if X_train.shape[0] == 0:
        mu = np.zeros(x_grid.shape[0])
        cov = kernel(x_grid)
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {
            "gp": None,
            "kernel": kernel,
            "kernel_label": kernel_label,
            "mu": mu,
            "sigma": sigma,
            "cov": cov,
            "X_train": X_train,
            "y_train": y_train,
            "auto_selected": False,
        }

    opt = "fmin_l_bfgs_b" if auto_select else None
    restarts = 5 if auto_select else 0

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=max(noise_variance, 1e-9),
        optimizer=opt,
        n_restarts_optimizer=restarts,
        random_state=42,
        normalize_y=False,
    )
    gp.fit(X_train, y_train)

    if auto_select and X_train.shape[0] >= 2:
        gp, selected_kernel, selected_label = select_best_kernel(X_train, y_train, params, noise_variance)
        kernel = selected_kernel
        kernel_label = selected_label

    mu, cov = gp.predict(x_grid, return_cov=True)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return {
        "gp": gp,
        "kernel": kernel,
        "kernel_label": kernel_label,
        "mu": mu,
        "sigma": sigma,
        "cov": cov,
        "X_train": X_train,
        "y_train": y_train,
        "auto_selected": auto_select and X_train.shape[0] >= 2,
    }


def sample_functions(model_info, x_grid, n_samples, seed):
    rng = np.random.default_rng(seed)
    gp = model_info["gp"]
    cov = model_info["cov"]
    mu = model_info["mu"]
    jitter = 1e-9 * np.eye(cov.shape[0])
    if gp is None:
        samples = rng.multivariate_normal(mu, cov + jitter, size=n_samples).T
    else:
        samples = gp.sample_y(x_grid, n_samples=n_samples, random_state=seed)
    return np.asarray(samples)


# =========================
# Plot builders
# =========================
def make_gp_figure(x_grid, y_true, model_info, points, next_x):
    X_train, y_train = model_info["X_train"], model_info["y_train"]
    mu, sigma = model_info["mu"], model_info["sigma"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_grid[:, 0], y=y_true, mode="lines", name="Ground Truth", line=dict(color="rgba(80,80,80,0.7)", dash="dash")))
    fig.add_trace(go.Scatter(x=x_grid[:, 0], y=mu, mode="lines", name="GP Mean", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_grid[:, 0], x_grid[::-1, 0]]),
        y=np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
        hoverinfo="skip",
        showlegend=True,
    ))
    if X_train.shape[0] > 0:
        fig.add_trace(go.Scatter(
            x=X_train[:, 0], y=y_train,
            mode="markers",
            name="Measured Points",
            marker=dict(color="red", size=10, line=dict(color="darkred", width=1)),
        ))
    if next_x is not None:
        fig.add_vline(x=next_x, line_width=2, line_dash="dot", line_color="orange")

    fig.update_layout(
        template="plotly_white",
        title="Gaussian Process Modeling (click this chart to add a point)",
        xaxis_title="Distance / x",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=55, b=20),
        height=430,
        font=dict(size=14)
    )
    fig.update_xaxes(range=[X_MIN, X_MAX])
    return fig


def make_samples_figure(x_grid, y_true, model_info, n_samples, seed):
    samples = sample_functions(model_info, x_grid, n_samples=n_samples, seed=seed)
    X_train, y_train = model_info["X_train"], model_info["y_train"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_grid[:, 0], y=y_true, mode="lines", name="Ground Truth", line=dict(color="rgba(80,80,80,0.6)", dash="dash")))
    colors = ["#0b84a5", "#e45756", "#2a9d8f", "#f4a261", "#6a4c93"]
    for i in range(samples.shape[1]):
        fig.add_trace(go.Scatter(
            x=x_grid[:, 0],
            y=samples[:, i],
            mode="lines",
            name=f"Sample {i + 1}",
            line=dict(color=colors[i % len(colors)], width=1.8),
            opacity=0.8,
        ))
    if X_train.shape[0] > 0:
        fig.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode="markers", name="Measured Points", marker=dict(color="red", size=9)))
    fig.update_layout(
        template="plotly_white",
        title="Posterior / Prior Samples",
        xaxis_title="Distance / x",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=55, b=20),
        height=430,
        font=dict(size=14)
    )
    fig.update_xaxes(range=[X_MIN, X_MAX])
    return fig


def make_acq_figure(x_grid, acquisition, acquisition_label, next_x):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_grid[:, 0], y=acquisition, mode="lines", name=acquisition_label, line=dict(color="green", width=3)))
    fig.add_trace(go.Scatter(x=x_grid[:, 0], y=np.zeros_like(x_grid[:, 0]), mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x_grid[:, 0],
        y=acquisition,
        fill="tozeroy",
        fillcolor="rgba(0, 128, 0, 0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Acquisition area",
        hoverinfo="skip",
        showlegend=False,
    ))
    if next_x is not None:
        fig.add_vline(x=next_x, line_width=2, line_dash="dash", line_color="orange")
    fig.update_layout(
        template="plotly_white",
        title="Acquisition Function",
        xaxis_title="Distance / x",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=55, b=20),
        height=320,
        font=dict(size=14)
    )
    fig.update_xaxes(range=[X_MIN, X_MAX])
    return fig


def make_cov_figure(x_grid, cov, kernel_label):
    x_vals = x_grid[:, 0]
    fig = go.Figure(data=go.Heatmap(x=x_vals, y=x_vals, z=cov, colorscale="Viridis", colorbar=dict(title="Covariance")))
    fig.update_layout(
        template="plotly_white",
        title=f"Covariance Matrix - {kernel_label}",
        xaxis_title="Distance / x",
        yaxis_title="Distance / x",
        margin=dict(l=20, r=20, t=55, b=20),
        height=320,
        font=dict(size=14)
    )
    return fig


def make_2d_gaussian_figure(mu1, mu2, var1, var2, cov_val, cond_on="None", cond_val=0.0):
    if var1 * var2 - cov_val**2 <= 0:
        cov_val = np.sign(cov_val) * np.sqrt(var1 * var2) * 0.999 

    cov_matrix = np.array([[var1, cov_val], [cov_val, var2]])
    try:
        rv = multivariate_normal([mu1, mu2], cov_matrix)
    except Exception:
        return go.Figure()
        
    x, y = np.mgrid[-6:6:0.1], np.mgrid[-6:6:0.1]
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    z = rv.pdf(pos)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.015, vertical_spacing=0.015
    )

    fig.add_trace(go.Contour(x=x, y=y, z=z, colorscale="Viridis", showscale=False), row=2, col=1)
    
    pdf1_marg = norm.pdf(x, mu1, np.sqrt(var1))
    pdf2_marg = norm.pdf(y, mu2, np.sqrt(var2))
    
    title_text = f"Continuous 2D Gaussian<br>Covariance = {cov_val:.2f}"
    annot_top = f"Marginal x1 (Var = {var1:.2f})"
    annot_right = f"Marginal x2 (Var = {var2:.2f})"

    if cond_on == "x1":
        cond_mu2 = mu2 + (cov_val / var1) * (cond_val - mu1)
        cond_var2 = var2 - (cov_val**2) / var1
        pdf2_cond = norm.pdf(y, cond_mu2, np.sqrt(max(cond_var2, 1e-9)))

        fig.add_trace(go.Scatter(x=x, y=pdf1_marg, fill='tozeroy', mode='lines', line_color='lightgray', name='Marginal x1'), row=1, col=1)
        fig.add_vline(x=cond_val, line=dict(color='red', dash='dash'), row=1, col=1)
        fig.add_vline(x=cond_val, line=dict(color='red', dash='dash'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=pdf2_marg, y=y, mode='lines', line_color='rgba(250, 128, 114, 0.4)', name='Marginal x2'), row=2, col=2)
        fig.add_trace(go.Scatter(x=pdf2_cond, y=y, fill='tozerox', mode='lines', line_color='red', name='P(x2 | x1)'), row=2, col=2)
        
        title_text = f"Conditioned on x1 = {cond_val:.2f}<br>P(x2 | x1) Mean = {cond_mu2:.2f}, Var = {cond_var2:.2f}"
        annot_right = f"P(x2 | x1) (Var = {cond_var2:.2f})"
        
    elif cond_on == "x2":
        cond_mu1 = mu1 + (cov_val / var2) * (cond_val - mu2)
        cond_var1 = var1 - (cov_val**2) / var2
        pdf1_cond = norm.pdf(x, cond_mu1, np.sqrt(max(cond_var1, 1e-9)))

        fig.add_trace(go.Scatter(x=pdf2_marg, y=y, fill='tozerox', mode='lines', line_color='lightgray', name='Marginal x2'), row=2, col=2)
        fig.add_hline(y=cond_val, line=dict(color='blue', dash='dash'), row=2, col=2)
        fig.add_hline(y=cond_val, line=dict(color='blue', dash='dash'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=x, y=pdf1_marg, mode='lines', line_color='rgba(250, 128, 114, 0.4)', name='Marginal x1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=pdf1_cond, fill='tozeroy', mode='lines', line_color='blue', name='P(x1 | x2)'), row=1, col=1)
        
        title_text = f"Conditioned on x2 = {cond_val:.2f}<br>P(x1 | x2) Mean = {cond_mu1:.2f}, Var = {cond_var1:.2f}"
        annot_top = f"P(x1 | x2) (Var = {cond_var1:.2f})"
        
    else:
        fig.add_trace(go.Scatter(x=x, y=pdf1_marg, fill='tozeroy', mode='lines', line_color='salmon', name='Marginal x1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdf2_marg, y=y, fill='tozerox', mode='lines', line_color='salmon', name='Marginal x2'), row=2, col=2)

    fig.update_layout(
        height=650, 
        showlegend=False, 
        template='plotly_white', 
        font=dict(size=14),
        title=dict(text=title_text, x=0.4, y=0.96, xanchor="center")
    )
    
    fig.layout.annotations = [
        dict(x=0.4, y=1.04, xref="paper", yref="paper", text=annot_top, showarrow=False, font=dict(size=16)),
        dict(x=1.04, y=0.4, xref="paper", yref="paper", text=annot_right, showarrow=False, font=dict(size=16), textangle=-90)
    ]
    
    fig.update_xaxes(range=[-5, 5], title_text="x1", row=2, col=1)
    fig.update_yaxes(range=[-5, 5], title_text="x2", row=2, col=1)
    return fig


# =========================
# Dash app
# =========================
app = Dash(__name__, suppress_callback_exceptions=True)
# Diese Zeile ist das, was Render/Gunicorn braucht:
server = app.server
app.title = "Interactive Bayesian Optimization"

app.layout = html.Div(
    [
        dcc.Store(id="point-store", data={"xs": [], "ys": []}),
        html.Div(
            [
                html.Div(id="bo-controls", children=[
                    html.H2("Interactive BO App", style={"marginTop": "0px"}),
                    html.Div(
                        "Click in the GP plot to add measurement points.",
                        style={"marginBottom": "16px", "lineHeight": "1.4"},
                    ),
                    html.Label("Ground truth model"),
                    dcc.Dropdown(
                        id="gt-model",
                        value="complex_sin",
                        clearable=False,
                        options=[
                            {"label": "Complex sine (gold-mine style)", "value": "complex_sin"},
                            {"label": "Simple sine", "value": "simple_sin"},
                            {"label": "Quadratic", "value": "quadratic"},
                            {"label": "Step", "value": "step"},
                        ],
                    ),
                    html.Br(),
                    html.Div(id="kernel-controls-group", children=[
                        html.Label("Kernel strategy"),
                        dcc.RadioItems(
                            id="kernel-strategy",
                            value="manual_single",
                            options=[
                                {"label": "Manual single kernel", "value": "manual_single"},
                                {"label": "Manual combined kernels", "value": "manual_combined"},
                                {"label": "Auto-select best kernel", "value": "auto_select"},
                            ],
                            style={"display": "grid", "gap": "6px"},
                        ),
                        html.Br(),
                        html.Div(id="single-kernel-group", children=[
                            html.Label("Single kernel"),
                            dcc.Dropdown(
                                id="kernel-type",
                                value="matern",
                                clearable=False,
                                options=[
                                    {"label": "RBF", "value": "rbf"},
                                    {"label": "Matern", "value": "matern"},
                                    {"label": "Periodic", "value": "periodic"},
                                    {"label": "Rational Quadratic", "value": "rq"},
                                    {"label": "Linear", "value": "linear"},
                                ],
                            ),
                            html.Br(),
                        ]),
                        html.Div(id="combined-kernel-group", children=[
                            html.Label("Combined kernels"),
                            dcc.Dropdown(
                                id="combined-kernels",
                                value=["linear", "rbf"],
                                multi=True,
                                clearable=False,
                                options=[
                                    {"label": "Linear", "value": "linear"},
                                    {"label": "RBF", "value": "rbf"},
                                    {"label": "Periodic", "value": "periodic"},
                                    {"label": "Matern", "value": "matern"},
                                    {"label": "Rational Quadratic", "value": "rq"},
                                ],
                            ),
                            html.Br(),
                            html.Label("Combination mode"),
                            dcc.RadioItems(
                                id="combined-mode",
                                value="sum",
                                options=[
                                    {"label": "+ sum", "value": "sum"},
                                    {"label": "* product", "value": "product"},
                                ],
                                style={"display": "flex", "gap": "20px"},
                            ),
                            html.Br(),
                        ]),
                        html.Div(id="length-scale-group", children=[
                            html.Label("Kernel length scale"),
                            dcc.Slider(id="length-scale", min=0.1, max=5.0, step=0.1, value=0.5, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="period-group", children=[
                            html.Label("Periodic period"),
                            dcc.Slider(id="period", min=1.0, max=10.0, step=0.5, value=5.0, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="matern-nu-group", children=[
                            html.Label("Matern nu"),
                            dcc.Slider(id="matern-nu", min=0.5, max=2.5, step=0.5, value=1.5, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="rq-alpha-group", children=[
                            html.Label("RQ alpha"),
                            dcc.Slider(id="rq-alpha", min=0.1, max=5.0, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="linear-sigma-0-group", children=[
                            html.Label("Linear sigma_0"),
                            dcc.Slider(id="linear-sigma-0", min=0.1, max=5.0, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="kernel-amplitude-group", children=[
                            html.Label("Kernel amplitude"),
                            dcc.Slider(id="kernel-amplitude", min=0.1, max=5.0, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Br(),
                        ]),
                        html.Div(id="poly-c-group", children=[
                            html.Label("Polynomial c"),
                            dcc.Slider(id="poly-c", min=0.1, max=5.0, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                        ], style={"display": "none"}),
                    ]),
                    html.Div(
                        id="acq-controls-group",
                        children=[
                            html.Hr(),
                            html.Label("Acquisition method"),
                            dcc.Dropdown(
                                id="acq-method",
                                value="ucb",
                                clearable=False,
                                options=[
                                    {"label": "Expected Improvement (EI)", "value": "ei"},
                                    {"label": "Probability of Improvement (PI)", "value": "pi"},
                                    {"label": "Upper Confidence Bound (UCB)", "value": "ucb"},
                                ],
                            ),
                            html.Br(),
                            html.Div(id="xi-group", children=[
                                html.Label("xi for EI / PI"),
                                dcc.Slider(id="xi", min=0.0, max=1.0, step=0.01, value=0.01, tooltip={"placement": "bottom", "always_visible": True}),
                                html.Br(),
                            ]),
                            html.Div(id="kappa-group", children=[
                                html.Label("kappa for UCB"),
                                dcc.Slider(id="kappa", min=0.1, max=5.0, step=0.1, value=2.0, tooltip={"placement": "bottom", "always_visible": True}),
                            ]),
                        ]
                    ),
                    html.Hr(),
                    html.Label("Max points"),
                    dcc.Slider(id="max-points", min=1, max=40, step=1, value=DEFAULT_MAX_POINTS, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Measurement noise variance"),
                    dcc.Slider(id="noise-variance", min=0.0, max=0.5, step=0.001, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Div(
                        id="sample-controls-group",
                        style={"marginTop": "20px"},
                        children=[
                            html.Label("Posterior samples shown"),
                            dcc.Slider(id="sample-count", min=1, max=10, step=1, value=3, tooltip={"placement": "bottom", "always_visible": True}),
                        ]
                    ),
                    html.Br(),
                    html.Button("Clear measurements", id="clear-btn", n_clicks=0, style={"width": "100%", "padding": "10px"}),
                ]),
                html.Div(id="two-d-controls", children=[
                    html.H2("2D Gaussian Playground", style={"marginTop": "0px"}),
                    html.Div(
                        "Play with the Mean and Covariance values to see how the 2D Gaussian changes.",
                        style={"marginBottom": "16px", "lineHeight": "1.4"},
                    ),
                    html.Label("Mean x1"),
                    dcc.Slider(id="mu1", min=-3, max=3, step=0.1, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Mean x2"),
                    dcc.Slider(id="mu2", min=-3, max=3, step=0.1, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Variance x1"),
                    dcc.Slider(id="var1", min=0.1, max=5, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Variance x2"),
                    dcc.Slider(id="var2", min=0.1, max=5, step=0.1, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Covariance"),
                    dcc.Slider(id="cov12", min=-4.9, max=4.9, step=0.1, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Hr(),
                    html.Label("Conditioning & Marginalization"),
                    dcc.RadioItems(
                        id="condition-on",
                        value="None",
                        options=[
                            {"label": "Just Marginals", "value": "None"},
                            {"label": "Condition on x1 (slice vertical)", "value": "x1"},
                            {"label": "Condition on x2 (slice horizontal)", "value": "x2"},
                        ],
                        style={"display": "grid", "gap": "6px", "marginBottom": "10px"}
                    ),
                    html.Div(id="cond-val-group", children=[
                        html.Label("Observed Value"),
                        dcc.Slider(id="condition-val", min=-5.0, max=5.0, step=0.1, value=0.0, tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={"display": "none"}),
                ], style={"display": "none"}),
                html.Div(id="null-model-controls", children=[
                    html.H2("Bayesian Interval Null Model", style={"marginTop": "0px"}),
                    html.Label("H0 Interval (c)"),
                    dcc.Slider(id="c-val", min=0.05, max=1.0, step=0.05, value=0.2, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Observed Effect/Mean"),
                    dcc.Slider(id="obs-effect", min=-3.0, max=3.0, step=0.1, value=0.8, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Prior Width (Std Dev)"),
                    dcc.Slider(id="prior-sd", min=0.5, max=3.0, step=0.1, value=1.2, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Posterior Width (Std Dev)"),
                    dcc.Slider(id="post-sd", min=0.1, max=1.5, step=0.1, value=0.4, tooltip={"placement": "bottom", "always_visible": True}),
                ], style={"display": "none"}),
            ],
            style={
                "width": "340px",
                "minWidth": "340px",
                "padding": "20px",
                "borderRight": "1px solid #ddd",
                "background": "#f8f9fb",
                "height": "100vh",
                "overflowY": "auto",
                "boxSizing": "border-box",
            },
        ),
        html.Div(
            [
                dcc.Tabs(id="tabs", value="tab-all", children=[
                    dcc.Tab(label="Kernels, Samples & Covariance", value="tab-kernel-cov"),
                    dcc.Tab(label="Acquisition", value="tab-acq"),
                    dcc.Tab(label="All Together", value="tab-all"),
                    dcc.Tab(label="2D Cov Matrix", value="tab-2d"),
                    dcc.Tab(label="Null Model", value="tab-null"),
                ]),
                html.Div(id="bo-view", children=[
                    html.Div(id="summary-box", style={"marginBottom": "14px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "background": "white"}),
                    html.Div(id="gp-container", children=dcc.Graph(id="gp-graph", config={"displayModeBar": True})),
                    html.Div(id="flex-container", children=[
                        html.Div(id="samples-container", children=dcc.Graph(id="samples-graph"), style={"flex": "1"}),
                        html.Div(id="acq-container", children=dcc.Graph(id="acq-graph"), style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "12px"}),
                    html.Div(id="cov-container", children=dcc.Graph(id="cov-graph")),
                ], style={"paddingTop": "15px"}),
                
                html.Div(id="two-d-view", children=[
                    dcc.Graph(id="2d-graph", style={"height": "80vh"})
                ], style={"display": "none", "paddingTop": "15px"}),
                
                html.Div(id="null-model-view", children=[
                    dcc.Graph(id="null-model-graph"),
                    html.Div(id="null-model-text", style={"marginTop": "20px", "fontSize": "16px", "padding": "20px", "background": "#f8f9fb", "borderRadius": "8px"})
                ], style={"display": "none", "paddingTop": "15px", "width": "100%"})
            ],
            style={"flex": "1", "padding": "18px", "background": "#ffffff"},
        ),
    ],
    style={"display": "flex", "fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("point-store", "data"),
    Input("gp-graph", "clickData"),
    Input("clear-btn", "n_clicks"),
    State("point-store", "data"),
    State("max-points", "value"),
    State("gt-model", "value"),
    prevent_initial_call=True,
)
def update_points(click_data, clear_clicks, store, max_points, gt_model):
    trigger = ctx.triggered_id
    store = store or {"xs": [], "ys": []}
    if trigger == "clear-btn":
        return {"xs": [], "ys": []}

    if trigger != "gp-graph" or not click_data:
        raise PreventUpdate

    if len(store["xs"]) >= int(max_points):
        return store

    point = click_data["points"][0]
    x_clicked = float(point["x"])
    x_clicked = round(x_clicked, 4)
    if any(np.isclose(x_clicked, old_x, atol=1e-6) for old_x in store["xs"]):
        return store

    y_true = get_ground_truth(gt_model)(np.array([x_clicked]))[0]
    store["xs"].append(x_clicked)
    store["ys"].append(float(y_true))
    return store


@app.callback(
    Output("gp-graph", "figure"),
    Output("samples-graph", "figure"),
    Output("acq-graph", "figure"),
    Output("cov-graph", "figure"),
    Output("summary-box", "children"),
    Input("point-store", "data"),
    Input("gt-model", "value"),
    Input("kernel-strategy", "value"),
    Input("kernel-type", "value"),
    Input("combined-kernels", "value"),
    Input("combined-mode", "value"),
    Input("acq-method", "value"),
    Input("xi", "value"),
    Input("kappa", "value"),
    Input("length-scale", "value"),
    Input("period", "value"),
    Input("matern-nu", "value"),
    Input("rq-alpha", "value"),
    Input("linear-sigma-0", "value"),
    Input("kernel-amplitude", "value"),
    Input("poly-c", "value"),
    Input("max-points", "value"),
    Input("noise-variance", "value"),
    Input("sample-count", "value"),
)
def render_dashboard(
    store,
    gt_model,
    kernel_strategy,
    kernel_type,
    combined_kernels,
    combined_mode,
    acq_method,
    xi,
    kappa,
    length_scale,
    period,
    matern_nu,
    rq_alpha,
    linear_sigma_0,
    kernel_amplitude,
    poly_c,
    max_points,
    noise_variance,
    sample_count,
):
    x_grid = np.linspace(X_MIN, X_MAX, GRID_POINTS).reshape(-1, 1)
    gt_func = get_ground_truth(gt_model)
    y_true = gt_func(x_grid).reshape(-1)

    controls = {
        "kernel_strategy": kernel_strategy,
        "kernel_type": kernel_type,
        "combined_mode": combined_mode,
        "combined_kernels": combined_kernels or [kernel_type],
        "noise_variance": noise_variance,
        "length_scale": length_scale,
        "period": period,
        "matern_nu": matern_nu,
        "rq_alpha": rq_alpha,
        "linear_sigma_0": linear_sigma_0,
        "kernel_amplitude": kernel_amplitude,
        "poly_c": poly_c,
    }

    points = store or {"xs": [], "ys": []}
    X_train, y_train = make_training_arrays(points)
    model_info = fit_model(points, controls, x_grid)

    y_best = float(np.max(y_train)) if y_train.size > 0 else 0.0
    if model_info["gp"] is None:
        acquisition_model = PriorPredictor(model_info["mu"], model_info["sigma"])
    else:
        acquisition_model = model_info["gp"]
    acquisition, acquisition_label = compute_acquisition(x_grid, acquisition_model, y_best, acq_method, xi, kappa)

    next_x = suggest_next_point(acquisition, x_grid, X_train[:, 0] if X_train.shape[0] > 0 else [])
    if X_train.shape[0] >= int(max_points):
        next_x = None

    gp_figure = make_gp_figure(x_grid, y_true, model_info, points, next_x)
    samples_figure = make_samples_figure(x_grid, y_true, model_info, n_samples=int(sample_count), seed=SEED + X_train.shape[0])
    acq_figure = make_acq_figure(x_grid, acquisition, acquisition_label, next_x)
    cov_figure = make_cov_figure(x_grid, model_info["cov"], model_info["kernel_label"])

    status_lines = [
        html.B("Current configuration"),
        html.Div(f"Ground truth: {gt_model}"),
        html.Div(f"Kernel: {model_info['kernel_label']}") ,
        html.Div(f"Acquisition: {acq_method.upper()} (xi={xi}, kappa={kappa})"),
        html.Div(f"Measured points: {len(points['xs'])} / {max_points}"),
    ]
    if next_x is not None:
        status_lines.append(html.Div(f"Suggested next x: {next_x:.4f}"))
    else:
        status_lines.append(html.Div("Maximum number of points reached or no valid next point left."))

    return gp_figure, samples_figure, acq_figure, cov_figure, status_lines


@app.callback(
    Output("bo-controls", "style"),
    Output("two-d-controls", "style"),
    Output("null-model-controls", "style"),
    Output("bo-view", "style"),
    Output("two-d-view", "style"),
    Output("null-model-view", "style"),
    Output("summary-box", "style"),
    Output("gp-container", "style"),
    Output("flex-container", "style"),
    Output("samples-container", "style"),
    Output("acq-container", "style"),
    Output("cov-container", "style"),
    Output("acq-controls-group", "style"),
    Output("sample-controls-group", "style"),
    Output("kernel-controls-group", "style"),
    Input("tabs", "value")
)
def switch_tabs(tab):
    if tab is None:
        tab = "tab-all"
        
    # Base hidden style using display: none
    hidden = {"display": "none"}
    
    bo_ctrl = hidden.copy()
    two_d_ctrl = hidden.copy()
    null_ctrl = hidden.copy()
    bo_view = hidden.copy()
    two_d_view = hidden.copy()
    null_view = hidden.copy()
    
    summary = hidden.copy()
    gp = hidden.copy()
    flex = hidden.copy()
    samples = hidden.copy()
    acq = hidden.copy()
    cov = hidden.copy()
    
    acq_controls = hidden.copy()
    sample_controls = hidden.copy()
    kernel_controls = hidden.copy()

    if tab == "tab-2d":
        two_d_ctrl = {"display": "block"}
        two_d_view = {"display": "block", "paddingTop": "15px", "width": "100%"}
    elif tab == "tab-null":
        null_ctrl = {"display": "block"}
        null_view = {"display": "block", "paddingTop": "15px", "width": "100%"}
    else:
        bo_ctrl = {"display": "block"}
        bo_view = {"display": "flex", "flexDirection": "column", "gap": "15px", "paddingTop": "15px", "width": "100%"}
        
        # summary AND gp are always plotted in BO mode so you can click & add points!
        summary = {"marginBottom": "0px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "background": "white"}
        gp = {"display": "block", "width": "100%"}
        
        flex_base = {"display": "flex", "gap": "12px", "width": "100%", "flexWrap": "wrap"}
        item_base = {"flex": "1 1 45%", "minWidth": "0px"}

        if tab == "tab-kernel-cov":
            flex = flex_base
            samples = item_base
            cov = {"display": "block", "width": "100%"}
            sample_controls = {"display": "block", "marginTop": "20px"}
            kernel_controls = {"display": "block"}
        elif tab == "tab-acq":
            flex = flex_base
            acq = item_base
            acq_controls = {"display": "block"}
        elif tab == "tab-all":
            flex = flex_base
            samples = item_base
            acq = item_base
            cov = {"display": "block", "width": "100%"}
            acq_controls = {"display": "block"}
            sample_controls = {"display": "block", "marginTop": "20px"}
            kernel_controls = {"display": "block"}
            
    return bo_ctrl, two_d_ctrl, null_ctrl, bo_view, two_d_view, null_view, summary, gp, flex, samples, acq, cov, acq_controls, sample_controls, kernel_controls


@app.callback(
    Output("single-kernel-group", "style"),
    Output("combined-kernel-group", "style"),
    Output("length-scale-group", "style"),
    Output("period-group", "style"),
    Output("matern-nu-group", "style"),
    Output("rq-alpha-group", "style"),
    Output("linear-sigma-0-group", "style"),
    Output("kernel-amplitude-group", "style"),
    Input("kernel-strategy", "value"),
    Input("kernel-type", "value"),
    Input("combined-kernels", "value")
)
def toggle_kernel_params(strategy, single_type, combined_types):
    show = {"display": "block"}
    hide = {"display": "none"}
    
    out_single = hide
    out_combined = hide
    out_length = hide
    out_period = hide
    out_matern = hide
    out_rq = hide
    out_linear = hide
    out_amp = hide

    if strategy != "auto_select":
        out_amp = show
        
        active_kernels = []
        if strategy == "manual_single":
            out_single = show
            active_kernels = [single_type]
        elif strategy == "manual_combined":
            out_combined = show
            active_kernels = combined_types if combined_types else []

        if any(k in ["rbf", "matern", "periodic", "rq"] for k in active_kernels):
            out_length = show
        if "periodic" in active_kernels:
            out_period = show
        if "matern" in active_kernels:
            out_matern = show
        if "rq" in active_kernels:
            out_rq = show
        if "linear" in active_kernels:
            out_linear = show

    return out_single, out_combined, out_length, out_period, out_matern, out_rq, out_linear, out_amp


@app.callback(
    Output("xi-group", "style"),
    Output("kappa-group", "style"),
    Input("acq-method", "value")
)
def toggle_acq_params(method):
    if method in ["ei", "pi"]:
        return {"display": "block"}, {"display": "none"}
    elif method == "ucb":
        return {"display": "none"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}


@app.callback(
    Output("cond-val-group", "style"),
    Input("condition-on", "value")
)
def toggle_cond_val(cond_var):
    if cond_var in ["x1", "x2"]:
        return {"display": "block", "marginTop": "10px"}
    return {"display": "none"}


@app.callback(
    Output("2d-graph", "figure"),
    Input("mu1", "value"),
    Input("mu2", "value"),
    Input("var1", "value"),
    Input("var2", "value"),
    Input("cov12", "value"),
    Input("condition-on", "value"),
    Input("condition-val", "value")
)
def update_2d_gaussian(mu1, mu2, var1, var2, cov12, cond_on, cond_val):
    return make_2d_gaussian_figure(mu1, mu2, var1, var2, cov12, cond_on, cond_val)


@app.callback(
    Output("null-model-graph", "figure"),
    Output("null-model-text", "children"),
    Input("c-val", "value"),
    Input("obs-effect", "value"),
    Input("prior-sd", "value"),
    Input("post-sd", "value")
)
def update_null_model(c_val, obs_effect, prior_sd, post_sd):
    x = np.linspace(-4, 4, 1000)
    prior_density = norm.pdf(x, 0, prior_sd)
    posterior_density = norm.pdf(x, obs_effect, post_sd)

    area_prior = norm.cdf(c_val, 0, prior_sd) - norm.cdf(-c_val, 0, prior_sd)
    area_post = norm.cdf(c_val, obs_effect, post_sd) - norm.cdf(-c_val, obs_effect, post_sd)
    bf_01 = area_post / area_prior if area_prior > 0 else np.inf

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=prior_density, name="Prior", line=dict(color="gray", width=3)))
    fig.add_trace(go.Scatter(x=x, y=posterior_density, name="Posterior", line=dict(color="blue", width=3)))

    x_fill = np.linspace(-c_val, c_val, 100)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([norm.pdf(x_fill, 0, prior_sd), np.zeros_like(x_fill)]),
        fill="toself", fillcolor="rgba(128,128,128,0.3)",
        line=dict(color="rgba(255,255,255,0)"), name="Prior Area", hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([norm.pdf(x_fill, obs_effect, post_sd), np.zeros_like(x_fill)]),
        fill="toself", fillcolor="rgba(0,0,255,0.3)",
        line=dict(color="rgba(255,255,255,0)"), name="Posterior Area", hoverinfo="skip"
    ))

    fig.add_vline(x=c_val, line=dict(color="red", dash="dash"))
    fig.add_vline(x=-c_val, line=dict(color="red", dash="dash"))

    fig.update_layout(
        title="Probability Mass in the Null Interval",
        xaxis_title="Effect Size (Delta)",
        yaxis_title="Density",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(size=14)
    )

    interpretation = "supports the null interval" if bf_01 > 1 else "speaks AGAINST the null interval"
    
    text_out = html.Div([
        html.Div(f"Area in Null Interval BEFORE (Prior): {area_prior*100:.2f} %", style={"marginBottom": "10px"}),
        html.Div(f"Area in Null Interval AFTER (Posterior): {area_post*100:.2f} %", style={"marginBottom": "10px"}),
        html.H3(f"Interval Bayes Factor (BF_01): {bf_01:.4f}", style={"color": "#2c3e50"}),
        html.Div(f"-> Data {interpretation}", style={"fontWeight": "bold", "color": "green" if bf_01 > 1 else "red"})
    ])

    return fig, text_out


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8052)
