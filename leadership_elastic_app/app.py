from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

# -----------------------------
# Model configuration
# -----------------------------
VIRTUES = [
    "Courage",
    "Transcendence",
    "Drive",
    "Collaboration",
    "Humanity",
    "Humility",
    "Integrity",
    "Temperance",
    "Justice",
    "Accountability",
]

VIRTUE_COLORS = {
    "Courage": "#ff6b6b",
    "Transcendence": "#9b7bff",
    "Drive": "#f2994a",
    "Collaboration": "#56ccf2",
    "Humanity": "#6fcf97",
    "Humility": "#bdbdbd",
    "Integrity": "#27ae60",
    "Temperance": "#2d9cdb",
    "Justice": "#f2c94c",
    "Accountability": "#eb5757",
}

THETA = np.linspace(0.0, 2.0 * np.pi, len(VIRTUES), endpoint=False)
THETA_BY_VIRTUE = {virtue: THETA[i] for i, virtue in enumerate(VIRTUES)}

DEFAULT_R = 0.45
DEFAULT_Z = 0.0
DEFAULT_CAPACITY = 1.8
MAX_CAPACITY = 3.2
MIN_CAPACITY = 0.4

# Nonlinear L2 compression (still proportional scaling)
COMPRESSION_EXPONENT = 1.15

# In-graph handle behavior
HANDLE_R_STEP = 0.06
HANDLE_Z_STEP = 0.12
HANDLE_OFFSET_VISUAL = 0.18
HANDLE_MARKER_SIZE = 14
HANDLE_SHAPE_RADIUS_2D = 0.05

# Optional training mode
TRAINING_DELTA = 0.01
TRAINING_BALANCE_THRESHOLD = 0.30
TRAINING_MIN_MEAN = 0.22


@dataclass
class ConstraintState:
    requested_r: np.ndarray
    requested_z: np.ndarray
    effective_r: np.ndarray
    effective_z: np.ndarray
    requested_norm: float
    effective_norm: float
    scale: float
    capacity: float
    mode: str


def enforce_capacity(
    r_values: list[float] | np.ndarray,
    z_values: list[float] | np.ndarray,
    capacity: float,
    mode: str,
) -> ConstraintState:
    """Apply global elastic capacity with nonlinear L2 proportional compression."""
    r = np.asarray(r_values, dtype=float)
    z = np.asarray(z_values, dtype=float)

    if mode == "2d":
        z = np.zeros_like(r)
        requested_norm = float(np.sqrt(np.sum(r**2)))
    else:
        requested_norm = float(np.sqrt(np.sum(r**2 + z**2)))

    safe_capacity = float(np.clip(capacity, MIN_CAPACITY, MAX_CAPACITY))

    if requested_norm <= safe_capacity or requested_norm <= 1e-12:
        scale = 1.0
    else:
        linear_ratio = safe_capacity / requested_norm
        scale = float(linear_ratio**COMPRESSION_EXPONENT)

    effective_r = r * scale
    effective_z = z * scale

    if mode == "2d":
        effective_norm = float(np.sqrt(np.sum(effective_r**2)))
    else:
        effective_norm = float(np.sqrt(np.sum(effective_r**2 + effective_z**2)))

    return ConstraintState(
        requested_r=r,
        requested_z=z,
        effective_r=effective_r,
        effective_z=effective_z,
        requested_norm=requested_norm,
        effective_norm=effective_norm,
        scale=scale,
        capacity=safe_capacity,
        mode=mode,
    )


def balanced_usage_score(state: ConstraintState) -> tuple[float, float]:
    """Return (mean magnitude, coefficient of variation). Lower CV means more balanced usage."""
    if state.mode == "2d":
        magnitudes = state.effective_r
    else:
        magnitudes = np.sqrt(state.effective_r**2 + state.effective_z**2)

    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))
    cv = std_mag / max(mean_mag, 1e-8)
    return mean_mag, cv


def _circle_points(radius: float, n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros_like(t)
    return x, y, z


def _sphere_mesh(radius: float, u_steps: int = 48, v_steps: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, u_steps)
    v = np.linspace(0.0, np.pi, v_steps)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _fan_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build triangle fan indices using vertex 0 as center and 1..n as boundary."""
    i = np.zeros(n, dtype=int)
    j = np.arange(1, n + 1, dtype=int)
    k = np.roll(j, -1)
    return i, j, k


def build_figure(state: ConstraintState) -> go.Figure:
    req_x = state.requested_r * np.cos(THETA)
    req_y = state.requested_r * np.sin(THETA)
    req_z = state.requested_z if state.mode == "3d" else np.zeros_like(req_x)

    eff_x = state.effective_r * np.cos(THETA)
    eff_y = state.effective_r * np.sin(THETA)
    eff_z = state.effective_z if state.mode == "3d" else np.zeros_like(eff_x)
    axis_limit_xy = max(1.0, state.capacity * 1.12)
    z_limit = 0.22 if state.mode == "2d" else 1.25

    fig = go.Figure()

    # Capacity boundary
    if state.mode == "3d":
        sx, sy, sz = _sphere_mesh(state.capacity)
        fig.add_trace(
            go.Surface(
                x=sx,
                y=sy,
                z=sz,
                colorscale=[[0.0, "#6f86a4"], [1.0, "#6f86a4"]],
                opacity=0.11,
                showscale=False,
                hoverinfo="skip",
                name="Capacity Sphere",
                lighting={"ambient": 0.35, "diffuse": 0.75, "specular": 0.45, "roughness": 0.25, "fresnel": 0.15},
                lightposition={"x": 130, "y": 90, "z": 250},
            )
        )

    cx, cy, cz = _circle_points(state.capacity)
    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="lines",
            line={"color": "#a9b7c8", "width": 3},
            name="Capacity Boundary",
            hovertemplate="Capacity boundary<extra></extra>",
        )
    )

    # Demand contour (before compression)
    fig.add_trace(
        go.Scatter3d(
            x=np.r_[req_x, req_x[0]],
            y=np.r_[req_y, req_y[0]],
            z=np.r_[req_z, req_z[0]],
            mode="lines",
            line={"color": "#ffad33", "width": 3, "dash": "dash"},
            name="Requested Shape",
            hovertemplate="Requested contour<extra></extra>",
        )
    )

    # Effective contour (after compression)
    fig.add_trace(
        go.Scatter3d(
            x=np.r_[eff_x, eff_x[0]],
            y=np.r_[eff_y, eff_y[0]],
            z=np.r_[eff_z, eff_z[0]],
            mode="lines",
            line={"color": "#35d0ff", "width": 5},
            name="Effective Shape",
            hovertemplate="Effective contour<extra></extra>",
        )
    )

    # Filled envelopes so compression is visually obvious ("balloon effect")
    req_fill_z = req_z if state.mode == "3d" else np.full_like(req_z, 0.008)
    eff_fill_z = eff_z if state.mode == "3d" else np.full_like(eff_z, 0.012)
    req_i, req_j, req_k = _fan_indices(len(req_x))
    eff_i, eff_j, eff_k = _fan_indices(len(eff_x))
    fig.add_trace(
        go.Mesh3d(
            x=np.r_[0.0, req_x],
            y=np.r_[0.0, req_y],
            z=np.r_[0.0, req_fill_z],
            i=req_i,
            j=req_j,
            k=req_k,
            color="#ffad33",
            opacity=0.18,
            name="Requested Envelope",
            hoverinfo="skip",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=np.r_[0.0, eff_x],
            y=np.r_[0.0, eff_y],
            z=np.r_[0.0, eff_fill_z],
            i=eff_i,
            j=eff_j,
            k=eff_k,
            color="#35d0ff",
            opacity=0.42,
            name="Effective Envelope",
            hoverinfo="skip",
            showscale=False,
            lighting={"ambient": 0.45, "diffuse": 0.72, "specular": 0.5, "roughness": 0.3, "fresnel": 0.1},
            lightposition={"x": 140, "y": 70, "z": 220},
        )
    )

    # Rays from Judgment to each virtue effective location
    for i, virtue in enumerate(VIRTUES):
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, eff_x[i]],
                y=[0.0, eff_y[i]],
                z=[0.0, eff_z[i]],
                mode="lines",
                line={"color": "#c2ccda", "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Virtue points
    for i, virtue in enumerate(VIRTUES):
        fig.add_trace(
            go.Scatter3d(
                x=[eff_x[i]],
                y=[eff_y[i]],
                z=[eff_z[i]],
                mode="markers+text",
                marker={"size": 15, "color": VIRTUE_COLORS[virtue], "line": {"color": "#0f1b2e", "width": 1.0}},
                text=[virtue],
                textposition="top center",
                textfont={"size": 11, "color": "#15243a"},
                name=virtue,
                hovertemplate=(
                    f"{virtue}<br>"
                    f"r: {state.effective_r[i]:.3f}<br>"
                    f"z: {state.effective_z[i]:.3f}<extra></extra>"
                ),
            )
        )

        # Interactive radial handles (click to adjust r request)
        base_r = float(state.requested_r[i])
        base_theta = float(THETA[i])
        base_z = float(state.requested_z[i] if state.mode == "3d" else eff_z[i])
        visual_r_plus = min(1.0, base_r + HANDLE_OFFSET_VISUAL)
        visual_r_minus = max(0.0, base_r - HANDLE_OFFSET_VISUAL)

        rp_x = visual_r_plus * math.cos(base_theta)
        rp_y = visual_r_plus * math.sin(base_theta)
        rm_x = visual_r_minus * math.cos(base_theta)
        rm_y = visual_r_minus * math.sin(base_theta)

        fig.add_trace(
            go.Scatter3d(
                    x=[rp_x],
                    y=[rp_y],
                    z=[base_z],
                mode="markers",
                marker={
                    "size": HANDLE_MARKER_SIZE,
                    "symbol": "diamond-open",
                    "color": VIRTUE_COLORS[virtue],
                    "line": {"color": VIRTUE_COLORS[virtue], "width": 2},
                },
                customdata=[[virtue, "r_plus"]],
                name=f"{virtue} r+",
                showlegend=False,
                hovertemplate=f"{virtue}<br>Click: increase radial magnitude<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                    x=[rm_x],
                    y=[rm_y],
                    z=[base_z],
                mode="markers",
                marker={
                    "size": HANDLE_MARKER_SIZE,
                    "symbol": "diamond",
                    "color": "#ffffff",
                    "line": {"color": VIRTUE_COLORS[virtue], "width": 1.5},
                },
                customdata=[[virtue, "r_minus"]],
                name=f"{virtue} r-",
                showlegend=False,
                hovertemplate=f"{virtue}<br>Click: decrease radial magnitude<extra></extra>",
            )
        )

        # Interactive vertical handles in 3D mode (click to adjust z request)
        if state.mode == "3d":
            z_plus = min(1.0, base_z + HANDLE_OFFSET_VISUAL)
            z_minus = max(-1.0, base_z - HANDLE_OFFSET_VISUAL)

            fig.add_trace(
                go.Scatter3d(
                    x=[req_x[i], req_x[i]],
                    y=[req_y[i], req_y[i]],
                    z=[z_minus, z_plus],
                    mode="lines",
                    line={"color": "#6d86ab", "width": 5, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[req_x[i]],
                    y=[req_y[i]],
                    z=[z_plus],
                    mode="markers",
                    marker={
                        "size": HANDLE_MARKER_SIZE + 4,
                        "symbol": "triangle-up",
                        "color": VIRTUE_COLORS[virtue],
                        "line": {"color": "#0f1b2e", "width": 1.2},
                    },
                    customdata=[[virtue, "z_plus"]],
                    showlegend=False,
                    hovertemplate=f"{virtue}<br>Click: move +Z (Leading Up)<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[req_x[i]],
                    y=[req_y[i]],
                    z=[z_minus],
                    mode="markers",
                    marker={
                        "size": HANDLE_MARKER_SIZE + 4,
                        "symbol": "triangle-down",
                        "color": VIRTUE_COLORS[virtue],
                        "line": {"color": "#0f1b2e", "width": 1.2},
                    },
                    customdata=[[virtue, "z_minus"]],
                    showlegend=False,
                    hovertemplate=f"{virtue}<br>Click: move -Z (Leading Down)<extra></extra>",
                )
            )

    # Judgment fixed at origin
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={"size": 11, "color": "#111111", "symbol": "diamond"},
            text=["Judgment"],
            textposition="bottom center",
            textfont={"size": 12, "color": "#111111"},
            name="Judgment (Fixed)",
            hovertemplate="Judgment (fixed origin)<extra></extra>",
        )
    )

    if state.mode == "2d":
        camera = {"eye": {"x": 0.0, "y": 0.0, "z": 2.5}}
        z_range = [-z_limit, z_limit]
    else:
        camera = {"eye": {"x": 1.55, "y": 1.35, "z": 1.1}}
        z_range = [-z_limit, z_limit]

    title_suffix = "2D Character Mode" if state.mode == "2d" else "3D Meta-Leadership Mode"

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
        font={"family": "Helvetica, Arial, sans-serif", "size": 15, "color": "#1b2a40"},
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 0.98,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": "#1b2a40"},
        },
        title={
            "text": f"Character Elastic Constraint System — {title_suffix}",
            "x": 0.01,
            "font": {"size": 17, "color": "#0f1f35"},
        },
        transition={"duration": 120, "easing": "linear"},
        scene={
            "camera": camera,
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35 if state.mode == "2d" else 1.0},
            "xaxis": {
                "title": "Leading Across (X)",
                "range": [-axis_limit_xy, axis_limit_xy],
                "showgrid": False,
                "zeroline": True,
                "zerolinecolor": "#9fb3cf",
                "color": "#1c2f48",
                "backgroundcolor": "#ffffff",
                "showbackground": False,
            },
            "yaxis": {
                "title": "Leading Across (Y)",
                "range": [-axis_limit_xy, axis_limit_xy],
                "showgrid": False,
                "zeroline": True,
                "zerolinecolor": "#9fb3cf",
                "color": "#1c2f48",
                "backgroundcolor": "#ffffff",
                "showbackground": False,
            },
            "zaxis": {
                "title": "Meta Axis ( +Up / -Down )",
                "range": z_range,
                "showgrid": False,
                "zeroline": True,
                "zerolinecolor": "#9fb3cf",
                "visible": True,
                "showticklabels": state.mode == "3d",
                "color": "#1c2f48",
                "backgroundcolor": "#ffffff",
                "showbackground": False,
            },
            "uirevision": "elastic-leadership",
        },
    )

    return fig


def controls_panel() -> html.Div:
    return html.Div(
        [
            html.H2("Leadership Elastic Controls", style={"marginTop": "0.2rem", "marginBottom": "0.4rem"}),
            html.Div(
                "Adjust virtue magnitudes, capacity, and meta-axis displacement. Judgment remains fixed at the origin.",
                style={"color": "#516781", "fontSize": "0.92rem", "marginBottom": "0.9rem"},
            ),
            html.Div(
                "In-graph handles: click ◇ to adjust radial magnitude; in 3D mode click ▲ / ▼ to move up/down.",
                style={
                    "color": "#2d3f59",
                    "fontSize": "0.86rem",
                    "background": "#eef4ff",
                    "border": "1px solid #d6e0ef",
                    "padding": "0.45rem 0.55rem",
                    "borderRadius": "8px",
                    "marginBottom": "0.8rem",
                },
            ),
            html.Label("Mode", style={"fontSize": "0.90rem", "fontWeight": 600, "color": "#23344f", "display": "block", "marginBottom": "0.25rem"}),
            dcc.RadioItems(
                id="mode-toggle",
                options=[
                    {"label": "2D Character Mode", "value": "2d"},
                    {"label": "3D Meta-Leadership Mode", "value": "3d"},
                ],
                value="2d",
                labelStyle={"display": "block", "margin": "0.25rem 0"},
                style={"marginBottom": "0.9rem"},
            ),
            html.Label("Global Capacity", style={"fontSize": "0.90rem", "fontWeight": 600, "color": "#23344f", "display": "block", "marginBottom": "0.25rem"}),
            dcc.Slider(
                id="capacity-slider",
                min=MIN_CAPACITY,
                max=MAX_CAPACITY,
                step=0.01,
                value=DEFAULT_CAPACITY,
                marks={
                    0.5: "0.5",
                    1.0: "1.0",
                    1.5: "1.5",
                    2.0: "2.0",
                    2.5: "2.5",
                    3.0: "3.0",
                },
                tooltip={"always_visible": False},
            ),
            html.Div(style={"height": "0.7rem"}),
            dcc.Checklist(
                id="training-mode",
                options=[{"label": "Training Mode (capacity grows with balanced usage)", "value": "on"}],
                value=[],
                style={"fontSize": "0.88rem", "color": "#2d3f59"},
                labelStyle={"display": "block"},
            ),
            html.Button(
                "Reset",
                id="reset-button",
                n_clicks=0,
                style={
                    "marginTop": "0.7rem",
                    "background": "#2f6fed",
                    "border": "1px solid #2a5fcb",
                    "color": "#ffffff",
                    "padding": "0.45rem 0.9rem",
                    "cursor": "pointer",
                    "borderRadius": "6px",
                },
            ),
            html.Hr(style={"borderColor": "#d6deea", "margin": "0.9rem 0"}),
            html.H4("Radial Magnitudes rᵢ", style={"margin": "0 0 0.55rem 0"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                virtue,
                                style={
                                    "fontSize": "0.86rem",
                                    "fontWeight": 600,
                                    "color": VIRTUE_COLORS[virtue],
                                    "marginBottom": "0.18rem",
                                },
                            ),
                            dcc.Slider(
                                id={"type": "r-slider", "virtue": virtue},
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=DEFAULT_R,
                                marks={0: "0", 0.5: "0.5", 1.0: "1"},
                                tooltip={"always_visible": False},
                            ),
                        ],
                        style={"marginBottom": "0.52rem"},
                    )
                    for virtue in VIRTUES
                ]
            ),
            html.Div(id="z-section", children=[
                html.H4("Meta Axis zᵢ", style={"margin": "0.85rem 0 0.55rem 0"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    virtue,
                                    style={
                                        "fontSize": "0.86rem",
                                        "fontWeight": 600,
                                        "color": VIRTUE_COLORS[virtue],
                                        "marginBottom": "0.18rem",
                                    },
                                ),
                                dcc.Slider(
                                    id={"type": "z-slider", "virtue": virtue},
                                    min=-1.0,
                                    max=1.0,
                                    step=0.01,
                                    value=DEFAULT_Z,
                                    marks={-1.0: "-1", 0: "0", 1.0: "+1"},
                                    tooltip={"always_visible": False},
                                ),
                            ],
                            style={"marginBottom": "0.52rem"},
                        )
                        for virtue in VIRTUES
                    ]
                ),
                html.Div(
                    "+Z: Leading Up, -Z: Leading Down, X-Y: Leading Across",
                    style={"fontSize": "0.82rem", "color": "#4c6281", "marginTop": "0.35rem"},
                ),
            ]),
        ],
        style={
            "width": "360px",
            "minWidth": "330px",
            "maxHeight": "95vh",
            "overflowY": "auto",
            "padding": "1rem 1rem 1rem 1rem",
            "background": "#f7f9fc",
            "borderRight": "1px solid #dde4ef",
        },
    )


app = Dash(__name__, title="Character Leadership Elastic Constraint System")
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id="last-capacity", data=DEFAULT_CAPACITY),
        dcc.Interval(id="training-interval", interval=1800, n_intervals=0, disabled=True),
        html.Div(
            [
                controls_panel(),
                html.Div(
                    [
                        html.Div(
                            "Tip: In 3D, click the large ▲ or ▼ handles to move each virtue up/down. Blue fill is effective (after compression); orange is requested.",
                            style={
                                "padding": "0.5rem 0.7rem 0.2rem 0.7rem",
                                "fontSize": "0.85rem",
                                "color": "#38506c",
                            },
                        ),
                        dcc.Graph(
                            id="leadership-graph",
                            animate=False,
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "responsive": True,
                            },
                            style={"height": "77vh", "width": "100%"},
                        ),
                        html.Div(id="capacity-status", style={"padding": "0.35rem 0.6rem 0.4rem 0.6rem"}),
                        html.Div(id="virtue-table", style={"padding": "0.2rem 0.6rem 0.9rem 0.6rem"}),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                ),
            ],
            style={"display": "flex", "height": "100vh", "width": "100vw", "background": "#ffffff", "color": "#1b2a40"},
        ),
    ]
)


@app.callback(Output("z-section", "style"), Input("mode-toggle", "value"))
def toggle_z_section(mode: str) -> dict[str, str]:
    if mode == "3d":
        return {"display": "block"}
    return {"display": "none"}


@app.callback(Output("training-interval", "disabled"), Input("training-mode", "value"))
def toggle_training_interval(training_mode: list[str]) -> bool:
    return "on" not in (training_mode or [])


@app.callback(
    Output({"type": "r-slider", "virtue": ALL}, "value"),
    Output({"type": "z-slider", "virtue": ALL}, "value"),
    Output("capacity-slider", "value", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_controls(n_clicks: int):
    if not n_clicks:
        raise PreventUpdate
    return [DEFAULT_R for _ in VIRTUES], [DEFAULT_Z for _ in VIRTUES], DEFAULT_CAPACITY


@app.callback(
    Output({"type": "r-slider", "virtue": ALL}, "value", allow_duplicate=True),
    Output({"type": "z-slider", "virtue": ALL}, "value", allow_duplicate=True),
    Input("leadership-graph", "clickData"),
    State("mode-toggle", "value"),
    State({"type": "r-slider", "virtue": ALL}, "value"),
    State({"type": "z-slider", "virtue": ALL}, "value"),
    prevent_initial_call=True,
)
def move_with_graph_handles(
    click_data: dict | None,
    mode: str,
    r_values: list[float],
    z_values: list[float],
):
    if not click_data or "points" not in click_data or not click_data["points"]:
        raise PreventUpdate

    point = click_data["points"][0]
    custom = point.get("customdata")
    if not custom or len(custom) != 2:
        raise PreventUpdate

    virtue = str(custom[0])
    action = str(custom[1])
    if virtue not in VIRTUES:
        raise PreventUpdate

    idx = VIRTUES.index(virtue)
    new_r = list(r_values)
    new_z = list(z_values)

    if action == "r_plus":
        new_r[idx] = round(min(1.0, new_r[idx] + HANDLE_R_STEP), 3)
    elif action == "r_minus":
        new_r[idx] = round(max(0.0, new_r[idx] - HANDLE_R_STEP), 3)
    elif action == "z_plus" and mode == "3d":
        new_z[idx] = round(min(1.0, new_z[idx] + HANDLE_Z_STEP), 3)
    elif action == "z_minus" and mode == "3d":
        new_z[idx] = round(max(-1.0, new_z[idx] - HANDLE_Z_STEP), 3)
    else:
        raise PreventUpdate

    return new_r, new_z


@app.callback(
    Output("capacity-slider", "value", allow_duplicate=True),
    Input("training-interval", "n_intervals"),
    State("training-mode", "value"),
    State({"type": "r-slider", "virtue": ALL}, "value"),
    State({"type": "z-slider", "virtue": ALL}, "value"),
    State("mode-toggle", "value"),
    State("capacity-slider", "value"),
    prevent_initial_call=True,
)
def grow_capacity_with_training(
    n_intervals: int,
    training_mode: list[str],
    r_values: list[float],
    z_values: list[float],
    mode: str,
    current_capacity: float,
):
    if n_intervals is None or "on" not in (training_mode or []):
        return no_update

    state = enforce_capacity(r_values, z_values, current_capacity, mode)
    mean_mag, cv = balanced_usage_score(state)

    if mean_mag >= TRAINING_MIN_MEAN and cv <= TRAINING_BALANCE_THRESHOLD:
        new_capacity = min(MAX_CAPACITY, float(current_capacity) + TRAINING_DELTA)
        return round(new_capacity, 3)

    return no_update


@app.callback(
    Output("leadership-graph", "figure"),
    Output("capacity-status", "children"),
    Output("virtue-table", "children"),
    Input({"type": "r-slider", "virtue": ALL}, "value"),
    Input({"type": "z-slider", "virtue": ALL}, "value"),
    Input("capacity-slider", "value"),
    Input("mode-toggle", "value"),
)
def update_system(
    r_values: list[float],
    z_values: list[float],
    capacity: float,
    mode: str,
):
    if not r_values or not z_values:
        raise PreventUpdate

    state = enforce_capacity(r_values, z_values, capacity, mode)
    fig = build_figure(state)

    mean_mag, cv = balanced_usage_score(state)
    compressed = state.scale < 0.999999
    compression_pct = (1.0 - state.scale) * 100.0

    status = html.Div(
        [
            html.Div(
                [
                    html.Span("Constraint Status: ", style={"fontWeight": 700}),
                    html.Span(
                        "Compressed" if compressed else "Within Capacity",
                        style={
                            "color": "#d97706" if compressed else "#15803d",
                            "fontWeight": 700,
                        },
                    ),
                ],
                style={"fontSize": "1rem", "marginBottom": "0.24rem"},
            ),
            html.Div(
                f"Requested Norm: {state.requested_norm:.3f} | Effective Norm: {state.effective_norm:.3f} | Capacity: {state.capacity:.3f}",
                style={"color": "#2f4764", "fontSize": "0.88rem", "marginBottom": "0.16rem"},
            ),
            html.Div(
                f"Global Scale Factor: {state.scale:.4f} ({compression_pct:.1f}% compression)",
                style={"color": "#4b6280", "fontSize": "0.84rem"},
            ),
            html.Div(
                f"Balanced Usage Signal — mean magnitude: {mean_mag:.3f}, CV: {cv:.3f}",
                style={"color": "#4b6280", "fontSize": "0.82rem", "marginTop": "0.24rem"},
            ),
            html.Div(
                "Graph controls: click white/outlined diamonds to contract/expand radial value. "
                + ("In 3D mode, click ▲ / ▼ handles to move up/down." if mode == "3d" else ""),
                style={"color": "#334f70", "fontSize": "0.82rem", "marginTop": "0.24rem"},
            ),
        ],
        style={
            "border": "1px solid #d5deea",
            "background": "#f8fbff",
            "padding": "0.6rem 0.7rem",
            "borderRadius": "8px",
        },
    )

    rows = []
    for i, virtue in enumerate(VIRTUES):
        req_mag = (
            state.requested_r[i]
            if mode == "2d"
            else math.sqrt(state.requested_r[i] ** 2 + state.requested_z[i] ** 2)
        )
        eff_mag = (
            state.effective_r[i]
            if mode == "2d"
            else math.sqrt(state.effective_r[i] ** 2 + state.effective_z[i] ** 2)
        )
        rows.append(
            html.Tr(
                [
                    html.Td(virtue, style={"color": VIRTUE_COLORS[virtue], "fontWeight": 600}),
                    html.Td(f"{state.requested_r[i]:.3f}"),
                    html.Td(f"{state.requested_z[i]:.3f}" if mode == "3d" else "0.000"),
                    html.Td(f"{req_mag:.3f}"),
                    html.Td(f"{state.effective_r[i]:.3f}"),
                    html.Td(f"{state.effective_z[i]:.3f}" if mode == "3d" else "0.000"),
                    html.Td(f"{eff_mag:.3f}"),
                ]
            )
        )

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Virtue"),
                        html.Th("r req"),
                        html.Th("z req"),
                        html.Th("|v| req"),
                        html.Th("r eff"),
                        html.Th("z eff"),
                        html.Th("|v| eff"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "0.82rem",
            "background": "#ffffff",
            "border": "1px solid #d5deea",
            "borderRadius": "8px",
            "overflow": "hidden",
        },
    )

    wrapper = html.Div(
        [
            html.Div("Virtue Values (requested vs effective)", style={"marginBottom": "0.35rem", "fontWeight": 600}),
            table,
        ]
    )

    return fig, status, wrapper


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
