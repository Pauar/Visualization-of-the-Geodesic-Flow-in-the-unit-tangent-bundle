import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------
# 1. TORUS PARAMETERS & SURFACE
# ---------------------------------------------------------
R = 3.5  
r = 1.8  
N = 100  

theta_grid = np.linspace(-np.pi, np.pi, N)
phi_grid = np.linspace(-np.pi, np.pi, N)
Theta, Phi = np.meshgrid(theta_grid, phi_grid)

X_torus = (R + r * np.cos(Phi)) * np.cos(Theta)
Y_torus = (R + r * np.cos(Phi)) * np.sin(Theta)
Z_torus = r * np.sin(Phi)

fig = go.Figure(data=[go.Surface(
    x=X_torus, y=Y_torus, z=Z_torus,
    colorscale='Ice', opacity=0.15, showscale=False
)])

# ---------------------------------------------------------
# 2. HELPER TO MAP H -> D -> TORUS
# ---------------------------------------------------------
def map_to_torus(Z, V):
    """Takes complex arrays Z (position) and V (velocity) in H, maps to 3D Torus."""
    # Transform to D
    z_D = (Z - 1j) / (Z + 1j)
    v_D = (2j / (Z + 1j)**2) * V
    
    # Map to Torus coords
    rho = np.abs(z_D)
    alpha = np.angle(z_D)
    psi = np.angle(v_D)
    
    X = (R + r * rho * np.cos(alpha)) * np.cos(psi)
    Y = (R + r * rho * np.cos(alpha)) * np.sin(psi)
    Z_3d = r * rho * np.sin(alpha)
    return X, Y, Z_3d

# ---------------------------------------------------------
# 3. CALCULATE THE CURVES
# ---------------------------------------------------------
# A. The Reference Geodesic (varying t, s=0)
t = np.linspace(-6, 6, 1000)
Z_geo = 1j * np.exp(t)
V_geo = 1j * np.exp(t)
X_geo, Y_geo, Z_geo_3d = map_to_torus(Z_geo, V_geo)

# B. The Stable Manifold U- (varying s, t=0)
s_minus = np.linspace(-100, 100, 1000)
Z_stable = 1j + s_minus
V_stable = 1j * np.ones_like(s_minus)
X_stable, Y_stable, Z_stable_3d = map_to_torus(Z_stable, V_stable)

# C. The Unstable Manifold U+ (varying s, t=0)
s_plus = np.linspace(-100, 100, 1000)
den = (s_plus * 1j + 1)
Z_unstable = 1j / den
V_unstable = 1j / (den**2)
X_unstable, Y_unstable, Z_unstable_3d = map_to_torus(Z_unstable, V_unstable)

# ---------------------------------------------------------
# 4. PLOT THE CURVES
# ---------------------------------------------------------
fig.add_trace(go.Scatter3d(
    x=X_geo, y=Y_geo, z=Z_geo_3d,
    mode='lines', name='Reference Geodesic',
    line=dict(color='black', width=8) # Changed from white to black for visibility
))

fig.add_trace(go.Scatter3d(
    x=X_stable, y=Y_stable, z=Z_stable_3d,
    mode='lines', name='Stable Manifold (U-)',
    line=dict(color='#00CCFF', width=8) # Cyan/Blue
))

fig.add_trace(go.Scatter3d(
    x=X_unstable, y=Y_unstable, z=Z_unstable_3d,
    mode='lines', name='Unstable Manifold (U+)',
    line=dict(color='#FF3366', width=8) # Pink/Red
))

# ---------------------------------------------------------
# 5. FINAL LAYOUT
# ---------------------------------------------------------
fig.update_layout(
    title=dict(text='Geodesic Flow vs. Stable & Unstable Manifolds in T¹D', font=dict(color='black')),
    scene=dict(
        xaxis=dict(title='', range=[-6, 6], showbackground=False, showticklabels=False),
        yaxis=dict(title='', range=[-6, 6], showbackground=False, showticklabels=False),
        zaxis=dict(title='', range=[-6, 6], showbackground=False, showticklabels=False),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.75, y=0.9, font=dict(color='black')), # Text changed to black
    paper_bgcolor='white', # Background changed to white
)

fig.show()