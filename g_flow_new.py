import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------
# 1. TORUS PARAMETERS & SURFACE
# ---------------------------------------------------------
R = 3.5  # Major radius (distance from center of hole to tube)
r = 1.8  # Minor radius (radius of the tube itself)
N = 100  # Resolution

theta_grid = np.linspace(-np.pi, np.pi, N)
phi_grid = np.linspace(-np.pi, np.pi, N)
Theta, Phi = np.meshgrid(theta_grid, phi_grid)

X_torus = (R + r * np.cos(Phi)) * np.cos(Theta)
Y_torus = (R + r * np.cos(Phi)) * np.sin(Theta)
Z_torus = r * np.sin(Phi)

# Create the figure with the Torus surface
fig = go.Figure(data=[go.Surface(
    x=X_torus, 
    y=Y_torus, 
    z=Z_torus,
    colorscale='Ice',
    opacity=0.2,       
    showscale=False
)])

# ---------------------------------------------------------
# 2. GEODESIC ORBIT CALCULATIONS
# ---------------------------------------------------------
def calculate_orbit(x0, y0, theta0, t_max=15, num_points=500):
    # Symmetrical time window from -t_max to +t_max
    t = np.linspace(-t_max, t_max, num_points)
    
    # Calculate Z_t (Position in H) using complex numbers (1j is the imaginary unit i)
    num_Z = (y0*np.cos(theta0) + x0*np.sin(theta0))*np.exp(t/2)*1j + (-y0*np.sin(theta0) + x0*np.cos(theta0))*np.exp(-t/2)
    den_Z = np.sin(theta0)*np.exp(t/2)*1j + np.cos(theta0)*np.exp(-t/2)
    Z_t = num_Z / den_Z
    
    # Calculate V_t (Velocity vector in H)
    V_t = (y0 * 1j) / (den_Z**2)
    
    # Map to the Unit Disk D (Cayley Transform)
    z_D = (Z_t - 1j) / (Z_t + 1j)
    v_D = (2j / (Z_t + 1j)**2) * V_t
    
    # Extract Coordinates for the 3D Torus
    rho = np.abs(z_D)
    alpha = np.angle(z_D)
    psi = np.angle(v_D)
    
    # Map to 3D Cartesian coordinates
    X_orbit = (R + r * rho * np.cos(alpha)) * np.cos(psi)
    Y_orbit = (R + r * rho * np.cos(alpha)) * np.sin(psi)
    Z_orbit = r * rho * np.sin(alpha)
    
    return X_orbit, Y_orbit, Z_orbit

# ---------------------------------------------------------
# 3. PLOT MULTIPLE ORBITS
# ---------------------------------------------------------
# Format: (x0, y0, theta0)
initial_states = [
    (0.0, 1.0, np.pi/4),   # Starts at i (maps to origin of D) -> Straight diameter
    (1.0, 2.0, np.pi/4),   # Off-center -> Curved arc
    (-1.0, 0.5, 0),        # Off-center -> Curved arc
    (2.0, 1.0, np.pi/2),   # Off-center -> Curved arc
    (0.0, 0.2, np.pi/3)    # Close to the real axis -> Deeply curved arc
]

colors = ['#FF3366', '#33CCFF', '#FFCC00', '#33FF66', '#CC33FF']

for i, (x0, y0, theta0) in enumerate(initial_states):
    X_orb, Y_orb, Z_orb = calculate_orbit(x0, y0, theta0, t_max=7, num_points=1000)
    
    # Add each trajectory to the Plotly figure
    fig.add_trace(go.Scatter3d(
        x=X_orb, 
        y=Y_orb, 
        z=Z_orb,
        mode='lines',
        name=f'z₀={x0}+{y0}i, θ₀={theta0:.2f}',
        line=dict(color=colors[i], width=8)
    ))

# ---------------------------------------------------------
# 4. FINAL LAYOUT & RENDERING
# ---------------------------------------------------------
fig.update_layout(
    title='Geodesic Flow Orbits in T¹D (Unit Tangent Bundle)',
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False), 
        aspectmode='data'  
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.8, y=0.9)
)

fig.show()