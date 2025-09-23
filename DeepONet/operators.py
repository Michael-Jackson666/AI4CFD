"""
Operator definitions and data generation for DeepONet training.
Common operators for learning mappings between function spaces.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import odeint
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class AntiderivativeOperator:
    """
    Antiderivative operator: maps f(x) to ∫f(x)dx
    """
    
    def __init__(self, domain=(-1, 1), n_sensors=50, n_queries=100):
        self.domain = domain
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        # Fixed sensor and query locations
        self.sensor_locations = np.linspace(domain[0], domain[1], n_sensors)
        self.query_locations = np.linspace(domain[0], domain[1], n_queries)
    
    def generate_data(self, n_samples=1000, max_freq=5):
        """
        Generate training data for antiderivative operator.
        
        Returns:
            branch_data: Function values at sensor locations [n_samples, n_sensors]
            trunk_data: Query locations [n_samples, n_queries, 1]
            output_data: Antiderivatives at query locations [n_samples, n_queries, 1]
        """
        branch_data = []
        output_data = []
        
        for _ in range(n_samples):
            # Generate random function as sum of sinusoids
            n_modes = np.random.randint(1, max_freq + 1)
            coeffs = np.random.normal(0, 1, n_modes)
            freqs = np.random.uniform(1, max_freq, n_modes)
            phases = np.random.uniform(0, 2*np.pi, n_modes)
            
            # Function values at sensor locations
            f_sensors = np.zeros(self.n_sensors)
            for i in range(n_modes):
                f_sensors += coeffs[i] * np.sin(freqs[i] * self.sensor_locations + phases[i])
            
            # Compute antiderivative at query locations
            antiderivative = np.zeros(self.n_queries)
            for i in range(n_modes):
                antiderivative += -coeffs[i] / freqs[i] * np.cos(freqs[i] * self.query_locations + phases[i])
            
            # Normalize antiderivative (set integration constant to make it zero at left boundary)
            antiderivative = antiderivative - antiderivative[0]
            
            branch_data.append(f_sensors)
            output_data.append(antiderivative)
        
        # Convert to numpy arrays
        branch_data = np.array(branch_data)
        trunk_data = np.tile(self.query_locations.reshape(1, -1, 1), (n_samples, 1, 1))
        output_data = np.array(output_data).reshape(n_samples, self.n_queries, 1)
        
        return branch_data, trunk_data, output_data


class HeatEquationOperator:
    """
    Heat equation operator: maps initial condition u0(x) to solution u(x,t)
    ∂u/∂t = α ∂²u/∂x²
    """
    
    def __init__(self, domain=(0, 1), time_domain=(0, 1), alpha=0.1, 
                 n_sensors=50, n_queries=100):
        self.domain = domain
        self.time_domain = time_domain
        self.alpha = alpha
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        # Sensor locations (spatial)
        self.sensor_locations = np.linspace(domain[0], domain[1], n_sensors)
        
        # Query locations (spatial + temporal)
        x_query = np.linspace(domain[0], domain[1], int(np.sqrt(n_queries)))
        t_query = np.linspace(time_domain[0], time_domain[1], int(np.sqrt(n_queries)))
        X_query, T_query = np.meshgrid(x_query, t_query)
        self.query_locations = np.stack([X_query.ravel(), T_query.ravel()], axis=1)
        self.n_queries = len(self.query_locations)
    
    def solve_heat_equation(self, u0, t_eval, n_spatial=200):
        """
        Solve heat equation using finite differences.
        """
        x = np.linspace(self.domain[0], self.domain[1], n_spatial)
        dx = x[1] - x[0]
        
        # Interpolate initial condition to grid
        u_init = np.interp(x, self.sensor_locations, u0)
        
        # Set up finite difference matrix (with homogeneous Dirichlet BC)
        diagonals = [np.ones(n_spatial-2), -2*np.ones(n_spatial-2), np.ones(n_spatial-2)]
        A = diags(diagonals, [-1, 0, 1], shape=(n_spatial-2, n_spatial-2)) / dx**2
        A = self.alpha * A.toarray()
        
        def heat_rhs(u_interior, t):
            return A @ u_interior
        
        # Solve ODE (interior points only, assuming homogeneous Dirichlet BC)
        u_interior_init = u_init[1:-1]
        u_interior_t = odeint(heat_rhs, u_interior_init, t_eval)
        
        # Add boundary conditions
        u_full = np.zeros((len(t_eval), n_spatial))
        u_full[:, 1:-1] = u_interior_t
        
        return x, u_full
    
    def generate_data(self, n_samples=1000, max_modes=5):
        """
        Generate training data for heat equation operator.
        """
        branch_data = []
        output_data = []
        
        for _ in range(n_samples):
            # Generate random initial condition
            n_modes = np.random.randint(1, max_modes + 1)
            coeffs = np.random.normal(0, 1, n_modes)
            
            # Initial condition as sum of sines (satisfies homogeneous BC)
            u0 = np.zeros(self.n_sensors)
            for i in range(n_modes):
                k = i + 1  # Mode number
                u0 += coeffs[i] * np.sin(k * np.pi * self.sensor_locations)
            
            # Solve heat equation
            t_unique = np.unique(self.query_locations[:, 1])
            x_grid, u_solution = self.solve_heat_equation(u0, t_unique)
            
            # Interpolate solution at query points
            u_query = np.zeros(self.n_queries)
            for j, (x_q, t_q) in enumerate(self.query_locations):
                # Find time index
                t_idx = np.argmin(np.abs(t_unique - t_q))
                # Interpolate in space
                u_query[j] = np.interp(x_q, x_grid, u_solution[t_idx])
            
            branch_data.append(u0)
            output_data.append(u_query)
        
        branch_data = np.array(branch_data)
        trunk_data = np.tile(self.query_locations.reshape(1, -1, 2), (n_samples, 1, 1))
        output_data = np.array(output_data).reshape(n_samples, self.n_queries, 1)
        
        return branch_data, trunk_data, output_data


class DarcyFlowOperator:
    """
    Darcy flow operator: maps permeability field κ(x,y) to pressure field p(x,y)
    -∇·(κ∇p) = f
    """
    
    def __init__(self, domain=((0, 1), (0, 1)), n_sensors=50, n_queries=100):
        self.domain = domain
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        # Create sensor grid (permeability field input)
        n_side = int(np.sqrt(n_sensors))
        self.n_side = n_side
        x_sensors = np.linspace(domain[0][0], domain[0][1], n_side)
        y_sensors = np.linspace(domain[1][0], domain[1][1], n_side)
        X_sensors, Y_sensors = np.meshgrid(x_sensors, y_sensors)
        self.sensor_grid = np.stack([X_sensors.ravel(), Y_sensors.ravel()], axis=1)
        
        # Query locations (pressure field output)
        n_query_side = int(np.sqrt(n_queries))
        x_query = np.linspace(domain[0][0], domain[0][1], n_query_side)
        y_query = np.linspace(domain[1][0], domain[1][1], n_query_side)
        X_query, Y_query = np.meshgrid(x_query, y_query)
        self.query_locations = np.stack([X_query.ravel(), Y_query.ravel()], axis=1)
        self.n_queries = len(self.query_locations)
    
    def solve_darcy_flow(self, kappa_field, source_strength=1.0):
        """
        Solve Darcy flow equation using finite differences.
        Simplified version with constant source term.
        """
        n = self.n_side
        h = 1.0 / (n - 1)  # Grid spacing
        
        # Reshape permeability field to 2D grid
        kappa_grid = kappa_field.reshape(n, n)
        
        # Set up finite difference system (simplified)
        # For full implementation, would need proper finite volume discretization
        
        # Create a simple Laplacian with variable coefficients (approximate)
        A = np.zeros((n*n, n*n))
        b = np.zeros(n*n)
        
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                
                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    # Boundary conditions (homogeneous Dirichlet)
                    A[idx, idx] = 1.0
                    b[idx] = 0.0
                else:
                    # Interior points: -∇·(κ∇p) = f
                    kappa_center = kappa_grid[i, j]
                    
                    # Approximate -∇·(κ∇p) with finite differences
                    A[idx, idx] = -4 * kappa_center / h**2
                    A[idx, idx-1] = kappa_center / h**2      # Left
                    A[idx, idx+1] = kappa_center / h**2      # Right
                    A[idx, idx-n] = kappa_center / h**2      # Down
                    A[idx, idx+n] = kappa_center / h**2      # Up
                    
                    b[idx] = -source_strength
        
        # Solve system
        try:
            p_solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            p_solution = np.linalg.lstsq(A, b, rcond=None)[0]
        
        return p_solution.reshape(n, n)
    
    def generate_data(self, n_samples=1000, kappa_range=(0.1, 2.0)):
        """
        Generate training data for Darcy flow operator.
        """
        branch_data = []
        output_data = []
        
        for _ in range(n_samples):
            # Generate random permeability field
            # Use log-normal distribution for physical realism
            log_kappa = np.random.normal(0, 0.5, self.n_sensors)
            kappa = np.exp(log_kappa)
            
            # Normalize to desired range
            kappa = kappa_range[0] + (kappa_range[1] - kappa_range[0]) * \
                   (kappa - kappa.min()) / (kappa.max() - kappa.min())
            
            # Solve Darcy flow
            pressure_grid = self.solve_darcy_flow(kappa)
            
            # Interpolate pressure at query points
            from scipy.interpolate import RegularGridInterpolator
            x_grid = np.linspace(self.domain[0][0], self.domain[0][1], self.n_side)
            y_grid = np.linspace(self.domain[1][0], self.domain[1][1], self.n_side)
            
            interp_func = RegularGridInterpolator((x_grid, y_grid), pressure_grid, 
                                                bounds_error=False, fill_value=0)
            
            pressure_query = interp_func(self.query_locations)
            
            branch_data.append(kappa)
            output_data.append(pressure_query)
        
        branch_data = np.array(branch_data)
        trunk_data = np.tile(self.query_locations.reshape(1, -1, 2), (n_samples, 1, 1))
        output_data = np.array(output_data).reshape(n_samples, self.n_queries, 1)
        
        return branch_data, trunk_data, output_data


class BurgersOperator:
    """
    Burgers' equation operator: maps initial condition to solution at time T
    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    """
    
    def __init__(self, domain=(-1, 1), final_time=1.0, nu=0.01, 
                 n_sensors=50, n_queries=100):
        self.domain = domain
        self.final_time = final_time
        self.nu = nu
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        self.sensor_locations = np.linspace(domain[0], domain[1], n_sensors)
        self.query_locations = np.linspace(domain[0], domain[1], n_queries)
    
    def solve_burgers(self, u0, nt=1000):
        """
        Solve Burgers' equation using finite differences.
        """
        # Spatial grid
        x = np.linspace(self.domain[0], self.domain[1], len(u0))
        dx = x[1] - x[0]
        
        # Time grid
        dt = self.final_time / nt
        
        # Initialize solution
        u = u0.copy()
        
        # Time stepping (simple explicit scheme)
        for _ in range(nt):
            u_new = u.copy()
            
            # Interior points
            for i in range(1, len(u)-1):
                # Burgers' equation discretization
                u_t = -u[i] * (u[i+1] - u[i-1]) / (2*dx) + \
                      self.nu * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
                u_new[i] = u[i] + dt * u_t
            
            # Boundary conditions (keep fixed)
            u_new[0] = u0[0]
            u_new[-1] = u0[-1]
            
            u = u_new
        
        return u
    
    def generate_data(self, n_samples=1000, amplitude_range=(0.5, 2.0)):
        """
        Generate training data for Burgers' operator.
        """
        branch_data = []
        output_data = []
        
        for _ in range(n_samples):
            # Generate smooth random initial condition
            # Use combination of smooth functions
            amplitude = np.random.uniform(*amplitude_range)
            center = np.random.uniform(-0.5, 0.5)
            width = np.random.uniform(0.2, 0.8)
            
            # Gaussian-like initial condition
            u0 = amplitude * np.exp(-(self.sensor_locations - center)**2 / width**2)
            
            # Solve Burgers' equation
            u_final = self.solve_burgers(u0)
            
            # Interpolate to query locations
            u_query = np.interp(self.query_locations, self.sensor_locations, u_final)
            
            branch_data.append(u0)
            output_data.append(u_query)
        
        branch_data = np.array(branch_data)
        trunk_data = np.tile(self.query_locations.reshape(1, -1, 1), (n_samples, 1, 1))
        output_data = np.array(output_data).reshape(n_samples, self.n_queries, 1)
        
        return branch_data, trunk_data, output_data


class AdvectionOperator:
    """
    Advection operator: maps initial condition to solution at time T
    ∂u/∂t + c∂u/∂x = 0
    """
    
    def __init__(self, domain=(0, 1), final_time=1.0, velocity=1.0,
                 n_sensors=50, n_queries=100):
        self.domain = domain
        self.final_time = final_time
        self.velocity = velocity
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        self.sensor_locations = np.linspace(domain[0], domain[1], n_sensors)
        self.query_locations = np.linspace(domain[0], domain[1], n_queries)
    
    def solve_advection(self, u0):
        """
        Solve advection equation analytically.
        Solution: u(x,t) = u0(x - ct)
        """
        # For periodic boundary conditions
        L = self.domain[1] - self.domain[0]
        shift = self.velocity * self.final_time
        
        # Shifted locations (with periodic wrapping)
        shifted_x = self.query_locations - shift
        shifted_x = np.mod(shifted_x - self.domain[0], L) + self.domain[0]
        
        # Interpolate initial condition at shifted locations
        u_final = np.interp(shifted_x, self.sensor_locations, u0, 
                           period=L)
        
        return u_final
    
    def generate_data(self, n_samples=1000, max_modes=5):
        """
        Generate training data for advection operator.
        """
        branch_data = []
        output_data = []
        
        for _ in range(n_samples):
            # Generate periodic initial condition
            n_modes = np.random.randint(1, max_modes + 1)
            coeffs_sin = np.random.normal(0, 1, n_modes)
            coeffs_cos = np.random.normal(0, 1, n_modes)
            
            u0 = np.zeros(self.n_sensors)
            for k in range(1, n_modes + 1):
                u0 += coeffs_sin[k-1] * np.sin(2*np.pi*k*self.sensor_locations) + \
                      coeffs_cos[k-1] * np.cos(2*np.pi*k*self.sensor_locations)
            
            # Solve advection
            u_final = self.solve_advection(u0)
            
            branch_data.append(u0)
            output_data.append(u_final)
        
        branch_data = np.array(branch_data)
        trunk_data = np.tile(self.query_locations.reshape(1, -1, 1), (n_samples, 1, 1))
        output_data = np.array(output_data).reshape(n_samples, self.n_queries, 1)
        
        return branch_data, trunk_data, output_data


def create_operator_data(operator_type, **kwargs):
    """
    Factory function to create operator data.
    
    Args:
        operator_type: Type of operator
        **kwargs: Additional parameters
    
    Returns:
        Operator instance
    """
    operators = {
        'antiderivative': AntiderivativeOperator,
        'heat': HeatEquationOperator,
        'darcy': DarcyFlowOperator,
        'burgers': BurgersOperator,
        'advection': AdvectionOperator
    }
    
    if operator_type not in operators:
        raise ValueError(f"Unknown operator type: {operator_type}. "
                        f"Available: {list(operators.keys())}")
    
    return operators[operator_type](**kwargs)


if __name__ == "__main__":
    # Test operator data generation
    print("Testing operator data generation...")
    
    # Test antiderivative operator
    print("\n1. Antiderivative Operator:")
    antiderivative_op = create_operator_data('antiderivative', n_sensors=50, n_queries=100)
    branch_data, trunk_data, output_data = antiderivative_op.generate_data(n_samples=10)
    print(f"Branch data shape: {branch_data.shape}")
    print(f"Trunk data shape: {trunk_data.shape}")
    print(f"Output data shape: {output_data.shape}")
    
    # Test heat equation operator
    print("\n2. Heat Equation Operator:")
    heat_op = create_operator_data('heat', n_sensors=30, n_queries=64)
    branch_data, trunk_data, output_data = heat_op.generate_data(n_samples=5)
    print(f"Branch data shape: {branch_data.shape}")
    print(f"Trunk data shape: {trunk_data.shape}")
    print(f"Output data shape: {output_data.shape}")
    
    # Test advection operator
    print("\n3. Advection Operator:")
    advection_op = create_operator_data('advection', n_sensors=50, n_queries=50)
    branch_data, trunk_data, output_data = advection_op.generate_data(n_samples=10)
    print(f"Branch data shape: {branch_data.shape}")
    print(f"Trunk data shape: {trunk_data.shape}")
    print(f"Output data shape: {output_data.shape}")
    
    print("\nAll operator tests completed successfully!")