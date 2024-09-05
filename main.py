import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hyp2f1

# Parameters
E_alpha = 3.5  # MeV
T_e_keV = 20   # keV
T_e = T_e_keV / 1000  # Convert keV to MeV
A_alpha = 4    # Atomic mass of alpha particles
A_i = 2.5      # Atomic mass of ions

# Calculate critical energy E_c in MeV
E_c = 59.2 * T_e / A_i**(2/3)

# Normalization constant A
def normalization_constant(E_alpha, E_c):
    integral, _ = quad(lambda E: 1 / (E * (1 + (E_c / E)**(3/2))), 0, E_alpha)
    return 1 / integral

A = normalization_constant(E_alpha, E_c)

# Slowing down PDF f(E)
def slowing_down_pdf(E, E_c, E_alpha, A):
    return A / (E * (1 + (E_c / E)**(3/2))) if E <= E_alpha else 0

# Maxwell-Boltzmann PDF f_MB(E)
def maxwell_boltzmann_pdf(E, T):
    return np.sqrt(4 * E / (np.pi * T**3)) * np.exp(-E / T)

# Calculate the mean energy for the slowing down PDF
def mean_energy_slowing_down(E_c, E_alpha, A):
    def integrand(E):
        return 1 / (1 + (E_c / E)**(3/2))
    
    integral, _ = quad(integrand, 0, E_alpha)
    mean_energy = A * integral
    
    # Hypergeometric function calculation
    x = (E_c / E_alpha)**(3/2)
    hypergeo_part = hyp2f1(-2/3, 1, 1/3, -x)
    
    # Mean energy formula
    mean_energy_formula = A * E_c * (E_alpha / E_c * hypergeo_part - 4 * np.pi / (3 * np.sqrt(3)))
    
    return mean_energy, mean_energy_formula

mean_E, mean_E_formula = mean_energy_slowing_down(E_c, E_alpha, A)

# Determine T for Maxwell-Boltzmann such that its mean energy matches slowing down PDF
T_MB = (2 / 3) * mean_E_formula

# Extended energy range for plotting (up to 4 MeV)
E_values = np.linspace(0.001, 4, 1000)  # Avoid division by zero

# Compute PDFs
f_slowing_down = np.array([slowing_down_pdf(E, E_c, E_alpha, A) for E in E_values])
f_MB = maxwell_boltzmann_pdf(E_values, T_MB)

# Plotting settings for PowerPoint
plt.figure(figsize=(8, 9))  # Aspect ratio 8:9 for the right half of a slide
plt.rcParams.update({'font.size': 18})  # Increase font size for all plot elements

plt.plot(E_values, f_slowing_down, label='Slowing Down')
plt.plot(E_values, f_MB, label='Maxwell-Boltzmann', linestyle='--')
plt.xlabel('Energy [MeV]')
plt.ylabel(r'Probability Density Function [MeV$^{-1}$]')
# plt.title('Slowing Down and Maxwell-Boltzmann Distributions')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels and title
# plt.show()
plt.savefig('main.png', bbox_inches='tight', dpi=300)

# Check if PDFs integrate to 1
integral_slowing_down, _ = quad(lambda E: slowing_down_pdf(E, E_c, E_alpha, A), 0, E_alpha)
integral_MB, _ = quad(lambda E: maxwell_boltzmann_pdf(E, T_MB), 0, np.inf)

print(f"Integral of Slowing Down PDF: {integral_slowing_down:.4f}")
print(f"Integral of Maxwell-Boltzmann PDF: {integral_MB:.4f}")
print(f"Mean energy of Slowing Down PDF (calculated): {mean_E:.4f} MeV")
print(f"Mean energy of Slowing Down PDF (formula): {mean_E_formula:.4f} MeV")
print(f"Ec = {E_c:.4f} MeV")