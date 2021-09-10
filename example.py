import matplotlib.pyplot as plt

# from lorentz_boost import boosts
import pandas as pd
import numpy as np


def angle(x1, y1, z1, x2, y2, z2):
    r1 = np.column_stack((x1, y1, z1))
    r2 = np.column_stack((x2, y2, z2))

    norm1 = np.linalg.norm(r1, axis=1)
    norm2 = np.linalg.norm(r2, axis=1)

    dot = np.sum(r1 * r2, axis=1)
    dot /= norm1 * norm2

    return np.arccos(dot)


def z_angle(x, y, z):
    norm = np.sqrt(x * x + y * y + z * z)
    return np.arccos(z / norm)


# Download some open data from CMS
dataframe = pd.read_csv("https://opendata.cern.ch/record/5203/files/Jpsimumu.csv")

# Find x,y,z, energy of muons
px1 = dataframe["px1"].to_numpy()
py1 = dataframe["py1"].to_numpy()
pz1 = dataframe["pz1"].to_numpy()
energy1 = dataframe["px1"].to_numpy()

px2 = dataframe["px2"].to_numpy()
py2 = dataframe["py2"].to_numpy()
pz2 = dataframe["pz2"].to_numpy()
energy2 = dataframe["px2"].to_numpy()

# Find centre of mass 4-vectors, assuming they came from a J/psi
jspi_mass_gev = 3096.900
jpsi_px = px1 + px2
jpsi_py = py1 + py2
jpsi_pz = pz1 + pz2

angles1 = z_angle(px1, py1, pz1)
angles2 = z_angle(px2, py2, pz2)
plt.hist(angles1, bins=100)
plt.hist(angles2, bins=100)
plt.show()

# Plot before
# Plot after
