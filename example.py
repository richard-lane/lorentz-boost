import matplotlib.pyplot as plt

from lorentz_boost import boosts
from lorentz_boost import utils
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


# Generate non boosted particle with phasespace
# Generate boosted particle with phasespace
# Boost with my thing
# Plot z angle

assert False

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
jpsi_mass_gev = 3096.900
jpsi_px = px1 + px2
jpsi_py = py1 + py2
jpsi_pz = pz1 + pz2
jpsi_energy = np.sqrt(jpsi_mass_gev ** 2 + jpsi_px ** 2 + jpsi_py ** 2 + jpsi_pz ** 2)

kw = {"bins": np.linspace(-200, 200), "histtype": "step"}
plt.hist(jpsi_px, **kw)
plt.hist(jpsi_py, **kw)
plt.hist(jpsi_pz, **kw)
# plt.hist(jpsi_energy, **kw)
plt.show()

angles1 = z_angle(px1, py1, pz1)
angles2 = z_angle(px2, py2, pz2)

kw = {"bins": 100, "histtype": "step"}
plt.hist(angles1, **kw)
plt.hist(angles2, **kw)

# Boost
particles1 = np.row_stack((px1, py1, pz1, energy1))
particles2 = np.row_stack((px2, py2, pz2, energy2))
targets = np.row_stack((jpsi_px, jpsi_py, jpsi_pz, jpsi_energy))
print(utils.gamma(jpsi_mass_gev * np.ones(len(jpsi_px)), jpsi_energy))

boosted1 = boosts.boosts(particles1, targets)
boosted2 = boosts.boosts(particles2, targets)

angles1 = z_angle(*boosted1[:3])
angles2 = z_angle(*boosted2[:3])
plt.hist(angles1, **kw)
plt.hist(angles2, **kw)
plt.show()

# Plot before
# Plot after
