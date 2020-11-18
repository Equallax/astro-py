import json
import os
import pathlib
import winsound

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
from progress.bar import IncrementalBar
from scipy.constants import G

output_dir = 'output'
filename = 'stress_testXtreme.mp4'

json_path = 'data.json'

time_in_seconds = 2
fps = 120

precision_factor = 2
frames = fps * time_in_seconds


# Een class word gemaakt. De hemellichamen worden allemaal object van deze class.
class CelestialBody:
    def __init__(self, name, orbiting, radius, mass, pos, momentum, colour=None):
        self.name = name
        self.orbiting = orbiting
        self.radius = radius
        self.mass = float(mass)  # kg
        self.pos = np.asarray(pos, dtype=np.float64)
        self.momentum = np.asarray(momentum, dtype=np.float64)
        self.force = np.zeros(3)
        self.colour = colour

        # Arrays worden gemaakt waar elke simulatie cycle de positie van een object bij word toegevoegd
        self.x_path = np.empty(1, dtype=np.float64)
        self.y_path = np.empty(1, dtype=np.float64)
        self.z_path = np.empty(1, dtype=np.float64)

        # Arrays krijgen de eerste positie al toegevoegd, zodat np.append later correct kan functioneren
        self.x_path[0] = self.pos[0]
        self.y_path[0] = self.pos[1]
        self.z_path[0] = self.pos[2]


class FancyBar(IncrementalBar):
    suffix = '%(percent).1f%% - %(eta)ds remaining. Frame: %(index)d/%(max)d'


bar = FancyBar(f'Creating {filename}', max=frames)

realistic = {
    'real_G': False,
    'real_calculations': False
}

# gebruikt de echte G
if realistic['real_G'] is True:
    def gravity(x, y):

        """
        Functie die ervoor zorgt dat gravitatiekracht tussen hemellichamen kan worden berekent
        Dat gebeurt hier met de echte gravitatieconstante
        """

        r_vec = x.pos - y.pos
        r_mag = np.linalg.norm(r_vec)
        r_hat = r_vec / r_mag
        # Calculate force magnitude.
        force_mag = G * x.mass * y.mass / r_mag ** 2
        # Calculate force vector.
        force_vec = -force_mag * r_hat

        return force_vec
else:
    # gebruikt 1 in plaats van G
    def gravity(x, y):

        """
        Functie die ervoor zorgt dat gravitatiekracht tussen hemellichamen kan worden berekent
        Dat gebeurt hier met het getal 1 in plaats van de echte de echte gravitatieconstante
        """

        r_vec = x.pos - y.pos
        r_mag = np.linalg.norm(r_vec)
        r_hat = r_vec / r_mag
        # Calculate force magnitude.
        force_mag = 1 * x.mass * y.mass / r_mag ** 2
        # Calculate force vector.
        force_vec = -force_mag * r_hat

        return force_vec

# Hemellichamen als objecten van de CelestialBody class initialiseren
# Hemellichamen als objecten van de CelestialBody class initialiseren
# een list maken van de planeten

with open(json_path, 'r', encoding='UTF-8') as json_file:
    json_dict = json.load(json_file)

bodies = [CelestialBody(data['name'], data['orbiting'], data['radius'], data['mass'], data['pos'], data['momentum'],
                        data['colour']) for name, data in json_dict.items()]

# stelt het plot in om 3d te tonen

fig = plt.figure()
fig.set_size_inches(25.6, 14.4)

ax = fig.add_subplot(111, projection='3d')
ax.grid(True, linestyle='-', color='0.75')
ax.view_init(elev=60, azim=-45)


# initialiseert het plot met de positie van de aarde
# scat = plt.scatter(Earth.pos[0], Earth.pos[1], 0)

# berekent de krachten op de planeten op een wijze vergelijkbaar met coach maar dan beter en in 3d
# de simulatie functie
def sim(scatter_plot):
    """
    Deze functie simuleert de beweging van de hemellichamen.
    Dat gebeurt behulp van de simulatie functie die ervoor zorgt dat krachten tussen de hemellichamen worden berekent.
    """

    dt = 0.001 / precision_factor
    t = 0

    def update_sim():
        current_body.force = force_list[i]
        current_body.momentum += current_body.force * dt
        current_body.pos += current_body.momentum / current_body.mass * dt

        # past trails toe om de vorige posities te kunnen zien
        current_body.x_path = np.append(current_body.x_path, current_body.pos[0])
        current_body.y_path = np.append(current_body.y_path, current_body.pos[1])
        current_body.z_path = np.append(current_body.z_path, current_body.pos[2])

    # gebruikt realistische berekeningen waarin alles met alles rekening houdt
    if realistic['real_calculations'] is True:
        force_list = []
        for current_body in bodies:
            temp = bodies.copy()
            temp.pop(temp.index(current_body))
            # past posities van hemellichamen aan op basis van de krachten die op hen werken
            forces = np.zeros(3, dtype=np.float64)
            for affecting_body in temp:
                forces += gravity(current_body, affecting_body)

            force_list.append(forces)

        for i, current_body in enumerate(bodies):
            update_sim()

        t += dt
    # gebruikt versimpelde berekeningen waarin de planeten alleen met de orbiting bodies rekening houden
    else:
        force_list = []
        for current_body in bodies:
            forces = np.zeros(3, dtype=np.float64)

            # past posities van hemellichamen aan op basis van de krachten die op hen werken
            if current_body.orbiting is not None and current_body is not bodies[0]:
                for i in current_body.orbiting:
                    forces += gravity(current_body, bodies[i])

                force_list.append(forces)
            else:
                force_list.append((0, 0, 0))

        for i, current_body in enumerate(bodies):
            if current_body.orbiting is not None and current_body is not bodies[0]:
                update_sim()

        t += dt
    bar.next()

    # stelt de grootte van het assenstelsel in
    plt.cla()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-8, 8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.set_clip_on(False)

    # zet de waardes van het stelsel in het plot
    for i in bodies:
        ax.plot(i.x_path[-100::], i.y_path[-100::], i.z_path[-100::], color=i.colour, zorder=3.9, alpha=0.5)
        if i.colour is not None:
            ax.scatter(i.pos[0], i.pos[1], i.pos[2], s=i.radius, color=i.colour, zorder=2)
            # ax.text(i.pos[0], i.pos[1], i.pos[2], s=i.name, zorder=10.,
            # verticalalignment='center_baseline', horizontalalignment='center', fontsize=8)

    return [scatter_plot]


if not os.path.isdir(os.path.join(pathlib.Path().absolute(), output_dir)):
    os.mkdir(output_dir)

animation = FuncAnimation(fig, func=sim, frames=frames, interval=1)

try:
    animation.save(output_dir + '\\' + filename, writer=FFMpegWriter(fps=fps))
except:
    print("FFMpeg was unavailable. Stopping program")

bar.finish()
winsound.PlaySound('SystemAsterisk', 0)
exit()
