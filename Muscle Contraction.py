#Importing everything
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
ACTIN_LENGTH = 10
MYOSIN_HEAD_COUNT = 6
POWER_STROKE_LENGTH = 0.5
CYCLE_LENGTH = 100

# Classes
class MyosinHead:
    def __init__(self, initial_position, power_stroke_length, cycle_length):
        self.initial_position = np.array(initial_position)
        self.power_stroke_length = power_stroke_length
        self.cycle_length = cycle_length
        self.cycle_position = 0
        self.attached = False
        self.atp_bound = False
        self.adp_pi = False

    def update_position(self, time, calcium_concentration):
        phase = (time % self.cycle_length) / self.cycle_length
        if phase < 0.2:
            # Tropomyosin blocks binding site
            self.attached = False
            self.atp_bound = False
            self.adp_pi = False
            self.cycle_position = 0
        elif 0.2 <= phase < 0.4:
            # Calcium ions bind, expose binding site
            self.attached = calcium_concentration > 0.1
            self.cycle_position = 0
        elif 0.4 <= phase < 0.6:
            # Myosin head attaches and performs power stroke
            self.attached = True
            self.cycle_position = (phase - 0.4) * 5 * self.power_stroke_length
        elif 0.6 <= phase < 0.8:
            # ATP binds, myosin head detaches
            self.attached = False
            self.atp_bound = True
            self.cycle_position = self.power_stroke_length
        else:
            # ATP hydrolysis, myosin head re-cocks
            self.attached = False
            self.atp_bound = False
            self.adp_pi = True
            self.cycle_position = (1 - phase) * 5 * self.power_stroke_length

        if self.attached:
            return self.initial_position + np.array([self.cycle_position, 0])
        else:
            return self.initial_position

class ATP:
    def __init__(self, position):
        self.position = np.array(position)
        self.bound = False
    
    def bind(self, myosin_head):
        myosin_head.attached = False
        self.bound = True

class MuscleSimulator:
    def __init__(self, actin_length, myosin_head_count, power_stroke_length, cycle_length):
        self.actin = np.array([[-actin_length/2, 0], [actin_length/2, 0]])
        self.myosin_heads = [MyosinHead([x, -0.5], power_stroke_length, cycle_length) 
                             for x in np.linspace(-actin_length/2, actin_length/2, myosin_head_count)]
        self.calcium_concentration = 0
        self.atps = [ATP([x, 1]) for x in np.linspace(-actin_length/2, actin_length/2, myosin_head_count)]

    # Function which applies the action potential
    def apply_action_potential(self, time):
        if time % 200 < 100:  # Simulate periodic action potential
            self.calcium_concentration = 1
        else:
            self.calcium_concentration = 0

    def update(self, time):
        self.apply_action_potential(time)
        positions = [head.update_position(time, self.calcium_concentration) for head in self.myosin_heads]
        for atp in self.atps:
            for head in self.myosin_heads:
                if np.allclose(head.update_position(time, self.calcium_concentration), head.initial_position):
                    atp.bind(head)
        return positions

# Initialize the muscle simulator
muscle = MuscleSimulator(ACTIN_LENGTH, MYOSIN_HEAD_COUNT, POWER_STROKE_LENGTH, CYCLE_LENGTH)

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-ACTIN_LENGTH/2 - 1, ACTIN_LENGTH/2 + 1)
ax.set_ylim(-2, 2)

actin_line, = ax.plot([], [], 'g-', lw=2)
myosin_heads, = ax.plot([], [], 'ro')
atps, = ax.plot([], [], 'bo')  # ATP molecules are blue
connections = []

# Labels
actin_label = ax.text(0, 0.2, 'Actin Filament', horizontalalignment='center', verticalalignment='center', color='green')
myosin_labels = [ax.text(0, 0, '', horizontalalignment='center', verticalalignment='center', color='red') for _ in range(MYOSIN_HEAD_COUNT)]
atp_labels = [ax.text(0, 0, '', horizontalalignment='center', verticalalignment='center', color='blue') for _ in range(MYOSIN_HEAD_COUNT)]

def init():
    actin_line.set_data(muscle.actin[:, 0], muscle.actin[:, 1])
    myosin_heads.set_data([], [])
    atps.set_data([], [])
    for label in myosin_labels:
        label.set_text('')
    for label in atp_labels:
        label.set_text('')
    return actin_line, myosin_heads, atps, actin_label, *myosin_labels, *atp_labels

def animate(i):
    positions = muscle.update(i)
    myosin_heads.set_data([pos[0] for pos in positions], [pos[1] for pos in positions])
    atps.set_data([atp.position[0] for atp in muscle.atps], [atp.position[1] for atp in muscle.atps])

    # Update myosin head labels
    for j, pos in enumerate(positions):
        myosin_labels[j].set_position((pos[0], pos[1] - 0.3))
        myosin_labels[j].set_text('Myosin Head')

    # Update ATP labels
    for j, atp in enumerate(muscle.atps):
        if atp.bound:
            atp_labels[j].set_text('ADP + Pi')
        else:
            atp_labels[j].set_text('ATP')
        atp_labels[j].set_position((atp.position[0], atp.position[1] + 0.3))

    # Remove old connections
    for line in connections:
        line.remove()
    connections.clear()

    # Draw connections between myosin heads and actin if attached
    for head in muscle.myosin_heads:
        if head.attached:
            line, = ax.plot([head.initial_position[0], head.initial_position[0] + head.cycle_position], 
                            [head.initial_position[1], 0], 'r-')
            connections.append(line)
        else:
            line, = ax.plot([head.initial_position[0], head.initial_position[0]], 
                            [head.initial_position[1], 0], 'b--')
            connections.append(line)

    # Move ATP molecules to myosin heads and simulate ATP hydrolysis
    for atp, head in zip(muscle.atps, muscle.myosin_heads):
        if not atp.bound:
            atp.position = head.initial_position + np.array([0, 1])
        else:
            atp.position = head.initial_position

    return actin_line, myosin_heads, atps, actin_label, *myosin_labels, *atp_labels, *connections

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=400, interval=50, blit=True)

plt.show()
