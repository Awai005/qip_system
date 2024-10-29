from fill_the_code import Controller
from render.render import Renderer

def main():
    #####################################################################################
     # Instantiate your controller in fill_the_code.py.
    SEC = 10

    your_controller = Controller(
        state_bins=[5, 5, 5, 3, 5, 3, 5, 3, 5, 3],  # Adjust as needed
        action_size=15,
        epsilon=1.5,
        epsilon_min=0.1,
        alpha=0.1,
        gamma=0.95,
        a_max=60,
        pretrained_q_table_path = 'current_q_table.npy',  # Start training with a fresh Q-table
        training=False
    )                            # Rendering period in seconds (e.g., 10).
    #####################################################################################

    renderer = Renderer(controller=your_controller, sec=SEC)
    transitions = renderer.render()

    # Save the transitions to a txt file.
    with open('transitions.txt', 'w') as f:
        for transition in transitions:
            f.write(transition[0])
            f.write(transition[1])
            f.write(transition[2])
            f.write('\n')


if __name__ == '__main__':
    main()

