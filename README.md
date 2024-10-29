# Quadruple Inverted Pendulum Controller

This project implements a controller for a **Quadruple Inverted Pendulum (QIP)** using **Q-Learning** and **Linear Quadratic Regulator (LQR)** methods.

## Instructions for Training Your Own Q-Table

**Set Training Mode**: In your main script, ensure the `training` parameter is set to `True` when initializing the `Controller`:

   ```python
   your_controller = Controller(
       pretrained_q_table_path=None,  # Start with a fresh Q-table
       training=True
   )
   ```
## Reward Function
Feel free to modify the reward function as desired.


## Future Work
For better scalability and performance, consider replacing the Q-Table with a neural network to approximate the Q-values. 
This approach handles continuous state spaces more efficiently and mitigates the "curse of dimensionality."

## Acknowledgement
