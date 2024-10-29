# Quadruple Inverted Pendulum Controller

[Porject PDF file]([https://doi.org/10.1016/j.engappai.2023.107518](https://drive.google.com/file/d/1jL_oVYQrVnmIAMw-khLq6Ku9Vv8g0Lue/view?usp=sharing)):

This project implements a controller for a **Quadruple Inverted Pendulum (QIP)** using **Q-Learning** and **Linear Quadratic Regulator (LQR)** methods.


## System Requirements
   ```
python == 3.9
matplotlib == 3.9.2
numpy == 2.0.2
   ```


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
- [N-Link_Pendulum Equations](https://blog.wolfram.com/2011/03/01/stabilized-n-link-pendulum/): A MATHEMATICA developed approach to derive equations of motion of n-link pendulum.
- [LQR_Inverted_Pendulum](https://ieeexplore.ieee.org/document/6508052): Stabilizing three link inverted Pendulum.
- [Mathwork's reinforcement_Learning](https://github.com/mathworks/Reinforcement-Learning-Inverted-Pendulum-with-QUBE-Servo2/tree/master): Mathworks Implementation
- [Reinforcement_Learning](https://zenodo.org/records/6582706): Reinforcement Learning for swing up
- [3-link Pendulum with reinforcement learning](https://doi.org/10.1016/j.engappai.2023.107518): Reinforcement learning for three link inverted Pendulum.

Finally Great thanks to the [KOH AI Competition](https://2024.iccas.org/?page_id=4431) Organizers for the opportunity to carry out this project
