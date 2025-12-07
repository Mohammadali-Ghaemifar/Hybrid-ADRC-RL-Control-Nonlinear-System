# Hybrid ADRC–DDPG Control of a Nonlinear Industrial System

This repository contains the MATLAB/Simulink implementation of a **hybrid ADRC–RL controller** for a **nonlinear industrial valve** with friction and time delay. The RL agent is trained and evaluated in closed loop with an **Active Disturbance Rejection Controller (ADRC)**, where **ADRC and RL act in series** during training and testing.

![Block Diagram](Block Diagram.png)

---

## 🧩 Simulation & Software Environment

- MATLAB & Simulink **R2024a**
- Reinforcement Learning Toolbox
- Custom nonlinear valve model with:
  - Static & dynamic friction
  - Process time delay (`L = 2.5 s`)
- Discrete-time plant:
  - Sampling time: `Ts = 1e-3 s`
  - First-order valve dynamics with gain `K = 3.8163` and time constant `tau = 156.46`
  - Delay implemented as `m = round(L/Ts)` samples
- ESO (Extended State Observer) discretized via simple Euler method with bandwidth `w0 = 0.5 rad/s`

All plant and observer parameters are exported to the base workspace using `exportToBase` and used inside the Simulink model `Train.slx`.

---

## 🧠 Control Architecture

The project implements and compares the following controllers:

- **PID** – classical baseline controller
- **ADRC** – ESO + nonlinear state error feedback
- **Hybrid ADRC–DDPG (proposed)**  
  - The RL agent and ADRC are **connected in series**:
    - The RL agent generates a **normalized pre-control signal**
    - This output is scaled and fed as the **input to the ADRC loop**
    - ADRC (with ESO) performs **disturbance estimation and robust tracking**
  - During training, the **full ADRC loop is active**, so the RL agent learns a policy **on top of ADRC**, not on the bare plant.

This design lets RL focus on improving tracking performance and transient behavior, while ADRC guarantees robustness and disturbance rejection.

---

## 📁 Project Files Overview

This project is intentionally kept minimal and consists of only three files:

### 1. `Main.m`
This is the **main training script** that initializes the plant model, ADRC parameters, RL environment, and DDPG agent.  
It starts the **hybrid ADRC–RL training process**, where:
- The RL agent learns in closed-loop with the ADRC controller.
- Plant dynamics, delay, ESO parameters, and scaling factors are defined here.
- Training results and the final trained agent are saved automatically.

---

### 2. `Train.slx`
This Simulink model implements the **training architecture of the hybrid ADRC–DDPG controller**.
- The **RL agent and ADRC are connected in series** inside this model.
- The RL block generates a normalized pre-control signal.
- This signal is processed by the ADRC loop (ESO + nonlinear control law).
- The complete closed-loop system is used during training.

---

### 3. `Test.slx`
This Simulink model is used **only for performance evaluation and comparison**.
- It compares:
  - **Hybrid ADRC–RL controller**
  - **Classical PID controller**
- The PID controller is **automatically tuned using the built-in Simulink PID Toolbox**.
- Both controllers are tested under identical:
  - Reference signals
  - Disturbances
  - Nonlinear valve dynamics

---

✅ Summary:
- `Main.m` → Training launcher  
- `Train.slx` → Hybrid ADRC–RL training model  
- `Test.slx` → Hybrid ADRC–RL vs tuned PID comparison
