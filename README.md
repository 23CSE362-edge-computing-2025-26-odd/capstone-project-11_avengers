#  IoT-Fog-Cloud Healthcare Monitoring System

##  Overview
This project proposes a **Cloud-Fog hybrid architecture** for real-time healthcare monitoring. By leveraging **edge computing (fog nodes)**, our system processes critical patient data close to the source, reducing latency and improving reliability. Non-urgent tasks are handled by the cloud for long-term storage and analytics.

---

##  Features
- Real-time patient monitoring (ECG, SpO₂, Heart Rate, BP).
- **Fog Broker** with task scheduling and allocation.
- **WSM (Weighted Sum Method)** for prioritization.
- **MBAR (Modified Best Available Resource)** for allocation.
- Simulation with **iFogSim** for performance evaluation.

---

##  System Architecture
- **IoT Devices** → Collect vitals.
- **Sink Layer** → Gateways & mobile devices.
- **Fog Layer** → Real-time processing with WSM + MBAR.
- **Cloud Layer** → Long-term storage & analytics.

---

##  Algorithms
- **WSM**: Prioritizes tasks based on latency, urgency, and energy.  
- **MBAR**: Allocates tasks to the most suitable fog node dynamically.

---

##  Simulation Plan
- Simulator: **iFogSim (Java)**  
- Workload: **300 healthcare tasks**  
- Scenarios: Random Allocation vs WSM vs WSM+MBAR  
- **Metrics**: Latency, Miss Ratio, Average Delay, Cost per Execution, Resource Utilization  

---

##  Results
- Reduced latency for critical tasks.  
- Lower miss ratio compared to cloud-only systems.  
- Better resource utilization and cost savings.  

---

##  Team Members
- **Nikshith** – Problem Understanding & Introduction  
- **Hemanth** – Architecture & Edge Solution  
- **Akhil** – Algorithms (WSM & MBAR)  
- **Tejeswar** – Benefits, Summary & Conclusion  

---

##  Future Scope
- Prototype using **Raspberry Pi / STM32 + sensors**.  
- Integration with real-time alert systems in hospitals.  
- Testing with **live patient sensor data**.  
