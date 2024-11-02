## 📊 Analysis of Viral Propagation and Cardiomyocyte Activity

This project explores two significant areas of cellular analysis using time-lapse imaging: assessing the propagation of a virus among neighboring cells and evaluating the contractile behavior of cardiomyocytes.

---

### 🦠 Problem 1: Measuring Viral Propagation Over Time

**📝 Dataset Overview**  
The dataset for this analysis is provided in the stack `problem1_viralpropagation.tif`, which consists of a two-channel time-lapse capturing:
- **🌀 Phase contrast** (Channel 1)
- **🦠 Viral expression** (Channel 2)

This dataset records the infection progression over time, where the second channel indicates viral expression levels within individual cells. The virus studied induces syncytium formation, causing single-nucleus cells to merge into multi-nucleated cells. The data was acquired using a 20x objective lens, with an approximate pixel size of 300 nm and a time interval of 15 minutes between frames.

**🔍 Analysis Objectives**
- **📈 Rate of Viral Expression Growth**: Quantify the rate at which viral expression increases in a cell after infection.
- **⏳ Infection Cycle Duration**: Measure the time it takes for a newly infected cell to reach viral titer levels comparable to previously infected cells within the field of view.
- **(💡 Bonus)** **Comparison of Nuclear Movement**: Analyze and compare the movement speed of nuclei between infected and uninfected cells.

---

### ❤️ Problem 2: Measuring Contractile Activity of Cardiomyocytes

**📝 Dataset Overview**  
The dataset `problem2_beatingcardiomyocytes.tif` is a label-free time-lapse recording showing the intrinsic anisotropy (birefringence) of cardiomyocytes, highlighting:
- The anisotropic myofibrils visible within cardiomyocytes against an isotropic background.

This data was captured at a high frame rate of 100 fps, with an approximate pixel size of 100 nm, providing detailed visual insights into cardiomyocyte contractility.

**🔍 Analysis Objectives**
- **🧭 Directional Movement Mapping**: Generate a map that visualizes the direction of movement of myofibrils and other cellular components.
- **📊 Magnitude of Movement Mapping**: Create a map that indicates the magnitude of movement for myofibrils and other cellular structures.
- **(💡 Bonus)** **Myofibril Orientation and Beat Alignment**: Evaluate the alignment between the orientation of myofibrils and the local direction of contraction, disregarding other cellular structures.

---

These analyses offer valuable insights into the dynamics of viral infection and the contractile behavior of cardiomyocytes, enhancing our understanding of cellular functions through advanced image analysis techniques.

---

💬 **For questions or suggestions, feel free to reach out via email:**
- 📧 mfazli@stanford.edu
- 📧 mfazli@meei.harvard.edu
