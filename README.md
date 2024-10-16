# FuxiCTR: Criteo Dataset Model Training & Experimentation

<p align="center">
  <img src="https://example.com/your-logo.png" alt="FuxiCTR" width="200"/>
</p>

<p align="center">
  <b>FuxiCTR</b> is an open-source framework designed for fast experimentation with deep learning models on click-through rate (CTR) prediction tasks.
</p>

---

## 🚀 Introduction

FuxiCTR is designed to streamline the process of training, tuning, and deploying machine learning models on large-scale datasets. This repository focuses on training models using the **Criteo dataset**, one of the largest publicly available datasets for CTR prediction.

This project integrates models like **xDeepFM**, **FiBiNet**, and **DeepFM**, enabling fast, customizable experiments with minimal configuration.

---

## 📂 Directory Structure

```bash
.
├── criteo_split/           # Processed Criteo dataset files
│   ├── train_1m.txt        # Training data (1 million samples)
│   ├── valid.csv           # Validation data
│   └── test.csv            # Test data
├── model_zoo/              # Collection of implemented models (xDeepFM, FiBiNet, etc.)
├── src/                    # Core code for data processing and model training
├── run_experiment.py       # Script to execute experiments
├── requirements.txt        # Dependencies for the project
└── README.md               # You're reading this!
🔧 Setup
1. Clone the repository:
bash
复制代码
git clone https://github.com/JR12138/fuxiCTR.git
cd fuxiCTR
2. Install dependencies:
bash
复制代码
pip install -r requirements.txt
3. Dataset preparation:
Ensure you have the Criteo dataset files split and placed in the criteo_split/ directory.

⚡️ How to Run Experiments
To start an experiment, simply run the following command:

bash
复制代码
python run_experiment.py --config config/xdeepfm.yaml
Modify the YAML configuration file to fine-tune hyperparameters for your model and dataset.

🔍 Model Overview
xDeepFM
xDeepFM combines both FM (Factorization Machines) and DNNs to model feature interactions at different levels. Its unique component is the CIN (Compressed Interaction Network), which explicitly captures high-order feature interactions.

Input: Dense and sparse features
Output: CTR prediction
Components:
FM Layer: Captures second-order feature interactions.
DNN Layer: Models deeper, non-linear interactions.
CIN: Captures high-order feature interactions.
More models can be found in the model_zoo/ directory, including DeepFM and FiBiNet.

📊 Results
We trained the xDeepFM model on the Criteo dataset (1M rows), and here are some sample results:

Model	AUC	Logloss
xDeepFM	0.804	0.441
DeepFM	0.798	0.446
FiBiNet	0.807	0.439
💡 Key Features
Flexible Configuration: Easily customize model and training parameters via YAML configuration files.
Model Zoo: Includes popular CTR models such as xDeepFM, FiBiNet, and more.
Efficient Data Processing: Handle large datasets like Criteo with built-in utilities for data splitting and preprocessing.
Multi-GPU Support: Train models efficiently using multiple GPUs for large-scale datasets.
🤝 Contributing
We welcome contributions! Please feel free to open issues or submit pull requests. Make sure to follow the contribution guidelines and code of conduct.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

<p align="center"> Made with ❤️ by [Your Name] </p> ```
