# Auto-Preprocessing-Engine-
# AutoCSVProcessor 🚀

**Automatic CSV Data Cleaning & Preprocessing Pipeline for Machine Learning**

AutoCSVProcessor is a lightweight Python utility that converts **raw CSV datasets into clean, organized, model-ready datasets** using an automated preprocessing pipeline.

It is designed to handle **any generic dataset** without prior knowledge of whether it is intended for **classification or regression** tasks.

The project demonstrates advanced Python concepts including **Object-Oriented Programming (OOP), decorators, and generators** to build a modular and extensible data processing system.

---

# ✨ Features

* Automatic CSV dataset loading
* Missing value handling
* Duplicate removal
* Outlier handling using IQR
* Automatic categorical encoding
* Feature scaling
* Multicollinearity detection and removal
* Low variance feature removal
* Export of clean **model-ready dataset**
* Step-by-step pipeline logging

---

# 🧠 Architecture

The system follows a **pipeline-based preprocessing architecture**.

```
CSV Dataset
    │
    ▼
Load Data
    │
    ▼
Remove Constant Columns
    │
    ▼
Remove Duplicate Rows
    │
    ▼
Handle Missing Values
    │
    ▼
Outlier Treatment
    │
    ▼
Categorical Encoding
    │
    ▼
Feature Scaling
    │
    ▼
Remove Multicollinearity
    │
    ▼
Remove Low Variance Features
    │
    ▼
Model-Ready Dataset
```

---

# 🏗 Project Structure

```
AutoCSVProcessor/
│
├── processor.py
├── example_dataset.csv
├── model_ready_dataset.csv
└── README.md
```

---

# 🚀 Usage

Import the processor and run the pipeline.

```python
from processor import AutoCSVProcessor

processor = AutoCSVProcessor("dataset.csv")

model_ready_data = processor.process()
```

The processed dataset will automatically be saved as:

```
model_ready_dataset.csv
```

---

# 🧪 Example Output

```
Running step: load_data
Shape: (1000, 12)

Running step: handle_missing
Completed: handle_missing

Running step: encode_categorical
Completed: encode_categorical

Final Shape: (1000, 9)
Saved to: model_ready_dataset.csv
```

---

# 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

# 🧩 Advanced Concepts Demonstrated

This project uses several advanced programming techniques:

### Object-Oriented Programming

Encapsulates preprocessing logic inside a reusable class.

### Decorators

Used for automatic logging of pipeline steps.

### Generators

Dynamically manages pipeline execution and step sequencing.

---

# 📊 Use Cases

This tool can be useful for:

* Rapid dataset preprocessing
* Machine learning experimentation
* Data science workflows
* Educational demonstrations of preprocessing pipelines
* Building AutoML-style systems

---

# 🚀 Future Improvements

* Automatic **target column detection**
* Automatic **classificatio**
