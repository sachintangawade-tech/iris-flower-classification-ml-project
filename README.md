# Iris Flower Classification — ML Project

This project demonstrates a complete classification pipeline using the Iris dataset from scikit-learn. The deliverables include a Jupyter Notebook (`iris_flower_classification.ipynb`) with EDA, multiple model training, evaluation, saving the best model, and a predict function for quick inference.

## Project highlights

- Load the Iris dataset and convert it into a `pandas.DataFrame`.
- Perform Exploratory Data Analysis (pairplot, heatmap, distributions).
- Train: Logistic Regression, SVM, Decision Tree, Random Forest.
- Evaluate models using accuracy, confusion matrix, and classification report.
- Save the best-performing model and provide a simple `predict()` helper function.

## How to run

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv; .\venv\Scripts\Activate; pip install -r requirements.txt
```

2. Start Jupyter Lab or Notebook and open `iris_flower_classification.ipynb`:

```powershell
jupyter notebook iris_flower_classification.ipynb
```

3. Run all the cells. The notebook saves the best model into `iris_best_model.joblib` when executed.

## Files
- `iris_flower_classification.ipynb`: The main notebook containing the full pipeline with explanations.
- `requirements.txt`: Required Python packages for running the notebook.
- `predict.py`: A helper script to load the saved model and run predictions on sample inputs.

## Author
- Project created as a portfolio-style exercise suitable for a BCA student learning machine learning.

## License
- MIT
