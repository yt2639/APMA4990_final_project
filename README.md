# APMA4990_final_project

This is the final project repo for APMA4990 course.

## Organization

- **Analysis:** Notebook for the project, models, and web application are in this folder.

- **Data:** Test data provided by Prof., raw data (3 million entries) queried through big-query, and test data joined with GSOD 2015 weather data are in this folder.

- **img:** Images used in the notebook are in this folder.

- **prediction:** Final predictions on the test data are in this folder.


## Notice

1. In **Analysis** folder, the web application is written on **Dash** (published by *Plotly*) and can only be run on local server. Specifically, under the correct python environment (for example, virtual-env), `cd` to the directory where the `Dash_web_applicatioin.py` locates, and then type in the `cmd`: `$ python Dash_web_applicatioin.py`. This is a demo image of the web app below.
![Web App Demo](img/web_app_demo.jpeg){width=50%}
<center> 
Web App Demo 
</center>

2. In **Analysis** folder, `Models.py` contains the constructed XGBoost, kNN, Lasso, and Ridge models.

3. In **Analysis/saved** folder, the trained models using features after feature selection are saved here.


## Division of responsibilities

**Yutao Tang:** Feature selection, ensemble model, web application.

**Yilin He:** Data cleaning, exploratory data analysis, XGBoost.

**Ashritha Eadara:** kNN, Lasso, Ridge.
