### utils.py Elaborado por Mario Paulín como parte del proyecto agritech  
### Este script se utiliza para definir funciones que facilitan el procesamiento de datos en el proyecto.

#Importamos las librerías necesarias
import json
import numpy as np 
import pandas as pd
from scipy.stats import boxcox
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from itertools import product
import random
import statsmodels.api as sm


### Funciones para EDA 

# Comienzo definiendo algunas funciones que se utilizaran para analizar los datos. 
def describe_categorical(data, column_name):
    """
    Muestra una tabla de conteos y un gráfico de barras para una variable categórica.

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - column_name: Nombre de la variable categórica a explorar
    """
    # Conteo y proporción de categorías
    counts = data[column_name].value_counts()
    proportions = data[column_name].value_counts(normalize=True) * 100

    # Crear un DataFrame para mostrar los resultados
    summary_table = pd.DataFrame({
        "Count": counts,
        "Proportion (%)": proportions
    })

    # Imprimir la tabla de resumen
    print(f"Conteo y Proporción de Categorías para '{column_name}':")
    print(summary_table)
    print("\n")

    # Gráfico de barras
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", hue=counts.index, legend=False)
    plt.title(f"Proporción de Categorías en '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel("Conteo")
    plt.show()


# Función para describir variables continuas
def describe_continuous(data, column_name):
    """
    Muestra estadísticas descriptivas y un boxplot para una variable continua.

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - column_name: Nombre de la columna continua a explorar
    """
    # Calculo de estadísticas descriptivas
    stats = {
        "Min": data[column_name].min(),
        "Max": data[column_name].max(),
        "Mean": data[column_name].mean(),
        "Median": data[column_name].median()
    }

    # Imprimir estadísticas descriptivas
    print(f"Estadísticas Descriptivas para '{column_name}':")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print("\n")

    # Plot boxplot
    sns.boxplot(x=data[column_name], color="skyblue")
    plt.title(f"Boxplot of '{column_name}'")
    plt.xlabel(column_name)
    plt.show()

    # Plot histograma
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column_name], kde=True, color="steelblue", bins=20)
    plt.title(f"Histograma de '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel("Frecuencia")
    plt.show()


# Funciones para exploarar la relacion entre el target y los predictores continuos. 
def association_continuous(train, column_name, target_name="target"):
    """
    Analiza la relacion entre el target y la variable candidato calculando la correlacion y generando un scatter plot.

    Parameters:
    - train: Pandas Dataframe con la data de entrenamiento.
    - column_name: Nombre de la columna continua a analizar.
    - target_name: Nombre de la columna objetivo (por defecto es 'target' para simplificar).
    """
    # Calculo de la correlacion con el target
    correlation = train[[column_name, target_name]].corr().iloc[0, 1]

    # Imprimir la correlación
    print(f"Correlación entre '{column_name}' y '{target_name}': {correlation:.2f}")

    # Scatter plot con linea de regresion
    plt.figure(figsize=(8, 5))
    sns.regplot(x=column_name, y=target_name, data=train, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    plt.title(f"Scatter Plot de '{column_name}' vs '{target_name}'")
    plt.xlabel(column_name)
    plt.ylabel(target_name)
    plt.show()



def transformaciones_contemporaneas(data, variables, lambda_path="boxcox_lambdas.json"):
    """
    Aplica transformaciones contemporáneas a un DataFrame.
    Estas transformaciones incluyen:
    - Box-Cox
    - Logaritmo
    - normalización
    - cuadrado normalizado
    - cubo normalizado
    - raíz cuadrada normalizada


    Parameters:
    - data: Pandas DataFrame que contiene los datos a transformar

    Returns:
    - DataFrame transformado
    """
    boxcox_lambdas = {}
    for var in variables:
        if var in data.columns:
            # Box-Cox
            data[f'{var}_boxcox'], lmbda = boxcox(data[var])
            boxcox_lambdas[var] = lmbda
            # Logaritmo. 
            data[f'{var}_log'] = np.log(data[var])
            # Normalización
            data[f'{var}_norm'] = (data[var] - data[var].mean()) / data[var].std()
            # Cuadrado normalizado
            data[f'{var}_square_norm'] = ((data[var] - data[var].mean()) / data[var].std()) ** 2
            # Cubo normalizado
            data[f'{var}_cube_norm'] = ((data[var] - data[var].mean()) / data[var].std()) ** 3
    # Guardar los lambdas en un archivo JSON
    with open(lambda_path, "w") as f:
        json.dump(boxcox_lambdas, f) 
               
    return data

def contemporaneas_polinomios(data, variables):
    """
    Crea características polinómicas contemporáneas para las variables especificadas.

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - variables: Lista de nombres de columnas para crear características polinómicas

    Returns:
    - DataFrame con las nuevas características polinómicas
    """
    for var in variables:
        if var in data.columns:
            data[f'{var}_square'] = data[var] ** 2
            data[f'{var}_cube'] = data[var] ** 3
    return data


def interacciones_contemporaneas(data, variables):
    """
    Crea interacciones contemporáneas entre las variables especificadas.
    Estas interacciones incluyen:
    - Razones
    - Productos

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - variables: Lista de nombres de columnas para crear interacciones

    Returns:
    - DataFrame con las nuevas interacciones
    """
    
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1 = variables[i]
            var2 = variables[j]
            if var1 in data.columns and var2 in data.columns:
                # Razón
                data[f'{var1}_over_{var2}'] = data[var1] / data[var2]
                # Producto
                data[f'{var1}_times_{var2}'] = data[var1] * data[var2]
    
    return data

### Funciones historicas

def calcular_historico(df, variables):
    """
    Calcula el histórico de variables por área y año.
    Agrega la media de los años anteriores para cada variable.

    Parameters:
    - df: Pandas DataFrame que contiene los datos
    - variables: Lista de nombres de columnas para calcular el histórico

    Returns:
    - DataFrame con las nuevas columnas de histórico
    """
    df = df.sort_values(['Area', 'Year'])

    for var in variables:
        if var in df.columns:
            # Histórico por área (excluyendo año actual)
            df[f'{var}_hist_mean'] = (
                df.groupby('Area')[var]
                .expanding()
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            # Media global previa (solo años anteriores)
            def global_prev_mean(row):
                mask = df['Year'] < row['Year']
                prev_vals = df.loc[mask, var]
                return prev_vals.mean() if not prev_vals.empty else np.nan
            df[f'{var}_global_prev_mean'] = df.apply(global_prev_mean, axis=1)
            # Combina: si no hay histórico de área, usa global previa
            df[f'{var}_hist_mean'] = df[f'{var}_hist_mean'].combine_first(df[f'{var}_global_prev_mean'])
    
    df = df.drop(columns=[f'{var}_global_prev_mean' for var in variables])
    
    return df


def cambio_porcentual(df, variable, historico_variable):
    """
    Calcula el cambio porcentual de una variable respecto a su histórico.

    Parameters:
    - df: Pandas DataFrame que contiene los datos
    - variable: Nombre de la columna de la variable actual
    - historico_variable: Nombre de la columna del histórico

    Returns:
    - Series con el cambio porcentual
    """
    df[variable+'_cambio_porcentual'] = ((df[variable] - df[historico_variable]) / df[historico_variable]) * 100
    return df

 
def cambio_logaritmo(df, variable, historico_variable):
    """
    Calcula el cambio logarítmico de una variable respecto a su histórico.

    Parameters:
    - df: Pandas DataFrame que contiene los datos
    - variable: Nombre de la columna de la variable actual
    - historico_variable: Nombre de la columna del histórico

    Returns:
    - Series con el cambio logarítmico
    """
    df[variable+'_cambio_logaritmico'] = np.log(df[variable]/df[historico_variable])
    return df

# Función para agregar variables dummy para la columna Area o pais 
def dummies_area(df: pd.DataFrame) -> pd.DataFrame:
    area_dummies = pd.get_dummies(df['Area'], prefix='Area')
    df = pd.concat([df, area_dummies], axis=1)
    return df

# Funciones para modelos 


# Funcion para ejecutar regresión lineal con validación cruzada temporal y loguear resultados en MLflow
def run_linear_regression_cv(
    train_df: pd.DataFrame,
    folds: list,
    features: list,
    experiment_name: str = "Yield_Prediction_Linear_full",
    run_name: str = "linear_regression_cv",
    model_type: str = "LinearRegression",
    tag_experiment: str = "linear_regression_cv",
    data_version: str = "v1.0",
    results_csv: str = "cv_results.csv"
):
    """
    Ejecuta validación cruzada temporal con regresión lineal y loguea resultados en MLflow.

    Args:
        train_df (pd.DataFrame): Datos de entrenamiento.
        folds (list): Lista de tuplas (train_idx, val_idx) para cada fold.
        features (list): Lista de variables independientes.
        experiment_name (str): Nombre del experimento en MLflow.
        run_name (str): Nombre de la corrida en MLflow.
        model_type (str): Tipo de modelo.
        tag_experiment (str): Tag para MLflow.
        data_version (str): Versión de los datos.
        results_csv (str): Nombre del archivo CSV de resultados.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        rmse_scores, mae_scores, r2_scores = [], [], []
        results = []
        for i, (train_idx, val_idx) in enumerate(folds):
            train_fold = train_df.loc[train_idx]
            val_fold = train_df.loc[val_idx]
            train_fold = train_fold.dropna(subset=features + ['yield_mean'])
            val_fold = val_fold.dropna(subset=features + ['yield_mean'])
            X_train, y_train = train_fold[features], train_fold['yield_mean']
            X_val, y_val = val_fold[features], val_fold['yield_mean']
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            row = {'fold': i+1, 'rmse': rmse, 'mae': mae, 'r2': r2}
            for f, c in zip(features, modelo.coef_):
                row[f'coef_{f}'] = c
            # Calcular p-valores con statsmodels
            X_train_sm = sm.add_constant(X_train)
            ols_model = sm.OLS(y_train, X_train_sm).fit()
            for f in ['const'] + features:
                row[f'pval_{f}'] = ols_model.pvalues.get(f, np.nan)
            results.append(row)
            print(f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
        mlflow.sklearn.log_model(modelo, "model")
        mlflow.log_param("features", features)
        mlflow.log_param("model_type", model_type)
        mlflow.set_tag("experiment", tag_experiment)
        mlflow.set_tag("data_version", data_version)
        results_df = pd.DataFrame(results)
        print("\nTabla de resultados por fold (con coeficientes y p-valores):")
        print(results_df)
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        mlflow.log_metric("cv_rmse_mean", np.mean(rmse_scores))
        mlflow.log_metric("cv_mae_mean", np.mean(mae_scores))
        mlflow.log_metric("cv_r2_mean", np.mean(r2_scores))

# Modelo de efectos aleatorios (MixedLM) con validación cruzada temporal y logging en MLflow
def run_mixedlm_cv(
    train_df: pd.DataFrame,
    folds: list,
    features: list,
    group_col: str = "Area",
    experiment_name: str = "Yield_Prediction_Linear_random_effects_subset",
    run_name: str = "linear_regression_cv",
    model_type: str = "MixedLM_Area",
    tag_experiment: str = "mixedlm_random_effects_cv",
    data_version: str = "v1.0",
    results_csv: str = "cv_results_mixedlm.csv"
):
    """
    Validación cruzada temporal con MixedLM (efectos aleatorios) y logging en MLflow.
    """
    import statsmodels.formula.api as smf
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        rmse_scores, mae_scores, r2_scores = [], [], []
        results = []
        for i, (train_idx, val_idx) in enumerate(folds):
            train_fold = train_df.loc[train_idx].copy()
            val_fold = train_df.loc[val_idx].copy()
            train_fold = train_fold.dropna(subset=features + ['yield_mean', group_col])
            val_fold = val_fold.dropna(subset=features + ['yield_mean', group_col])
            train_fold[group_col] = train_fold[group_col].astype('category')
            val_fold[group_col] = val_fold[group_col].astype('category')
            Xy_train = train_fold[features + ['yield_mean', group_col]].copy()
            Xy_val = val_fold[features + ['yield_mean', group_col]].copy()
            formula = 'yield_mean ~ ' + ' + '.join(features)
            try:
                mixed_model = smf.mixedlm(formula, Xy_train, groups=Xy_train[group_col])
                mixed_result = mixed_model.fit(reml=False, method='lbfgs')
                y_pred = mixed_result.predict(Xy_val)
                rmse = root_mean_squared_error(Xy_val['yield_mean'], y_pred)
                mae = mean_absolute_error(Xy_val['yield_mean'], y_pred)
                r2 = r2_score(Xy_val['yield_mean'], y_pred)
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)
                row = {'fold': i+1, 'rmse': rmse, 'mae': mae, 'r2': r2}
                for f in mixed_result.params.index:
                    row[f'coef_{f}'] = mixed_result.params[f]
                    row[f'pval_{f}'] = mixed_result.pvalues.get(f, np.nan)
                results.append(row)
                print(f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
            except Exception as e:
                print(f"Fold {i+1}: Error en MixedLM: {e}")
                row = {'fold': i+1, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
                results.append(row)
        mlflow.log_param("features", features)
        mlflow.log_param("model_type", model_type)
        mlflow.set_tag("experiment", tag_experiment)
        mlflow.set_tag("data_version", data_version)
        results_df = pd.DataFrame(results)
        print("\nTabla de resultados por fold (con coeficientes y p-valores):")
        print(results_df)
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        mlflow.log_metric("cv_rmse_mean", np.nanmean(rmse_scores))
        mlflow.log_metric("cv_mae_mean", np.nanmean(mae_scores))
        mlflow.log_metric("cv_r2_mean", np.nanmean(r2_scores))

### Modelo de efectos fijos con validación cruzada temporal y logging en MLflow
def run_fixed_effects_cv(
    train_df: pd.DataFrame,
    folds: list,
    features_base: list,
    group_col: str = "Area",
    experiment_name: str = "Yield_Prediction_Linear_fixed_effects_subset",
    run_name: str = "fixed_effects_cv",
    model_type: str = "LinearRegression_FixedEffects_Area",
    tag_experiment: str = "fixed_effects_cv",
    data_version: str = "v1.0",
    results_csv: str = "cv_results_fixed_effects.csv"
):
    """
    Validación cruzada temporal con efectos fijos (dummies para Area) y logging en MLflow.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        rmse_scores, mae_scores, r2_scores = [], [], []
        results = []
        for i, (train_idx, val_idx) in enumerate(folds):
            train_fold = train_df.loc[train_idx]
            val_fold = train_df.loc[val_idx]
            train_fold = train_fold.dropna(subset=features_base + ['yield_mean', group_col])
            val_fold = val_fold.dropna(subset=features_base + ['yield_mean', group_col])
            # Crear dummies para Area
            X_train = pd.get_dummies(train_fold[features_base + [group_col]], drop_first=True)
            X_val = pd.get_dummies(val_fold[features_base + [group_col]], drop_first=True)
            # Alinear columnas por si faltan dummies en val
            X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
            y_train = train_fold['yield_mean']
            y_val = val_fold['yield_mean']
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            row = {'fold': i+1, 'rmse': rmse, 'mae': mae, 'r2': r2}
            for f, c in zip(X_train.columns, modelo.coef_):
                row[f'coef_{f}'] = c
            results.append(row)
            print(f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
        mlflow.sklearn.log_model(modelo, "model")
        mlflow.log_param("features", features_base + [group_col])
        mlflow.log_param("model_type", model_type)
        mlflow.set_tag("experiment", tag_experiment)
        mlflow.set_tag("data_version", data_version)
        results_df = pd.DataFrame(results)
        print("\nTabla de resultados por fold (con coeficientes):")
        print(results_df)
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        mlflow.log_metric("cv_rmse_mean", np.mean(rmse_scores))
        mlflow.log_metric("cv_mae_mean", np.mean(mae_scores))
        mlflow.log_metric("cv_r2_mean", np.mean(r2_scores))

### Modelo random forest con validación cruzada temporal y logging en MLflow
def run_random_forest_cv(
    train_df: pd.DataFrame,
    folds: list,
    features: list,
    experiment_name: str = "Yield_Prediction_RandomForest_subset",
    run_name: str = "random_forest_cv",
    model_type: str = "RandomForestRegressor",
    tag_experiment: str = "random_forest_cv",
    data_version: str = "v1.0",
    results_csv: str = "cv_results_rf.csv",
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
):
    """
    Validación cruzada temporal con Random Forest y logging en MLflow.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        rmse_scores, mae_scores, r2_scores = [], [], []
        results = []
        for i, (train_idx, val_idx) in enumerate(folds):
            train_fold = train_df.loc[train_idx]
            val_fold = train_df.loc[val_idx]
            train_fold = train_fold.dropna(subset=features + ['yield_mean'])
            val_fold = val_fold.dropna(subset=features + ['yield_mean'])
            X_train, y_train = train_fold[features], train_fold['yield_mean']
            X_val, y_val = val_fold[features], val_fold['yield_mean']
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            row = {'fold': i+1, 'rmse': rmse, 'mae': mae, 'r2': r2}
            for f, imp in zip(features, rf.feature_importances_):
                row[f'featimp_{f}'] = imp
            results.append(row)
            print(f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
        mlflow.sklearn.log_model(rf, "model")
        mlflow.log_param("features", features)
        mlflow.log_param("model_type", model_type)
        mlflow.set_tag("experiment", tag_experiment)
        mlflow.set_tag("data_version", data_version)
        results_df = pd.DataFrame(results)
        print("\nTabla de resultados por fold (con importancias de variables):")
        print(results_df)
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        mlflow.log_metric("cv_rmse_mean", np.mean(rmse_scores))
        mlflow.log_metric("cv_mae_mean", np.mean(mae_scores))
        mlflow.log_metric("cv_r2_mean", np.mean(r2_scores))