(ccs:references:introduction)=
# Concrete Compressive Strength
The strength properties of concrete, such as compressive strength depend on many process variables (such as curation time), material feed (binder, slag, water etc.) and their quality. It is important for concrete to possess the right compressive strength depending on the application. For example, concrete used in the construction of roads and bridges, require much higher strength than for example, the concrete used for steps and pavements. Contractors who are typically the buyers require concrete with strict specifications, and process engineers in the concrete mixing plant need to make sure to use appropriate process parameters and feeds are used to produce the right concrete blocks. Since this process happens in batch mode, samples from each batch is tested to ensure the concrete meets the strength requirements.

Testing concrete samples for right compressive strength requires a destructive form of testing. Often it is difficult to be certain about the right combination of feeds and process parameters, especially if the ingredients are sourced differently. Applying AI / ML capabilities to estimate the concrete strength is one of the analytic insights process engineers could use effectively to determine the right combination of process parameters that is most likely to result in the right compressive strength. This also helps in avoiding destroying samples, wastage, thereby increase yield and expensive recalls. In this demo, we cover a use case on concrete comprehensive strength prediction using the data from [UCI repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).

The dataset contains 8 columns such as cement, water, superplasticizer etc. that are used in the process, and concrete comprehensive strength as target variable. We consider regression and classification problems on the target variable. Regression model predicts the concrete comprehensive strength for a given input mixture combination. Classification model predicts whether the given input mixture combination outputs acceptable concrete comprehensive strengh or not. 

For this tutorial, we use gradient boost classifier and regressor models. We use aix360's [Nearest Neighborhood Contrastive Explainer (NNContrastiveExplainer)](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/nncontrastive/nncontrastive.py) to identify viable mixture combination for acceptable concrete comprehensive strength. Further, we perform detailed analysis on impact of the ingredients on the predicted concrete comprehensive strengh using [Grouped Conditional Expectation Explainer (GroupedCEExplainer)](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/gce/gce.py).

For more algorithmic details on NNContrastive and GroupedCE, you can refer to [Nearest Neighbor Contrastive explanation](ccs:references:nncontrastive) and [Grouped Conditional Expectation Explainer](ccs:references:gce) sections respectively.

To start this hands on demo, skip to [Instructions](ccs:references:instructions).

(ccs:references:nncontrastive)=
#### Nearest Neighbor Contrastive explanation
- Nearest Neighbor Contrastive explanation method is an exemplar based constrastive explanation method which provides feasible or realizable contrastive instances. 
- For a given model, exemplar/representative dataset, and query point, it computes the closest point, within the representative dataset, that has a different prediction compared to the query point (with respect to the model). 
- The closeness is defined using an AutoEncoder and ensures a robust and faithful neighborhood even in case of high-dimensional feature space or noisy datasets. 
- This explanation method can also be used in the model-free case, where the model predictions are replaced by (user provided) ground truth. 

(ccs:references:gce)=
#### Grouped Conditional Expectation Explainer
- Grouped Conditional Expectation (GroupedCE) Explainer is a local, model-agnostic explainer that generates Grouped Conditional Expectation (CE) plots for a given instance and a set of features. 
- The set of features can be either a subset of the input covariates defined by the user or the top K features based on the importance provided by a global explainer (e.g., SHAP). 
- The explainer produces 3D plots, containing the model output when pairs of features vary simultaneously. 
- If a single feature is provided, the explainer produces standard 2D ICE plots, where only one feature is perturbed at a time.

(ccs:references:instructions)=
## Instructions
- Create a new notebook `concrete_comprehensive_strength.ipynb` in Jupyter lab to run the demo on tabular regression and classification use cases. Refer to the instructions in [prerequisites](prereq:references:verify_installation). 
- Follow all the below sections and execute the code by pasting to the newly created notebook `concrete_comprehensive_strength.ipynb`.

### Imports

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
import os
import warnings
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.gridspec as gridspec
import plotly.io as pio
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image
from aix360.algorithms.nncontrastive import NearestNeighborContrastiveExplainer
from aix360.algorithms.gce.gce import GroupedCEExplainer


# disable warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

np.random.RandomState(seed=42)
```

### Load Dataset

Download the dataset from [UCI repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) using the below code snippet. For use-case and dataset description, refer to [introduction](ccs:references:introduction).

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
column_names = ['Cement (kg in a m^3 mixture)',
       'Blast Furnace Slag (kg in a m^3 mixture)',
       'Fly Ash (kg in a m^3 mixture)',
       'Water (kg in a m^3 mixture)',
       'Superplasticizer (kg in a m^3 mixture)',
       'Coarse Aggregate (kg in a m^3 mixture)',
       'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)',
       'Concrete compressive strength(MPa, megapascals) ']

pd_data = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    )

pd_data = pd_data.dropna()
pd_data.columns = column_names

display(pd_data)

CCSThreshold = 28

pd_data_train, pd_data_test = train_test_split(pd_data, test_size=0.3, shuffle=True,random_state=42)

feature_names = pd_data_train.columns[:-1]
X_train = pd_data_train.values[:,:-1]
y_train = pd_data_train.values[:,-1]
y_train_cls = np.array(y_train<=CCSThreshold)

X_test = pd_data_test.values[:,:-1]
y_test = pd_data_test.values[:,-1]
y_test_cls = np.array(y_test<=CCSThreshold)
```

### Train Classification Model

Train a GradientBoostingClassifier to predict if the concrete comprehensive strength is above the accepted quality threshold (28 MPa).

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python

model = GradientBoostingClassifier().fit(X_train,y_train_cls)

print('Train Accuracy score : ', model.score(X_train,y_train_cls))
print('Test Accuracy score : ', model.score(X_test,y_test_cls))

```

### Initialize NearestNeighborContrastiveExplainer With Black Box Model
[NearestNeighborContrastiveExplainer](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/nncontrastive/nncontrastive.py) generates contrastive explanation close to the exemplar dataset for a given dataset. This exemplar dataset represents viable contrastive examples for the selected use case. The `NearestNeighborContrastiveExplainer` uses the input model to detect classes in the input data. And these classes are used to build the exemplar dataset for a given test instance. 

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
neighbors = 3
embedding_dim = 4
layers_config = []
epochs = 500
random_seed = 42
explainer_with_model = NearestNeighborContrastiveExplainer(model=model.predict,
                                              embedding_dim=embedding_dim,
                                              layers_config=layers_config,
                                              neighbors=neighbors)

history = explainer_with_model.fit(
    pd_data_train[feature_names],
    epochs=epochs,
    numeric_scaling=None,
    random_seed=random_seed,
)

p_str = f'Epochs: {epochs}'
for key in history.history:
    p_str = f'{p_str}\t{key}: {history.history[key][-1]:.4f}'
print(p_str)
```

### Compute NNContrastive Explanation
Find the nearest contrastive for a given instance. Consider a random instance from the set of non-acceptable instances (class 0) and compute the explanation.

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python

y_test_pred = np.array(model.predict(X_test))
nonacceptable_predictions_indices = np.where(y_test_pred == 0)[0]
rand_index = np.random.choice(nonacceptable_predictions_indices)
instance_to_explain = pd_data_test[feature_names].iloc[rand_index:rand_index+1]
nearest_acceptable_contrastive = explainer_with_model.explain_instance(instance_to_explain)
nearest_acceptable_contrastive[0]["neighbors"][0]
```

### Plot NearestNeighborContrastive Explanation

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
def plot_matrix(matrix, vmin=None, vmax=None, precision=3, rotation=0):
    if len(matrix.shape) == 1:
        matrix = matrix[np.newaxis, :]

    if vmin is None:
        vmin = np.min(matrix)
    if vmax is None:
        vmax = np.max(matrix)

    fig, ax = plt.subplots(figsize=(20, 4))
    # ax_mat = ax.matshow(matrix, vmin=vmin, vmax=vmax, aspect="auto", cmap="RdYlGn")
    ax_mat = ax.matshow(matrix, vmin=vmin, vmax=vmax, cmap="RdYlGn")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i, j]
            ax.text(
                j,
                i,
                str(np.round(c, precision)),
                va="center",
                ha="center",
                rotation=rotation,
                color="black",
                fontweight="bold",
                fontsize="x-large",
            )
    fig.colorbar(ax_mat, ax = ax)

delta = np.array(nearest_acceptable_contrastive[0]["neighbors"][0]) - np.array(instance_to_explain)

plot_matrix(delta,vmin=-1*np.max(np.abs(delta)), vmax=np.max(np.abs(delta)), rotation = 90)
plt.xticks(np.arange(len(feature_names)), feature_names,rotation = 90, fontsize='large')
plt.yticks([])
plt.tight_layout()
plt.title('delta between the selected instance and the nearest acceptable contrastive',
          fontsize='x-large',
          fontweight='bold')
plt.show()
```

### Train Regression Model

Train a GradientBoostingRegressor to predict concrete comprehensive strength for a given mixture combination.

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
model = GradientBoostingRegressor().fit(X_train,y_train)

print('Train R2 score : ', model.score(X_train,y_train))
print('Test R2 score : ', model.score(X_test,y_test))
```

### Compute Individual Conditional Expectation (ICE)

Initialize [GroupedCEExplainer](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/gce/gce.py) to compute ICE scores for each feature. The explanation is a local explanation computed for single instance. 

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python


n_test_samples = 25 # number of instances to explain
n_samples = 100 # number of samples to generate for selected feature

ice_explanations = {}
for i, feature_col in enumerate(feature_names):
    ice_explainer = GroupedCEExplainer(model=model.predict,
                                       data=X_train, 
                                       feature_names=feature_names,
                                         n_samples=n_samples,
                                         features_selected=[feature_col],
                                        random_seed=22
                                  )
    ice_explanations[feature_col] = []
    for i in range(n_test_samples):
        ice_explanations[feature_col].append(ice_explainer.explain_instance(instance=X_test[[i], :]))
```

### Plot ICE Explanation

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
def plot_ice_explanation(explanation, title):
    feature_names = list(explanation.keys())
    
    fig = plt.figure(layout='constrained', figsize=(15,10))
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios= [1, int(len(feature_names)/3)], top=0.9)
    gs2 = gridspec.GridSpecFromSubplotSpec(int(np.ceil(len(feature_names)/3)), 3, subplot_spec=gs0[1])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[4.5, 10, 4.5])
    # print()
    for i, feature_col in enumerate(feature_names):
        if i > 0:
            gs = gs2[i-1]
        else:
            gs = gs1[1]


        ax = fig.add_subplot(gs)
        delta_avg = 0
        delta_ice_values = []
        for i in range(len(explanation[feature_col])):
            delta_ice = explanation[feature_col][i]["ice_value"] - explanation[feature_col][i]["ice_value"][0]
            delta_ice_values.append(delta_ice)
            ax.plot(explanation[feature_col][i]["feature_value"], delta_ice, color='b', alpha=0.2)


        delta_ice_values = np.asarray(delta_ice_values)
        # average ice value across test instances
        delta_avg = np.mean(delta_ice_values, axis=0)
        ax.plot(explanation[feature_col][0]["feature_value"], delta_avg, color='r')  

        ax.set_title("{}".format(feature_col))

        if i==0:
            ax.set_ylabel('Model Prediction')
            ax.set_xlabel('Feature Value')

    fig.suptitle(title, fontsize='x-large')
    plt.show()

plot_ice_explanation(ice_explanations, title="GroupedCE (Individual Conditional Expectation) Explanation Plot for {}-{} Test Instances.".format(0, n_test_samples-1))
```

### Compute Grouped Conditional Expectation (GCE)
Initialize GroupedCEExplainer to compute GCE scores for each feature. The explanation is a local explanation computed for single instance. Each cell in the resulting grid explains how a combination of feature values (example BPM and BP) impact the model prediction.

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
n_samples = 100
top_k_features = 4

# initialization
groupedce_explainer = GroupedCEExplainer(model=model.predict,
                                         data=X_train, 
                                         feature_names=feature_names,
                                         n_samples=n_samples,
                                         top_k_features=top_k_features,
                                        random_seed=22)

y_test_pred_regression = model.predict(X_test)
nonacceptable_predictions_indices = np.where(y_test_pred_regression <= CCSThreshold)[0]
rand_index = np.random.choice(nonacceptable_predictions_indices)
groupedce_instance_to_explain = pd_data_test[feature_names].iloc[rand_index:rand_index+1]
groupedce_explanation = groupedce_explainer.explain_instance(instance=groupedce_instance_to_explain)
```

### Plot GCE Explanation

Paste the below code snippet into a cell in `concrete_comprehensive_strength.ipynb` in Jupyter lab and run the cell.

```python
def plot_gce_explanation(
    explanation,
    plot_width: int = 250,
    plot_height: int = 250,
    plot_bgcolor: str = "white",
    plot_line_width: int = 2,
    plot_instance_size: int = 15,
    plot_instance_color: str = "firebrick",
    plot_instance_width: int = 4,
    plot_contour_coloring: str = "heatmap",
    plot_contour_color: Union[str, List[Tuple[float, str]]] = "Portland",
    renderer="notebook",
    title=None,
    **kwargs,
):

    exp_data = explanation
    feat_dict = {k: len(exp_data[k].keys()) for k in exp_data['selected_features']}
    features = sorted(feat_dict, key=lambda l: feat_dict[l])
    n_feat = len(features)

    specs = [
        [{} if i <= j else None for j in range(n_feat - 1)] for i in range(n_feat - 1)
    ]

    fig = make_subplots(
        rows=n_feat - 1,
        cols=n_feat - 1,
        specs=specs,
        shared_xaxes="columns",
        shared_yaxes="rows",
        column_titles=features[1:],
        row_titles=features[:-1],
    )

    for x_i in range(n_feat):
        for y_i in range(n_feat):
            if y_i < x_i:
                x_feat = features[x_i]
                y_feat = features[y_i]
                z = exp_data[x_feat][y_feat]["gce_values"]
                x_g = exp_data[x_feat][y_feat]["x_grid"]
                y_g = exp_data[x_feat][y_feat]["y_grid"]
                fig.add_trace(
                    go.Contour(
                        z=z,
                        x=x_g,
                        y=y_g,
                        connectgaps=True,
                        line_smoothing=0.5,
                        contours_coloring=plot_contour_coloring,
                        contours_showlabels=True,
                        line_width=plot_line_width,
                        coloraxis="coloraxis1",
                        hovertemplate="<b>"
                        + str(x_feat)
                        + "</b>: %{x:.2f}<br>"
                        + "<b>"
                        + str(y_feat)
                        + "</b>: %{y:.2f}<br>"
                        + "<b>prediction</b>: %{z:.2f}<br><extra></extra>",
                    ),
                    row=y_i + 1,
                    col=x_i,
                )
                if "current_values" in exp_data[x_feat][y_feat]:
                    x = exp_data[x_feat][y_feat]["current_values"][x_feat]
                    y = exp_data[x_feat][y_feat]["current_values"][y_feat]
                    fig.add_trace(
                        go.Scatter(
                            mode="markers",
                            marker_symbol="x",
                            x=[x],
                            y=[y],
                            marker_color=plot_instance_color,
                            marker_line_color=plot_instance_color,
                            marker_size=plot_instance_size,
                            marker_line_width=plot_instance_width,
                            showlegend=False,
                            hovertemplate="{}: {:.2f}<br> {}: {:.2f}<extra></extra>".format(
                                x_feat, x, y_feat, y
                            ),
                        ),
                        row=y_i + 1,
                        col=x_i,
                    )

    fig.update_layout(
        height=(n_feat - 1) * plot_height,
        width=(n_feat - 1) * plot_width,
        plot_bgcolor=plot_bgcolor,
        coloraxis_autocolorscale=False,
        coloraxis_colorscale=plot_contour_color,
        title_text=title
        
    )
    return fig

title="Grouped Conditional Expectation (GCE) Plots for Test Instance {} and Selected Features {}".format(i, groupedce_explanation['selected_features'])

fig = plot_gce_explanation(groupedce_explanation, title=title)
dataset_plot_bytes = fig.to_image(format="png", width=1400, height=1400)
Image(dataset_plot_bytes)
```

Congratulations!! and Thank You for attending the tutorial!!