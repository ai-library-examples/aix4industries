# AI Explainability 360 Toolkit for Time-Series and Industrial Use Cases (KDD 2023)

With the growing adoption of AI, trust and explainability have become critical which has attracted a lot of research attention over the past decade and has led to the development of many popular AI explainability libraries such as AIX360, Alibi, OmniXAI, etc. Despite that, applying explainability techniques in practice often poses challenges such as lack of consistency between explainers, semantically incorrect explanations, or scalability. Furthermore, one of the key modalities that has been less explored, both from the algorithmic and practice point of view, is time-series. Several application domains involve time-series including Industry 4.0, asset monitoring, supply chain or finance to name a few.

The [AIX360 library](https://github.com/Trusted-AI/AIX360) has been incubated by the Linux Foundation AI & Data open-source projects and it has gained significant popularity: its public GitHub repository has over 1.3K stars and has been broadly adopted in the academic and applied settings. Motivated by industrial applications, large scale client projects and deployments in software products in the areas of IoT, asset management or supply chain, the AIX360 library has been recently expanded significantly to address the above challenges. AIX360 now includes new techniques including support for time-series modality introducing time series based explainers such as TS-LIME, TS Saliency, and TS-ICE. It also introduces improvements in generating model agnostic, consistent, diverse, and scalable explanations, and new algorithms for tabular data. 

In this hands-on tutorial, we provide an overview of the library with the focus on the latest additions, time series explainers and use cases such as forecasting, time series anomaly detection or classification, and hands-on demonstrations based on industrial use-cases selected to demonstrate practical challenges and how they are addressed. The audience will be able to evaluate different types of explanations with a focus on practical aspects motivated by real deployments.

## Demo Books
Access the demo workbooks at https://ai-library-examples.github.io/aix4industries/books/intro.html. 

## Tutorial Website
Tutorial website can be accessed at https://sites.google.com/view/kdd2023-aix4industries.

## Setup Local
Refer [Setup Instructions](https://pages.github.ibm.com/Giridhar-Ganapavarapu/aix-books/books/prerequisites.html) to review pre-requisites and prepare your environment for the tutorial.

### Other resources
- https://research.ibm.com/publications/ai-explainability-360-toolkit-for-time-series-and-industrial-use-cases
- Explainability algorithms used in this demo
    - [TSSaliency](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/tssaliency/tssaliency.py)
    - [TSLime](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/tslime/tslime.py)
    - [TSICE](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/tsice/tsice.py)
    - [NNContrastive](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/nncontrastive/nncontrastive.py)
    - [GCE](https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/gce/gce.py)
