# Juxtapose

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#versioning">Versioning</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Welcome to Juxtapose, a Python tool that can be used to compare gene co-expression networks (GCNs). Juxtapose, together with different similarity measures, can be utilized for comparative transcriptomics between a set of organisms. While we focus on its application to comparing co-expression networks across species in evolutionary studies, Juxtapose is also generalizable to co-expression network comparisons across tissues or conditions within the same species. A word embedding strategy commonly used in natural language processing was utilized in order to generate gene embeddings based on walks made throughout the GCNs. 

You may also suggest changes by forking this repo and creating a pull request or opening an issue. 

<!-- GETTING STARTED -->
## Getting Started

The following steps will guide you through the process of running Juxtapose on your local machine or an [AWS spot instance](https://aws.amazon.com/ec2/spot/?cards.sort-by=item.additionalFields.startDateTime&cards.sort-order=asc).

### Prerequisites

The main dependencies of Juxtapose are gensim, multiprocessing, numpy, pandas, and scipy. See requirements.txt for the complete list of requirements.

### Installation

Juxtapose can be installed using the following commands.

 ```sh
  conda create -n juxtapose python=3.6
  ```

To install requirements, the following command can be run.

```sh
  make setup
  ```
  ### Adding more tests

New tests should be added as modules where their names start with test_ under test directory.

<!-- USAGE EXAMPLES -->
## Usage

In order to run Juxtapose, two JSON files are required containing the desired parameters for (1) creating an anchored network using a set of genes and making walks through this network and (2) running an embedding method to obtain pairwise local distances between genes as well as a global similarity between networks, results and visualizations from biclustering local pairwise distances. 

These operations can be done individually, or run_all.sh can be used to run through all of the steps if the JSON files are provided as follows.

Let us take an example of embedding a simple line network.

To run the anchoring step, we also require the genes/nodes of the network that will be used as the anchor points in the networks that are going to be compared. As the networks will be compared, these synthetics structures that are attached to the real networks should be the same.

Finally, running X will calculate the local similarity measures between genes and bicluster these results. If a full co-expression network is used and it is not possible to generate the complete matrix, there is also an option to select only a percentage of each bicluster in order to make a representative visualization. Also, the global similarity is reported and saved in X.

Larger networks will not be possible to compare on many machines due to the large memory requirements as the number of edges in the networks increases. As such, we recommend an AWS spot instance for more affordable resources. In order to set up an instance that will work for a larger network, e.g. 10,000+ genes, one option would be to select 
EC2 Dashboard
Spot request

```sh
sudo apt update
sudo apt install python3-pip
python3 -m pip install --user numpy scipy matplotlib
pip3 install --upgrade gensim
pip3 install seaborn
pip3 install -U scikit-learn
pip3 install torch torchvision
 ```

```sh
 mkdir experiment
 chmod -R 777 experiment
 ```
 After the volume is attached to the spot instance, the code can be downloaded and Juxtapose can be set up as was done above.

<!-- Versioning -->
## Versioning

We use [Semantic Versioning 2.0.0](http://semver.org/) for versioning.


<!-- CONTACT -->
## Contact

**Katie Ovens** - katie.ovens@usask.ca
Project Link: [https://github.com/klovens/juxtapose](https://github.com/klovens/juxtapose)
