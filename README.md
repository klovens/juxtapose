# Juxtapose

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
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

The following steps will guide you through the process of running Juxtapose on your local machine or an AWS spot instance.

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
