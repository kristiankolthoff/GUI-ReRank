# GUI-ReRank: Enhancing GUI Retrieval with Multi-Modal LLM-based GUI Reranking

This repository contains the supplementary material and implementation for our work on enhancing GUI retrieval using multi-modal LLM-based GUI reranking to the **ASE 2025** conference demonstration track. 

## Introduction and Overview

<img src="https://raw.githubusercontent.com/kristiankolthoff/GUI-ReRank/refs/heads/master/data/images/overview_gui_rerank.png" width="100%">

*GUI-ReRank* provides an embedding-based GUI retrieval approach for natural language requirements decomposed into multiple search dimensions and constraints, to enable GUI search over complex search queries. Moreover, it provides a highly effective multi-modal LLM-based GUI reranking approach supporting image or text inputs for obtaining high-quality rerankings given the requirements. In addition, *GUI-ReRank* facilitates the integration of arbitrary GUI repositories into the search engine through an automated and customizable LLM-based GUI annotation and embedding pipeline.

We provide a detailed explanation and show the functionality of *GUI-ReRank* in our YouTube video:

<a href="https://www.youtube.com/watch?v=_7x9UCh82ug" target="_blank">
  <img src="https://raw.githubusercontent.com/kristiankolthoff/GUI-ReRank/refs/heads/master/data/images/gui_rerank_youtube_thumbnail.png" alt="Watch the video" style="max-width:100%;"/>
</a>


The project is structured as follows:

- **gui_rerank**: This directory contains the core Python-based implementation of the retrieval models, reranking architecture, and evaluation scripts described in our work. It includes modules for dataset building, embeddings, LLM-based reranking, evaluation, and more

- **webapp**: This directory contains a prototypical Django web application that demonstrates our approach in a web-based setting. It includes dataset management, search, and settings modules, as well as templates for the user interface

- **rico**: This directory contains the *Rico* dataset import folders. The `images` subfolder should hold all Rico image files (e.g., 1.jpg, 2.jpg, ...), and the `dataset` subfolder should contain annotation and embedding files required for importing the *Rico* dataset into *GUI-ReRank*

- **annotations**: This directory contains annotation files for the *Rico* dataset (spanning 48.066 GUIs), where each annotation is a mapping and the keys are the IDs of the *Rico* GUIs and the values are the annotations for multiple search dimensions. This dataset is primarily used inside *GUI-ReRank* for enabling search over the *Rico* GUI dataset, however, might also be valuable for other research directions.



## Installation & Setup

This section will guide you through setting up *GUI-ReRank* on your machine, including all dependencies and how to get the system running both with and without datasets.

### 1. Clone the Repository

First, download the codebase to your local machine using git:

```sh
git clone https://github.com/kristiankolthoff/GUI-ReRank.git && cd GUI-ReRank
```

### 2. Install Docker & Docker Compose

*GUI-ReRank* uses Docker to simplify setup and ensure all dependencies are installed correctly. Please make sure you have both [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system before proceeding. These tools allow you to run the application and its services (MySQL, Redis, Celery, etc.) in isolated containers.

---

## Running *GUI-ReRank*

You can start the application and explore its features even before importing any datasets. This is the base version without *Rico* and can already be used to annotate your own datasets and make them searchable within the app.

1. **Start the required services (MySQL, Redis, Celery) with Docker Compose:**
   Run the following command in your project root to build and start the necessary containers in the background:
   ```sh
   docker-compose up -d --build
   ```
   This command will:
   - Build the Docker images if they do not exist yet.
   - Start the MySQL database, Redis server, and Celery worker in the background.
   - The `-d` flag (detached mode) allows you to continue using the same terminal for the next steps while the services keep running.

2. **Set up a Python 3.10.18 virtual environment and install dependencies:**
   The application itself (Django/Daphne) is run locally, not in Docker. You need to create a virtual environment, activate it, and install dependencies using Poetry. Below are two options:

   - **With Conda:**
     Create and activate a new environment called `gui-rerank` with Python 3.10.18, then install dependencies:
     ```sh
     conda create -n gui-rerank python=3.10.18 -y && conda activate gui-rerank && cd gui_rerank && python -m pip install --upgrade pip && pip install poetry && poetry install
     ```
     This will create and activate the environment, upgrade pip, install Poetry, and install all project dependencies as specified in `pyproject.toml`.

   - **With venv:**
     Create and activate a new virtual environment called `gui-rerank` using venv, then install dependencies:
     ```sh
     python3.10 -m venv gui-rerank && source gui-rerank/bin/activate && cd gui_rerank && python -m pip install --upgrade pip && pip install poetry && poetry install
     ```
     This will create and activate the environment, upgrade pip, install Poetry, and install all project dependencies as specified in `pyproject.toml`.

3. **Run Django migrations:**
   After installing dependencies, you need to apply database migrations. From the project root, run:
   ```sh
   cd webapp && python manage.py migrate
   ```
   This will apply all necessary migrations to set up the database schema.

4. **Start the Daphne server:**
   Finally, start the Daphne server to run the web application:
   ```sh
   daphne -b 0.0.0.0 -p 8000 config.asgi:application
   ```
   The web interface will be available at [http://localhost:8000](http://localhost:8000).

5. **Configure API keys in the running app:**
   After starting the Daphne server and opening the web interface, you must first provide at least your OpenAI API key in the application settings and save it. This is required for the core functionality. If you wish to use Google or Anthropic models, you can also provide those API keys in the settings. However, an OpenAI API key is required in any case. Once you have saved your API key(s), you can begin adding datasets or conducting searches in the app.

---

## Importing the *Rico* Dataset

To use *GUI-ReRank* with real data, you can import the large-scale publicly available *Rico* dataset. Follow these steps to prepare and import the dataset:

### Step 1: Download and Prepare *Rico* Images
- Download the *Rico* images archive (`unique_uis.tar.gz`) from the official source:
  [Rico GUI Dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz)
- Place the `unique_uis.tar.gz` file directly into the `rico/images` directory in your project root.
- Extract the archive inside `rico/images`. After extraction, you will have a folder structure like:
  - `rico/images/unique_uis/`
    - `combined/`
      - `1.jpg`, `2.jpg`, ...
- The `combined` folder inside `unique_uis` contains all the Rico images. You do not need to move the images; just note the full path to the `combined` folder for the import step.

### Step 2: Download *Rico* Annotation and Embedding Data
- Download the annotation and embedding data (provided by our project and available through Zenodo) into the `rico/data` directory in your project root. This data includes metadata and embeddings required for the import: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16451923.svg)](https://doi.org/10.5281/zenodo.16451923)
- The `rico/data` folder should now contain all necessary metadata and embedding files for the *Rico* dataset import.
- Make sure you know the full path to this folder for the import step.

### Step 3: Ensure MySQL and Redis are running
- Before importing, make sure the database and Redis server are running. If you have not already started them, use:
  ```sh
  docker-compose up -d
  ```
  This will start the required services in the background and you can continue using the same terminal.

### Step 4: Import the *Rico* Dataset
- Run the import command locally (not in Docker). This will process the images and metadata, create the necessary database entries, and copy the images to the correct location in the app's media folder:
  ```sh
  python manage.py import_dataset --dataset_path ./rico/data --name "Rico GUI Dataset" --images_path ./rico/images/unique_uis/combined
  ```
  - The `--dataset_path` argument should point to the `rico/data` directory (e.g., `./rico/data`).
  - The `--images_path` argument should point to the `combined` folder inside `rico/images/unique_uis` (e.g., `./rico/images/unique_uis/combined`).
  - The `--name` argument sets the name for the imported dataset in the application. We recommend using `"Rico GUI Dataset"` as the name.

This process will import the Rico dataset into *GUI-ReRank* and make it available for search and reranking within the application.

---

## Notes
- The `/rico/images` and `/rico/dataset` folders must be present in your project root and will be mounted into the container automatically (see `docker-compose.yml`). However, these files can be removed after importing the *Rico* dataset into *GUI-ReRank*.
- The import command must be run **after** MySQL and Redis are up and running, otherwise the import will fail to connect to the database.
- If you want to clean up orphan containers (leftover containers from previous runs), you can run:
  ```sh
  docker-compose down --remove-orphans
  ```
- For large imports, ensure you have sufficient disk space and memory available on your machine.

---

## Troubleshooting
- **File access issues:** If you encounter issues with file access, make sure the `/rico/images` and `/rico/dataset` folders are correctly mounted and accessible from within the container. You can check this by opening a shell in the container and listing the contents:
  ```sh
  docker-compose run --rm --entrypoint /bin/bash app
  ls /rico/images
  ls /rico/dataset
  ```
- **Database connection errors:** Ensure MySQL and Redis are running before starting the import or the web application.
- **Running management commands:** If you need to run other Django management commands or debug inside the container, you can start a shell as shown above and run any command interactively.
- **Performance:** For very large imports, the process may take some time. Monitor your system resources and consider running the import on a machine with sufficient RAM and disk space.


