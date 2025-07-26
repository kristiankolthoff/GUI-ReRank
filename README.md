# GUI-ReRank: Enhancing GUI Retrieval with Multi-Modal LLM-based GUI Reranking

This repository contains the supplementary material and implementation for our work on enhancing GUI retrieval using multi-modal LLM-based GUI reranking to the **ASE 2025** conference demonstration track. 

## Introduction and Overview

<img src="https://raw.githubusercontent.com/kristiankolthoff/GUI-ReRank/refs/heads/master/data/images/overview_gui_rerank.png" width="100%">

*GUI-ReRank* provides an embedding-based GUI retrieval approach for natural language requirements decomposed into multiple search dimensions and constraints, to enable GUI search over complex search queries. Moreover, it provides a highly effective multi-modal LLM-based GUI reranking approach supporting image or text inputs for obtaining high-quality rerankings given the requirements. In addition, *GUI-ReRank* facilitates the integration of arbitrary GUI repositories into the search engine through an automated and customizable LLM-based GUI annotation and embedding pipeline.

We provide a detailed explanation and show the functionality of *GUI-ReRank* in our YouTube video:

<a href="https://www.youtube.com/watch?v=_7x9UCh82ug" target="_blank">
  <img src="https://raw.githubusercontent.com/kristiankolthoff/GUI-ReRank/refs/heads/master/data/images/gui_rerank_youtube_thumbnail.png" alt="Watch the video" style="border:1px solid #ccc; border-radius:8px; padding:4px; max-width:100%;"/>
</a>


The project is structured as follows:

- **gui_rerank**: This directory contains the core Python-based implementation of the retrieval models, reranking architecture, and evaluation scripts described in our work. It includes modules for dataset building, embeddings, LLM-based reranking, evaluation, and more

- **webapp**: This directory contains a prototypical Django web application that demonstrates our approach in a web-based setting. It includes dataset management, search, and settings modules, as well as templates for the user interface

- **rico**: This directory contains the *Rico* dataset import folders. The `images` subfolder should hold all Rico image files (e.g., 1.jpg, 2.jpg, ...), and the `dataset` subfolder should contain annotation and embedding files required for importing the *Rico* dataset into *GUI-ReRank*



## Installation & Setup

This section will guide you through setting up *GUI-ReRank* on your machine, including all dependencies and how to get the system running both with and without datasets.

### 1. Clone the Repository

First, download the codebase to your local machine using git:

```sh
git clone https://github.com/kristiankolthoff/GUI-ReRank.git
cd gui_rerank
```

### 2. Install Docker & Docker Compose

*GUI-ReRank* uses Docker to simplify setup and ensure all dependencies are installed correctly. Please make sure you have both [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system before proceeding. These tools allow you to run the application and its services (MySQL, Redis, Celery, etc.) in isolated containers.

---

## Running *GUI-ReRank*

You can start the application and explore its features even before importing any datasets. This is the base version without *Rico* and can already be used to annotate your own datasets and make them searchable within the app.

1. **Start all services (MySQL, Redis, Celery, and the web app):**
   Run the following command in your project root to build and start all necessary containers:
   ```sh
   docker-compose up --build
   ```
   This command will:
   - Build the Docker images if they do not exist yet.
   - Start the MySQL database, Redis server, Celery worker, and the Django web application.
   - The web interface will be available at [http://localhost:8000](http://localhost:8000).

2. **Access the application:**
   Open your browser and go to [http://localhost:8000](http://localhost:8000). You can now use *GUI-ReRank*, but please note that there will be no datasets loaded yet, so search and ranking features may not return results until you annotate or import data.

---

## Importing the *Rico* Dataset

To use *GUI-ReRank* with real data, *GUI-ReRank* ships with the large-scale publicly available *Rico* dataset prepared for searching within *GUI-ReRank*, which consists of a diverse set of mobile UI images from Android. Follow these steps to prepare and import the dataset:

### Step 1: Download and Prepare *Rico* Images
- **Images directory path:** `/rico/images`
- Download the *Rico* images archive from the official source:
  [Rico GUI Dataset](https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz)
- Unzip the archive. After extraction, **move all image files directly into the `/rico/images` folder** in your project root (e.g., `/rico/images/1.jpg`, `/rico/images/2.jpg`, ...). This ensures the import script can find them easily.
- The *Rico* dataset also contains JSON files for each image. These are not needed for *GUI-ReRank* and can be ignored and kept within the same directory.

### Step 2: Download *Rico* Annotation and Embedding Data
- **Dataset directory path:** `/rico/dataset`
- Download the annotation and embedding data (provided by our project) into the `/rico/dataset` folder in your project root. This data includes metadata and embeddings required for the import.
- **[TODO: Add download link for our dataset here]**
- The `/rico/dataset` folder should contain all necessary metadata and embedding files for the *Rico* dataset import.

### Step 3: Start Only MySQL and Redis
- Before importing, you need to have the database and Redis server running. Start them in the background with:
```sh
docker-compose up -d mysql redis
```
- This command will start only the MySQL and Redis containers, which are required for the import process. The `-d` flag runs them in detached mode so you can continue using your terminal.

### Step 4: Import the *Rico* Dataset
- Now you can run the import command, which will process the images and metadata, create the necessary database entries, and copy the images to the correct location in the app's media folder:
```sh
docker-compose run --rm --entrypoint "" app python manage.py import_dataset --dataset_path /rico/dataset --images_path /rico/images --name "Rico GUI Dataset"
```
- This command runs the Django management command `import_dataset` inside the `app` container, bypassing the default entrypoint so only the import runs.
- The `--dataset_path` and `--images_path` arguments tell the script where to find the data and images.
- The `--name` argument sets the name for the imported dataset in the application.
- The `--rm` flag ensures the container is removed after the import completes.

---

## Notes
- The `/rico/images` and `/rico/dataset` folders must be present in your project root and will be mounted into the container automatically (see `docker-compose.yml`).
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


