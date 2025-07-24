import pandas as pd
from pathlib import Path
import os
from gui_rerank.models.screen_image import ScreenImage
from gui_rerank.annotator.llm_annotator import LLMAnnotator
from gui_rerank.llm.llm import LLM
from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings
from gui_rerank.ranking.dataset_builder import DatasetBuilder
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def print_progress(progress: int, message: str, dataset_name: str):
    print(f"[{dataset_name}] {progress}% - {message}")

def batch_list(lst: List, batch_size: int) -> List[List]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

if __name__ == "__main__":
    # 1. Load the CSV and extract UI Numbers
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/rico_filtered/filtered_guis.csv'))
    df = pd.read_csv(csv_path)
    #ui_numbers = df["UI Number"].astype(str).tolist()[0:10000] # full_run_1
    #ui_numbers = df["UI Number"].astype(str).tolist()[10000:20000] # full_run_2
    #ui_numbers = df["UI Number"].astype(str).tolist()[20000:30000] # full_run_3
    #ui_numbers = df["UI Number"].astype(str).tolist()[30000:40000] # full_run_4
    ui_numbers = df["UI Number"].astype(str).tolist()[40000:50000] # full_run_5
    print(f"Loaded {len(ui_numbers)} Rico GUIs.")
    del df  # Free memory

    # 2. Create ScreenImage objects
    base_image_path = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")
    screen_images = [
        ScreenImage(id=ui_num, filelocation=str(base_image_path / f"{ui_num}.jpg"))
        for ui_num in ui_numbers
    ]

    print(f"Loaded {len(screen_images)} Rico ScreenImages.")
    print(f"First screen images: {screen_images[:5]}")

    # 3. Set up LLM, annotator, embedding model, and dataset builder
    llm = LLM(model_name=LLM.MODEL_GPT_4_1)
    annotator = LLMAnnotator(llm=llm)
    embedding_model = LangChainEmbeddings(model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, dimensions=3072)
    builder = DatasetBuilder(annotator, embedding_model)

    # 4. Set dataset and checkpoint paths
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/checkpoints/'))

    # 5. Build and save the dataset
    rico_batch_size = 1000
    annotation_batch_size = 5
    batches = batch_list(screen_images, rico_batch_size)
    print(f"Loaded {len(batches)} Rico batches.")
    datasets = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                builder.create,
                batch,
                f"batch_{i}",
                dataset_path,
                checkpoint_path,
                progress_callback=print_progress,
                batch_size=annotation_batch_size
            )
            for i, batch in enumerate(batches)
        ]
        for i, future in enumerate(as_completed(futures)):
            ds = future.result()
            ds.save()
            print(f"Batch {i} completed: {ds}")
            datasets.append(ds)

    # Merge all datasets
    if datasets:
        merged_dataset = datasets[0].merge_many(datasets)
        print(merged_dataset)
        merged_dataset.name = "rico_filtered_merged"
        merged_dataset.save()
        print("Merged dataset saved as 'rico_filtered_merged'.")
    else:
        print("No datasets were created.")