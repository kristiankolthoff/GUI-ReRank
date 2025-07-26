# *Rico* Annotations

This directory contains annotation files for the *Rico* dataset. Each annotation is a mapping where the keys are the IDs of the *Rico* GUIs and the values are the annotations for multiple search dimensions. Note that this is computed on a filtered set of *Rico* spanning 48.066 GUIs. Annotations were created with GPT4.1 (accessed in July 2025) and a tempereature setting of 0.05. Each annotation entry includes:

- **general_description**: A summary of all other search dimensions for the GUI.
- **domain**: The application domain of the GUI.
- **functionality**: The main functionality provided by the GUI.
- **design**: Design-related aspects of the GUI.
- **gui_components**: The GUI components present in the interface.
- **text**: Text content found in the GUI.
- **nsfw**: Whether the GUI contains not-safe-for-work content.

These annotations are used to enable multi-dimensional search and retrieval within *GUI-ReRank*.  
