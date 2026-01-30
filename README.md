# Machine Learning 2

Repository for the Machine Learning 2 course: homework assignments and the Pix2Pix project.

## Project structure

| Folder | Description |
|--------|-------------|
| **HW1** | Homework 1 — notebook, PDF, pickle outputs (`HW1_Q2.pkl`, `HW1_Q3.pkl`), and `HW1_Q3a.py` |
| **HW2** | Homework 2 — emotion classification; notebook, PDF, and data in `HW2_data/` (`trainEmotions.csv`, `testEmotions.csv`) |
| **HW3** | Homework 3 — GAN-related content; notebook, PDF, and `Tutorial9_GAN.html` |
| **HW4** | Homework 4 — GAN implementation; notebook, PDF, instructions, trained model (`fixed_GAN_Model.pkl`), loss plots, and generated images; includes `requirements.txt` |
| **Project Pix2Pix Summary and Code** | Pix2Pix project: summary report, notebook, guidelines, and result images (see that folder’s [README](Project%20Pix2Pix%20Summary%20and%20Code/README.md)) |

## Running the notebooks

- **HW1–HW3**: Open the corresponding `HW*.ipynb` in Jupyter or a compatible environment; no special setup beyond standard ML libraries.
- **HW4**: Install dependencies from `HW4/requirements.txt` (PyTorch, etc.) before running `HW4.ipynb`.
- **Pix2Pix**: Use the notebook and guidelines in `Project Pix2Pix Summary and Code/`; see that folder’s README for details.

## Contents at a glance

- **HW1**: Course assignment 1 (theory + code, pickle exports).
- **HW2**: Emotion classification using `trainEmotions.csv` and `testEmotions.csv`.
- **HW3**: GAN tutorial and assignment (includes `Tutorial9_GAN.html`).
- **HW4**: GAN training and evaluation — generator/discriminator loss plots and generated samples in `Genrated_GAN_Photos/` and `Gen and Dis Loss Plots/`.
- **Project Pix2Pix**: Image-to-image translation (e.g. facades); report in `Imagination to Image-nation.pdf`, code in `Project's Notebook.ipynb`, results in `photos/`.
