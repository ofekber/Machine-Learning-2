# Project Pix2Pix — Imagination to Image-nation

Pix2Pix (image-to-image translation) project for Machine Learning 2: summary report, code, and result images.

## Contents

| Item | Description |
|------|-------------|
| **Imagination to Image-nation.pdf** | Project summary and report |
| **Project's Notebook.ipynb** | Full implementation: data download (e.g. facades dataset), model, training, and evaluation |
| **Project ML2 2025 guidelines.pdf** | Course project guidelines |
| **photos/** | Loss curves and sample outputs |

## Photos folder

- **Loss plots**: Generator and discriminator losses (standard and cosine variants), e.g. `Gen Loss.png`, `Gen GAN Loss.png`, `Gen_Cos Loss.png`, `Gen GAN_Cos Loss.png`, `Dis Loss.png`, `Dis_Cos Loss.png`, `L1 Loss.png`, `Cosine Loss.png`.
- **Epoch results**: Sample generations at different epochs — `epoch_original_results.png`, `epoch_cosine_results.png`.

## Running the project

1. Open `Project's Notebook.ipynb` in Jupyter or a compatible environment.
2. The notebook downloads the facades dataset from the Berkeley Pix2Pix page (e.g. `facades.tar.gz`) when run.
3. Ensure PyTorch (and any other dependencies used in the notebook) is installed.
4. Run cells in order to train and evaluate the Pix2Pix model; outputs and plots are saved as in the notebook (e.g. into `photos/` or as specified in the code).

For full details and methodology, see **Imagination to Image-nation.pdf** and **Project ML2 2025 guidelines.pdf**.
