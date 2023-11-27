# Use the official PyTorch image as a parent image
FROM huggingface/transformers-pytorch-gpu:latest

# Install git, pip then Hugging Face transformers
# RUN apt-get update && apt-get install -y git
# RUN pip install --upgrade pip
# RUN pip install transformers datasets

# Optional: Install additional dependencies for Jupyter Notebook, if you want an interactive environment
# RUN pip install jupyter

# Optional: Install additional common data science libraries
# RUN pip install pandas numpy scikit-learn matplotlib seaborn
