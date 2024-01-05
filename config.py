from pathlib import Path

def get_config():
    return {
        "batch_size": 4, # should be 8
        "num_epochs": 10, # should be 20
        "lr": 0.001, # should be 10**-4
        "seq_len": 400,
        "tgt_seq_len": 50, 
        "d_model": 512, #should be 512
        "datasource": 'cnn_dailymail',
        "text_src": "article",
        "text_tgt": 'highlights',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])