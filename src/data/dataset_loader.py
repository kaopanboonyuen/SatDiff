from datasets import load_dataset

def load_satellite_dataset(config):
    dataset = load_dataset("imagefolder", data_dir=config['dataset']['train_dir'])
    return dataset["train"]