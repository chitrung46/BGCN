import pandas as pd

class SRDataset:
    def __init__(self, args):
        super(SRDataset, self).__init__()
        self.args = args

    def load_dataset(self, dataset_name):
        pass

    def preprocess(self, dataset_name):
        if dataset_name == 'Beauty':
            df = pd.read_json('All_Beauty.jsonl.gz', compression='gzip', lines=True)

            
        