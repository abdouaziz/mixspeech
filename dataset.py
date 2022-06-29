from datasets import load_dataset
from datasets import ClassLabel, Dataset, DatasetDict 

class Data :

    def save_dataset(self,train_file=None, dev_file = None , test_file=None ,  json_name =None , type_to_save='jsonl'):

        if train_file is  None:
            raise ValueError('train_file is None')
            
        dataset = DatasetDict({'train': Dataset.from_pandas(train_file) , 'dev': Dataset.from_pandas(dev_file) , 'test': Dataset.from_pandas(test_file)})

        for split , item in dataset.items():
            item.to_json(f"{json_name}-{split}.{type_to_save}")


    def load_dataset(self,train_json =None , dev_json = None , test_json = None , type_to_load='json'):
        if train_json is  None:
            raise ValueError('train json is None')


        json_data_file = { "train": train_json , "dev": dev_json , "test": test_json}

        dataset_loaded = load_dataset(type_to_load, data_files=json_data_file)


        return dataset_loaded
            