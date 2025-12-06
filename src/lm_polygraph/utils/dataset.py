import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional


class Dataset:
    """
    Seq2seq dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(self, x: List[str], y: List[str], batch_size: int, z: Optional[List[int]] = None):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
            z (Optional[List[int]]): an optional list of labels (e.g., int64).
        """
        self.x = x
        self.y = y
        self.z = z # Store the optional label list
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], Optional[List[int]]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str], Optional[List[int]]]]: iterates over batches in dataset,
                returns list of input texts, list of corresponding output texts, and optional list of labels.
        """
        # Determine the length of the lists to iterate over
        data_len = len(self.x)
        
        for i in range(0, data_len, self.batch_size):
            x_batch = self.x[i : i + self.batch_size]
            y_batch = self.y[i : i + self.batch_size]
            
            # Check if labels are present and yield the batch of labels if they are
            if self.z is not None:
                z_batch = self.z[i : i + self.batch_size]
                yield (x_batch, y_batch, z_batch)
            else:
                # If no labels, yield None for the third element to maintain structure
                yield (x_batch, y_batch, None)


    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        if self.z is not None:
            self.z = [self.z[i] for i in indices]
        return self

    def train_test_split(self, test_size: int, seed: int, split: str = "train"): # 'split' here is different from hf dataset split
        """
        Samples dataset into train and test parts.

        Parameters:
            test_size (int): size of test dataset,
            seed (int): seed to perform random splitting with,
            split (str): either 'train' or 'test'. If 'train', lefts only train data in the current dataset object.
                If 'test', left only test data. Default: 'train'.

        Returns:
            Tuple: train/test input, target, and optional label texts list.
        """
        # Combine x, y, and z (if present) for splitting
        data_to_split = [np.array(self.x), np.array(self.y)]
        has_labels = self.z is not None
        if has_labels:
            data_to_split.append(np.array(self.z))

        # Perform the split
        split_results = train_test_split(
            *data_to_split, # Unpack the lists
            test_size=test_size,
            random_state=seed,
        )
        
        # Determine the indices for x, y, and z in the results
        num_fields = 2 + has_labels
        X_train, X_test = split_results[0], split_results[num_fields]
        y_train, y_test = split_results[1], split_results[num_fields + 1]
        
        if has_labels:
            z_train, z_test = split_results[2], split_results[num_fields + 2]


        if split == "train": # keep only train part in the current dataset object
            self.x = X_train.tolist()
            self.y = y_train.tolist()
            if has_labels:
                self.z = z_train.tolist()
        else: # keep only test part in the current dataset object
            self.x = X_test.tolist()
            self.y = y_test.tolist()
            if has_labels:
                self.z = z_test.tolist()

        # Construct the return tuple: (X_train, X_test, y_train, y_test, [z_train, z_test])
        return_tuple = (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )
        if has_labels:
            return_tuple = return_tuple + (z_train.tolist(), z_test.tolist())
            
        return return_tuple


    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        current_len = len(self.x)
        if current_len < size:
            indices = list(range(current_len))
        else:
            if size < 1:
                size = int(size * current_len)
            indices = np.random.choice(current_len, size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        z_column: Optional[str] = None, # Added z_column argument
        **kwargs,
    ):
        """
        Creates the dataset from .CSV table.

        Parameters:
            csv_path (str): path to .csv table,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch.
            z_column (Optional[str]): optional name of column to take labels from.
        """
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()
        z = csv[z_column].tolist() if z_column and z_column in csv.columns else None

        if len(prompt):
            x = [prompt.format(text=text) for text in x]

        return Dataset(x, y, batch_size, z=z) # Pass z to the constructor

    @staticmethod
    def load_hf_dataset(
        path: Union[str, List[str]],
        split: str, # 'split' here is just the hf dataset split to use, not for dataset train/test split
        **kwargs,
    ):
        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = path
            dataset = hf_dataset.load_from_disk(path)
        elif isinstance(path, str):
            dataset_name = path
            dataset = load_dataset(path, split=split, **kwargs)
        else:
            dataset_name = path[0]
            dataset = load_dataset(*path, split=split, **kwargs)

        return dataset_name, dataset

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        description: str = "",
        mmlu_max_subject_size: int = 100,
        n_shot: int = 0,
        few_shot_split: str = "train",
        few_shot_prompt: Optional[str] = None,
        instruct: bool = False,
        split: str = "test",
        size: int = None,
        **kwargs,
    ):
        """
        Creates the dataset from Huggingface datasets.
        
        If 'CV-MedBench' is in dataset_name, it will attempt to retrieve a 'label' column
        and create the Dataset object with the 'z' attribute set.

        Parameters:
            dataset_path (str): HF path to dataset,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch,
            prompt (str): prompt template to use for input texts (default: ''),
            split (str): dataset split to take data from (default: 'test'),
            size (Optional[int]): size to subsample dataset to. If None, the full dataset split will be taken.
                Default: None.
        """
        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)

        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))

        x, y, z = [], [], None # Initialize z as None

        if "allenai/c4" in dataset_name.lower():
            for inst in dataset:
                if len(inst[x_column]) <= 1024:
                    x.append(inst[x_column])
                    y.append(inst[y_column])
        else:
            x = dataset[x_column]
            if y_column is not None:
                y = dataset[y_column]
            else:
                y = ["" for _ in range(len(x))]
            
            # --- START: CV-MedBench specific logic ---
            # Check for CV-MedBench and the 'label' feature
            if "CV-MedBench" in dataset_name and "label" in dataset.features:
                z = dataset["label"] # Assuming 'label' is the column name for the int64 label
            # --- END: CV-MedBench specific logic ---


        return Dataset(x, y, batch_size, z=z) # Pass z to the constructor

    @staticmethod
    def load(path_or_path_and_files: Union[str, List[str]], *args, **kwargs):
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            path_or_path_and_files (str or List[str]): local path to .csv table or HF path to dataset.
        """
        if isinstance(path_or_path_and_files, str) and os.path.isfile(
            path_or_path_and_files
        ):
            # Pass through any z_column argument if present
            return Dataset.from_csv(path_or_path_and_files, *args, **kwargs)
            
        # from_datasets is called with the same arguments, which will handle the CV-MedBench case internally
        return Dataset.from_datasets(path_or_path_and_files, *args, **kwargs)