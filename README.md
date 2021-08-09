## Data

Raw downloaded data (from CIFAR-10 dataset) is in the `cifar-10-batches-py` folder.

To process data for training, run `python ./scripts/data_process.py`.
Processed data is quite large, 1.48 GB, so to actually train on the data, you need to process it yourself.
The output data should be located in the `data` folder. Specifically, the training data are:
 - train_batch_1
 - train_batch_2
 - train_batch_3
 - train_batch_4
 - train_batch_5
and the test data is in test_batch.
Each batch is a dictionary, where the "x" key holds the input features (lightness channel L*) and the "y" key holds the corresponding output features (a* and b* channels).
 
On average system, takes 15.468s to complete data processing.