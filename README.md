Steps to reproduce:

1. Run the preprocess script on the dataset (train and test).
2. Run the preparation script on the processed dataset.
3. Run the "whisper_lora_train" file to train the model on the dataset.
4. Run the test script to evaluate the fine-tuned model. (Also run it on the base model if you don't know the base model's performance on the dataset)
   May have some slight variations if retraining due to the seed

If testing a model by itself (without running through the training, etc), just run the test script while making sure the path is set to the model's directory to evaluate the WER.
