try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary
    
import timm

try:
    import wandb
except:
    !pip install wandb
    import wandb
# wandb.login()



import torch


"""
Contains functions for training and testing a PyTorch model.
"""
from torchinfo import summary
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import math
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import numpy as np 
import os
import logging
import seaborn as sns


class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
            
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)  

    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]


class Engine():

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 dirpath,
                 early_stopping = False,
                 config = None):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.early_stopping = early_stopping
        self.counter = 0
        self.early_stop = False # type: ignore
        self.best_score = None
        
        self.dirpath = dirpath

        # Copy your config
        self.config = config

    def model_summary(self,
                      input_size,
                      col_width = 20,):

        return summary(self.model,
                input_size = input_size, # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
                col_names = ["input_size", "output_size", "num_params", "trainable"],
                col_width = col_width,
                row_settings = ["var_names"])
    

    def check_early_stop(self,
                   val_loss,
                   delta,
                   verbose,
                   patience):

        score = -val_loss
        # print(verbose)
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + delta:
            self.counter += 1

            if verbose:
                print(f"Early stopping counter: {self.counter} out of {patience}")

            if self.counter >= patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.early_stop = False
            self.counter = 0

    # convenience funtion to log predictions for a batch of test images
    def log_test_predictions(self, images, labels, outputs, predicted, test_table, log_counter):
        #  obtain confidence scores for all classes
        scores = F.softmax(outputs.data, dim=1)
        log_scores = scores.cpu().numpy()
        log_images = images.cpu().numpy()
        log_labels = labels.cpu().numpy()
        log_preds = predicted.cpu().numpy()
        # adding ids based on the order of the images
        _id = 0
        for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
            # add required info to data table:
            # id, image pixels, model's guess, true label, scores for all classes
            img_id = str(_id) + "_" + str(log_counter)
            img = i.transpose(1, 2, 0)
            test_table.add_data(img_id, wandb.Image(img), p, l, *s)
            _id += 1
            if _id == self.config.batch_size:
                break

            
    def train_step(self, epoch):
#         correct = 0
        y_tr_true, y_tr_pred= [], []
        # Put model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        n_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.batch_size)

        # Loop through data loader data batches
        for i, data in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')):
            # Send data to target device
            X_train, y_train = data
            
            y_tr_true.extend(y_train) # collect all training labels

            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)

            # 1. Forward pass

            outputs = self.model(X_train)
            _, y_pred = torch.max(outputs, 1) # max along an axis gives both max val as well as index
            y_tr_pred.extend(y_pred.data.cpu())

            # 2. Calculate  and accumulate loss
            loss = self.loss_fn(outputs, y_train)

            train_loss += loss.item() 

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()
            
        # Adjust metrics to get average loss and accuracy per batch
        train_loss /= len(self.train_dataloader)
        train_acc = accuracy_score(y_tr_true, y_tr_pred)


        # Log Training info in W&B
        metrics ={
                      "train/train_loss": train_loss,
                      "train/epoch": (i + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                      "train/train_acc": train_acc,
                  }
        if self.config:
            wandb.log(metrics)

        return train_loss, train_acc


    def test_step(self, epoch):

        y_ts_true,  y_ts_pred = [], []
        log_counter = 0
        # Put model in eval mode
        
        columns=["Id", "Image", "Predicted", "Actual"]
        for digit in range(3):
            columns.append("score_" + str(digit))
        test_table = wandb.Table(columns=columns)

        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for i, data in enumerate(tqdm(self.test_dataloader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')):
                # Send data to target device
                X_test, y_test = data
                y_ts_true.extend(y_test) # collect all training labels

                y_test = y_test.to(self.device)
                X_test = X_test.to(self.device)

                # 1. Forward pass
                outputs = self.model(X_test)
                _, y_pred = torch.max(outputs, 1) # max along an axis gives both max val as well as index

                y_ts_pred.extend(y_pred.data.cpu())

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(outputs, y_test)
                test_loss += loss.item() 

                # Calculate and accumulate accuracy
                self.log_test_predictions(X_test, y_test, outputs, y_pred, test_table, log_counter)
                log_counter += 1

         # ✨ W&B: Log predictions table to wandb
        wandb.log({"test_predictions" : test_table})
        
        # Adjust metrics to get average loss and accuracy per batch
        test_loss/= len(self.test_dataloader)
        test_acc = accuracy_score(y_ts_true, y_ts_pred)



        # Log Testing info in W&B
        metrics ={
                  "test/test_loss": test_loss,
                  "test/test_acc": test_acc,
                  }

        if self.config:
            wandb.log(metrics)
            
        

        return test_loss, test_acc


    def evaluate(self,
                eval_dataloader):  
        
        self.eval_dataloader = eval_dataloader
        
        y_ts_true,  y_ts_pred = [], []
        
        self.model.eval()  # Set the model to evaluation mode
        running_loss =  0.0
        correct_predictions =  0
        total_predictions =  0

        with torch.no_grad():  # Disable gradient computation
            for i, data in enumerate(tqdm(self.eval_dataloader, unit='batch')):
                
                inputs, labels = data
                
                y_ts_true.extend(labels)
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                # Compute the loss and update the running loss
                running_loss += loss.item() 

                # Compute the number of correct predictions
                _, y_pred = torch.max(outputs,  1)
                y_ts_pred.extend(y_pred.data.cpu())

        # Compute the average loss and accuracy
        avg_loss = running_loss / len(self.eval_dataloader)

        return avg_loss, (y_ts_true, y_ts_pred)
    
    def train(self,
              train_dataloader,
              test_dataloader,
              epochs=1,
              delta = 0,
              patience = 10,
              verbose = False):

        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        best_val_loss = float('inf')
        # Create empty results dictionary
        results = {"epoch":[],
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }

        # Make sure model on target device
        self.model.to(self.device)

        checkpoint_saver = CheckpointSaver(dirpath=self.dirpath, decreasing=True, top_n=5)
        
        # Loop through training and testing steps for a number of epochs
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_step(epoch)
            test_loss, test_acc = self.test_step(epoch)

            # Print out what's happening
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )

            # Update results dictionary
            results["epoch"].append(epoch+1)
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            results["train_acc"].append(train_acc)
            results["test_acc"].append(test_acc)

            checkpoint_saver(self.model, epoch+1, test_loss)

            if self.early_stopping:
                self.check_early_stop(test_loss, delta, verbose, patience)
                if self.early_stop:
                    print("Early Stopping")
                    break

        # Mark the run as finished
        wandb.finish()
        # Return the filled results at the end of the epochs
        
        return results
    
    