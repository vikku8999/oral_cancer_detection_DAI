from sklearn.model_selection import StratifiedKFold

"""
Contains functions for training and testing a PyTorch model.
"""
from torchinfo import summary
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim



class KFold():
#     wandb.init(
#     project="ressnet50-v1",
#     config={
#             "epochs": EPOCH,
#             "batch_size": BATCH_SIZE,
#             "lr": ALPHA,
#             "architecture": "CNN",
#             })

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 early_stopping = False):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.early_stopping = early_stopping
        self.counter = 0
        self.early_stop = False # type: ignore
        self.best_score = None

#         # Copy your config
#         self.config = wandb.config

    def check_early_stop(self,
                   val_loss,
                   delta,
                   verbose,
                   patience,
                   epoch):

        score = -val_loss
        # print(verbose)
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + delta and epoch < self.epochs:
            self.counter += 1

            if verbose:
                print(f"Early stopping counter: {self.counter} out of {patience}")

            if self.counter >= patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.early_stop = False
            self.counter = 0
            

    def train_epoch(self, epoch):
        
        y_tr_true, y_tr_pred= [], []
        train_loss = 0.0
        
        model.train()
        for i, data in enumerate(tqdm(self.trainloader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')):
            
            images, labels = data
            y_tr_true.extend(labels) # collect all training labels
            images,labels = images.to(device),labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            
            y_tr_pred.extend(predictions)
           
        
        train_correct =  accuracy_score(y_tr_true, y_tr_pred)

        return train_loss,train_correct

    def valid_epoch(self, epoch):
        
        valid_loss = 0.0
        y_tr_true, y_tr_pred= [], []
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.testloader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')):
                
                images, labels = data
                y_tr_true.extend(labels) # collect all training labels
                images,labels = images.to(device),labels.to(device)
                
                output = model(images)
                loss=loss_fn(output,labels)
                valid_loss+=loss.item()*images.size(0)
                scores, predictions = torch.max(output.data,1)
                y_tr_pred.extend(prediction)
        
        val_correct = accuracy_score(y_tr_true, y_tr_pred)

        return valid_loss,val_correct
    
    def test(self,
            test_data,):
        self.test_dataset = test_data
        
        valid_loss = 0.0
        y_tr_true, y_tr_pred= [], []
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.testloader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')):
                
                images, labels = data
                y_tr_true.extend(labels) # collect all training labels
                images,labels = images.to(device),labels.to(device)
                
                output = model(images)
                loss=loss_fn(output,labels)
                valid_loss+=loss.item()*images.size(0)
                scores, predictions = torch.max(output.data,1)
                y_tr_pred.extend(prediction)
        
        val_correct = accuracy_score(y_tr_true, y_tr_pred)

        return valid_loss,val_correct
    
    def train(self,
              train_data,
              epochs=1,
              k_folds=5,
              delta = 0,
              patience = 10,
              verbose = False):

        self.epochs = epochs
        self.dataset = train_data


        # Create empty results dictionary
        results = {"epoch":[],
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }

        # Define the K-fold Cross Validator
        kfold =  StratifiedKFold(n_splits= k_folds, shuffle=True, random_state=42)
        
        
        
        labels = [t[1] for t in self.dataset]

        # Make sure model on target device
        self.model.to(self.device)

        # Loop through training and testing steps for a number of epochs
        print('--------------------------------')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.dataset, labels)):
            print('Fold {}'.format(fold + 1))
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            
            # Define data loaders for training and testing data in this fold
            self.trainloader = torch.utils.data.DataLoader(
                              self.dataset, 
                              batch_size = BATCH_SIZE,
                              sampler = train_subsampler)

            self.testloader = torch.utils.data.DataLoader(
                              self.dataset,
                              batch_size = BATCH_SIZE, 
                              sampler = test_subsampler)

            for epoch in range(self.epochs):
                train_loss, train_correct = self.train_epoch(epoch)
                test_loss, test_correct = self.valid_epoch(epoch)

                train_loss = train_loss / len(self.trainloader.sampler)
                train_acc = train_correct * 100
                test_loss = test_loss / len(self.testloader.sampler)
                test_acc = test_correct * 100
                                                                                                            

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



                if self.early_stopping:
                    self.check_early_stop(test_loss, delta, verbose, patience, epoch)
                    if self.early_stop:
                        print("Early Stopping")
                        break

        # Mark the run as finished
#         wandb.finish()
        # Return the filled results at the end of the epochs
        return results