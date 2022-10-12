from typing import Optional, Sequence
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim import SGD, AdamW
from torchmetrics import Accuracy
from torchvision.models import resnet50

class DownstreamLinearModule_sweep(pl.LightningModule):
    def __init__(
        self, 
        hypers, 
        backbone_weights: str,
        num_classes: int,
        batch_size: int,
        lr_decay_steps: Optional[Sequence[int]] = None,
        metric: str = 'accuracy',
        task: str = 'linear_eval',
        **kwargs,
    ):
        super().__init__()
        self.backbone_weights = backbone_weights
        self.num_classes = num_classes
        self.max_epochs = hypers.epochs
        self.batch_size = batch_size
        self.lr = hypers.lr *batch_size/256 # Sweeping Learning rate value
        print("learning Rate of", self.lr)
        self.weight_decay = hypers.weight_decay # Sweeping weight decay value
        self.lr_decay_steps = lr_decay_steps
        self.metric = metric
        self.task = task
        self.scheduler = hypers.lr_scheduler ## Sweeping learning rate schedule
        self.__build_model()
        
        ## Plugin Modules
        self.loss_module = nn.CrossEntropyLoss()
        self.mean_acc = Accuracy(num_classes=num_classes, average='macro')
        self.accuracy_5= Accuracy(top_k=5)
        self.accuracy= Accuracy()


    def __build_model(self):

        # 1. Backbone
        backbone = resnet50(pretrained=True)
        ## Loading weight Configure
        state = torch.load(self.backbone_weights)["state_dict"]  
        for k in list(state.keys()):
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)


        ## ---------- Method 2 -------------
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
   
        # linear model
        self.classifier = nn.Linear(2048, self.num_classes)
    
    def forward(self, x):
        ##----Method 1 --------
        # 1. Feature extraction
        # x = self.feature_extractor(x)
        # # 2. Classification
        # x = self.fc(x)
        ## ----Method 2 ------
        if self.task=="finetune": 
            representations = self.feature_extractor(x).flatten(1)
        else: 
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        #representations=self.feature_extractor(x).flatten(1)
        x  = self.classifier(representations)
        #x=F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # if self.task == 'linear_eval':
        #     self.feature_extractor.eval()
        
        # else:
        #     self.feature_extractor.train(True)
        # self.feature_extractor.eval()
        # 1. Forward pass
        x, y = batch
        #y=y.softmax(dim=1)
        #y_logits = self.forward(x)
        y_logits = self.forward(x)
        print(f"the ground truth label {y[0]}")
        print(f"The prediction with softmax {y_logits[0].argmax(dim=-1)}")

        # 2. Compute loss
        train_loss = F.cross_entropy(y_logits, y)
        #train_loss=self.loss_module(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            log = {"train_loss": train_loss, "train_acc1": acc1, "train_acc5": acc5}
            # acc=(y_logits.argmax(dim=-1) == y).float().mean()#
            # self.log_dict("train_acc", acc, on_step=False, on_epoch=True)
            # self.log_dict("train_loss", train_loss, prog_bar=True)
        elif self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            log = {"train_loss": train_loss, "train_acc1": acc_1,}
        else:
            mean_acc = self.mean_acc(y_logits, y)
            log = {"train_loss": train_loss, "train_mean_acc": mean_acc}
        
        self.log_dict(log, on_epoch=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        batch_size = x.size(0)
        y_logits = self.forward(x)
        #y_logits = self.forward(x)

        # 2. Compute loss
        val_loss = F.cross_entropy(y_logits, y)
        #val_loss=self.loss_module(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}
  
        if self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            acc_5 =self.accuracy_5(y_logits, y)
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_acc1": acc_1,"val_acc5": acc_5, }
        else:
            mean_acc = self.mean_acc(y_logits, y)
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_mean_acc": mean_acc}
        
        return results

    def validation_epoch_end(self, outs):
        
        val_loss = self.__weighted_mean(outs, "val_loss", "batch_size")
        
        if self.metric == 'accuracy_1_5':
            val_acc1 = self.__weighted_mean(outs, "val_acc1", "batch_size")
            val_acc5 = self.__weighted_mean(outs, "val_acc5", "batch_size")
            log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.metric == 'accuracy_1_5_torchmetric':
            avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
            avg_acc_1 = torch.stack([x["val_acc1"] for x in outs]).mean()
            avg_acc_5 = torch.stack([x["val_acc5"] for x in outs]).mean()
            log = {"ptl/val_loss": avg_loss, "ptl/val_accuracy_1": avg_acc_1,"ptl/val_accuracy_5": avg_acc_5}
        
        else:
            val_mean_acc = self.__weighted_mean(outs, "val_mean_acc", "batch_size")
            log = {"val_loss": val_loss, "val_mean_acc": val_mean_acc}

        self.log_dict(log, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        batch_size = x.size(0)
        y_logits = self.forward(x)

        # 2. Compute loss
        test_loss = F.cross_entropy(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_acc1": acc1, "test_acc5": acc5}
        
        if self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            acc_5 =self.accuracy_5(y_logits, y)
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_acc1": acc_1, "test_acc5": acc_5}
        else:
            mean_acc = self.mean_acc(y_logits, y)
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_mean_acc": mean_acc}
        
        return results

    def test_epoch_end(self, outs):
        test_loss = self.__weighted_mean(outs, "test_loss", "batch_size")
        if self.metric == 'accuracy_1_5':
            test_acc1 = self.__weighted_mean(outs, "test_acc1", "batch_size")
            test_acc5 = self.__weighted_mean(outs, "test_acc5", "batch_size")
            log = {"test_loss": test_loss, "test_acc1": test_acc1, "val_acc5": test_acc5}
        
        if self.metric == 'accuracy_1_5':
            test_loss = torch.stack([x["test_loss"] for x in outs]).mean()
            test_acc_1 = torch.stack([x["test_acc1"] for x in outs]).mean() 
            test_acc_5 = torch.stack([x["test_acc5"] for x in outs]).mean()   
            log = {"ptl/test_loss": test_loss, "ptl/test_accuracy_1": test_acc_1, "ptl/test_accuracy_5": test_acc_5}
        
        else:
            test_mean_acc = self.__weighted_mean(outs, "test_mean_acc", "batch_size")
            log = {"test_loss": test_loss, "test_mean_acc": test_mean_acc}

        self.log_dict(log, sync_dist=True)

    def configure_optimizers(self):
        # parameters = list(self.parameters())
        # trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.task == 'finetune':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        else:
            #optimizer = LBFGS(self.parameters(), lr=self.lr )
            #optimizer = SGD(self.classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
            optimizer = AdamW(self.classifier.parameters(), lr=self.lr, betas=(0.9,0.999), weight_decay=self.weight_decay)

        #optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        
        if self.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler == 'reduce':
            #scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=20, factor=0.5,),  "monitor": "ptl/val_loss"}
            return [optimizer], [scheduler]        
        else: 
            return optimizer
            

    def __accuracy_at_k(self, outputs, targets, top_k=(1, 5)):
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def __weighted_mean(self, outputs, key, batch_size_key):
        value = 0
        n = 0
        for out in outputs:
            value += out[batch_size_key] * out[key]
            n += out[batch_size_key]
        value = value / n
        return value.squeeze(0)

class DownstreamLinearModule(pl.LightningModule):
    def __init__(
        self, 
        backbone_weights: str,
        num_classes: int,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        scheduler: str,
        metric: str ,
        task: str ,
        lr_decay_steps: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_weights = backbone_weights
        self.num_classes = num_classes
        self.max_epochs = epochs
        self.batch_size = batch_size
        self.lr = lr *batch_size/256
        print("learning Rate of", self.lr)
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.metric = metric
        self.task = task
        self.scheduler = scheduler
        self.__build_model()

        ## Plugin Modules
        self.loss_module = nn.CrossEntropyLoss()
        self.mean_acc = Accuracy(num_classes=num_classes, average='macro')
        self.accuracy_5= Accuracy(top_k=5)
        self.accuracy= Accuracy()

    # def __build_model(self):

    #     # 1. Backbone
    #     backbone = resnet50()
    #     backbone.fc = nn.Identity()
    #     #backbone.load_from_checkpoint(self.backbone_weights).pk
    #     state = torch.load(self.backbone_weights, map_location=torch.device('cpu'))#["state_dict"]

    #     for k in list(state.keys()):
    #         if "backbone" in k:
    #             state[k.replace("backbone.", "")] = state[k]
    #         del state[k]
    #     backbone.load_state_dict(state, strict=False)
    #     self.feature_extractor = backbone

    #     if hasattr(backbone, "inplanes"):
    #         feature_dim = backbone.inplanes
    #     else:
    #         feature_dim = backbone.num_features

    #     if self.task == 'linear_eval' or self.task=="ImageNet_linear":
    #         for p in self.feature_extractor.parameters():
    #             p.requires_grad = False
    #     else:
    #         for p in self.feature_extractor.parameters():
    #             p.requires_grad = True

    #     # 2. Classifier
    #     self.fc = nn.Linear(feature_dim, self.num_classes)

    #     # 3. Display how many trainable parameters
    #     parameters = list(self.parameters())
    #     trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
    #     print(f'Trainable parameters : {len(trainable_parameters)}')
    #     print(f'Total parameters : {len(parameters)}')
    
    # def forward(self, x):
    #     # 1. Feature extraction
    #     #x = self.feature_extractor(x)
        
    #     if self.task=="linear_eval" or self.task=="ImageNet_linear": 
    #         self.feature_extractor.eval()
    #         with torch.no_grad():
    #             representations = self.feature_extractor(x)#.flatten(1)
    #     else: 
    #         representations = self.feature_extractor(x)#.flatten(1)
        
    #     #representations=self.feature_extractor(x).flatten(1)
            
    #     # x= self.classifier(representations)
    
    #     # return x


    #     # 2. Classification
    #     #x = self.fc(x)
        
    #     x = self.fc(representations)
    #     x=F.softmax(x, dim=1)
    #     return x



    def __build_model(self):

        # 1. Backbone
        backbone = resnet50(pretrained=True)
        ## Loading weight Configure
        state = torch.load(self.backbone_weights)["state_dict"]  
        for k in list(state.keys()):
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)


        ## ---------- Method 2 -------------
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
   
        # linear model
        self.classifier = nn.Linear(2048, self.num_classes)

 
        # # 3. Display how many trainable parameters
        # parameters = list(self.parameters())
        # trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        
        # print(f'Total parameters : {len(parameters)}')
        # if self.task == 'linear_eval':
        #     print(f'Encoder Trainable parameters : {len(trainable_parameters)}')
        #     print(f'Linear Trainable parameters : {self.fc.parameters()}')
       

    def forward(self, x):
        ##----Method 1 --------
        # 1. Feature extraction
        # x = self.feature_extractor(x)
        # # 2. Classification
        # x = self.fc(x)
        ## ----Method 2 ------
        if self.task=="finetune": 
            representations = self.feature_extractor(x).flatten(1)
        else: 
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        #representations=self.feature_extractor(x).flatten(1)
            
        x  = self.classifier(representations)
        #x=F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # if self.task == 'linear_eval':
        #     self.feature_extractor.eval()
        
        # else:
        #     self.feature_extractor.train(True)
        # self.feature_extractor.eval()
        # 1. Forward pass
        x, y = batch
        #y=y.softmax(dim=1)

        #print(y.shape)
        # print(x.shape)
        #y_logits = self.forward(x)
        y_logits = self.forward(x)
        print(f"the ground truth label {y[0]}")
        print(f"The prediction with softmax {y_logits[0].argmax(dim=-1)}")

        # 2. Compute loss
        train_loss = F.cross_entropy(y_logits, y)
        #train_loss=self.loss_module(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            log = {"train_loss": train_loss, "train_acc1": acc1, "train_acc5": acc5}
            # acc=(y_logits.argmax(dim=-1) == y).float().mean()#
            # self.log_dict("train_acc", acc, on_step=False, on_epoch=True)
            # self.log_dict("train_loss", train_loss, prog_bar=True)
        elif self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            log = {"train_loss": train_loss, "train_acc1": acc_1,}
        else:
            mean_acc = self.mean_acc(y_logits, y)
            log = {"train_loss": train_loss, "train_mean_acc": mean_acc}
        
        self.log_dict(log, on_epoch=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        batch_size = x.size(0)
        y_logits = self.forward(x)
        #y_logits = self.forward(x)

        # 2. Compute loss
        val_loss = F.cross_entropy(y_logits, y)
        #val_loss=self.loss_module(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
           
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}
            # acc= (y==y_logits).float().mean
            # self.log_dict("val_acc", acc)
        if self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            acc_5 =self.accuracy_5(y_logits, y)
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_acc1": acc_1,"val_acc5": acc_5, }
        else:
            mean_acc = self.mean_acc(y_logits, y)
            results = {"batch_size": batch_size, "val_loss": val_loss, "val_mean_acc": mean_acc}
        
        return results

    def validation_epoch_end(self, outs):
        
        val_loss = self.__weighted_mean(outs, "val_loss", "batch_size")
        
        if self.metric == 'accuracy_1_5':
            val_acc1 = self.__weighted_mean(outs, "val_acc1", "batch_size")
            val_acc5 = self.__weighted_mean(outs, "val_acc5", "batch_size")
            log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.metric == 'accuracy_1_5_torchmetric':
            avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
            avg_acc_1 = torch.stack([x["val_acc1"] for x in outs]).mean()
            avg_acc_5 = torch.stack([x["val_acc5"] for x in outs]).mean()

            log = {"ptl/val_loss": avg_loss, "ptl/val_accuracy_1": avg_acc_1,"ptl/val_accuracy_5": avg_acc_5}
        
        else:
            val_mean_acc = self.__weighted_mean(outs, "val_mean_acc", "batch_size")
            log = {"val_loss": val_loss, "val_mean_acc": val_mean_acc}

        self.log_dict(log, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # 1. Forward pass
        x, y = batch
        batch_size = x.size(0)
        y_logits = self.forward(x)

        # 2. Compute loss
        test_loss = F.cross_entropy(y_logits, y)

        # 3. Compute accuracy:
        if self.metric == 'accuracy_1_5':
            acc1, acc5 = self.__accuracy_at_k(y_logits, y, top_k=(1, 5))
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_acc1": acc1, "test_acc5": acc5}
        
        if self.metric == 'accuracy_1_5_torchmetric':
            acc_1 = self.accuracy(y_logits, y)
            acc_5 =self.accuracy_5(y_logits, y)
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_acc1": acc_1, "test_acc5": acc_5}
        else:
            mean_acc = self.mean_acc(y_logits, y)
            results = {"batch_size": batch_size, "test_loss": test_loss, "test_mean_acc": mean_acc}
        
        return results

    def test_epoch_end(self, outs):
        test_loss = self.__weighted_mean(outs, "test_loss", "batch_size")
        if self.metric == 'accuracy_1_5':
            test_acc1 = self.__weighted_mean(outs, "test_acc1", "batch_size")
            test_acc5 = self.__weighted_mean(outs, "test_acc5", "batch_size")
            log = {"test_loss": test_loss, "test_acc1": test_acc1, "val_acc5": test_acc5}
        
        if self.metric == 'accuracy_1_5':
            test_loss = torch.stack([x["test_loss"] for x in outs]).mean()
            test_acc_1 = torch.stack([x["test_acc1"] for x in outs]).mean() 
            test_acc_5 = torch.stack([x["test_acc5"] for x in outs]).mean()   
            log = {"ptl/test_loss": test_loss, "ptl/test_accuracy_1": test_acc_1, "ptl/test_accuracy_5": test_acc_5}
        
        else:
            test_mean_acc = self.__weighted_mean(outs, "test_mean_acc", "batch_size")
            log = {"test_loss": test_loss, "test_mean_acc": test_mean_acc}

        self.log_dict(log, sync_dist=True)

    def configure_optimizers(self):
        # parameters = list(self.parameters())
        # trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.task == 'finetune':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        else:
            #optimizer = LBFGS(self.parameters(), lr=self.lr )
            #optimizer = SGD(self.classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
            optimizer = AdamW(self.classifier.parameters(), lr=self.lr, betas=(0.9,0.999), weight_decay=self.weight_decay)

        #optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        
        if self.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler == 'reduce':
            #scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=20, factor=0.5,),  "monitor": "ptl/val_loss"}

            return [optimizer], [scheduler]        
        else: 
            return optimizer
            

    def __accuracy_at_k(self, outputs, targets, top_k=(1, 5)):
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
    def __weighted_mean(self, outputs, key, batch_size_key):
        value = 0
        n = 0
        for out in outputs:
            value += out[batch_size_key] * out[key]
            n += out[batch_size_key]
        value = value / n
        return value.squeeze(0)
        