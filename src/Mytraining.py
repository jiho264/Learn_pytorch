import torch
import tqdm


class DoTraining:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scaler=None,
        scheduler=None,
        earlystopper=None,
        device="cuda",
        logs=None,
        file_path=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.earlystopper = earlystopper
        self.device = device
        self.logs = logs
        self.file_path = file_path

    def Forward_train(self, dataloader):
        # Training loop @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(dataloader, desc="train"):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(dataloader)
        train_acc = correct / total

        return train_loss, train_acc

    def Forward_eval(self, dataloader, test=False):
        self.model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            if "MyResNet34" in str(self.model.named_modules):
                for images, labels in tqdm.tqdm(dataloader, desc="eval"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    eval_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            else:
                for images, labels in dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    eval_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

        eval_loss /= len(dataloader)
        eval_acc = correct / total

        return eval_loss, eval_acc

    def Save(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "earlystopper": self.earlystopper.state_dict(),
            "logs": self.logs,
        }

        torch.save(checkpoint, "logs/" + self.file_path + ".pth.tar")
        # print(f"Saved PyTorch Model State to [logs/{file_path}.pth.tar]")

        return

    def SingleEpoch(
        self, train_dataloader, valid_dataloader=None, test_dataloader=None
    ):
        if valid_dataloader == None and test_dataloader == None:
            raise ValueError("No any valid/test dataloader")

        train_loss, train_acc = self.Forward_train(train_dataloader)
        self.logs["train_loss"].append(train_loss)
        self.logs["train_acc"].append(train_acc)

        self.logs["lr_log"].append(self.optimizer.param_groups[0]["lr"])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        if valid_dataloader != None:
            valid_loss, valid_acc = self.Forward_eval(valid_dataloader)
            self.logs["valid_loss"].append(valid_loss)
            self.logs["valid_acc"].append(valid_acc)
            print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%")

        if test_dataloader != None:
            test_loss, test_acc = self.Forward_eval(test_dataloader, test=True)
            self.logs["test_loss"].append(test_loss)
            self.logs["test_acc"].append(test_acc)
            print(f"Test  Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        ############################################################################

        # Save the model (checkpoint) and logs
        # self.Save(self.file_path)

        if valid_dataloader != None:
            eval_loss = valid_loss
        if valid_dataloader == None and test_dataloader != None:
            eval_loss = test_loss

        # Learning rate scheduler
        if self.scheduler.__class__ == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler.step(eval_loss)
        elif self.scheduler.__class__ == torch.optim.lr_scheduler.MultiStepLR:
            _tmp_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            if _tmp_lr != self.optimizer.param_groups[0]["lr"]:
                print(
                    "Learning Rate has changed : Now is",
                    self.optimizer.param_groups[0]["lr"],
                )

        return eval_loss
