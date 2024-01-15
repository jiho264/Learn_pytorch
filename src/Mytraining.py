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
        device="cuda",
        logs=None,
        file_path=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.device = device
        self.logs = logs
        self.file_path = file_path
        self.valid_loss = 0.0

    def Forward_train(self, dataset):
        # Training loop @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm.tqdm(dataset):
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

        train_loss = running_loss / len(dataset)
        train_acc = correct / total

        self.logs["train_loss"].append(train_loss)
        self.logs["train_acc"].append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        return

    def Forward_eval(self, dataset, test=False):
        self.model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataset:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_loss /= len(dataset)
        valid_acc = correct / total

        if test == True:
            self.logs["test_loss"].append(valid_loss)
            self.logs["test_acc"].append(valid_acc)
            print(f"Test  Loss: {valid_loss:.4f} | Test Acc: {valid_acc*100:.2f}%")
        else:
            self.valid_loss = valid_loss
            self.logs["valid_loss"].append(self.valid_loss)
            self.logs["valid_acc"].append(valid_acc)
            print(
                f"Valid Loss: {self.valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%"
            )

        return

    def Save(self, file_path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "logs": self.logs,
        }

        torch.save(checkpoint, "logs/" + file_path + ".pth.tar")
        print(f"Saved PyTorch Model State to [logs/{file_path}.pth.tar]")

    def SingleEpoch(self, train_dataloader, valid_dataloader, test_dataloader=None):
        self.Forward_train(train_dataloader)
        self.Forward_eval(valid_dataloader)
        if test_dataloader != None:
            self.Forward_eval(test_dataloader, test=True)

        # Save the model (checkpoint) and logs
        self.Save(self.file_path)
        # Learning rate scheduler
        self.scheduler.step(self.valid_loss)

        return
