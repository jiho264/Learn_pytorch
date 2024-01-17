import matplotlib.pyplot as plt


class LogViewer:
    def __init__(self, logs):
        self.logs = logs
        pass

    def draw(self, start=0, range=999999):
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        if range == 999999:
            range = len(self.logs["train_loss"])
            # 첫 번째 그래프: Training and Test Loss
            axs[0].plot(self.logs["train_loss"][start:], label="Training Loss")
            axs[0].plot(self.logs["valid_loss"][start:], label="Validation Loss")
            axs[0].plot(self.logs["test_loss"][start:], label="Test Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("Training, Validation and Test Loss")
            axs[0].legend()

            # 두 번째 그래프: Training and Test Accuracy
            axs[1].plot(self.logs["train_acc"][start:], label="Training Accuracy")
            axs[1].plot(self.logs["valid_acc"][start:], label="Validation Accuracy")
            axs[1].plot(self.logs["test_acc"][start:], label="Test Accuracy")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].set_title("Training, Validation and Test Accuracy")
            axs[1].legend()

            # 그래프를 보여줍니다.
            plt.tight_layout()
            plt.show()

        elif range != 999999 and start + range < len(self.logs["train_loss"]):
            # 첫 번째 그래프: Training and Test Loss
            axs[0].plot(self.logs["train_loss"], label="Training Loss")
            axs[0].plot(self.logs["valid_loss"], label="Validation Loss")
            axs[0].plot(self.logs["test_loss"], label="Test Loss")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("Training, Validation and Test Loss")
            axs[0].set_xlim([start, start + range])
            axs[0].legend()

            # 두 번째 그래프: Training and Test Accuracy
            axs[1].plot(self.logs["train_acc"], label="Training Accuracy")
            axs[1].plot(
                self.logs["valid_acc"],
                label="Validation Accuracy",
            )
            axs[1].plot(self.logs["test_acc"], label="Test Accuracy")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Accuracy")
            axs[1].set_title("Training, Validation and Test Accuracy")
            axs[1].set_xlim([start, start + range])
            axs[1].legend()

            # 그래프를 보여줍니다.
            plt.tight_layout()
            plt.show()
        else:
            print("Out of range")

    def print_len(self):
        print("Num of train logs : ", len(self.logs["train_loss"]))
        print("Num of train logs : ", len(self.logs["train_acc"]))
        print("Num of valid logs : ", len(self.logs["valid_loss"]))
        print("Num of valid logs : ", len(self.logs["valid_acc"]))
        print("Num of test  logs : ", len(self.logs["test_loss"]))
        print("Num of test  logs : ", len(self.logs["test_acc"]))
        print("Num of lr    logs : ", len(self.logs["lr_log"]))

    def print_all(self):
        if len(self.logs["valid_loss"]) == 0 and len(self.logs["test_loss"]) != 0:
            for i in range(len(self.logs["train_loss"])):
                print(
                    f"{i+1} epoch: train_loss={self.logs['train_loss'][i]:.4f}, train_acc={self.logs['train_acc'][i]:.4f}, test_loss={self.logs['test_loss'][i]:.4f}, test_acc={self.logs['test_acc'][i]:.4f}, lr={self.logs['lr_log'][i]:.4f}"
                )
        elif len(self.logs["valid_loss"]) != 0 and len(self.logs["test_loss"]) == 0:
            for i in range(len(self.logs["train_loss"])):
                print(
                    f"{i+1} epoch: train_loss={self.logs['train_loss'][i]:.4f}, train_acc={self.logs['train_acc'][i]:.4f}, valid_loss={self.logs['valid_loss'][i]:.4f}, valid_acc={self.logs['valid_acc'][i]:.4f}, lr={self.logs['lr_log'][i]:.4f}"
                )
        else:
            for i in range(len(self.logs["train_loss"])):
                print(
                    f"{i+1} epoch: train_loss={self.logs['train_loss'][i]:.4f}, train_acc={self.logs['train_acc'][i]:.4f}, test_loss={self.logs['test_loss'][i]:.4f}, test_acc={self.logs['test_acc'][i]:.4f}, valid_loss={self.logs['valid_loss'][i]:.4f}, valid_acc={self.logs['valid_acc'][i]:.4f}, lr={self.logs['lr_log'][i]:.4f}"
                )
