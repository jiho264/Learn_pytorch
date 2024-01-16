import os


def clear_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


# directory = "/home/lee/Desktop/Learn_pytorch"

clear_directory(os.path.join(directory, "logs"))
clear_directory(os.path.join(directory, "models"))
