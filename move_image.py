import os

dir = ['55','60','65','70','80']
def separate_image(folder,without_folder, with_folder):
    for i in dir:
        path = os.path.join(folder,i)
        for filename in os.listdir(path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                input_file_path = os.path.join(path, filename)
                if filename.startswith('p'):
                    output_file_path = os.path.join(with_folder, filename)
                else:
                    output_file_path = os.path.join(without_folder, filename)
                os.rename(input_file_path, output_file_path)


if __name__ == "__main__":
    pass