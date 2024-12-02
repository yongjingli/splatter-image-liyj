import tarfile

import chardet

def extract_tar_gz_with_error_handling(file_path):
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            for member in tar.getmembers():
                try:
                    tar.extract(member)
                except tarfile.TarError as e:
                    print(f"解压文件 {member.name} 时出错: {e}")
        print("文件解压完成，部分错误已跳过。")
    except FileNotFoundError:
        print("文件不存在。")
    except tarfile.TarError as e:
        print(f"解压过程出现严重错误: {e}")

def extract_tar_gz_with_error_handling2(file_path):
    buffer_size = 8192  # 可根据实际情况调整缓冲区大小
    try:
        with open(file_path, 'r', buffering=buffer_size) as file:
            with tarfile.open(fileobj=file, mode="r:gz") as tar:
                for member in tar.getmembers():
                    try:
                        tar.extract(member)
                    except tarfile.TarError as e:
                        print(f"解压文件 {member.name} 时出错: {e}")
        print("文件解压完成，部分错误已跳过。")
    except FileNotFoundError:
        print("文件不存在。")
    except tarfile.TarError as e:
        print(f"解压过程出现严重错误: {e}")


def unzip_objaverse():
    file_path = "views_release.tar.gz"
    # extract_tar_gz_with_error_handling(file_path)
    extract_tar_gz_with_error_handling2(file_path)

def check_encode():
    file_path = "views_release.tar.gz"
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        print(result['encoding'])


if __name__ == "__main__":
    print("Start")

    # tar --ignore-failed-read -xf archive.tar
    # -x：表示解压缩（extract）。
    # -f：后面跟着要解压的文件名。
    # --ignore - failed - read：在读取文件时，如果遇到错误，继续处理后续文件，而不是立即退出。

    # tar --ignore-failed-read -xf archive.tar 2>&1 | grep -i "error"
    # 2>&1：将标准错误输出（stderr）重定向到标准输出（stdout），这样错误信息将会与正常输出一起显示。
    # | grep -i "error"：通过管道将输出传递给 grep，并只显示包含 "error" 的行。-i 选项使得搜索不区分大小写。

    # sudo apt-get install p7zip-full
    # 7z x xxx.tar.gz

    # 在 7z 命令中，“-y” 选项（代表 “yes”）用于在所有查询（例如覆盖现有文件等情况）时都自动回答 “是”，并且在遇到错误时尝试继续解压。
    # 例如，你可以使用 “7z x -y large_file.tar.gz -o target_directory” 命令。这里 “x” 是解压操作，“-y” 强制继续，“-o target_directory” 指定解压后的输出目录。
    # 不过要注意，使用 “-y” 选项可能会导致一些问题被掩盖。例如，如果文件损坏非常严重，继续解压可能会导致程序出现其他异常行为，或者生成的文件可能无法正常使用

    # gzip -d
    # https://www.runoob.com/linux/linux-comm-gzip.html
    # unzip_objaverse()
    check_encode()
    print("End")
