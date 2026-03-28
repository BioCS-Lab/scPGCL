import numpy as np

def inspect_npy(file_path):
    try:
        # 尝试不带 allow_pickle 加载
        data = np.load(file_path)
        print(f"文件 {file_path} 加载成功！")
        print("数据类型:", type(data))

        if isinstance(data, np.ndarray):
            print("这是一个 numpy.ndarray")
            print("数据形状:", data.shape)
            print("数据 dtype:", data.dtype)
            print("前5条数据:\n", data[:5])
        else:
            print("未知的数据结构:", type(data))

    except ValueError:
        # 如果出错，说明可能是字典或其他对象
        print("直接加载失败，尝试使用 allow_pickle=True ...")
        data = np.load(file_path, allow_pickle=True).item()
        print("数据类型:", type(data))
        if isinstance(data, dict):
            print("这是一个字典，包含以下键:")
            for k in data.keys():
                print(f"  - {k} : {type(data[k])}, 形状: {getattr(data[k], 'shape', None)}")
        else:
            print("不是字典，内容:", data)

if __name__ == "__main__":
    file_path = r"C:\Users\Administrator\PycharmProjects\pythonProject\ReGCL-main\embedding\SCoPE2_Specht_embedding_final.npy"  # 改成你的路径
    inspect_npy(file_path)
