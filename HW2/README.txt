机器学习与数据挖掘 · 第二次大作业

先打开《第二次大作业-题面.pdf》。请解压整个压缩包，并保留 data 文件夹结构。

建议使用 Python 3.11 或 3.12。在解压目录打开终端，运行：

    python -m pip install -r requirements.txt
    python starter.py

从教师处领取个人题码，再导出对应的四张图。将下面的 P001 替换成自己的题码：

    python starter.py --code P001

starter.py 会训练原始像素的逻辑回归基线，并在 demo_images 目录保存示例图。
个人题运行命令会额外保存本人的两对图，并打印相应样本编号、真实类别和背景变化类型。
在读入模板基础上补齐六组对照、重构图、个人实验和测试预测导出。

数据均为教师提供的合成灰度图。类别 0 表示横向划痕，类别 1 表示纵向划痕。
数组无需解压为单张图片。读入函数返回 NumPy 数组：

    train.npz / val.npz:
        images: (1800,16,16) / (400,16,16)，uint8
        ids:    每张图的唯一编号
        labels: 每张图的类别 0 或 1
    test.npz:
        images: (400,16,16)，uint8
        ids:    每张图的唯一编号
    personal_pairs.npz:
        codes:   (300,)，题码 P001 至 P300
        images:  (300,2,2,16,16)，按题码、配对、图像 A/B 排列
        ids:     (300,2,2)，图像编号
        labels:  (300,2)，每对图共用一个真实标签
        changes: (300,2)，brightness 表示亮度变化，texture 表示纹理变化

训练使用每张图的 256 个像素值。用 starter.pixels 把像素转换为 [0,1] 范围并展平。
图片编号用于对照与提交。个人题及测试图只在模型训练完成后用于评估。
测试标签由教师保管，提交 test_predictions.csv 后由教师计算测试错误率。
