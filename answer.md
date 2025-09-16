对你的回复：

输入图像大小是多少？是否都是类似 128×128 或 256×256？

【512*512，但数据集我是金字塔卷积核的方式从细节到整体地采集数据，所以有128*128的，也有256*256的，并在我打标记的时候都reshape为512*512，我觉得这算是简化了全局图像分辨率大小不一的简单解决方式？】

你希望输出固定长度轨迹（如40点）还是支持变长？

【固定长度，40个点，且模型只负责预测38个点，起终点需要和输入的一样（也就是固定），我希望以此帮助模型更快找到较优解】

是否有预期引入哪些辅助输入通道（如SDF、梯度、A*成本图等）？

【我没设想过，你可以帮我研究一下是否值得引如这些嘛？我是用A*算法生成路径，但算法本身精度不高容易穿墙，你可以研究看看引入什么比较好，或者不引入】

对解码方式有偏好吗？是否偏好自回归式轨迹生成，还是一次性预测全部路径点？

【能够在3050 8GB上训练即可，我比较喜欢一次性预测全部路径点，但如果你发现自回归比较好，那就尝试一下】

是否期望模型支持多样性路径生成（如min-of-K）？

【你研究看看好不好用，这个你决定】

你倾向采用单阶段训练，还是三阶段结构（教师-学生-偏好微调）？

【我希望精度高，所以你研究哪个精度高就用哪个，不过（教师-学生-偏好RL微调）可能会更好？因为我目前有看到一些吧diffusion和RL结合的模型，所以我觉得可以尝试CVAE+RL微调，也就是有三个选择：1.CVAE+RL微调 2.CVAE+教师-学生-偏好RL微调 3.纯CVAE+RL微调】

rob@rob-OT:~/code/SwinPath_BiCVAE_framework$ tree
.
├── checkpoints
├── csv_output_dir
├── data_cfg.yaml
├── loginfo
├── main.py
├── models
├── readme.md
├── requirements.txt
└── src
    ├── backbones
    │   ├── c_attn_decoder.py
    │   ├── cvae_core.py
    │   ├── deform_cross_attn.py
    │   ├── dpo.py
    │   ├── lantent_diffusion.py
    │   ├── posenc.py
    │   ├── refine_attn.py
    │   ├── rnn.py
    │   ├── swin_unet_encoder.py
    │   ├── utils.py
    │   └── vae.py
    ├── configs.py
    ├── data_loader.py
    ├── loss.py
    ├── model
    │   ├── model.py
    │   ├── perception.py
    ├── trainer.py
    └── utils
        ├── functions.py
        ├── logger.py
        ├── teacher.py
        └── val_aug.py

