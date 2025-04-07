# 声源定位

## 项目结构
```plaintext
project/
├── SpeechSSLdata.py          # 生成声源定位数据集
├── load_data.py              # 数据加载模块
├── max_corr_backbone.py      # 模型主干网络
├── train.py                  # 模型训练脚本
├── eval.py                   # 模型测试脚本
└── SRP.py                    # SRP对比方法
```

## 使用方法
1. 生成数据集
2. 训练模型
3. 测试模型效果
4. 对比方法
