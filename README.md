# verl_learn

verl (HybridFlow) RLHF 框架学习资料整理

## 内容

### 1. 论文资源

- **HybridFlow_RLHF_Framework.pdf** - HybridFlow 论文原版 (EuroSys 2025)
- **verl_intro_chinese.html** - verl 中文介绍博客文章

### 2. PPO训练流程动画

- **ppo_training_animation.html** - PPO训练流程动画演示（基础版）
- **ppo_training_animation_detailed.html** - PPO训练流程动画（详细版，对应源码）

动画功能：
- 可视化PPO训练9个步骤
- 显示函数调用链和源码行号
- Worker架构联动展示
- DataProto数据流转可视化

### 3. 源码架构走读指导

- **verl_architecture_guide.md** - verl源码架构与走读指南

包含：
- 目录结构总览
- 核心架构图
- 推荐走读顺序（5个阶段）
- 核心类和函数速查表
- PPO训练流程详解
- 关键设计模式
- 配置系统介绍
- 扩展指南

## 使用方法

1. 打开 `ppo_training_animation.html` 或 `ppo_training_animation_detailed.html` 查看动画
2. 阅读 `verl_architecture_guide.md` 了解架构
3. 参考 `HybridFlow_RLHF_Framework.pdf` 理解理论基础

## 相关链接

- [verl GitHub](https://github.com/verl-project/verl)
- [HybridFlow Paper (arxiv)](https://arxiv.org/abs/2409.19256)
- [DeepWiki verl 文档](https://deepwiki.com/verl-project/verl)