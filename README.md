<img width="186" height="254" alt="image" src="https://github.com/user-attachments/assets/87e4c8bb-1d0a-48fa-80c6-ef562e08b056" /># SAM3
# 4.11
## image feature失真情况
做了一下imagefeature在
- QWen原始模型ViT pre-merger（dim=1024）
- QWen模型微调后ViT pre-merger（dim=1024）
- decoder输入的 projected（dim=256）
三个状态处的可视化（PCA）
<img width="377" height="1056" alt="image" src="https://github.com/user-attachments/assets/b880240f-95a5-4e0c-93b6-b10db08a4554" />

**计算方式为：以原始的qwen模型的premerger图像特征为基准，计算余弦相似度loss（orig是没有微调的qwen的premerger）**
通过30个sample的评估，
三个状态处的loss为：
<img width="387" height="128" alt="image" src="https://github.com/user-attachments/assets/17b67c82-9425-49da-92d1-2aff19f5271b" />

所以直接拿矩阵投影的失真很严重

不过可以通过三个状态处的特征可视化图发现目前projector是可以保持图中实例的大部分特征，但是边缘和实例中心不够饱满
<img width="185" height="261" alt="image" src="https://github.com/user-attachments/assets/aadd8fb0-657d-4ac1-b340-16c53864ea4b" />
<img width="186" height="254" alt="image" src="https://github.com/user-attachments/assets/1ee8f6fb-37fc-4a2b-a618-443248aa6f53" />
这个右下角的人的肚子都是空白的



## 分割目标注意力


```markdown
 你的模型目前存在的问题

  一、CosSim 可视化仍然接近全红 — 说明 projector 没有学到有效的空间区分

  看 0008_features.jpg 和 0014_features.jpg 的 CosSim 列：
  - Orig(1024d) 和 FT(1024d) 的 CosSim 几乎全红，说明在高维空间里，所有 token 和 GT 中心的余弦相似度差异极小。这意味着 ViT 的 pre-merger 特征主要编码的是全局语义而不是空间区分性 — 所有位置的 token
  在语义上都很"像"。
  - Proj(256d) 的 CosSim 稍有变化但仍然很红，projector 只引入了非常微弱的空间区分。

  二、L2 Norm 反转 (norm_corr_ft_proj = -0.175) — projector 编码方式与原始特征对立

  统计数据显示 mean_norm_corr_ft_proj = -0.175，标准差极小(0.004)，说明这个反转是系统性的，不是偶然：
  - 微调前后的 norm 高度相关 (0.847)
  - 但经过 projector 后 norm 完全反转

  这意味着 projector 学到了一种"反转编码"：原来激活强的区域经过投影后反而变弱。从 0014_features.jpg 的 L2Norm 列可以看到：Orig 和 FT 的 norm 分布类似（蓝色=高激活区大致对应物体），而 Proj 的 norm
  分布明显不同。

  三、Target Token 的语义定位能力分化严重

  从 target_sim 图看：

  0014（"mom"）— 比较好的例子：
  - CosSim overlay 显示 target token 在 256d 空间中确实大致指向了人物区域（overlay 中人物区域偏绿/暖色）
  - 预测 mask（红色）覆盖了大部分人物区域
  - 但 CosSim 阈值化后（Row 2 左下红色区域）仍然很散，和 GT 重合度不高

  0020（"left guy"）— 很差的例子：
  - cos_corr_ft_proj = 0.239，projector 几乎完全破坏了空间结构
  - 预测 mask（红色）覆盖的区域偏右，和 GT（绿色，左边的人）有显著偏差
  - CosSim 阈值化后的红色区域分散在全图，target token 没有精确定位到目标

  0008（"little girl"）— 中等：
  - 预测 mask（红色/粉色）大致落在小女孩区域但边界不准
  - 目标较小时分割效果更差

  四、Grad-CAM 显示梯度分散 — 模型不知道该"看"哪里

  从 0008_gradcam_fft.jpg 和 0014_gradcam_fft.jpg：
  - GradCAM pre-proj(1024d)：梯度散布在整个图上，没有聚焦于目标区域
  - GradCAM post-proj(256d)：稍好一些（0014 中有模糊的人形轮廓），但远不如理想中应该看到的"目标区域高亮"效果
  - GradCAM overlay：叠加到原图后可以看到梯度热点和目标区域只有部分重叠

  这说明 seg_head 的 mask_decoder 在做预测时，没有准确聚焦到正确的空间位置，梯度信号分散。

  五、FFT 频谱分析显示 projector 引入高频噪声

  从 0014_gradcam_fft.jpg 的 Row 2（FFT 行）：
  - Orig 和 FT 的 FFT 频谱类似，以低频为主（中心亮）
  - Proj 的 FFT 频谱明显更分散，高频分量增多 — 验证了之前的"空心效应"假说：projector 强化了边缘（高频）但丢失了内部填充（低频）

  六、核心结论

  ┌─────────────────────┬────────────────┬────────────────────────────────────┐
  │        指标         │       值       │                含义                │
  ├─────────────────────┼────────────────┼────────────────────────────────────┤
  │ cos_corr(orig→ft)   │ 0.893 ± 0.017  │ 微调对 ViT 空间结构影响较小 ✓      │
  ├─────────────────────┼────────────────┼────────────────────────────────────┤
  │ cos_corr(ft→proj)   │ 0.733 ± 0.231  │ projector 造成严重且不稳定的失真 ✗ │
  ├─────────────────────┼────────────────┼────────────────────────────────────┤
  │ cos_corr(orig→proj) │ 0.627 ± 0.243  │ 累积失真更大 ✗                     │
  ├─────────────────────┼────────────────┼────────────────────────────────────┤
  │ norm_corr(ft→proj)  │ -0.175 ± 0.004 │ 系统性 norm 反转 ⚠                 │
  └─────────────────────┴────────────────┴────────────────────────────────────┘

  根本问题不在微调（orig→ft 保持了 0.89 的一致性），而在 projector（ft→proj 降到 0.73 且方差巨大 0.23）。

  projector（vision_proj: 1024→256→256 的两层 MLP+GELU）是瓶颈：
  1. 维度压缩太激进（1024→256，4倍压缩），丢失空间信息
  2. 学到了边缘偏好（BCE+Dice 梯度集中在边界导致），表现为高频噪声增多、物体内部空心
  3. 空间一致性不稳定（标准差 0.23），对不同图像/不同位置的目标，投影质量差异很大

  建议的改进方向：
  - 增大 projector 中间维度（如 1024→512→256），减缓信息压缩
  - 加入 spatial-aware loss（如 feature matching loss）约束投影前后空间结构一致性
  - 或者参考 SAM2 的 FPN 式多尺度融合，不要一步到位压缩
```
# 4.10
效果
![00002](https://github.com/user-attachments/assets/efcec69a-9b7d-4de9-99dc-2ecdf706afc2)
![00018](https://github.com/user-attachments/assets/6918d37e-e19c-4982-8a1f-e41f81c97cf3)
![00004](https://github.com/user-attachments/assets/eaa86011-639e-4290-bbe5-fd1f84b858ed)

![12347](https://github.com/user-attachments/assets/a48411e3-6a0e-4405-8097-d5210dd38d99)
![12349](https://github.com/user-attachments/assets/0b49b0bf-0407-4260-b84e-1fa9003350cc)
![12345](https://github.com/user-attachments/assets/0002ad81-382c-4e7c-bde5-d6fdc78682cd)
![12348](https://github.com/user-attachments/assets/0d90cd13-ecb0-4b25-ba08-062164b84e44)
![12346](https://github.com/user-attachments/assets/34862a6e-4ff5-4b08-b10b-cf233ef8da70)
![12350](https://github.com/user-attachments/assets/d1bfdf93-cc15-488a-bfc1-d896764c79ba)


# pipeline
``` markdown
模型架构
                    ┌──────────────────────────────────────────────┐
                    │                输入样本                       │
                    │ image + text(prompt / <anchor> / <target_1>) │
                    └──────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────────┐
                    │     Qwen3VLVisionModelWithIntermediate       │
                    │ patch_embed + pos_embed + vision blocks      │
                    └──────────────────────────────────────────────┘
                           │                              │
                           │                              │
                           │ pre-merger last_hidden_state │ post-merger merger输出
                           │ 高分辨率视觉token            │ 与LLM图像占位对齐
                           ▼                              ▼
                _casam_image_embeds_pre           _casam_image_embeds
                           │                              │
                           │                              └────→ 送入Qwen LLM做多模态生成
                           │
                           ▼
                ┌──────────────────────────────────────────────┐
                │                CasamSegHead                  │
                │ vision_proj + vision_norm                    │
                │ target_proj                                  │
                │ SAM3 PromptEncoder                           │
                │ SAM3 MaskDecoder                             │
                └──────────────────────────────────────────────┘
                    │                 │                 │
                    │                 │                 │
                    │                 │                 └─ no_mask_embed → dense_prompt
                    │                 └─ target token hidden → text_prompt
                    └─ pre-merger image tokens → image_embeddings
                                      │
                                      ▼
                         SAM3 MaskDecoder / TwoWayTransformer
                                      │
                                      ▼
                           low-res masks → 上采样 → 插值回原图
                                      │
                                      ▼
                                  分割 mask
  训练损失: L_total = L_text + L_anchor + L_seg(Focal + Dice)
```

# 4.9
- [ ] 混合ade20k||cocostuff||mapillary||paco_lvis||pascal_part||refcoco||refcocog||refcoco+一共360000条进行3个epoch训练

config：上调image_min_pixels（256->512）。
昨天使用240000条ade20k||cocostuff||mapillary||paco_lvis||pascal_part数据混合训练一个epoch，giou提升0.06，ciou提升0.2（相对resize mask+只在refcoco上训）。

![00006](https://github.com/user-attachments/assets/773c6206-f03a-472b-a7c7-c364e51906de)
![000061](https://github.com/user-attachments/assets/7314a8da-f107-474d-aa78-801f01368c9f)

上图为mask-resize
下图为图像-resize

# 4.8工作

编译好了flash-Attention，训练时间缩短三分之一

claude推荐使用一个pixel shuffle，ablation看一下效果。这个是走SAM3内部自带的high_res_features，从visionEncoder上采样提取特征（上采样地很猛）。
```markdown
 MaskDecoder内部：
  image_embedding + sparse_prompt → TwoWayTransformer交叉注意力
      → ConvTranspose2d 4x上采样
      → + high_res_features (这里才和PixelShuffle的输出相加)
      → mask预测
```
- 4.9 暂时先不ablation，增加数据规模增加训练轮数看看能否合格。

# 下一步的ablation

- [ ] loss：尝试boundary loss
- [ ] anchorloss的pre-merger和post-merger

# debug
## decoder的input

SAM3的decoder需要输入两种feature
- text prompt
- image feature

QWen的vision encoder处理图片后，进行merger从而提高计算效率降低开销。

之前的bug是提取Post-merger的图像特征，但是这样发现在同样训练参数下loss会升高。

更新：修改成了pre-merger的图像特征。

## 训练过程中loss的计算方式
### anchorcosineloss
anchorcosineloss是从某句话的回答中提取出<anchor\_1>位置，查询到HiddenState，去和经过visionEncoder的图像的真实anchor区域pooling

之前的bug是在LLM层中提取anchorfeature，再和<anchor\_1>这一token的hiddenstate对齐（这俩实际上是一个东西），

更新：当backward到anchorloss的时候，读取anchor的mask并且计算anchor区域，在Post-merger后的imagefeature中做这一区域的池化计算，和<anchor_\1>token的Hiddenstate做cosine，目的是让LLM里的<anchor\_1>token 的hiddenstate去靠近视觉区域语义。

- [ ] TODO：对比mask的vision feature使用post/pre-merger谁能带来更好的focus和seg效果？

### segloss
segloss目前的计算公式为：

$$segloss= 2.0*loss_{BCE}+0.5*loss_{Dice}+0.5*loss_{IoU}$$

之前的bug是直接使用了逐像素IoU作为loss，发现不对劲就删除了。

更新：目前的IoUloss是SAM3原生的用于提高mask分割质量的一个loss（非主导loss），这个在SAM3的forward中负责选出多个mask中最佳的那一个

- [x] Ablation：IoUloss是否有用 没有用

### boundary loss
Claude给我推荐：如果分辨率低的话，可以试试boundary loss，可以做一下Ablation

## ZeRo3的切片问题
ckpt的保存和最终权重保存会崩溃，提示权重为(\[0,\])，后面直接让AI修了也不报错了。


## 数据集问题
<target_1>这个token出现的位置太靠前，比如在这一个中：

```json
"conversations": [
      {
        "from": "human",
        "value": "<image>\nSegment the sink that is to the bottom left of the doorknob."
      },
      {
        "from": "gpt",
        "reasoning": "I first find the doorknob<anchor_1>, then I look for the sink<target_1> to its bottom left.",
        "value": "It's sink<target_1>."
      }
    ]
```
虽然tgt_1确实跟在sink后并且传到decoder中，但是忽视了后续的“to its bottom left”，从而导致自回归的时候就丢失了这一信息，tgt_1的输出实际上是有缺陷的。


```markdown
  024×768图片输入示例

  ┌──────┬───────────────────────┬─────────────────────────────────────────────────────────┐
  │ 步骤 │         操作          │                       输入 → 输出                       │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 1    │ smart_resize          │ 1024×768 → 1008×756                                     │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 2    │ PatchEmbed (14×14)    │ 1008×756 → 72×54网格 = 3888 tokens, 1024维              │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 3    │ ViT Block 3,6,9,12    │ 缓存4份 [3888, 1024]                                    │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 4    │ ViT Block 24 (最后层) │ last_hidden_state [3888, 1024]                          │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 5    │ PatchMerger (2×2)     │ [3888, 1024] → [972, 2048]，网格36×27                   │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 6    │ LLM处理               │ 972 visual + N text tokens → hidden_states [B, S, 2048] │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 7    │ vision_proj           │ [3888, 1024] → [1, 256, 72, 54]                         │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 8    │ target_proj           │ [1, 2048] → [1, 256]                                    │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 9    │ PixelShuffle s0 (4x)  │ [3888, 2048] → [1, 32, 288, 216]                        │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 10   │ PixelShuffle s1 (2x)  │ [3888, 2048] → [1, 64, 144, 108]                        │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 11   │ MaskDecoder           │ 72×54 → 上采样 → 288×216                                │
  ├──────┼───────────────────────┼─────────────────────────────────────────────────────────┤
  │ 12   │ bilinear              │ 288×216 → 768×1024（原图尺寸）                          │
  └──────┴───────────────────────┴─────────────────────────────────────────────────────────┘
```
# 新设计
<anchor\_1>到<anchor\_8>特殊token用于标记anchor，让模型学习到anchor特征；<target\_1>用于标记分割目标，提取目标hiddenstate用于分割。

$$loss = w_{text}*TextLoss+ w_{anchor}*AnchorCosineLoss + w_{seg}*TargetSegLoss(BCE+Dice+IoU)$$

# 在refcoco上进行Train和eval

## v1效果
![](./version1_example2.jpg)
![](./version1_example.jpg)


## v2效果
![](./version2_example1.jpg)
![](./version2_example2.jpg)
![](./version2_example3.jpg)

# anchor-target simple example
```json
{
    "image": "images/coco2017train/000000522418.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nSegment the sink that is to the bottom left of the doorknob."
      },
      {
        "from": "gpt",
        "reasoning": "I first find the doorknob<anchor_1>, then I look for the sink to its bottom left<target_1>.",
        "value": "It's sink<target_1>."
      }
    ],
    "masks": {
      "anchor": [
        {
          "token": "<anchor_1>",
          "id": "lvis:25333"
        }
      ],
      "target": [
        {
          "token": "<target_1>",
          "id": "lvis:25334"
        }
      ]
    }
  }
```
mask通过\[DatasetName:ID\]映射查询

这个样例数据过于简单，还需增强
