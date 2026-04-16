# SAM3

# 4.16
## debug
### bug1
nn.MultiheadAttention 内部有 self.out_proj = NonDynamicallyQuantizableLinear，这是一个子模块，ZeRO-3会分别跟踪它。当 activation checkpoint recompute 时，ZeRO-3 的 submodule hook 没有正确释放。

修改：把所有 MultiheadAttentionWrapper 替换成自定义的纯 nn.Linear 实现，完全绕开 nn.MultiheadAttention。


### bug2
因为box/points等geometry prompt我们用不到，如果直接设置为空会报错dtype不对，所以将每一个geometry的入口都做一个提前的对齐：
- points_direct_project(points) 改为先对齐到 self.points_direct_project.weight.dtype
- points_pos_enc_project(enc) 改为先对齐到 self.points_pos_enc_project.weight.dtype
- boxes_direct_project(boxes) 改为先对齐到 self.boxes_direct_project.weight.dtype
- boxes_pos_enc_project(enc) 改为先对齐到 self.boxes_pos_enc_project.weight.dtype

### bug3
在model_misc.py中in_proj_weight 把 Q、K、V 合并投影，但 cross attention 时 key 的输入可能维度不同（kdim != embed_dim）。需要分开Q、K、V 的投影权重。

# 4.15

发现一个多分割目标的数据集：
https://github.com/jdg900/MMR

如果后面要拓展多target的分割可以拿来用

## v2的大量数据训练
加入SAM3encoder，训练配置为：
- 192k数据，refcoco按照50%去采样，其余数据占剩下50%
- 保守学习率，冻结SAM3Encoder、QwenEncoder、训练SAM3neck、LLM、SAM3decoder
- 在refcocoeval上进行测试
- batchsize为64，image_min_pixels为768，epoch为1
- loss为1的text+1的seg，seg为2的BCE+0.5的Dice


因为采样这个机制，refcoco大概12w条数据只有96000条被采样到了，所以这一个epoch只能过一遍而且没过完全。

得到的ciou大约为0.6x，通过分割可视化可以看到

### 一些好的样本：

![00006](https://github.com/user-attachments/assets/fbd7126b-56ce-4d5b-9b07-7b02c6158ab8)

![00041](https://github.com/user-attachments/assets/c3b78015-ac5f-40dd-b256-fa9540225a79)

### 一些欠分割的样本：

![00005](https://github.com/user-attachments/assets/d6ea2cfe-1e8c-4e49-b0d2-069914c1948a)
![00038](https://github.com/user-attachments/assets/0230f8b0-c7d2-40b7-beca-460811ee9001)

### 一些过分割的样本：

![00008](https://github.com/user-attachments/assets/44815f8b-5ed0-4962-bc54-6dae725e39f3)
![00007](https://github.com/user-attachments/assets/b9eae2c7-18e1-4c1d-9c3f-cff179b04e83)


### 训练曲线：

<img width="963" height="772" alt="image" src="https://github.com/user-attachments/assets/e83b2cd5-c755-4e76-ae22-19c365b3b170" />

## v2的一些小测试

因为这个ciou太拉跨了（可能是因为模型变更加复杂，而且SAM3的Encoder和QWEN的Encoder不一样，所以需要多训才能让targetProjector和decoder充分发挥性能，但是目前的训练量有点少）。

所以就做了个测试：只修改数据量和epoch（50个）保持其他所有训练配置不变：
- 只训练一个refcoco样本，复制64张作为一个batch，最后也在这一张图上进行eval，但是ciou没眼看（0.2）
- 然后修改了BCE权重到0.5，Dice权重到2（就是互换了权重），发现segloss降低得更快,然后ciou暴涨（但还是很低，0.6）

所以从这个得出结论：BCE大的话，mask基本都龟缩在GT的内部，调高DICE反而能减少这种现象。

但是ciou上不去，claude也说模型没啥问题。




# 4.11
## image feature失真情况
做了一下imagefeature在
- QWen原始模型ViT pre-merger（dim=1024）
- QWen模型微调后ViT pre-merger（dim=1024）
- decoder输入的 projected（dim=256）
三个状态处的可视化（PCA）

<img width="377" height="1056" alt="image" src="https://github.com/user-attachments/assets/b880240f-95a5-4e0c-93b6-b10db08a4554" />

**计算方式为：以原始的qwen模型的premerger图像特征为基准，计算余弦相似度loss（orig是没有微调的qwen的premerger）**

通过30个sample的评估，三个状态处的loss为：

<img width="387" height="128" alt="image" src="https://github.com/user-attachments/assets/17b67c82-9425-49da-92d1-2aff19f5271b" />

所以直接拿矩阵投影的失真很严重

不过可以通过三个状态处的特征可视化图发现目前projector是可以保持图中实例的大部分特征，但是边缘和实例中心不够饱满

<img width="185" height="261" alt="image" src="https://github.com/user-attachments/assets/aadd8fb0-657d-4ac1-b340-16c53864ea4b" />

<img width="186" height="254" alt="image" src="https://github.com/user-attachments/assets/1ee8f6fb-37fc-4a2b-a618-443248aa6f53" />

这个右下角的人的肚子都是空白的


## 分割目标注意力可视化
还在做，但是能够阶段性地发现模型注意不到/注意错分割目标（不过这个好像就是我们能够改进的点和创新点）


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
