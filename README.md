# SAM3


# 4.20
## SAM3训练问题
看到github上一个项目是 https://github.com/Sompote/SAM3_LoRA 里面是拿COCO数据集微调的，一共有21个category，每一张图片都是多个GT mask。它这个数据没有单GT的，也没考虑这种想法。

而且SAM3的数据

## Projector失真问题
能不能加进去一个将一个targettoken（2048维）信息分离成几乎无损、多维的、适配SAM3 Language space的多个token 序列（256维）。因为SAM3的textEncoder就支持不定长度的输入，所以SAM3的decoder实际上接受的是一堆feature向量去分割，而并非仅限于一个。

### HYPERSEG
类似于这样

<img width="661" height="783" alt="image" src="https://github.com/user-attachments/assets/a98cd220-91ac-4d53-a80a-159f9f6dd1ef" />

<img width="236" height="246" alt="c7346108-3173-4f4f-9aab-a9865bdd9e6a" src="https://github.com/user-attachments/assets/c4251a6a-f54c-4496-b87e-ab65ec2967da" />

## 为什么自回归
让多个Projector将targettoken硬生生分离出来256维度的token，不论加上什么限制，都没有办法解决语义坍缩的问题。

这个问题已经在第一轮debug中验证，让target token直接映射到8个256维token去，反而会将target token密集的视觉语义信息丢失（从8改到1的时候，seg_loss将近减少了一倍）

自回归模型能够看之前生成的token去生成下一个，比如例子“a boy handing a cup on the chair”，这一个代表“boy”的phrase会被LLM浓缩到一个2048维度的向量去（类似于一张图片），然后一个小的decoder就会读取这个“图片”，去生成离散的特征：[boy] [chair] [cup] [on the chair]（这里的on the chair就是空间关系的语义信息，并非是生硬的一个phrase）。我们可以设置一个max length，让这个decoder读取从LLM出来的target，去生成这些特征，然后喂入SAM3。

这样不仅符合SAM3原生的那种textembedding模式，而且尽可能保留了信息。

## 数据或者自回归方式？
### 重头训一个小decoder（AI估计要增加的参数为5M-10M）
可以从已有的CoA数据中让AI提纯<target>之前描述目标的完整phrase，比如

```markdown
"segment the man on the left of the woman" the left man<target> -> [the left man] -> [man] [on the left] | length = 2
"who may be the soccer player in the image" the man wearing the soccer shoes<target> -> [man wearing soccer shoes] -> [man] [wearing] [soccer shoes] | length = 3

```
只需要训练这个映射的stage2到stage3就行，先训得差不多直接接上去再连这SAM3的decoder一起训就行

### 直接从LLM中提取
有点像：我感觉本质上类似让MLLM生成target_0到16，然后分别过一个proj[旺柴]

让模型生成不定长度的<feature>在<target>后？

这样的好处是不用再训一个单独的模块，但是数据成本差不多




# 4.17
## SAM3最大bug今日出现
在之前的训练中忽略了SAM3原生的多mask分割机制，这一块既没有冻结也没有训练，没有冻结是因为梯度会在这些模块中传播，没有训练是因为没有loss去监督，导致这个模块会被随机化。因为这个bug导致昨晚的BCE和DICE所有的模型的ciou几乎都为0。这个分割机之中有200个query slots，分别对不同

## bug详解
虽然是说既没有被训，也没有被冻结，但是实际上的bug比这个还要抽象。在训练的过程中，我们实际上启用了第0个query，然后关闭了其他199个query，所以能够发现seg_loss和能够显著降低，并且训练曲线看起来也比较正常。但是在训练完后开启inference后，所有样本的ciou几乎为0，只有1-2个的ciou稍微高一点。

这个bug就是因为我们在训练的时候，实际上只训练了1一个queryhead，所以在我们的数据集上只有这个queryhead有用。但是在推理的时候，200个queryhead全部开启，其中只有一个是学习过我们数据的，这样以来推理的过程就是一个中奖率为$0.005$的一个抽奖环节，要是抽到了我们的queryhead则ciou高，抽不到就直接瞎分割。从mask的可视化可以看出来：

<img width="1028" height="334" alt="image" src="https://github.com/user-attachments/assets/140a6523-70d5-4bb8-87da-7bc2b40b9ce1" />

这次的GT应该是运动员，但是抽到的query刚好是对足球敏感的（SAM3预训练过的），经过<target_1> token的干扰就分割了人的脚和足球

所以总结一下这个是针对单目标分割时候的优化不到位，有种头疼砍头的效果。


## query到底是怎么回事
1. 每个query是一个256维的可学习向量（query_embed），预训练时已经学会了不同语义的表示。
2. 在 Decoder 里200 个 query 同时和图像特征做 cross-attention：query × 图像特征 → 每个 query 学到"自己负责的区域在哪里"
3. 打分（dot_prod_scoring）text prompt × query → 相似度分数（pred_logits）分数最高的 query 被选中，输出它对应的 mask。

核心：query 不需要重新训练，只需要让 dot_prod_scoring 学会"text prompt → 选哪个 query"这个映射。



## query的forward

### 在model_builder.py中的decoder构建
<img width="258" height="426" alt="image" src="https://github.com/user-attachments/assets/9e830ea0-0a8c-40cc-9363-f62a04b42b7c" />

### 在sam3_image.py中传入decoder参与运算
<img width="769" height="624" alt="image" src="https://github.com/user-attachments/assets/4350d239-a148-4a45-b252-4f95da9e9065" />

### 在decoder.py中参与cross-Attention的forward
<img width="622" height="553" alt="image" src="https://github.com/user-attachments/assets/956b113f-487f-483a-9709-b00624b618b7" />

### 在sam3_image.py中经过前序处理成为一个objectfeature
<img width="822" height="854" alt="image" src="https://github.com/user-attachments/assets/8c7ce3cf-632e-48e9-89bd-028b68b95aa0" />

### 在maskformer_segmentation.py中与图片pixel-level做点积相似度得到mask
<img width="755" height="645" alt="image" src="https://github.com/user-attachments/assets/958b6701-bba1-49ca-b0bf-fcaaea3d79ef" />

## query的backward
在forward到maskSegmentation后，模型产出三种loss：第一种是maskloss（BCE+Dice），第二种是IoUloss，第三种是scoringloss。实际上IoUloss由maskloss计算而来，但是maskloss不参与query的监督，IoUloss和scoringlos参与query的backward。

1. Mask Loss（主要）
  casam_loss.py:507-516：
  with torch.no_grad():
      iou = inter / (union + 1e-6)   # [N_q]
      best_q = iou.argmax().item()   # 选 IoU 最高的 query
  然后对 best_q 对应的 mask logit 计算 BCE + Dice loss。梯度从 mask loss 反传，经过 segmentation_head → decoder →
  query_embed，更新 query 权重。

2. Scoring Loss（辅助）
  casam_loss.py:548-558：
  target = torch.zeros_like(score_logits)
  target[best_q_idx] = best_iou.clamp(0, 1)   # soft label = IoU 值
  loss_cls = F.binary_cross_entropy_with_logits(score_logits, target, reduction="mean")
  loss = loss + 0.1 * loss_cls_total / num_masks
  pred_logits 由 DotProductScoring（dot_prod_scoring）计算：query hidden state 和 text prompt 做 dot product。这个 loss
  教模型"哪个 query 和当前 text prompt 最匹配"。

4. 训练流程一句话

query_embed的200个向量通过 decoder 的 cross-attention 与图像和文本交互，产生 mask 和置信度；mask loss 监督 mask质量，scoring loss 监督 query 选择；两个 loss 的梯度共同更新 query_embed 权重，使每个query逐渐专注于不同类型的语义区域。
  
## debug思路
发现了训练完的模型ciou直接接近0，分析是inference的时候**targettoken选错了/权重保存错误/训练目标搞错了**，然后问claude发现有一个这个query机制，于是加入了Fakequery（query的最终目标）和finalquery（200个真实的query）的监测，发现Fakequery的ciou还真挺高的，但是finalquery的几乎都为0。 于是通过查代码发现我们的训练措施有误，然后修改了query的个数，让全部的query都参与训练。并且增加了监督query的scoringloss和Iouloss。经过同样的训练发现finalciou（真实的分割效果）翻了一倍，Fakeciou（理想的分割效果）也达到了和之前使用QwenEncoder的同样水平（除去极难样本）。

<img width="685" height="71" alt="image" src="https://github.com/user-attachments/assets/7805d1f8-3c28-4cab-9dac-e03a219a361b" />
## debug效果

### debug之前
<img width="1165" height="630" alt="image" src="https://github.com/user-attachments/assets/591b6531-fdc4-4bf8-a7e7-e83272041d9c" />


### debug之后
<img width="1014" height="598" alt="image" src="https://github.com/user-attachments/assets/fc71f9f9-c5d7-4df2-bc86-80cb21218377" />

现在的finalciou涨了将近3倍（从0.1->0.3）


## 一些反思
### 可以只用 1 个 query 吗？
  技术上可以，在 forward() 里强制只取 query 0：
  pred_masks = out["pred_masks"][:, 0:1]  # 只用 query 0
  但这样做的问题：
  1. 预训练权重失效：SAM3 的 200 个 query 是预训练好的，每个 query 的 query_embed 权重代表不同的语义先验。强制只用 query
   0，相当于丢弃了 199 个预训练好的语义槽位，从零开始训练 query 0
  2. decoder 结构依赖 200 个 query：decoder 的 cross-attention 是 200 个 query 同时和图像特征交互，减少 query 数量会改变
   attention 的分布，影响特征提取质量
  3. 之前已经验证过：我们之前固定用 query 0 训练，ciou 只有 0.028，正是因为 query 0 不一定是最适合当前任务的 query
  正确做法：
  保留 200 个 query，但只监督 1 个（IoU matching 选出的那个）。让 dot_prod_scoring 学会在推理时也选到同一个
  query。这样既利用了预训练的语义先验，又适配了我们的单目标任务。

### 训练过程中发现loss降低不同于之前冻结query的时候（textloss极端不稳定到最后却极速收敛，segloss下降速度极快过了几个step就几乎为0.0x），能否分开训？
你的诊断很准确。症状分析：

  - text_loss 突然升高：LLM 和 seg_head 的梯度互相干扰
  - grad_norm 极小但 cIoU 不理想：模型收敛到了一个局部最优，query_embed 没有被有效监督
  - 根本原因：target_proj 还没学好时，query 通过 cross-attention 收到的 text signal 是噪声，query 学不到有意义的分布

  两阶段方案

  阶段 1（~500-1000 steps）：冻结 query_embed，只训练 target_proj + transformer

  目标：让 target_proj 先学会产生有意义的 language_features，让 encoder fusion 适配。

  阶段 2（正式训练）：解冻 query_embed，降低 LR，联合训练

  此时 text prompt 已经稳定，query 能通过 cross-attention 学到真实的语义分布。


### 能否减少到20个query使用？
  技术上可以，num_queries=200 是 model_builder.py:179 的参数，改成 20 即可。

  但有个根本问题

  SAM3 的预训练权重里，query_embed 是 Embedding(200, 256)。如果改成 20，预训练权重无法加载，这 20 个 query
  从随机初始化开始训练，等于放弃了 SAM3 预训练的所有先验知识。

  我们的场景需要多少 query？

  我们是单目标分割（每次只找一个 <target_1>），理论上 1 个 query 就够。但：

  - 200 个 query 里，SAM3 预训练已经让不同 query 专注于不同语义类型（人、物体、背景等）
  - IoU matching 会自动找到最合适的那个 query
  - 减少 query 数量不会加速太多（瓶颈在 ViT encoder，不在 decoder query 数量）

  结论

  不建议减少。原因：
  1. 破坏预训练权重加载
  2. 性能收益几乎为零（decoder 计算量相对 ViT 很小）
  3. 200 个 query 的多样性对 IoU matching 有帮助

  如果显存是问题，更有效的方式是减少 LANG_SEQ（当前 16）或冻结更多 SAM3 模块。
# 4.16

## 问题
- [x]1. 对BCE和Dice来说，能否拿之前的QwenVisionEncoder版尝试配比？因为我感觉现在SAM3和QWEN的两个Encoder不一样，这个loss的配比没有办法参考只有Qwen的，但是目前的训练时间比较长，Qwen更高效一点。
2. 

## 训练
刚刚已经全部debug完成，运行了训练，参数为下：
- GLOBAL_BATCH_SIZE=32，BATCH_PER_DEVICE=1，GRAD_ACCUM_STEPS=4
- 学习率：总4e-5，merger5e-5，seghead（SAM3decoder和target_projector）5e-5
- 数据量为256k，和之前0.725的QwenVisionEncoder版对齐
- SAM3权重从Huggingface上申请到了，SAM3的ViT已经换成了ViT-L
- $loss=1.0*l_{text}+1.0*l_{seg}$,$l_{seg}=1.0*l_{BCE}+1.0*l_{Dice}$(均衡一点试试，到时候拿小数据跑出来最佳配比）
- 像素最大最小都为448

<img width="303" height="22" alt="image" src="https://github.com/user-attachments/assets/cb1e18f4-22c5-4618-a71a-259865c5dff2" />

## debug

## bug-让SAM3的torch的分布式适配deepspeed
### bug0
主要问题就是SAM3的很多权重以及接口没有办法适配deepspeed的格式。如果要硬接就会对SAM3进行大改，相对来说AI的幻觉要更多，所以解决方案围绕在保留SAM3源码尽可能不改动，只改动一些小的模块，不适配deepspeed就不适配吧，只要修改到能跑就行。所以对SAM3权重还是按照pytorch的读取方式，除了以下三个bug之外，主要都是ZeRo3的切片问题（好修），以及SAM3权重读取和deepspeed不兼容的问题，这两个问题都通过模仿LISA解决了。


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
