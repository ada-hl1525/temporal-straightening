# Project Understanding Notes

这份笔记是给自己看的，用来理解我们现在的项目到底在做什么、结果说明了什么、以及 final report / viva 应该怎么讲。

## 1. 项目最初想做什么

最初的 project plan 想研究：

> AI-generated videos 是否能保持基本物理规律？

原计划里有三类物理行为：

- object permanence / occlusion：物体被遮挡后是否还保持身份和轨迹；
- gravity-driven motion：物体是否按照重力连续运动；
- collision-induced motion change：碰撞后运动方向和速度是否合理改变。

原计划的方法是：先生成视频，再人工打 physical-consistency score，然后用 pretrained visual embeddings 看 latent trajectory 里有没有对应的变化。

## 2. 为什么后来改成 simulated pendulum

我们实际尝试 AI-generated toy videos 后发现一个问题：

> AI 生成的视频本身太不可控。

比如提示它生成 wrong physics，它不一定真的生成我们想要的错误；有时候 correct 和 wrong 都很模糊，有时候视频只是视觉上奇怪，但不是明确的物理错误。

所以如果直接分析这些 AI 视频，很难判断：

- latent trajectory 的异常来自物理错误；
- 还是来自生成模型没有听懂 prompt；
- 或者只是视频画面质量差。

因此我们把主实验改成 controlled simulated videos。这样 correct / wrong physics 是人为设定的，ground truth 更可靠。

这不是偏离原计划，而是让原计划里的 evaluation framework 更可控、更科学。

## 3. 现在项目的核心问题

现在的研究问题可以理解成：

> 在可控物理视频中，pretrained vision encoders 的 latent trajectory 是否能反映物理一致性差异？

更具体地说：

- correct physics 和 wrong physics 的 latent trajectory 是否不同；
- 哪些 trajectory metrics 最有用；
- 哪些 encoder 更敏感；
- feature map / attention map 是否能解释模型关注了哪里。

## 4. 当前数据是什么

主数据集是 simulated pendulum videos。

当前主要结果来自：

- 48 个视频；
- 9 个 encoder；
- 432 条 encoder-video evaluation rows；
- 4968 个 pairwise trajectory comparisons。

视频包括：

- single pendulum correct；
- single pendulum wrong；
- double pendulum correct；
- double pendulum wrong。

wrong physics 类型包括：

- energy gain；
- periodic kick；
- joint kick；
- reverse gravity；
- zero gravity；
- overdamping；
- midpoint impulse。

这些 wrong cases 都是可控的，不是靠 AI prompt 碰运气生成的。

## 5. evaluation pipeline 在做什么

每个视频会被均匀采样成若干帧。每一帧输入 vision encoder，得到一个 embedding。

所以一个视频会变成：

```text
frame_1, frame_2, ..., frame_T
      ↓
z_1, z_2, ..., z_T
```

也就是 latent space 里的一条 trajectory。

然后我们分析这条 trajectory 的形状和变化。

## 6. 主要 trajectory metrics 是什么

### Straightness

straightness 看的是：

> 从第一帧到最后一帧的直接距离，占整条路径长度的比例。

如果 trajectory 很直，straightness 高；如果绕来绕去，straightness 低。

但 pendulum 本来就是周期运动，所以 correct video 的 latent trajectory 也可能弯曲或回环。因此 straightness 不是最可靠的物理指标。

### Curvature

curvature 看的是：

> 连续 latent movement 的方向变化有多大。

它可以反映轨迹是否弯曲，但同样受周期运动影响，所以也不是最稳定。

### Mean Step Distance

mean step distance 看的是：

> 相邻两帧 embedding 平均变化多大。

这是目前最有用的指标之一。wrong physics 往往会导致帧与帧之间 latent movement 更大。

### Std Step Distance

std step distance 看的是：

> frame-to-frame movement 是否稳定。

如果视频里有 sudden kick、gravity change、impulse，latent movement 会出现 spike，因此 std step distance 会变大。

## 7. 当前最重要的结果

最核心结论：

> Wrong-physics videos tend to produce larger and less stable frame-to-frame movement in latent space.

换句话说，物理错误的视频在 latent trajectory 里更容易出现跳变、不稳定、变化幅度变大。

目前最有用的指标是：

- mean_step_distance；
- std_step_distance。

straightness 和 curvature 没那么可靠。

## 8. Encoder 对比怎么理解

从当前结果看，DINOv2 family 最强。

按 mean step distance 的 AUC：

- DINOv2-base: 0.802；
- CLIP-B/32: 0.785；
- DINOv2-small: 0.769。

按 wrong-minus-correct mean step distance delta：

- DINOv2-base: +4.527；
- DINOv2-small: +4.491；
- DINOv2-large: +4.478。

所以可以说：

> DINOv2 gives the strongest and most consistent response to controlled physical violations.

CLIP 也有信号，尤其 CLIP-B/32。

MAE 最弱，说明不是所有 pretrained image encoders 都适合做 temporal physical consistency analysis。

## 9. Wrong Type 结果怎么理解

最容易被 latent metrics 捕捉到的是 abrupt changes，例如：

- single periodic kick；
- single impulse at midpoint；
- double impulse at midpoint；
- double joint kick；
- double reverse gravity。

这些错误会造成明显的运动突变，所以 latent step distance 会变大。

更平滑的错误，比如 energy gain 或 overdamping，可能不容易被简单 embedding metric 捕捉。

这说明我们的方法目前更擅长检测：

> temporal discontinuity / abrupt motion anomaly

而不一定能稳定检测所有细微的 physical implausibility。

## 10. Pairwise Separation 图怎么理解

pairwise analysis 不是只看单个视频的 metric，而是比较不同视频 trajectory 之间的距离。

它有三种 pair：

- correct-correct：两个正确视频之间的正常差异；
- correct-wrong：正确视频和错误视频之间的差异；
- wrong-wrong：两个错误视频之间的差异。

如果 correct-wrong 明显大于 correct-correct，说明 wrong physics 的 latent trajectory 超出了正常物理变化范围。

当前结果大致支持这一点，但分布还有 overlap。

所以要谨慎表述：

> The signal is useful as a diagnostic indicator, but not yet a complete classifier.

## 11. Feature Map / Attention Map 在看什么

`feature_attention_examples.tar.gz` 是 qualitative case study。

它里面有：

- sampled frames；
- feature-change maps；
- attention maps。

feature-change map 看的是：

> 哪些图像区域的 patch-level feature 在相邻帧之间变化最大。

attention map 看的是：

> encoder 的 class token 比较关注哪些 patch。

目前的 qualitative montage 可以说明：

- feature changes 经常出现在 pendulum bob / rod 附近；
- DINOv2 和 CLIP 有可视化 attention；
- SigLIP 有 feature-change map，但当前没有保存 attention overlay。

这些图不能单独证明模型“理解物理”，但可以作为辅助解释：

> latent trajectory 的变化不是完全随机的，它至少在一些 case 中和运动物体区域有关。

## 12. Report 应该怎么讲主线

最终 report 的逻辑应该是：

1. 原始问题：video generation 需要 physical consistency evaluation。
2. 初步尝试：AI-generated toy videos 可以跑通 pipeline，但物理标签不可靠。
3. 方法调整：使用 controlled simulated pendulum videos 建立可靠 benchmark。
4. 主实验：用多个 pretrained visual encoders 抽 frame embeddings。
5. 分析：把每个视频看成 latent trajectory。
6. 结果：wrong physics 通常导致更大的 frame-to-frame latent movement。
7. encoder comparison：DINOv2 最稳定，CLIP 也有效，MAE 较弱。
8. qualitative evidence：feature/attention maps 支持部分空间解释。
9. limitation：目前主要覆盖 gravity/dynamics，还没有完整覆盖 collision 和 occlusion。
10. future work：扩展到 simulated collision、occlusion，再迁移回 AI-generated videos。

## 13. Viva 可以怎么说

可以这样说：

> My original plan was to evaluate whether generated videos preserve simple physical regularities, such as gravity, collision, and object permanence. During the project, I found that AI-generated videos were too noisy and did not always follow the prompts, so the physical labels were not reliable. I therefore refined the project by introducing controlled simulated videos. This allowed me to create correct and deliberately wrong physical dynamics, and then test whether pretrained vision encoders show different latent trajectories for these cases.

然后接：

> The main finding is that wrong-physics videos tend to produce larger and less stable frame-to-frame movement in latent space. This is especially clear for mean step distance and step-distance variability. DINOv2 models show the strongest response, while MAE is much weaker. This suggests that latent trajectory analysis can be a useful diagnostic tool, although it is not yet a complete physical reasoning evaluator.

## 14. 现在还缺什么

为了更贴近原 project plan，最自然的补充是：

### Collision

生成两个球碰撞的视频：

- correct elastic collision；
- no reaction after contact；
- wrong bounce direction；
- energy gain after collision。

这对应原计划里的 collision-induced motion change。

### Occlusion

生成小球或摆锤经过遮挡板的视频：

- correct reappearance；
- disappear behind occluder；
- reappear at wrong location；
- change identity/color after occlusion。

这对应原计划里的 object permanence。

这两个不需要做成很大。可以作为 smaller extension，让 report 能完整呼应原计划的三类物理行为。

## 15. 一句话总结

这个项目现在可以总结为：

> We started from evaluating physical consistency in generated videos, but moved to controlled simulated videos to obtain reliable ground truth. Using latent trajectories from pretrained vision encoders, we found that physically wrong dynamics usually produce larger and less stable frame-to-frame representation changes, especially in DINOv2 and CLIP. This supports latent trajectory analysis as an interpretable diagnostic tool for physical consistency, while also showing that collision and occlusion should be added as future controlled benchmarks.

