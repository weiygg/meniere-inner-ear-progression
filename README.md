# 梅尼埃病多模态疾病进展建模

本项目实现横断面数据驱动的 pseudo-temporal P-EBM 工作流。第一轮仅执行数据/文件审计、患者-耳朵队列、分割 QC、内耳形态学与半规管几何提取，以及官方 P-EBM 模拟复现。横断面结果不得解释为真实自然病程、因果顺序或状态转移概率。

## 运行

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config\project_config.example.yaml config\project_config.yaml
# 编辑 config\project_config.yaml，填写本机的私有数据路径
.\run_all.ps1
```

全部路径从 `config/project_config.yaml` 读取，Python 代码使用 `pathlib`。原始临床表、图像和 mask 只读；中间结果、正式结果和日志分别写入 `results_md_progression/intermediate`、`results_md_progression/final` 和 `results_md_progression/logs`。

## 第一轮输出

- `01_data_audit/`：文件清单、临床数据审计、变量字典、缺失数据、重复/冲突及患者-耳朵关联。
- `02_cohort/`：patient-level index-ear 主队列、all-ear 敏感性队列及队列构建审计。若单/双侧状态未确认，两个 unilateral 队列 CSV 仅保留表头，并在审计表中标记为未构建。
- `03_morphometry/`：分割 QC、三平面 montage、基础三维形态、半规管中心线/管径/曲率/扭率、平面拟合、管间夹角、结构间距离、双侧内在差异和失败记录。
- `04_pebm/`：固定官方仓库版本的串行/同时事件模拟复现与患者 stage 概率归一化测试。
- `NEED_CONFIRMATION.md`：触发停止条件的问题清单。
- `first_round_summary.md`：首轮 10 项控制台/文件汇总。

## 当前停止规则

只要 `NEED_CONFIRMATION.md` 仍含阻断项，真实临床 P-EBM、临床-影像融合、anatomical endotype 和最终 bootstrap 均不运行。当前分割目录的 `subNNN` 与两个临床来源中的数值 ID 不能唯一对应，且 index ear、编码方向、事件正常/异常参考组及权威分割批次尚未全部确认。

官方 P-EBM 源码固定在 `results_md_progression/intermediate/vendor/pebm`，项目通过包装器调用未修改的上游 `EventOrder_pebm`。复现报告记录 branch、commit、Python/NumPy 版本和兼容性说明。

## 2026-07-31 丽水训练—浙二外部验证试点

用户确认了新的中心划分：`data/丽水-xjj内耳分割4.rar` 为丽水影像训练库（200人、400耳），其余压缩包为浙二外部影像；临床 Excel 第1张 `丽水` 和第3张 `浙二` 分别作为内部与外部临床表。

```powershell
.\.venv\Scripts\python.exe src\22_audit_reorganized_data.py --config config\project_config.yaml --data-dir data
.\.venv\Scripts\python.exe src\23_run_clinical_pebm_validation.py --config config\project_config.yaml
```

试点结果位于 `results_md_progression/final/clinical_pebm_external_validation_20260731/`。该分析以“唯一存在 AAO-HNS 分期的耳”为受累耳代理，以配对另一耳为参考代理；这不等同于确认健康耳。模型只在丽水拟合，浙二不重拟合混合分布、事件顺序或阈值。400耳是影像训练库分母，完整临床 P-EBM 子集为丽水56人/112耳，外部验证为浙二94人/188耳。

当前结果仍属于横断面 pseudo-temporal 探索，不能解释为真实自然病程或因果顺序。浙二外部 AUC 接近随机水平，因此不支持稳定的跨中心受累耳区分，也不替代对临床编码、双侧状态和外部人工分割参考的进一步确认。

## GitHub 仓库范围与隐私

本仓库仅保存可复现代码、测试、依赖声明和脱敏配置模板。临床表、DICOM/NIfTI 影像、分割 mask、患者级中间结果、模型权重、压缩数据包以及本机真实路径配置均由 `.gitignore` 排除，不应上传到 GitHub。

因此，读取本仓库可以审阅研究设计和实现，但不能仅凭仓库重建患者级数据或复现需要受控医学数据的最终数值结果。横断面 P-EBM 结果仅表示 pseudo-temporal ordering，不代表纵向进展或因果关系；外部验证流程中的预处理、ROI、模型参数、阈值和后处理应保持冻结。
