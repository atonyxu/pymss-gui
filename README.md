# pymss GUI

基于 [pymss](https://github.com/pymss-project/pymss) 库构建的音乐音源分离图形界面工具，使用 Tkinter 实现，方便在桌面环境中进行人声/伴奏等音轨的分离。

## 功能特性

- **音源分离推理**：选择输入音频、输出目录、模型与运行设备，一键完成分离。
- **官方模型管理**：自动获取 pymss 支持的模型列表，可预先下载模型权重。
- **自定义模型支持**：自动扫描「模型位置」目录下（含子目录）的自定义权重，识别条件为同名的 `*.ckpt` 文件 + `*.yaml` 配置，配置中需包含 `model_type`（缺失时默认 `bs_roformer`）。自定义模型通过 `MSSeparator` 手动构造加载，并自动读取配置中的推理参数（如 `batch_size`、`dim_t`，以及将 `num_overlap` 转换为 `overlap_size`）。
- **下载状态标识**：已下载的模型在下拉列表中以 `[已下载]` 前缀标记，并优先排序。
- **模型分组排序**：下拉列表中自定义模型置顶，其后为已下载模型，再为未下载模型。
- **多种设备支持**：`auto` / `cpu` / `cuda` / `mps` / `mlx`。
- **多种输出格式**：`wav` / `flac` / `mp3` / `m4a`。
- **推理选项**：TTA、按文件夹保存、音频归一化。
- **音频融合 (Ensemble)**：将多个分离结果加权融合。
- **模型信息查看**：在「模型信息」选项卡查看全部模型详情与下载状态，并可下载当前所选官方模型。
- **配置持久化**：界面设置自动保存到 `~/.pymss_gui_config.json`。

## 环境要求

- Python 3.8+
- [pymss](https://pypi.org/) 库
- Tkinter（大多数 Python 发行版已内置）
- PyYAML（`pip install pymss` 一般已包含；用于解析自定义模型配置）

## 安装

```bash
pip install pymss
```

> Linux 用户如缺少 Tkinter，可安装系统包，例如：
> `sudo apt install python3-tk`

## 使用方法

```bash
python pymss_gui.py
```

程序启动后，「模型位置」默认显示为 pymss 库的默认权重目录（通常为 `~/.cache/pymss/models`）。

### 推理选项卡 (Inference)

1. 选择输入音频文件（支持 wav / mp3 / flac / m4a）。
2. 选择输出文件夹。
3. 从下拉列表选择模型（自定义模型以 `[自定义] 名称` 显示，已下载模型以 `[已下载] 名称` 显示，置顶优先）。
4. 按需设置设备、输出格式及推理选项。
5. 点击「开始推理」，日志区域实时显示进度，可随时「停止」。
6. 未下载的官方模型可点击「下载所选模型」预先拉取权重。

### 其他功能选项卡 (Tools)

- **音频融合**：选择至少两个音频文件并指定输出，执行加权融合。

### 模型信息选项卡 (Models)

- 列出全部模型（含自定义模型与官方模型），通过 `list_models` 获取官方模型。
- 选中模型后在文本框中显示其架构、类型、目标音轨、下载状态等详情。
- 点击「下载当前模型」可下载当前选中的官方模型（自定义模型已位于本地，无需下载）。

## 自定义模型说明

自定义模型需满足：

- 权重文件为 `*.ckpt`，且同级目录存在同名配置文件 `*.yaml` 或 `*.yml`。
- 配置文件中需包含 `model_type` 字段，取值参考 pymss 支持的类型：`bs_roformer` / `mel_band_roformer` / `htdemucs` / `mdx23c` / `bandit` / `bandit_v2` / `scnet` / `apollo` / `vr`（缺失时默认 `bs_roformer`）。
- 支持 MSST 风格的 `!!python/tuple` / `!!python/list` 标签。

示例配置（节选）：

```yaml
model_type: bs_roformer
inference:
  batch_size: 2
  dim_t: 1101
  num_overlap: 2   # 会自动转换为 overlap_size
```

推理时程序会读取该配置的 `inference` 段作为推理参数。

## 配置文件

界面参数会自动保存在用户主目录下的 `~/.pymss_gui_config.json`，下次启动时自动恢复。

## 许可证

MIT License
