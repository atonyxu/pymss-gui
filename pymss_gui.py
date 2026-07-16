#!/usr/bin/env python3
"""pymss GUI - Tkinter interface for music source separation with pymss."""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from pymss import list_models, MSSeparator, download_model, get_model_entry
from pymss.model_registry import model_root, model_path_for, auxiliary_paths_for

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pymss_gui_config.json")


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


class RedirectText:
    """Write-only stream that appends to a tkinter Text widget."""

    def __init__(self, widget, log_func):
        self.widget = widget
        self.log_func = log_func

    def write(self, text):
        if text:
            self.log_func(text)

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pymss GUI - Music Source Separation")
        self.geometry("820x680")
        self.minsize(720, 560)
        self.cfg = load_config()
        self.model_entries = []
        self.inference_thread = None
        self.stop_event = threading.Event()

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_main = ttk.Frame(self.notebook)
        self.tab_extra = ttk.Frame(self.notebook)
        self.tab_models = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="推理 (Inference)")
        self.notebook.add(self.tab_extra, text="其他功能 (Tools)")
        self.notebook.add(self.tab_models, text="模型信息 (Models)")

        self._build_main_tab()
        self._build_extra_tab()
        self._build_models_tab()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Tab 1: Inference
    # ------------------------------------------------------------------ #
    def _build_main_tab(self):
        f = self.tab_main
        pad = {"padx": 6, "pady": 4}

        # Input files
        ttk.Label(f, text="输入文件 (wav / mp3):").grid(row=0, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar(value=self.cfg.get("input_files", ""))
        self.input_entry = ttk.Entry(f, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(f, text="选择...", command=self._choose_inputs).grid(row=0, column=2, **pad)
        ttk.Button(f, text="清空", command=lambda: self.input_var.set("")).grid(row=0, column=3, **pad)

        # Output folder
        ttk.Label(f, text="输出文件夹:").grid(row=1, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar(value=self.cfg.get("output_folder", ""))
        ttk.Entry(f, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(f, text="选择...", command=self._choose_output).grid(row=1, column=2, **pad)

        # Model selection
        ttk.Label(f, text="模型:").grid(row=2, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(f, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(f, text="刷新列表", command=self._refresh_models).grid(row=2, column=2, **pad)

        # python env
        ttk.Label(f, text="Python 环境:").grid(row=3, column=0, sticky="w", **pad)
        self.pyenv_var = tk.StringVar(value=self.cfg.get("pyenv", sys.executable))
        self.pyenv_entry = ttk.Entry(f, textvariable=self.pyenv_var)
        self.pyenv_entry.grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(f, text="选择...", command=self._choose_pyenv).grid(row=3, column=2, **pad)

        # model location
        ttk.Label(f, text="模型位置:").grid(row=4, column=0, sticky="w", **pad)
        try:
            default_model_dir = str(model_root())
        except Exception:
            default_model_dir = ""
        self.modeldir_var = tk.StringVar(value=self.cfg.get("model_dir") or default_model_dir)
        modeldir_entry = ttk.Entry(f, textvariable=self.modeldir_var)
        modeldir_entry.grid(row=4, column=1, sticky="ew", **pad)
        modeldir_entry.bind("<FocusOut>", lambda e: self._refresh_download_states())
        ttk.Button(f, text="选择...", command=self._choose_modeldir).grid(row=4, column=2, **pad)

        # Options
        f2 = ttk.LabelFrame(f, text="推理选项")
        f2.grid(row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        self.device_var = tk.StringVar(value=self.cfg.get("device", "auto"))
        ttk.Label(f2, text="设备:").pack(side="left", padx=4)
        ttk.Combobox(f2, textvariable=self.device_var, state="readonly",
                     values=["auto", "cpu", "cuda", "mps", "mlx"], width=8).pack(side="left", padx=2)
        self.format_var = tk.StringVar(value=self.cfg.get("output_format", "wav"))
        ttk.Label(f2, text="格式:").pack(side="left", padx=4)
        ttk.Combobox(f2, textvariable=self.format_var, state="readonly",
                     values=["wav", "flac", "mp3", "m4a"], width=8).pack(side="left", padx=2)
        self.tta_var = tk.BooleanVar(value=self.cfg.get("tta", False))
        ttk.Checkbutton(f2, text="TTA", variable=self.tta_var).pack(side="left", padx=6)
        self.asfolder_var = tk.BooleanVar(value=self.cfg.get("save_as_folder", True))
        ttk.Checkbutton(f2, text="按文件夹保存", variable=self.asfolder_var).pack(side="left", padx=6)
        self.normalize_var = tk.BooleanVar(value=self.cfg.get("normalize", False))
        ttk.Checkbutton(f2, text="归一化", variable=self.normalize_var).pack(side="left", padx=6)

        # Buttons
        bf = ttk.Frame(f)
        bf.grid(row=6, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        self.run_btn = ttk.Button(bf, text="开始推理", command=self._start_inference)
        self.run_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(bf, text="停止", command=self._stop_inference, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(bf, text="下载所选模型", command=self._download_model).pack(side="left", padx=4)

        # Log
        ttk.Label(f, text="日志:").grid(row=7, column=0, sticky="w", **pad)
        self.log_text = scrolledtext.ScrolledText(f, height=14, wrap="word")
        self.log_text.grid(row=8, column=0, columnspan=4, sticky="nsew", padx=6, pady=4)
        self.log_text.config(state="disabled")

        f.rowconfigure(8, weight=1)
        f.columnconfigure(1, weight=1)

        self._refresh_models()

    def _choose_inputs(self):
        files = filedialog.askopenfilenames(
            title="选择音频文件",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("All", "*.*")])
        if files:
            self.input_var.set(";".join(files))

    def _choose_output(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.output_var.set(d)

    def _choose_pyenv(self):
        f = filedialog.askopenfilename(
            title="选择 python 可执行文件",
            filetypes=[("Python", "python python3 python.exe"), ("All", "*.*")])
        if f:
            self.pyenv_var.set(f)

    def _choose_modeldir(self):
        d = filedialog.askdirectory(title="选择模型目录")
        if d:
            self.modeldir_var.set(d)

    def _refresh_models(self):
        try:
            self.model_entries = list_models(supported=True)
        except Exception as e:
            self._log(f"[错误] 获取模型列表失败: {e}\n")
            return
        downloaded = [m for m in self.model_entries if self._is_downloaded(m)]
        pending = [m for m in self.model_entries if m not in downloaded]
        self.model_entries = downloaded + pending
        names = [("* " + m.name) if m in downloaded else m.name for m in self.model_entries]
        self.model_combo["values"] = names
        saved = self.cfg.get("model_name", "")
        saved_disp = ("* " + saved) if saved else ""
        if saved_disp in names:
            self.model_var.set(saved_disp)
        elif downloaded:
            self.model_var.set("* " + downloaded[0].name)
        elif names:
            self.model_var.set(names[0])
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: None)
        self._log(f"已加载 {len(names)} 个可用模型（已下载 {len(downloaded)} 个）。\n")

    def _selected_entry(self):
        name = self.model_var.get().lstrip("* ").strip()
        for m in self.model_entries:
            if m.name == name:
                return m
        return None

    def _is_downloaded(self, entry):
        model_dir = self.modeldir_var.get().strip() or None
        try:
            paths = [model_path_for(entry, model_dir)] + auxiliary_paths_for(entry, model_dir)
        except Exception:
            return False
        return all(os.path.exists(str(p)) for p in paths)

    def _refresh_download_states(self):
        if getattr(self, "model_entries", None):
            self._refresh_models()
        if getattr(self, "catalog_entries", None):
            self._refresh_catalog()

    def _download_model(self):
        entry = self._selected_entry()
        if not entry:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        model_dir = self.modeldir_var.get().strip() or None
        self._log(f"开始下载模型: {entry.name} ...\n")

        def worker():
            try:
                download_model(entry.name, model_dir=model_dir, source="modelscope")
                self._log(f"模型下载完成: {entry.name}\n")
            except Exception as e:
                self._log(f"[错误] 下载失败: {e}\n")
            finally:
                self.after(0, self._refresh_download_states)

        threading.Thread(target=worker, daemon=True).start()

    def _save_cfg(self):
        self.cfg.update({
            "input_files": self.input_var.get(),
            "output_folder": self.output_var.get(),
            "model_name": self.model_var.get().lstrip("* ").strip(),
            "pyenv": self.pyenv_var.get(),
            "model_dir": self.modeldir_var.get(),
            "device": self.device_var.get(),
            "output_format": self.format_var.get(),
            "tta": self.tta_var.get(),
            "save_as_folder": self.asfolder_var.get(),
            "normalize": self.normalize_var.get(),
        })
        save_config(self.cfg)

    def _start_inference(self):
        if self.inference_thread and self.inference_thread.is_alive():
            messagebox.showinfo("提示", "推理正在进行中")
            return
        inputs = [p for p in self.input_var.get().split(";") if p.strip()]
        if not inputs:
            messagebox.showwarning("提示", "请选择输入文件")
            return
        output = self.output_var.get().strip()
        if not output:
            messagebox.showwarning("提示", "请选择输出文件夹")
            return
        entry = self._selected_entry()
        if not entry:
            messagebox.showwarning("提示", "请选择模型")
            return
        os.makedirs(output, exist_ok=True)
        self._save_cfg()

        args = dict(
            model_name=entry.name,
            model_dir=self.modeldir_var.get().strip() or None,
            device=self.device_var.get(),
            output_format=self.format_var.get(),
            use_tta=self.tta_var.get(),
            save_as_folder=self.asfolder_var.get(),
            download=False,
            store_dirs=output,
            inference_params={"normalize": self.normalize_var.get()},
        )

        self.stop_event.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log(f"开始推理: 模型={entry.name}, 文件数={len(inputs)}\n")

        def worker():
            try:
                sep = MSSeparator.from_model_name(**args)
                for path in inputs:
                    if self.stop_event.is_set():
                        self._log("[停止] 用户取消推理。\n")
                        break
                    self._log(f"处理: {os.path.basename(path)}\n")
                    sep.process_folder(path)
                    self._log(f"完成: {os.path.basename(path)}\n")
                if not self.stop_event.is_set():
                    self._log("全部推理完成。\n")
            except Exception as e:
                self._log(f"[错误] {e}\n")
            finally:
                self.after(0, self._finish_inference)

        self.inference_thread = threading.Thread(target=worker, daemon=True)
        self.inference_thread.start()

    def _finish_inference(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _stop_inference(self):
        self.stop_event.set()
        self._log("[请求停止...]\n")

    # ------------------------------------------------------------------ #
    # Tab 2: Extra tools
    # ------------------------------------------------------------------ #
    def _build_extra_tab(self):
        f = self.tab_extra
        pad = {"padx": 6, "pady": 4}

        # Ensemble
        lf = ttk.LabelFrame(f, text="音频融合 (Ensemble)")
        lf.pack(fill="x", padx=6, pady=6)
        ttk.Label(lf, text="输入文件 (多个, 逗号/分号分隔):").pack(anchor="w", **pad)
        self.ens_in_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.ens_in_var).pack(fill="x", **pad)
        ttk.Button(lf, text="选择文件...", command=self._choose_ens_inputs).pack(anchor="w", **pad)
        ttk.Label(lf, text="输出文件:").pack(anchor="w", **pad)
        self.ens_out_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.ens_out_var).pack(fill="x", **pad)
        ttk.Button(lf, text="选择输出...", command=self._choose_ens_output).pack(anchor="w", **pad)
        ttk.Button(lf, text="执行融合", command=self._run_ensemble).pack(anchor="w", **pad)

        # Info / help
        lf3 = ttk.LabelFrame(f, text="说明")
        lf3.pack(fill="both", expand=True, padx=6, pady=6)
        info = ("pymss GUI 基于 pymss 库构建。\n"
                "· 推理选项卡: 选择输入音频、输出目录、模型与运行环境，点击『开始推理』。\n"
                "· 模型选择通过 pymss list_models 获取可用模型。\n"
                "· 『Python 环境』用于记录外部解释器路径（当前进程运行在 "
                f"{sys.executable}）。\n"
                "· 『下载所选模型』可预先拉取模型权重到模型位置。\n"
                "· 其他功能: 音频融合。\n"
                "· 模型信息: 在『模型信息』选项卡查看全部模型详情与下载状态。\n")
        ttk.Label(lf3, text=info, justify="left", wraplength=740).pack(anchor="w", **pad)

    def _choose_ens_inputs(self):
        files = filedialog.askopenfilenames(
            title="选择待融合音频",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a"), ("All", "*.*")])
        if files:
            self.ens_in_var.set(";".join(files))

    def _choose_ens_output(self):
        f = filedialog.asksaveasfilename(
            title="选择融合输出文件",
            defaultextension=".wav",
            filetypes=[("Wav", "*.wav"), ("Flac", "*.flac"), ("Mp3", "*.mp3"), ("M4a", "*.m4a")])
        if f:
            self.ens_out_var.set(f)

    def _run_ensemble(self):
        import re
        ins = [p for p in re.split(r"[;,\n]", self.ens_in_var.get()) if p.strip()]
        out = self.ens_out_var.get().strip()
        if len(ins) < 2:
            messagebox.showwarning("提示", "请选择至少两个输入文件")
            return
        if not out:
            messagebox.showwarning("提示", "请选择输出文件")
            return

        def worker():
            try:
                from pymss import save_ensemble_audio
                save_ensemble_audio(ins, out, weights=[1.0] * len(ins))
                self._log(f"[融合] 已保存到 {out}\n")
                self.after(0, lambda: messagebox.showinfo("完成", f"融合完成:\n{out}"))
            except Exception as e:
                self._log(f"[融合错误] {e}\n")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Tab 3: Model catalog
    # ------------------------------------------------------------------ #
    def _build_models_tab(self):
        f = self.tab_models
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(f)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="选择模型:").pack(side="left", padx=4)
        self.cat_model_var = tk.StringVar()
        self.cat_model_combo = ttk.Combobox(top, textvariable=self.cat_model_var, state="readonly")
        self.cat_model_combo.pack(side="left", fill="x", expand=True, padx=4)
        self.cat_model_combo.bind("<<ComboboxSelected>>", lambda e: self._show_catalog_info())
        ttk.Button(top, text="刷新列表", command=self._refresh_catalog).pack(side="left", padx=4)

        self.only_supported_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="仅支持的模型", variable=self.only_supported_var,
                        command=self._refresh_catalog).pack(side="left", padx=4)

        self.cat_info_text = scrolledtext.ScrolledText(f, wrap="word")
        self.cat_info_text.pack(fill="both", expand=True, **pad)
        self.cat_info_text.config(state="disabled")

        bf = ttk.Frame(f)
        bf.pack(fill="x", **pad)
        self.cat_download_btn = ttk.Button(bf, text="下载当前模型", command=self._download_catalog_model)
        self.cat_download_btn.pack(side="left", padx=4)

        self.catalog_entries = []
        self._refresh_catalog()

    def _download_catalog_model(self):
        name = self.cat_model_var.get().lstrip("* ").strip()
        entry = next((m for m in self.catalog_entries if m.name == name), None)
        if not entry:
            messagebox.showwarning("提示", "请先选择一个模型")
            return
        model_dir = self.modeldir_var.get().strip() or None
        self.cat_download_btn.config(state="disabled")
        self._log(f"开始下载模型: {entry.name} ...\n")

        def worker():
            try:
                download_model(entry.name, model_dir=model_dir, source="modelscope")
                self._log(f"模型下载完成: {entry.name}\n")
            except Exception as e:
                self._log(f"[错误] 下载失败: {e}\n")
            finally:
                self.after(0, lambda: (self.cat_download_btn.config(state="normal"),
                                       self._refresh_download_states()))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_catalog(self):
        try:
            self.catalog_entries = list_models(supported=self.only_supported_var.get())
        except Exception as e:
            self._log(f"[错误] 获取模型列表失败: {e}\n")
            return
        downloaded = [m for m in self.catalog_entries if self._is_downloaded(m)]
        pending = [m for m in self.catalog_entries if m not in downloaded]
        self.catalog_entries = downloaded + pending
        names = [("* " + m.name) if m in downloaded else m.name for m in self.catalog_entries]
        self.cat_model_combo["values"] = names
        if names:
            if self.cat_model_var.get() not in names:
                self.cat_model_var.set(names[0])
            self.cat_model_combo.bind(
                "<<ComboboxSelected>>",
                lambda e: self._show_catalog_info())
            self._show_catalog_info()

    def _show_catalog_info(self):
        name = self.cat_model_var.get().lstrip("* ").strip()
        entry = next((m for m in self.catalog_entries if m.name == name), None)
        if not entry:
            return
        downloaded = self._is_downloaded(entry)
        lines = [f"名称: {entry.name}",
                 f"别名: {', '.join(entry.aliases)}",
                 f"类别: {entry.category_path}",
                 f"架构: {entry.architecture}",
                 f"模型类型: {entry.model_type}",
                 f"目标音轨: {entry.target_stem}",
                 f"支持: {entry.supported}",
                 f"已下载: {'是' if downloaded else '否'}",
                 f"大小: {entry.size_bytes} bytes",
                 f"配置: {entry.config_relpath}",
                 f"权重: {entry.relpath}"]
        self.cat_info_text.config(state="normal")
        self.cat_info_text.delete("1.0", "end")
        self.cat_info_text.insert("1.0", "\n".join(lines))
        self.cat_info_text.config(state="disabled")

    # ------------------------------------------------------------------ #
    # Log & misc
    # ------------------------------------------------------------------ #
    def _log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _on_close(self):
        self._save_cfg()
        self.stop_event.set()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
