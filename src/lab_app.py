import concurrent.futures
import hashlib
import io
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
import requests
import ttkbootstrap as ttk
from PIL import Image, ImageTk

from .lab1_processing import (
    compress_manual,
    gray_to_rgb,
    invert_hsi_intensity,
    load_rgb_image,
    rediscretize_one_pass,
    rediscretize_two_pass,
    rgb_channel_as_image,
    rgb_to_hsi_intensity,
    save_gray_image,
    save_rgb_image,
    split_rgb_channels,
    stretch_manual,
)
from .lab2_processing import adaptive_threshold_minmax, rgb_to_grayscale_weighted
from .lab2_processing import fetch_sample_image_paths as fetch_lab2_sample_paths
from .lab3_processing import (
    boost_diff,
    diff_abs,
    diff_xor,
    fringe_erase_black,
)
from .lab3_processing import (
    fetch_sample_image_paths as fetch_lab3_sample_paths,
)
from .lab4_processing import fetch_sample_image_paths as fetch_lab4_sample_paths
from .lab4_processing import kayali_edges
from .lab_constants import PREVIEW_MAX_SIZE
from .lab_state import ImageState


class ImageLabApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        self.lab_titles = {
            "Lab1": "Лабораторная работа №1 — Цветовые модели и передискретизация",
            "Lab2": "Лабораторная работа №2 — Обесцвечивание и бинаризация изображений",
            "Lab3": "Лабораторная работа №3 — Фильтрация изображений",
            "Lab4": "Лабораторная работа №4 — Выделение контуров",
        }
        self.root.title(self.lab_titles["Lab1"])
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        self.state = ImageState()
        self._left_preview = None
        self._right_preview = None
        self.active_lab = "Lab1"
        self._lab_tabs = {}
        self.lab3_filtered = None
        self.lab3_diff = None
        self.lab3_input_preview = None
        self.lab4_gray = None
        self.lab4_gx = None
        self.lab4_gy = None
        self.lab4_g = None
        self.lab4_binary = None
        self.sample_image_paths = []
        self.sample_sidebar_open = True
        self.sample_thumbs = []
        self.sample_gallery_items = []
        self.sample_scroll_speed = 4.0
        self._thumb_loader_thread = None
        self._thumb_cancel = False
        self._thumb_executor = None
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sample_cache_dir = os.path.join(base_dir, "resources", "sample_cache")
        self.sample_thumb_dir = os.path.join(base_dir, "resources", "sample_thumbs")
        os.makedirs(self.sample_cache_dir, exist_ok=True)
        os.makedirs(self.sample_thumb_dir, exist_ok=True)

        self.info_var = tk.StringVar(value="Изображение не загружено")
        self.m_var = tk.StringVar(value="2")
        self.n_var = tk.StringVar(value="2")
        self.k_var = tk.StringVar(value="1")
        self.lab2_window_var = tk.IntVar(value=3)
        self.lab2_window_label_var = tk.StringVar(value="Окно: 3×3")
        self.lab3_boost_diff_var = tk.IntVar(value=0)
        self.lab4_threshold_var = tk.StringVar(value="60")
        self.log_visible = True
        self.controls_visible = True

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self._build_sidebar(outer)

        self._build_samples_sidebar(outer)

        main = ttk.Frame(outer, padding=12)
        main.pack(side="left", fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 10))

        ttk.Button(
            top, text="Открыть PNG/BMP", command=self.open_image, bootstyle="primary"
        ).pack(side="left")

        ttk.Button(
            top,
            text="Сохранить результат",
            command=self.save_processed,
            bootstyle="success",
        ).pack(side="left", padx=8)

        ttk.Label(top, textvariable=self.info_var).pack(side="left", padx=14)

        self.log_toggle_btn = ttk.Button(
            top, text="Скрыть журнал", command=self.toggle_log, bootstyle="secondary"
        )
        self.log_toggle_btn.pack(side="right")

        self.controls_toggle_btn = ttk.Button(
            top, text="Скрыть панель", command=self.toggle_controls, bootstyle="secondary"
        )
        self.controls_toggle_btn.pack(side="right", padx=(0, 6))

        body = ttk.Frame(main)
        body.pack(fill="both", expand=True)

        self.controls_outer = ttk.LabelFrame(body, text="Операции")
        self.controls_outer_parent = body
        self.controls_outer_pack = {"side": "left", "fill": "y", "padx": (0, 12)}
        self.controls_outer.pack(**self.controls_outer_pack)
        controls = ttk.Frame(self.controls_outer, padding=10)
        controls.pack(fill="both", expand=True)

        self.lab1_controls = ttk.Frame(controls)
        self.lab2_controls = ttk.Frame(controls)
        self.lab3_controls = ttk.Frame(controls)
        self.lab4_controls = ttk.Frame(controls)
        self._build_lab1_controls(self.lab1_controls)
        self._build_lab2_controls(self.lab2_controls)
        self._build_lab3_controls(self.lab3_controls)
        self._build_lab4_controls(self.lab4_controls)
        self.lab1_controls.pack(fill="both", expand=True)

        self.preview_wrap = ttk.Frame(body)
        self.preview_wrap.pack(side="left", fill="both", expand=True)

        previews = ttk.Frame(self.preview_wrap)
        previews.pack(fill="both", expand=True)

        left_box_outer = ttk.LabelFrame(previews, text="До")
        left_box_outer.pack(side="left", fill="both", expand=True, padx=(0, 8))
        left_box = ttk.Frame(left_box_outer, padding=8)
        left_box.pack(fill="both", expand=True)

        self.before_label = ttk.Label(left_box, anchor="center")
        self.before_label.pack(fill="both", expand=True)

        right_box_outer = ttk.LabelFrame(previews, text="После")
        right_box_outer.pack(side="left", fill="both", expand=True)
        right_box = ttk.Frame(right_box_outer, padding=8)
        right_box.pack(fill="both", expand=True)

        self.after_label = ttk.Label(right_box, anchor="center")
        self.after_label.pack(fill="both", expand=True)

        self.log_box_outer = ttk.LabelFrame(self.preview_wrap, text="Журнал")
        self.log_box_outer.pack(fill="x", pady=(10, 0))
        log_box = ttk.Frame(self.log_box_outer, padding=8)
        log_box.pack(fill="both", expand=True)

        self.log = tk.Text(
            log_box,
            height=9,
            wrap="word",
            bg="#111111",
            fg="#f2f2f2",
            insertbackground="#f2f2f2",
            relief="flat",
            borderwidth=0,
        )
        self.log.pack(fill="both", expand=True)

        self._append_log("✅ Перед выполнением операций откройте PNG/BMP изображение")

    def _build_lab1_controls(self, parent) -> None:
        self._build_color_section(parent)
        self._build_sampling_section(parent)

    def _build_samples_sidebar(self, parent) -> None:
        right_wrap = ttk.Frame(parent)
        right_wrap.pack(side="right", fill="y")

        toggle_bar = ttk.Frame(right_wrap, width=22)
        toggle_bar.pack(side="right", fill="y")
        toggle_bar.pack_propagate(False)
        self.sample_toggle_btn = ttk.Button(
            toggle_bar,
            text="◀",
            command=self._toggle_samples_sidebar,
            bootstyle="secondary",
        )
        self.sample_toggle_btn.place(relx=0.5, rely=0.5, anchor="center")

        self.sample_sidebar = ttk.Frame(right_wrap, width=320)
        self.sample_sidebar.pack(side="right", fill="y")
        self.sample_sidebar.pack_propagate(False)

        header = ttk.Frame(self.sample_sidebar, padding=(8, 8, 8, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Samples").pack(side="left")
        self.sample_fetch_btn = ttk.Button(
            header, text="⟳", command=self.fetch_samples, bootstyle="secondary"
        )
        self.sample_fetch_btn.pack(side="right")

        self.sample_canvas = tk.Canvas(self.sample_sidebar, highlightthickness=0)
        self.sample_scroll = ttk.Scrollbar(
            self.sample_sidebar, orient="vertical", command=self.sample_canvas.yview
        )
        self.sample_canvas.configure(yscrollcommand=self.sample_scroll.set)
        self.sample_scroll.pack(side="right", fill="y")
        self.sample_canvas.pack(side="left", fill="both", expand=True)

        self.sample_gallery = ttk.Frame(self.sample_canvas)
        self.sample_canvas_window = self.sample_canvas.create_window(
            (0, 0), window=self.sample_gallery, anchor="nw"
        )

        def on_configure(_event) -> None:
            self.sample_canvas.configure(scrollregion=self.sample_canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            self.sample_canvas.itemconfigure(
                self.sample_canvas_window, width=event.width
            )

        self.sample_gallery.bind("<Configure>", on_configure)
        self.sample_canvas.bind("<Configure>", on_canvas_configure)

        def _bind_wheel(_event) -> None:
            self.sample_canvas.bind_all("<MouseWheel>", self._on_sample_mousewheel)
            self.sample_canvas.bind_all("<Button-4>", self._on_sample_mousewheel)
            self.sample_canvas.bind_all("<Button-5>", self._on_sample_mousewheel)

        def _unbind_wheel(_event) -> None:
            self.sample_canvas.unbind_all("<MouseWheel>")
            self.sample_canvas.unbind_all("<Button-4>")
            self.sample_canvas.unbind_all("<Button-5>")

        self.sample_sidebar.bind("<Enter>", _bind_wheel)
        self.sample_sidebar.bind("<Leave>", _unbind_wheel)

    def _on_sample_mousewheel(self, event) -> str:
        top, _bottom = self.sample_canvas.yview()
        if event.num == 4:
            delta = -1.0
        elif event.num == 5:
            delta = 1.0
        else:
            delta = -float(event.delta) / 120.0

        move = (delta * 0.04) * self.sample_scroll_speed
        new_top = top + move
        if new_top < 0:
            new_top = 0.0
        if new_top > 1.0:
            new_top = 1.0
        self.sample_canvas.yview_moveto(new_top)
        return "break"

    def _toggle_samples_sidebar(self) -> None:
        if self.sample_sidebar_open:
            self.sample_sidebar.pack_forget()
            self.sample_sidebar_open = False
            self.sample_toggle_btn.configure(text="▶")
        else:
            self.sample_sidebar.pack(side="right", fill="y")
            self.sample_sidebar_open = True
            self.sample_toggle_btn.configure(text="◀")

    def _refresh_sample_gallery(self) -> None:
        for widget in self.sample_gallery.winfo_children():
            widget.destroy()
        self.sample_thumbs = []
        self.sample_gallery_items = []

        if not self.sample_image_paths:
            ttk.Label(self.sample_gallery, text="Нет sample-данных").grid(
                row=0, column=0, padx=8, pady=8
            )
            return

        cols = 3
        thumb_size = 70
        for idx, url in enumerate(self.sample_image_paths):
            row = idx // cols
            col = idx % cols
            cell = tk.Canvas(
                self.sample_gallery,
                width=thumb_size,
                height=thumb_size,
                bg="#4a4a4a",
                highlightthickness=0,
            )
            cell.grid(row=row, column=col, padx=4, pady=4)
            cell.bind("<Button-1>", lambda _e, u=url: self.load_sample_from_url(u))
            self.sample_gallery_items.append({"url": url, "canvas": cell, "image": None})

        self._start_thumb_loader()

    def _start_thumb_loader(self) -> None:
        self._thumb_cancel = True
        if self._thumb_executor is not None:
            try:
                self._thumb_executor.shutdown(wait=False)
            except Exception:
                pass
        self._thumb_cancel = False

        items = list(self.sample_gallery_items)
        self._thumb_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        def fetch(url: str):
            thumb = self._load_cached_thumb(url)
            if thumb is not None:
                return thumb

            full = self._load_cached_image(url)
            if full is None:
                try:
                    resp = requests.get(url, timeout=(3, 8))
                    resp.raise_for_status()
                    full = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    self._save_cache_image(url, full)
                except Exception:
                    return None

            thumb = full.copy()
            thumb.thumbnail((70, 70))
            self._save_cache_thumb(url, thumb)
            return thumb

        def on_done(fut, item):
            if self._thumb_cancel:
                return
            img = fut.result()
            if img is None:
                return

            def update() -> None:
                if self._thumb_cancel:
                    return
                photo = ImageTk.PhotoImage(img)
                canvas = item["canvas"]
                canvas.delete("all")
                canvas.create_image(35, 35, image=photo)
                canvas.image = photo
                self.sample_thumbs.append(photo)

            self.root.after(0, update)

        for item in items:
            fut = self._thumb_executor.submit(fetch, item["url"])
            fut.add_done_callback(lambda f, it=item: on_done(f, it))

    def load_sample_from_url(self, url: str) -> None:
        if not url:
            messagebox.showinfo(
                "Нет sample", "Сначала загрузите sample-данные и выберите файл."
            )
            return
        try:
            image = self._load_cached_image(url)
            if image is None:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                self._save_cache_image(url, image)
            arr = np.array(image, dtype=np.uint8)
            self.state.path = url
            self.state.image = arr
            self.state.processed = None
            self.lab3_filtered = None
            self.lab3_diff = None
            self.lab3_input_preview = None
            self.lab4_gray = None
            self.lab4_gx = None
            self.lab4_gy = None
            self.lab4_g = None
            self.lab4_binary = None
            name = url.split("/")[-1] or "sample"
            self.info_var.set(f"{name} — {arr.shape[1]}x{arr.shape[0]}")
            self._update_previews(arr, None)
            self._append_log(f"Загружен sample: {url}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _cache_paths(self, url: str) -> tuple[str, str]:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        img_path = os.path.join(self.sample_cache_dir, f"{key}.png")
        thumb_path = os.path.join(self.sample_thumb_dir, f"{key}.png")
        return img_path, thumb_path

    def _load_cached_image(self, url: str) -> Image.Image | None:
        img_path, _ = self._cache_paths(url)
        if os.path.exists(img_path):
            try:
                return Image.open(img_path).convert("RGB")
            except Exception:
                return None
        return None

    def _save_cache_image(self, url: str, img: Image.Image) -> None:
        img_path, _ = self._cache_paths(url)
        try:
            img.save(img_path, format="PNG")
        except Exception:
            pass

    def _load_cached_thumb(self, url: str) -> Image.Image | None:
        _, thumb_path = self._cache_paths(url)
        if os.path.exists(thumb_path):
            try:
                return Image.open(thumb_path).convert("RGB")
            except Exception:
                return None
        return None

    def _save_cache_thumb(self, url: str, img: Image.Image) -> None:
        _, thumb_path = self._cache_paths(url)
        try:
            img.save(thumb_path, format="PNG")
        except Exception:
            pass

    def _build_lab2_controls(self, parent) -> None:
        gray_outer = ttk.LabelFrame(parent, text="1. Обесцвечивание (полутон)")
        gray_outer.pack(fill="x", pady=(0, 10))
        gray_box = ttk.Frame(gray_outer, padding=8)
        gray_box.pack(fill="both", expand=True)

        ttk.Label(
            gray_box,
            text="Взвешенное усреднение каналов RGB → 1 яркостный канал.",
            wraplength=230,
            justify="left",
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            gray_box,
            text="Преобразовать в полутоновое",
            command=self.convert_to_grayscale,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        bin_outer = ttk.LabelFrame(parent, text="2. Бинаризация (адаптивная)")
        bin_outer.pack(fill="x")
        bin_box = ttk.Frame(bin_outer, padding=8)
        bin_box.pack(fill="both", expand=True)

        ttk.Label(
            bin_box,
            text="Метод минимаксного усреднения. Варианты окна: 3×3 и 25×25.",
            wraplength=230,
            justify="left",
        ).pack(fill="x", pady=(0, 6))

        ttk.Label(bin_box, textvariable=self.lab2_window_label_var).pack(
            anchor="w", pady=(0, 2)
        )
        self.lab2_window_scale = ttk.Scale(
            bin_box,
            from_=3,
            to=25,
            orient="horizontal",
            command=self._on_lab2_window_change,
        )
        self.lab2_window_scale.set(3)
        self.lab2_window_scale.pack(fill="x", pady=(0, 8))

        ttk.Button(
            bin_box,
            text="Бинаризовать",
            command=self.adaptive_binarize,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

    def _build_lab3_controls(self, parent) -> None:
        filt_outer = ttk.LabelFrame(parent, text="1. Фильтрация (стирание бахромы)")
        filt_outer.pack(fill="x", pady=(0, 10))
        filt_box = ttk.Frame(filt_outer, padding=8)
        filt_box.pack(fill="both", expand=True)

        ttk.Label(
            filt_box,
            text="Окно 3×3, стирание чёрной бахромы (max-фильтр).",
            wraplength=230,
            justify="left",
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            filt_box,
            text="Применить фильтр",
            command=self.apply_lab3_filter,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        diff_outer = ttk.LabelFrame(parent, text="2. Разностное изображение")
        diff_outer.pack(fill="x")
        diff_box = ttk.Frame(diff_outer, padding=8)
        diff_box.pack(fill="both", expand=True)

        ttk.Checkbutton(
            diff_box,
            text="Усилить разницу (x10) для полутона",
            variable=self.lab3_boost_diff_var,
            onvalue=1,
            offvalue=0,
            bootstyle="round-toggle",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Button(
            diff_box,
            text="Показать фильтр",
            command=self.show_lab3_filtered,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            diff_box,
            text="Показать разницу",
            command=self.show_lab3_diff,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            diff_box,
            text="Сохранить разницу",
            command=self.save_lab3_diff,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

    def _build_lab4_controls(self, parent) -> None:
        edges_outer = ttk.LabelFrame(parent, text="1. Выделение контуров (Кайяли 3×3)")
        edges_outer.pack(fill="x", pady=(0, 10))
        edges_box = ttk.Frame(edges_outer, padding=8)
        edges_box.pack(fill="both", expand=True)

        ttk.Label(
            edges_box,
            text="Оператор Кайяли 3×3, G = sqrt(Gx^2 + Gy^2).",
            wraplength=230,
            justify="left",
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            edges_box,
            text="Рассчитать градиенты",
            command=self.compute_lab4_edges,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        show_outer = ttk.LabelFrame(parent, text="2. Просмотр результатов")
        show_outer.pack(fill="x")
        show_box = ttk.Frame(show_outer, padding=8)
        show_box.pack(fill="both", expand=True)

        ttk.Label(show_box, text="Порог для G (0–255):").pack(anchor="w")
        ttk.Entry(show_box, textvariable=self.lab4_threshold_var).pack(
            fill="x", pady=(0, 6)
        )

        ttk.Button(
            show_box,
            text="Показать полутон",
            command=self.show_lab4_gray,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            show_box,
            text="Показать Gx",
            command=self.show_lab4_gx,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            show_box,
            text="Показать Gy",
            command=self.show_lab4_gy,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            show_box,
            text="Показать G",
            command=self.show_lab4_g,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            show_box,
            text="Показать бинаризованный G",
            command=self.show_lab4_binary,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

    def fetch_samples(self) -> None:
        if self.active_lab == "Lab2":
            self._fetch_samples(fetch_lab2_sample_paths, "Lab2")
        elif self.active_lab == "Lab3":
            self._fetch_samples(fetch_lab3_sample_paths, "Lab3")
        elif self.active_lab == "Lab4":
            self._fetch_samples(fetch_lab4_sample_paths, "Lab4")
        else:
            self._fetch_samples(fetch_lab2_sample_paths, "Lab2")

    def fetch_lab2_samples(self) -> None:
        self._fetch_samples(fetch_lab2_sample_paths, "Lab2")

    def fetch_lab3_samples(self) -> None:
        self._fetch_samples(fetch_lab3_sample_paths, "Lab3")

    def fetch_lab4_samples(self) -> None:
        self._fetch_samples(fetch_lab4_sample_paths, "Lab4")

    def _fetch_samples(self, fetch_fn, label: str) -> None:
        try:
            paths = fetch_fn()
            self.sample_image_paths = paths
            self._refresh_sample_gallery()
            if not self.sample_sidebar_open:
                self._toggle_samples_sidebar()
            self._append_log(f"{label}: загружено sample-изображений: {len(paths)}")
            if paths:
                self._append_log(f"{label}: первый sample: {paths[0]}")
            messagebox.showinfo("Готово", f"{label}: загружено {len(paths)} файлов.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def toggle_log(self) -> None:
        if self.log_visible:
            self.log_box_outer.pack_forget()
            self.log_toggle_btn.configure(text="Показать журнал")
            self.log_visible = False
        else:
            self.log_box_outer.pack(fill="x", pady=(10, 0))
            self.log_toggle_btn.configure(text="Скрыть журнал")
            self.log_visible = True
        self.controls_visible = True

    def toggle_controls(self) -> None:
        if self.controls_visible:
            self.controls_outer.pack_forget()
            self.controls_toggle_btn.configure(text="Показать панель")
            self.controls_visible = False
        else:
            self.controls_outer.pack(in_=self.controls_outer_parent, before=self.preview_wrap, **self.controls_outer_pack)
            self.controls_toggle_btn.configure(text="Скрыть панель")
            self.controls_visible = True

    def _build_sidebar(self, parent) -> None:
        sidebar_bg = "#151515"
        sidebar = tk.Frame(parent, bg=sidebar_bg, width=36)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tabs = ["Lab1", "Lab2", "Lab3", "Lab4"]
        for name in tabs:
            is_active = name == self.active_lab
            tab_bg = "#1f4f99" if is_active else "#202020"
            fg = "#ffffff" if is_active else "#c7c7c7"
            font = ("TkDefaultFont", 10, "bold" if is_active else "normal")
            tag = f"tab_{name}"

            width = 36
            height = 90
            pad = 4
            tab = tk.Canvas(
                sidebar,
                width=width,
                height=height,
                bg=sidebar_bg,
                highlightthickness=0,
            )
            rect_id = tab.create_rectangle(
                pad,
                pad,
                width - pad,
                height - pad,
                fill=tab_bg,
                outline="",
                tags=(tag,),
            )
            text_id = tab.create_text(
                width / 2,
                height / 2,
                text=name,
                angle=90,
                fill=fg,
                font=font,
                tags=(tag,),
            )
            tab.tag_bind(tag, "<Button-1>", lambda _event, n=name: self._on_lab_tab(n))
            tab.configure(cursor="hand2")
            tab.pack(side="top", fill="x", pady=4)
            self._lab_tabs[name] = {
                "canvas": tab,
                "rect": rect_id,
                "text": text_id,
            }

    def _on_lab_tab(self, name: str) -> None:
        if name in ("Lab1", "Lab2", "Lab3", "Lab4"):
            self._set_active_lab(name)
            return
        messagebox.showwarning("TBD", f"{name} пока не реализована.")

    def _set_active_lab(self, name: str) -> None:
        if name == self.active_lab:
            return
        self.active_lab = name
        self.root.title(self.lab_titles.get(name, self.root.title()))

        self.lab1_controls.pack_forget()
        self.lab2_controls.pack_forget()
        self.lab3_controls.pack_forget()
        self.lab4_controls.pack_forget()

        if name == "Lab1":
            self.lab1_controls.pack(fill="both", expand=True)
        elif name == "Lab2":
            self.lab2_controls.pack(fill="both", expand=True)
        elif name == "Lab3":
            self.lab3_controls.pack(fill="both", expand=True)
        else:
            self.lab4_controls.pack(fill="both", expand=True)

        for tab_name, meta in self._lab_tabs.items():
            is_active = tab_name == self.active_lab
            tab_bg = "#1f4f99" if is_active else "#202020"
            fg = "#ffffff" if is_active else "#c7c7c7"
            font = ("TkDefaultFont", 10, "bold" if is_active else "normal")
            canvas = meta["canvas"]
            canvas.itemconfigure(meta["rect"], fill=tab_bg)
            canvas.itemconfigure(meta["text"], fill=fg, font=font)

    def _build_color_section(self, parent) -> None:
        color_box_outer = ttk.LabelFrame(parent, text="1. Цветовые модели")
        color_box_outer.pack(fill="x", pady=(0, 10))
        color_box = ttk.Frame(color_box_outer, padding=8)
        color_box.pack(fill="both", expand=True)

        ttk.Button(
            color_box,
            text="Сохранить R, G, B как отдельные изображения",
            command=self.save_rgb_components,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            color_box,
            text="Преобразовать в HSI и сохранить яркость I",
            command=self.save_hsi_intensity,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

        ttk.Button(
            color_box,
            text="Инвертировать яркость I в исходном изображении",
            command=self.invert_intensity,
            bootstyle="secondary",
        ).pack(fill="x", pady=3)

    def _build_sampling_section(self, parent) -> None:
        sampling_box_outer = ttk.LabelFrame(parent, text="2. Передискретизация")
        sampling_box_outer.pack(fill="x")
        sampling_box = ttk.Frame(sampling_box_outer, padding=8)
        sampling_box.pack(fill="both", expand=True)

        ttk.Label(sampling_box, text="M (растяжение):").pack(anchor="w")
        ttk.Entry(sampling_box, textvariable=self.m_var).pack(fill="x", pady=(0, 6))

        ttk.Label(sampling_box, text="N (сжатие):").pack(anchor="w")
        ttk.Entry(sampling_box, textvariable=self.n_var).pack(fill="x", pady=(0, 6))

        ttk.Label(sampling_box, text="K (итоговый коэффициент, K=M/N):").pack(
            anchor="w"
        )
        ttk.Entry(sampling_box, textvariable=self.k_var).pack(fill="x", pady=(0, 8))

        ttk.Button(
            sampling_box,
            text="Растянуть в M раз",
            command=self.stretch_image,
            bootstyle="info",
        ).pack(fill="x", pady=3)

        ttk.Button(
            sampling_box,
            text="Сжать в N раз",
            command=self.compress_image,
            bootstyle="info",
        ).pack(fill="x", pady=3)

        ttk.Button(
            sampling_box,
            text="Передискретизация K=M/N в два прохода",
            command=self.rediscretize_two_pass_action,
            bootstyle="info",
        ).pack(fill="x", pady=3)

        ttk.Button(
            sampling_box,
            text="Передискретизация в K раз за один проход",
            command=self.rediscretize_one_pass_action,
            bootstyle="info",
        ).pack(fill="x", pady=3)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _require_image(self) -> np.ndarray:
        if self.state.image is None:
            raise ValueError("Сначала надо открыть изображение.")
        return self.state.image

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение", filetypes=[("Images", "*.png *.bmp")]
        )
        if not path:
            return

        try:
            image = load_rgb_image(path)
            self.state.path = path
            self.state.image = image
            self.state.processed = None
            self.lab3_filtered = None
            self.lab3_diff = None
            self.lab3_input_preview = None
            self.lab4_gray = None
            self.lab4_gx = None
            self.lab4_gy = None
            self.lab4_g = None
            self.lab4_binary = None

            self.info_var.set(
                f"{os.path.basename(path)} — {image.shape[1]}x{image.shape[0]}"
            )
            self._update_previews(image, None)
            self._append_log(f"Открыто изображение: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def save_processed(self) -> None:
        if self.state.processed is None:
            messagebox.showinfo("Нет результата", "Сначала выполните операцию.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("BMP image", "*.bmp")],
        )
        if not path:
            return

        try:
            if self.state.processed.ndim == 2:
                save_gray_image(self.state.processed, path)
            else:
                save_rgb_image(self.state.processed, path)
            self._append_log(f"Результат сохранён: {path}")
            messagebox.showinfo("✅", "Результат сохранён.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _preview_image(self, arr: np.ndarray) -> ImageTk.PhotoImage:
        if arr.ndim == 2:
            pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        else:
            pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        pil.thumbnail(PREVIEW_MAX_SIZE)
        return ImageTk.PhotoImage(pil)

    def _update_previews(
        self, original: Optional[np.ndarray], processed: Optional[np.ndarray]
    ) -> None:
        if original is not None:
            self._left_preview = self._preview_image(original)
            self.before_label.configure(image=self._left_preview)
        else:
            self.before_label.configure(image="")
            self._left_preview = None

        if processed is not None:
            self._right_preview = self._preview_image(processed)
            self.after_label.configure(image=self._right_preview)
        else:
            self.after_label.configure(image="")
            self._right_preview = None

    def _set_processed(self, arr: np.ndarray, action_name: str) -> None:
        self.state.processed = arr
        self._update_previews(self.state.image, arr)
        self._append_log(action_name)

    def _set_processed_with_original(
        self, original: np.ndarray, arr: np.ndarray, action_name: str
    ) -> None:
        self.state.processed = arr
        self._update_previews(original, arr)
        self._append_log(action_name)

    def _ask_save_base(self, suffix: str) -> Optional[str]:
        source_name = (
            "result"
            if not self.state.path
            else os.path.splitext(os.path.basename(self.state.path))[0]
        )

        return filedialog.asksaveasfilename(
            title="Сохранить файл",
            initialfile=f"{source_name}_{suffix}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("BMP image", "*.bmp")],
        )

    def save_rgb_components(self) -> None:
        try:
            image = self._require_image()
            r, g, b = split_rgb_channels(image)

            path_r = self._ask_save_base("R")
            if not path_r:
                return
            path_g = self._ask_save_base("G")
            if not path_g:
                return
            path_b = self._ask_save_base("B")
            if not path_b:
                return

            save_rgb_image(rgb_channel_as_image(r, 0), path_r)
            save_rgb_image(rgb_channel_as_image(g, 1), path_g)
            save_rgb_image(rgb_channel_as_image(b, 2), path_b)

            self._set_processed(
                rgb_channel_as_image(r, 0), "Сохранены компоненты R, G, B."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def save_hsi_intensity(self) -> None:
        try:
            image = self._require_image()
            intensity = rgb_to_hsi_intensity(image)

            path = self._ask_save_base("HSI_I")
            if not path:
                return

            save_gray_image(intensity, path)
            self._set_processed(
                gray_to_rgb(intensity), "Сохранена яркостная компонента I из HSI."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def invert_intensity(self) -> None:
        try:
            image = self._require_image()
            result = invert_hsi_intensity(image)
            self._set_processed(
                result, "Яркость I инвертирована в исходном изображении."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def convert_to_grayscale(self) -> None:
        try:
            image = self._require_image()
            gray = rgb_to_grayscale_weighted(image)
            self._set_processed(gray, "Преобразовано в полутоновое изображение.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def adaptive_binarize(self) -> None:
        try:
            image = self._require_image()
            if self.state.processed is not None and self.state.processed.ndim == 2:
                gray = self.state.processed
            else:
                gray = rgb_to_grayscale_weighted(image)
            window = self.lab2_window_var.get()
            binary = adaptive_threshold_minmax(gray, window=window)
            self._set_processed(
                binary, f"Бинаризация min-max (окно {window}x{window})."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _on_lab2_window_change(self, value: str) -> None:
        try:
            val = float(value)
        except ValueError:
            return
        window = 3 if val < 14 else 25
        if self.lab2_window_var.get() != window:
            self.lab2_window_var.set(window)
            self.lab2_window_label_var.set(f"Окно: {window}×{window}")
        if abs(val - window) > 0.5:
            self.lab2_window_scale.set(window)

    def apply_lab3_filter(self) -> None:
        try:
            image = self._require_image()
            source = self.state.processed if self.state.processed is not None else image
            if source.ndim == 3:
                gray = rgb_to_grayscale_weighted(source)
            else:
                gray = source

            filtered = fringe_erase_black(gray, window=3)

            if self._is_binary(gray):
                diff = diff_xor(self._to_binary(gray), self._to_binary(filtered))
            else:
                diff = diff_abs(gray, filtered)
                if self.lab3_boost_diff_var.get():
                    diff = boost_diff(diff, factor=10.0)

            self.lab3_filtered = filtered
            self.lab3_diff = diff
            self.lab3_input_preview = gray
            self._set_processed_with_original(
                gray, filtered, "Lab3: фильтр 'стирание бахромы' (окно 3x3)."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def show_lab3_filtered(self) -> None:
        if self.lab3_filtered is None or self.lab3_input_preview is None:
            messagebox.showinfo("Нет результата", "Сначала примените фильтр.")
            return
        self._update_previews(self.lab3_input_preview, self.lab3_filtered)

    def show_lab3_diff(self) -> None:
        if self.lab3_diff is None or self.lab3_input_preview is None:
            messagebox.showinfo("Нет результата", "Сначала примените фильтр.")
            return
        self._update_previews(self.lab3_input_preview, self.lab3_diff)

    def save_lab3_diff(self) -> None:
        if self.lab3_diff is None:
            messagebox.showinfo("Нет результата", "Сначала примените фильтр.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить разницу",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("BMP image", "*.bmp")],
        )
        if not path:
            return
        try:
            save_gray_image(self.lab3_diff, path)
            self._append_log(f"Разностное изображение сохранено: {path}")
            messagebox.showinfo("✅", "Разностное изображение сохранено.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _is_binary(self, gray: np.ndarray) -> bool:
        return np.all((gray == 0) | (gray == 255))

    def _to_binary(self, gray: np.ndarray) -> np.ndarray:
        return np.where(gray >= 128, 255, 0).astype(np.uint8)

    def compute_lab4_edges(self) -> None:
        try:
            image = self._require_image()
            if image.ndim == 3:
                gray = rgb_to_grayscale_weighted(image)
            else:
                gray = image

            gx, gy, g = kayali_edges(gray)

            self.lab4_gray = gray
            self.lab4_gx = gx
            self.lab4_gy = gy
            self.lab4_g = g
            self.lab4_binary = self._lab4_binarize(g)

            self._set_processed_with_original(
                image, g, "Lab4: рассчитаны Gx, Gy и G (Кайяли 3x3)."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def show_lab4_gray(self) -> None:
        if not self._lab4_ready():
            return
        self.state.processed = self.lab4_gray
        self._update_previews(self.state.image, self.lab4_gray)

    def show_lab4_gx(self) -> None:
        if not self._lab4_ready():
            return
        self.state.processed = self.lab4_gx
        self._update_previews(self.state.image, self.lab4_gx)

    def show_lab4_gy(self) -> None:
        if not self._lab4_ready():
            return
        self.state.processed = self.lab4_gy
        self._update_previews(self.state.image, self.lab4_gy)

    def show_lab4_g(self) -> None:
        if not self._lab4_ready():
            return
        self.state.processed = self.lab4_g
        self._update_previews(self.state.image, self.lab4_g)

    def show_lab4_binary(self) -> None:
        if not self._lab4_ready():
            return
        try:
            self.lab4_binary = self._lab4_binarize(self.lab4_g)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        self.state.processed = self.lab4_binary
        self._update_previews(self.state.image, self.lab4_binary)

    def _lab4_ready(self) -> bool:
        if self.lab4_g is None or self.lab4_gray is None:
            messagebox.showinfo("Нет результата", "Сначала рассчитайте градиенты.")
            return False
        return True

    def _lab4_binarize(self, g: np.ndarray) -> np.ndarray:
        threshold = self._read_threshold(self.lab4_threshold_var.get())
        return np.where(g >= threshold, 255, 0).astype(np.uint8)

    def _read_float(self, value: str, name: str) -> float:
        try:
            result = float(value.replace(",", "."))
        except ValueError:
            raise ValueError(f"Параметр {name} должен быть числом.")
        if result <= 0:
            raise ValueError(f"Параметр {name} должен быть больше 0.")
        return result

    def _read_threshold(self, value: str) -> int:
        try:
            result = int(float(value.replace(",", ".")))
        except ValueError:
            raise ValueError("Порог должен быть числом.")
        if not 0 <= result <= 255:
            raise ValueError("Порог должен быть в диапазоне 0–255.")
        return result

    def stretch_image(self) -> None:
        try:
            image = self._require_image()
            m = self._read_float(self.m_var.get(), "M")
            result = stretch_manual(image, m)
            self._set_processed(result, f"Выполнено растяжение в {m} раз.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def compress_image(self) -> None:
        try:
            image = self._require_image()
            n = self._read_float(self.n_var.get(), "N")
            result = compress_manual(image, n)
            self._set_processed(result, f"Выполнено сжатие в {n} раз.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def rediscretize_two_pass_action(self) -> None:
        try:
            image = self._require_image()
            m = self._read_float(self.m_var.get(), "M")
            n = self._read_float(self.n_var.get(), "N")
            result = rediscretize_two_pass(image, m, n)
            self._set_processed(
                result, f"Передискретизация в два прохода: M={m}, N={n}, K={m / n:.4f}."
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def rediscretize_one_pass_action(self) -> None:
        try:
            image = self._require_image()
            k = self._read_float(self.k_var.get(), "K")
            result = rediscretize_one_pass(image, k)
            self._set_processed(result, f"Передискретизация за один проход: K={k}.")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
