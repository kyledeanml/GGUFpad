#!/usr/bin/env python3
"""
GGUFpad - Local GGUF LLM Studio & Workstation
Styled after PythonPad: dark neon theme (#0b0f19), headerbar with linked controls,
multi-format model file browser, sidebar inspector, interactive prompt/inference console,
and bottom statusbar.
Designed for Linux Mint Cinnamon.
"""

import sys
import os
import threading
import json
import time
import cairo
from typing import Optional, List, Dict, Any, Tuple

sys.dont_write_bytecode = True
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Pango', '1.0')

from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib, Gio
from gguf_parser import GGUFReader, SUPPORTED_MODEL_EXTENSIONS
from gguf_engine import GGUFEngine
from gguf_context import ContextFileLoader, ContextAttachment, SUPPORTED_CONTEXT_CATEGORIES


class GGUFpadWindow(Gtk.ApplicationWindow):
    """Main window for GGUFpad, matching PythonPad's aesthetics and controls."""

    APP_NAME = "GGUFpad"
    APP_VERSION = "2.1"

    def __init__(self, app, initial_file=None):
        super().__init__(application=app, title=self.APP_NAME)
        self.set_default_size(1160, 780)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Set WM Class and Icons
        self.set_wmclass("ggufpad", "GGUFpad")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufpad.svg")
        if os.path.exists(icon_path):
            try:
                Gtk.Window.set_default_icon_from_file(icon_path)
                self.set_icon_from_file(icon_path)
                pixbufs = [
                    GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, sz, sz, True)
                    for sz in (16, 24, 32, 48, 64, 128, 256)
                ]
                Gtk.Window.set_default_icon_list(pixbufs)
                self.set_icon_list(pixbufs)
            except Exception:
                pass

        # Dark mode preference
        settings = Gtk.Settings.get_default()
        if settings:
            settings.set_property("gtk-application-prefer-dark-theme", True)

        self.engine = GGUFEngine()
        self.current_reader = None
        self.current_model_path = None
        self.chat_history = []
        self.context_attachments = []
        self._watermark_pixbuf = None
        self._watermark_loaded = False

        default_models_dir = os.path.expanduser("~/Documents/models")
        if not os.path.isdir(default_models_dir):
            default_models_dir = os.path.expanduser("~/Documents")
        self.models_directory = default_models_dir

        self._apply_dark_neon_theme()
        self._build_header_bar()
        self._build_main_area()
        self._build_status_bar()
        self._setup_shortcuts()

        self.connect("delete-event", self._on_delete_event)

        # Populate the model browser list on startup
        self.refresh_model_browser()

        if initial_file and os.path.isfile(initial_file):
            self.load_model(initial_file)

        self.show_all()
        self._refresh_context_tray()

    def _on_delete_event(self, widget, event):
        self.engine.stop_server()
        return False

    def _setup_shortcuts(self):
        accel = Gtk.AccelGroup()
        self.add_accel_group(accel)
        key, mod = Gtk.accelerator_parse("<Control>o")
        accel.connect(key, mod, Gtk.AccelFlags.VISIBLE, lambda *a: self.on_open_file())
        key_b, mod_b = Gtk.accelerator_parse("<Control>b")
        accel.connect(key_b, mod_b, Gtk.AccelFlags.VISIBLE, lambda *a: self.on_browse_models_dialog())
        key_l, mod_l = Gtk.accelerator_parse("<Control>l")
        accel.connect(key_l, mod_l, Gtk.AccelFlags.VISIBLE, lambda *a: self.on_load_context_files())

    def _apply_dark_neon_theme(self):
        """PythonPad dark neon theme (#0b0f19) with cyan/neon accents."""
        css = b"""
        /* GGUFpad PythonPad-Style Dark Neon Theme */
        window, dialog, .background {
            background-color: #0b0f19;
            color: #ffffff;
        }

        headerbar {
            background: linear-gradient(to bottom, #111827, #0b0f19);
            border-bottom: 1px solid rgba(0, 240, 255, 0.35);
            box-shadow: 0 1px 6px rgba(0, 240, 255, 0.12);
            color: #ffffff;
            padding: 4px 6px;
        }
        headerbar .title {
            color: #ffffff;
            font-weight: bold;
        }
        headerbar .subtitle {
            color: #38bdf8;
            font-size: 85%;
        }

        button {
            background-color: #131d33;
            color: #ffffff;
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 4px;
            padding: 4px 8px;
            transition: all 150ms ease;
        }
        button:hover {
            background-color: #1a2744;
            border-color: #00f0ff;
            color: #00f0ff;
            box-shadow: 0 0 6px rgba(0, 240, 255, 0.35);
        }
        button:checked, button:active {
            background-color: #034b75;
            border-color: #00f0ff;
            color: #ffffff;
            box-shadow: 0 0 8px rgba(0, 240, 255, 0.45);
        }

        .linked > button {
            border-color: rgba(0, 240, 255, 0.25);
        }
        .linked > button:hover {
            border-color: #00f0ff;
        }

        .gpu-badge {
            background-color: #0b291b;
            color: #4ade80;
            border: 1px solid #16a34a;
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: bold;
            font-size: 11px;
        }

        .format-badge {
            background-color: #034b75;
            color: #00f0ff;
            border: 1px solid rgba(0, 240, 255, 0.4);
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 10px;
            font-weight: bold;
        }

        .context-tray {
            background-color: #0d1526;
            border: 1px solid rgba(0, 240, 255, 0.35);
            border-radius: 6px;
            padding: 6px 10px;
        }

        .context-chip {
            background-color: #13223b;
            border: 1px solid rgba(0, 240, 255, 0.45);
            border-radius: 4px;
            padding: 2px 6px;
        }

        .model-row-btn {
            background-color: #0d1526;
            border: 1px solid rgba(0, 240, 255, 0.18);
            border-radius: 4px;
            padding: 5px 8px;
        }
        .model-row-btn:hover {
            background-color: #162440;
            border-color: #00f0ff;
        }

        paned > separator {
            background-color: #070a14;
            border-left: 1px solid rgba(0, 240, 255, 0.25);
            border-right: 1px solid rgba(0, 240, 255, 0.25);
            min-width: 6px;
        }
        paned > separator:hover {
            background-color: #00f0ff;
            box-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
        }

        .statusbar, #statusbar {
            background-color: #070a14;
            border-top: 1px solid rgba(0, 240, 255, 0.2);
            color: #94a3b8;
            padding: 3px 10px;
        }
        .statusbar label, #statusbar label {
            color: #cbd5e1;
            font-size: 88%;
        }

        .sidebar-panel {
            background-color: #0d1322;
            border-right: 1px solid rgba(0, 240, 255, 0.2);
        }

        .meta-card {
            background-color: #0a0e1a;
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 6px;
            padding: 10px;
        }

        .section-header {
            font-size: 11px;
            font-weight: bold;
            color: #00f0ff;
            letter-spacing: 0.8px;
            margin-top: 10px;
            margin-bottom: 4px;
        }

        textview, textview text {
            background-color: #070a14;
            color: #e2e8f0;
            font-family: monospace;
        }

        entry {
            background-color: #0a0e1a;
            color: #ffffff;
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 4px;
            padding: 4px 8px;
        }
        entry:focus {
            border-color: #00f0ff;
            box-shadow: 0 0 5px rgba(0, 240, 255, 0.4);
        }

        .chat-bubble-user {
            background-color: #034b75;
            color: #ffffff;
            border: 1px solid #00f0ff;
            border-radius: 8px 8px 2px 8px;
            padding: 8px 12px;
        }

        .chat-bubble-assistant {
            background-color: #101726;
            color: #f1f5f9;
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 8px 8px 8px 2px;
            padding: 10px 14px;
        }

        .chat-meta {
            font-size: 11px;
            color: #38bdf8;
            font-weight: bold;
        }

        scrollbar {
            background-color: #070a14;
        }
        scrollbar slider {
            background-color: #1e293b;
            border-radius: 4px;
            min-width: 8px;
        }
        scrollbar slider:hover {
            background-color: #0284c7;
        }
        scrollbar slider:active {
            background-color: #00f0ff;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        screen = Gdk.Screen.get_default()
        if screen:
            Gtk.StyleContext.add_provider_for_screen(
                screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _build_header_bar(self):
        self.header = Gtk.HeaderBar()
        self.header.set_show_close_button(True)
        self.header.set_title(self.APP_NAME)
        self.header.set_subtitle("GGUF Studio & Multimodal Inference")
        self.set_titlebar(self.header)

        # File actions: Open Model, Browse Model Files (Multi-format), Clear Chat
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(btn_box.get_style_context(), "linked")

        self.btn_open = Gtk.Button.new_from_icon_name("document-open-symbolic", Gtk.IconSize.BUTTON)
        self.btn_open.set_tooltip_text("Open Model File (Ctrl+O)")
        self.btn_open.connect("clicked", lambda w: self.on_open_file())
        btn_box.pack_start(self.btn_open, False, False, 0)

        self.btn_browse_models = Gtk.Button()
        browse_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        browse_icon = Gtk.Image.new_from_icon_name("folder-visiting-symbolic", Gtk.IconSize.BUTTON)
        browse_lbl = Gtk.Label(label="Model Files")
        browse_box.pack_start(browse_icon, False, False, 0)
        browse_box.pack_start(browse_lbl, False, False, 0)
        self.btn_browse_models.add(browse_box)
        self.btn_browse_models.set_tooltip_text("Browse & Filter Model Files of Various Types (.gguf, .safetensors, .pt, .bin, .onnx) (Ctrl+B)")
        self.btn_browse_models.connect("clicked", lambda w: self.on_browse_models_dialog())
        btn_box.pack_start(self.btn_browse_models, False, False, 0)

        self.btn_load_context = Gtk.Button()
        ctx_hdr_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        ctx_hdr_icon = Gtk.Image.new_from_icon_name("document-new-symbolic", Gtk.IconSize.BUTTON)
        ctx_hdr_lbl = Gtk.Label(label="Load Context")
        ctx_hdr_box.pack_start(ctx_hdr_icon, False, False, 0)
        ctx_hdr_box.pack_start(ctx_hdr_lbl, False, False, 0)
        self.btn_load_context.add(ctx_hdr_box)
        self.btn_load_context.set_tooltip_text(
            "Load Various File Types into Context (.pdf, .docx, .py, .csv, .xlsx, .json, .md, images, etc.) (Ctrl+L)"
        )
        self.btn_load_context.connect("clicked", lambda w: self.on_load_context_files())
        btn_box.pack_start(self.btn_load_context, False, False, 0)

        self.btn_clear = Gtk.Button.new_from_icon_name("edit-clear-all-symbolic", Gtk.IconSize.BUTTON)
        self.btn_clear.set_tooltip_text("Clear Session History")
        self.btn_clear.connect("clicked", lambda w: self.clear_chat())
        btn_box.pack_start(self.btn_clear, False, False, 0)

        self.header.pack_start(btn_box)

        # GPU acceleration status badge
        gpu_info = self.engine.get_gpu_info()
        gpu_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        if gpu_info["cuda_available"]:
            gpu_label = Gtk.Label(label=f"⚡ {gpu_info['device_name']} ({gpu_info['vram_gb']}GB VRAM)")
        else:
            gpu_label = Gtk.Label(label="💻 CPU Inference")
        gpu_label.get_style_context().add_class("gpu-badge")
        gpu_box.pack_start(gpu_label, False, False, 0)
        self.header.pack_end(gpu_box)

        # Right control group: Sidebar toggle
        ctrl_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(ctrl_box.get_style_context(), "linked")

        self.btn_toggle_sidebar = Gtk.ToggleButton()
        self.btn_toggle_sidebar.set_image(Gtk.Image.new_from_icon_name("view-paged-symbolic", Gtk.IconSize.BUTTON))
        self.btn_toggle_sidebar.set_active(True)
        self.btn_toggle_sidebar.set_tooltip_text("Toggle Model Inspector Sidebar")
        self.btn_toggle_sidebar.connect("toggled", self._on_toggle_sidebar)
        ctrl_box.pack_start(self.btn_toggle_sidebar, False, False, 0)

        self.header.pack_end(ctrl_box)

    def _build_main_area(self):
        self.paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.paned.set_position(360)
        self.add(self.paned)

        # Left Column: Model Inspector & Browser Sidebar
        self.sidebar_scroll = Gtk.ScrolledWindow()
        self.sidebar_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.sidebar_scroll.get_style_context().add_class("sidebar-panel")

        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar_box.set_margin_start(14)
        sidebar_box.set_margin_end(14)
        sidebar_box.set_margin_top(14)
        sidebar_box.set_margin_bottom(14)
        self.sidebar_scroll.add(sidebar_box)
        self.paned.pack1(self.sidebar_scroll, False, False)

        # 1. Model Files Browser Section
        browser_hdr_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        lbl_browser = Gtk.Label(label="MODEL FILES BROWSER", xalign=0)
        lbl_browser.get_style_context().add_class("section-header")
        browser_hdr_row.pack_start(lbl_browser, True, True, 0)

        btn_pick_dir = Gtk.Button.new_from_icon_name("folder-open-symbolic", Gtk.IconSize.MENU)
        btn_pick_dir.set_tooltip_text("Select Models Directory")
        btn_pick_dir.connect("clicked", lambda w: self.on_select_models_folder())
        browser_hdr_row.pack_end(btn_pick_dir, False, False, 0)

        btn_refresh_dir = Gtk.Button.new_from_icon_name("view-refresh-symbolic", Gtk.IconSize.MENU)
        btn_refresh_dir.set_tooltip_text("Rescan Model Files")
        btn_refresh_dir.connect("clicked", lambda w: self.refresh_model_browser())
        browser_hdr_row.pack_end(btn_refresh_dir, False, False, 0)

        sidebar_box.pack_start(browser_hdr_row, False, False, 0)

        browser_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        browser_card.get_style_context().add_class("meta-card")

        # Type Filter Combo
        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        filter_lbl = Gtk.Label(label="File Type:", xalign=0)
        filter_lbl.get_style_context().add_class("chat-meta")
        self.type_filter_combo = Gtk.ComboBoxText()
        self.type_filters = [
            ("All Model Types", SUPPORTED_MODEL_EXTENSIONS),
            ("GGUF (.gguf)", (".gguf",)),
            ("SafeTensors (.safetensors)", (".safetensors",)),
            ("PyTorch (.pt, .pth)", (".pt", ".pth")),
            ("Binary Weights (.bin)", (".bin",)),
            ("ONNX (.onnx)", (".onnx",)),
        ]
        for label_text, _exts in self.type_filters:
            self.type_filter_combo.append_text(label_text)
        self.type_filter_combo.set_active(0)
        self.type_filter_combo.connect("changed", lambda c: self.refresh_model_browser())
        filter_row.pack_start(filter_lbl, False, False, 0)
        filter_row.pack_start(self.type_filter_combo, True, True, 0)
        browser_card.pack_start(filter_row, False, False, 0)

        # Scrollable list of discovered models
        self.model_list_scroll = Gtk.ScrolledWindow()
        self.model_list_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.model_list_scroll.set_min_content_height(135)
        self.model_list_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.model_list_scroll.add(self.model_list_box)
        browser_card.pack_start(self.model_list_scroll, False, False, 0)

        sidebar_box.pack_start(browser_card, False, False, 0)

        # 2. Model Info Card
        lbl_meta = Gtk.Label(label="MODEL SPECIFICATION", xalign=0)
        lbl_meta.get_style_context().add_class("section-header")
        sidebar_box.pack_start(lbl_meta, False, False, 0)

        self.meta_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.meta_card.get_style_context().add_class("meta-card")
        self.meta_labels = {}
        for key in ["Model", "Arch", "Params", "Format", "Context", "Blocks", "File Size"]:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            k_lbl = Gtk.Label(label=f"{key}:", xalign=0)
            k_lbl.get_style_context().add_class("chat-meta")
            v_lbl = Gtk.Label(label="—", xalign=1)
            v_lbl.set_selectable(True)
            v_lbl.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
            v_lbl.set_max_width_chars(24)
            row.pack_start(k_lbl, True, True, 0)
            row.pack_start(v_lbl, False, False, 0)
            self.meta_card.pack_start(row, False, False, 0)
            self.meta_labels[key] = v_lbl
        sidebar_box.pack_start(self.meta_card, False, False, 0)

        # 3. Hardware & Offload Tuning (Temperature slider removed)
        lbl_tuning = Gtk.Label(label="INFERENCE & GPU TUNING", xalign=0)
        lbl_tuning.get_style_context().add_class("section-header")
        sidebar_box.pack_start(lbl_tuning, False, False, 0)

        tune_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        tune_card.get_style_context().add_class("meta-card")

        # GPU Offload Slider
        self.gpu_layers_label = Gtk.Label(label="GPU Layers (-ngl): 33 (Max RTX 3060)", xalign=0)
        self.gpu_layers_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 80, 1)
        self.gpu_layers_scale.set_value(33)
        self.gpu_layers_scale.connect(
            "value-changed",
            lambda s: self.gpu_layers_label.set_text(f"GPU Layers (-ngl): {int(s.get_value())}")
        )
        tune_card.pack_start(self.gpu_layers_label, False, False, 0)
        tune_card.pack_start(self.gpu_layers_scale, False, False, 0)

        sidebar_box.pack_start(tune_card, False, False, 0)

        # 4. System Role Template
        lbl_role = Gtk.Label(label="SYSTEM ROLE TEMPLATE", xalign=0)
        lbl_role.get_style_context().add_class("section-header")
        sidebar_box.pack_start(lbl_role, False, False, 0)

        self.sys_combo = Gtk.ComboBoxText()
        self.sys_templates = {
            "General Assistant": "You are a knowledgeable, direct, and concise AI assistant.",
            "Python & CUDA Engineer": "You are an expert Linux Mint, Python 3, and CUDA systems engineer. Provide clean, production-ready code.",
            "3D & Vision Specialist": "You are an expert in Blender, 3D modeling, rendering, and computer vision.",
            "Creative Writer": "You are an imaginative creative collaborator and writer.",
        }
        for name in self.sys_templates.keys():
            self.sys_combo.append_text(name)
        self.sys_combo.set_active(0)
        self.sys_combo.connect("changed", self._on_template_changed)
        sidebar_box.pack_start(self.sys_combo, False, False, 0)

        self.sys_prompt_entry = Gtk.TextView()
        self.sys_prompt_entry.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.sys_prompt_entry.get_buffer().set_text(self.sys_templates["General Assistant"])
        sys_scroll = Gtk.ScrolledWindow()
        sys_scroll.set_min_content_height(65)
        sys_scroll.add(self.sys_prompt_entry)
        sidebar_box.pack_start(sys_scroll, False, False, 0)

        # Right Column: Chat Workspace
        workspace_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.paned.pack2(workspace_box, True, False)

        # Message Scroll Area with translucent Yoga-Pad + Floating GPU watermark
        self.chat_scroll = Gtk.ScrolledWindow()
        self.chat_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.chat_scroll.connect_after("draw", self._on_draw_chat_watermark)

        # Overlay chat messages over the canvas
        self.chat_messages_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        self.chat_messages_box.set_margin_start(20)
        self.chat_messages_box.set_margin_end(20)
        self.chat_messages_box.set_margin_top(16)
        self.chat_messages_box.set_margin_bottom(16)
        self.chat_scroll.add(self.chat_messages_box)
        workspace_box.pack_start(self.chat_scroll, True, True, 0)

        # Welcome Card
        self._add_system_message(
            "Welcome to **GGUFpad**.\n\n"
            "• Click any model in the **Model Files Browser** on the left (or click **Model Files / Open**) to load `.gguf`, `.safetensors`, `.pt`, `.bin`, or `.onnx` files.\n"
            "• Click **Load File / Load Context (Ctrl+L)** to attach documents (`.pdf`, `.docx`, `.odt`), code (`.py`, `.js`, `.cpp`, `.ipynb`), data (`.csv`, `.xlsx`, `.json`, `.yaml`), or images into context.\n"
            "• Hardware acceleration is connected to your **NVIDIA GeForce RTX 3060 (12GB VRAM)**.\n"
            "• Type your prompt in the editor below and press **Enter** (or click Send)."
        )

        # Bottom Input Bar
        input_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        input_container.set_margin_start(16)
        input_container.set_margin_end(16)
        input_container.set_margin_bottom(10)
        input_container.set_margin_top(6)

        # Active Context Files Tray (shown when files are loaded into context)
        self.context_tray_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.context_tray_box.get_style_context().add_class("context-tray")

        ctx_title = Gtk.Label(label="CONTEXT FILES:", xalign=0)
        ctx_title.get_style_context().add_class("chat-meta")
        self.context_tray_box.pack_start(ctx_title, False, False, 0)

        self.context_chips_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        chips_scroll = Gtk.ScrolledWindow()
        chips_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        chips_scroll.set_min_content_height(28)
        chips_scroll.add(self.context_chips_box)
        self.context_tray_box.pack_start(chips_scroll, True, True, 0)

        self.btn_clear_context = Gtk.Button(label="Clear Context")
        self.btn_clear_context.set_tooltip_text("Remove all files from active context")
        self.btn_clear_context.connect("clicked", lambda w: self.clear_context_files())
        self.context_tray_box.pack_end(self.btn_clear_context, False, False, 0)

        input_container.pack_start(self.context_tray_box, False, False, 0)

        input_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        # Load File into Context Button (right next to prompt input)
        self.btn_attach_context = Gtk.Button()
        att_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        att_box.set_valign(Gtk.Align.CENTER)
        att_icon = Gtk.Image.new_from_icon_name("document-new-symbolic", Gtk.IconSize.BUTTON)
        att_lbl = Gtk.Label(label="Load File")
        att_lbl.get_style_context().add_class("chat-meta")
        att_box.pack_start(att_icon, False, False, 0)
        att_box.pack_start(att_lbl, False, False, 0)
        self.btn_attach_context.add(att_box)
        self.btn_attach_context.set_tooltip_text(
            "Load Various File Types into Context (.pdf, .docx, .py, .csv, .xlsx, .json, .md, images, etc.) (Ctrl+L)"
        )
        self.btn_attach_context.connect("clicked", lambda w: self.on_load_context_files())
        input_row.pack_start(self.btn_attach_context, False, False, 0)

        self.input_entry = Gtk.TextView()
        self.input_entry.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.input_entry.get_style_context().add_class("meta-card")
        self.input_entry.connect("key-press-event", self._on_input_key_press)

        input_scroll = Gtk.ScrolledWindow()
        input_scroll.set_min_content_height(55)
        input_scroll.set_max_content_height(130)
        input_scroll.add(self.input_entry)
        input_row.pack_start(input_scroll, True, True, 0)

        # Send Button
        self.btn_send = Gtk.Button.new_from_icon_name("mail-send-symbolic", Gtk.IconSize.BUTTON)
        self.btn_send.set_tooltip_text("Send Prompt (Enter)")
        self.btn_send.connect("clicked", lambda w: self.on_send())
        input_row.pack_start(self.btn_send, False, False, 0)

        input_container.pack_start(input_row, False, False, 0)
        workspace_box.pack_start(input_container, False, False, 0)

    def refresh_model_browser(self):
        """Scans self.models_directory (and common model paths) for model files of selected types."""
        for child in self.model_list_box.get_children():
            self.model_list_box.remove(child)

        idx = self.type_filter_combo.get_active()
        if idx < 0 or idx >= len(self.type_filters):
            idx = 0
        allowed_exts = self.type_filters[idx][1]

        found_files = []
        if os.path.isdir(self.models_directory):
            try:
                for root, _dirs, files in os.walk(self.models_directory):
                    for fname in sorted(files):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in allowed_exts:
                            full_p = os.path.join(root, fname)
                            try:
                                sz_gb = os.path.getsize(full_p) / (1024 ** 3)
                            except OSError:
                                sz_gb = 0.0
                            found_files.append((fname, full_p, ext.lstrip(".").upper(), sz_gb))
            except Exception:
                pass

        if not found_files:
            empty_lbl = Gtk.Label(
                label=f"No matching model files in:\n{self.models_directory}",
                xalign=0,
            )
            empty_lbl.set_line_wrap(True)
            empty_lbl.get_style_context().add_class("chat-meta")
            self.model_list_box.pack_start(empty_lbl, False, False, 4)
        else:
            for fname, full_p, badge_txt, sz_gb in found_files:
                btn = Gtk.Button()
                btn.get_style_context().add_class("model-row-btn")
                btn.set_tooltip_text(full_p)
                row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

                badge = Gtk.Label(label=badge_txt)
                badge.get_style_context().add_class("format-badge")
                row.pack_start(badge, False, False, 0)

                name_lbl = Gtk.Label(label=fname, xalign=0)
                name_lbl.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
                name_lbl.set_max_width_chars(22)
                row.pack_start(name_lbl, True, True, 0)

                sz_lbl = Gtk.Label(label=f"{sz_gb:.1f}G", xalign=1)
                sz_lbl.get_style_context().add_class("chat-meta")
                row.pack_end(sz_lbl, False, False, 0)

                btn.add(row)
                btn.connect("clicked", lambda _b, p=full_p: self.load_model(p))
                self.model_list_box.pack_start(btn, False, False, 0)

        self.model_list_box.show_all()

    def on_select_models_folder(self):
        dialog = Gtk.FileChooserDialog(
            title="Select Directory Containing Models",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT,
        )
        if os.path.isdir(self.models_directory):
            dialog.set_current_folder(self.models_directory)
        resp = dialog.run()
        if resp == Gtk.ResponseType.ACCEPT:
            folder = dialog.get_filename()
            if folder and os.path.isdir(folder):
                self.models_directory = folder
                self.refresh_model_browser()
        dialog.destroy()

    def on_browse_models_dialog(self):
        """Shows a dedicated multi-format Model Files Browser dialog."""
        dialog = Gtk.Dialog(
            title="Model Files Browser — All Formats",
            transient_for=self,
            flags=0,
        )
        dialog.add_buttons(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)
        dialog.set_default_size(680, 440)

        content = dialog.get_content_area()
        content.set_spacing(10)
        content.set_margin_start(14)
        content.set_margin_end(14)
        content.set_margin_top(14)
        content.set_margin_bottom(14)

        top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dir_lbl = Gtk.Label(label=f"Folder: {self.models_directory}", xalign=0)
        dir_lbl.set_ellipsize(Pango.EllipsizeMode.START)
        top_bar.pack_start(dir_lbl, True, True, 0)

        filter_combo = Gtk.ComboBoxText()
        for label_text, _exts in self.type_filters:
            filter_combo.append_text(label_text)
        filter_combo.set_active(self.type_filter_combo.get_active())
        top_bar.pack_start(filter_combo, False, False, 0)

        change_dir_btn = Gtk.Button(label="Change Folder...")
        top_bar.pack_start(change_dir_btn, False, False, 0)
        content.pack_start(top_bar, False, False, 0)

        # TreeView / ListStore of model files
        store = Gtk.ListStore(str, str, str, str)
        tree = Gtk.TreeView(model=store)
        for idx, col_title in enumerate(["Format", "Model Filename", "Size", "Full Path"]):
            renderer = Gtk.CellRendererText()
            if idx == 1:
                renderer.set_property("ellipsize", Pango.EllipsizeMode.MIDDLE)
            col = Gtk.TreeViewColumn(col_title, renderer, text=idx)
            col.set_resizable(True)
            if idx == 1:
                col.set_expand(True)
            tree.append_column(col)

        def populate_store():
            store.clear()
            f_idx = filter_combo.get_active()
            if f_idx < 0 or f_idx >= len(self.type_filters):
                f_idx = 0
            allowed = self.type_filters[f_idx][1]
            if os.path.isdir(self.models_directory):
                for root, _dirs, files in os.walk(self.models_directory):
                    for fname in sorted(files):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in allowed:
                            full_p = os.path.join(root, fname)
                            try:
                                sz_gb = f"{os.path.getsize(full_p) / (1024**3):.2f} GB"
                            except OSError:
                                sz_gb = "—"
                            store.append([ext.lstrip(".").upper(), fname, sz_gb, full_p])

        populate_store()
        filter_combo.connect("changed", lambda _c: populate_store())

        def on_change_folder(_b):
            self.on_select_models_folder()
            dir_lbl.set_text(f"Folder: {self.models_directory}")
            populate_store()

        change_dir_btn.connect("clicked", on_change_folder)

        def on_row_activated(tv, path, _col):
            model_iter = store.get_iter(path)
            if model_iter:
                full_p = store.get_value(model_iter, 3)
                dialog.response(Gtk.ResponseType.CLOSE)
                self.load_model(full_p)

        tree.connect("row-activated", on_row_activated)

        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.add(tree)
        content.pack_start(scroll, True, True, 0)

        hint_lbl = Gtk.Label(
            label="Double-click any model file to inspect & load it into GGUFpad.",
            xalign=0,
        )
        hint_lbl.get_style_context().add_class("chat-meta")
        content.pack_start(hint_lbl, False, False, 0)

        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def _get_watermark_pixbuf(self, target_size=360):
        """Load and cache the official GGUFpad SVG logo (GPU floating on yoga pad)."""
        if not self._watermark_loaded:
            self._watermark_loaded = True
            svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufpad.svg")
            if os.path.exists(svg_path):
                try:
                    self._watermark_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                        svg_path, target_size, target_size, True
                    )
                except Exception:
                    self._watermark_pixbuf = None
        return self._watermark_pixbuf

    def _on_draw_chat_watermark(self, widget, cr):
        """Render the translucent GGUFpad floating-GPU yoga pad watermark in the chat canvas background."""
        pixbuf = self._get_watermark_pixbuf(360)
        if pixbuf is None:
            return False

        try:
            alloc = widget.get_allocation()
            view_w = alloc.width
            view_h = alloc.height
            pw = pixbuf.get_width()
            ph = pixbuf.get_height()

            if view_w < 150 or view_h < 150:
                return False

            x = max(10, (view_w - pw) * 0.5)
            y = max(10, (view_h - ph) * 0.55)

            cr.save()
            cr.set_operator(cairo.OPERATOR_OVER)
            Gdk.cairo_set_source_pixbuf(cr, pixbuf, x, y)
            cr.paint_with_alpha(0.22)
            cr.restore()
        except Exception:
            pass
        return False

    def _build_status_bar(self):
        self.statusbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        self.statusbar.set_name("statusbar")
        self.statusbar.get_style_context().add_class("statusbar")

        self.status_model_label = Gtk.Label(label="Model: None loaded", xalign=0)
        self.status_gpu_label = Gtk.Label(label="GPU: Ready (NVIDIA RTX 3060)", xalign=0)
        self.status_tokens_label = Gtk.Label(label="Tokens: 0", xalign=1)

        self.statusbar.pack_start(self.status_model_label, True, True, 0)
        self.statusbar.pack_start(self.status_gpu_label, True, True, 0)
        self.statusbar.pack_end(self.status_tokens_label, False, False, 0)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.remove(self.paned)
        main_vbox.pack_start(self.paned, True, True, 0)
        main_vbox.pack_start(self.statusbar, False, False, 0)
        self.add(main_vbox)

    def _on_toggle_sidebar(self, btn):
        if btn.get_active():
            self.sidebar_scroll.show()
            self.paned.set_position(360)
        else:
            self.sidebar_scroll.hide()

    def _on_template_changed(self, combo):
        name = combo.get_active_text()
        if name in self.sys_templates:
            self.sys_prompt_entry.get_buffer().set_text(self.sys_templates[name])

    def on_open_file(self):
        dialog = Gtk.FileChooserDialog(
            title="Open Model File",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT
        )

        if os.path.isdir(self.models_directory):
            dialog.set_current_folder(self.models_directory)

        filter_models = Gtk.FileFilter()
        filter_models.set_name("All Supported Models (*.gguf, *.safetensors, *.pt, *.pth, *.bin, *.onnx)")
        for ext in SUPPORTED_MODEL_EXTENSIONS:
            filter_models.add_pattern(f"*{ext}")
        dialog.add_filter(filter_models)

        filter_gguf = Gtk.FileFilter()
        filter_gguf.set_name("GGUF Models (*.gguf)")
        filter_gguf.add_pattern("*.gguf")
        dialog.add_filter(filter_gguf)

        filter_st = Gtk.FileFilter()
        filter_st.set_name("SafeTensors Models (*.safetensors)")
        filter_st.add_pattern("*.safetensors")
        dialog.add_filter(filter_st)

        filter_pt = Gtk.FileFilter()
        filter_pt.set_name("PyTorch / Binary Models (*.pt, *.pth, *.bin)")
        filter_pt.add_pattern("*.pt")
        filter_pt.add_pattern("*.pth")
        filter_pt.add_pattern("*.bin")
        dialog.add_filter(filter_pt)

        filter_onnx = Gtk.FileFilter()
        filter_onnx.set_name("ONNX Models (*.onnx)")
        filter_onnx.add_pattern("*.onnx")
        dialog.add_filter(filter_onnx)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All Files (*)")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            filename = dialog.get_filename()
            dialog.destroy()
            self.load_model(filename)
        else:
            dialog.destroy()

    def load_model(self, file_path: str):
        try:
            self.current_reader = GGUFReader(file_path)
            self.current_model_path = file_path
            summary = self.current_reader.get_summary()

            self.meta_labels["Model"].set_text(summary["Filename"])
            self.meta_labels["Arch"].set_text(summary["Architecture"])
            self.meta_labels["Params"].set_text(summary["Parameters"])
            self.meta_labels["Format"].set_text(summary["Quantization"])
            self.meta_labels["Context"].set_text(str(summary["Context Length"]))
            self.meta_labels["Blocks"].set_text(str(summary["Layers (Blocks)"]))
            self.meta_labels["File Size"].set_text(summary["Size"])

            blocks = summary["Layers (Blocks)"]
            if blocks > 0:
                self.gpu_layers_scale.set_range(0, blocks + 4)
                self.gpu_layers_scale.set_value(blocks)

            self.status_model_label.set_text(f"Model: {summary['Filename']} ({summary['Quantization']})")

            self._add_system_message(
                f"Loaded Model: **{summary['Filename']}**\n\n"
                f"• **Architecture**: {summary['Architecture']}\n"
                f"• **Parameters**: {summary['Parameters']}\n"
                f"• **Format / Quantization**: {summary['Quantization']}\n"
                f"• **Context Window**: {summary['Context Length']} tokens\n"
                f"• **Transformer Blocks**: {blocks} (Configured for RTX 3060 offloading)"
            )

            # Pre-warm llama-server in the background if this is a GGUF model
            if self.current_reader.is_gguf:
                ngl = int(self.gpu_layers_scale.get_value())
                self.status_gpu_label.set_text("GPU: Loading model into RTX 3060 VRAM...")

                def bg_warm():
                    started = self.engine.start_llama_server(
                        model_path=file_path,
                        n_gpu_layers=ngl,
                        port=8080,
                        on_ready=lambda: GLib.idle_add(
                            self.status_gpu_label.set_text, "GPU: Ready & Loaded (RTX 3060 CUDA)"
                        ),
                    )
                    if started and self.engine.wait_until_ready(timeout=60.0):
                        GLib.idle_add(
                            self.status_gpu_label.set_text, "GPU: Ready & Loaded (RTX 3060 CUDA)"
                        )

                threading.Thread(target=bg_warm, daemon=True).start()
        except Exception as e:
            self._add_system_message(f"⚠️ Error reading model file: {str(e)}")

    def _add_system_message(self, text: str):
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bubble.get_style_context().add_class("chat-bubble-assistant")

        meta = Gtk.Label(label="GGUFpad System", xalign=0)
        meta.get_style_context().add_class("chat-meta")

        content = Gtk.Label(label=text, xalign=0)
        content.set_line_wrap(True)
        content.set_selectable(True)

        bubble.pack_start(meta, False, False, 0)
        bubble.pack_start(content, False, False, 0)
        self.chat_messages_box.pack_start(bubble, False, False, 0)
        self.chat_messages_box.show_all()
        self._scroll_to_bottom()

    def _add_user_message(self, text: str):
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bubble.set_halign(Gtk.Align.END)
        bubble.get_style_context().add_class("chat-bubble-user")

        meta = Gtk.Label(label="You", xalign=1)
        meta.get_style_context().add_class("chat-meta")

        content = Gtk.Label(label=text, xalign=1)
        content.set_line_wrap(True)
        content.set_selectable(True)

        bubble.pack_start(meta, False, False, 0)
        bubble.pack_start(content, False, False, 0)
        self.chat_messages_box.pack_start(bubble, False, False, 0)
        self.chat_messages_box.show_all()
        self._scroll_to_bottom()

    def on_load_context_files(self):
        """Opens a multi-format FileChooserDialog to load one or more files into conversation context."""
        dialog = Gtk.FileChooserDialog(
            title="Load File(s) into Conversation Context",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT,
        )
        dialog.set_select_multiple(True)

        docs_dir = os.path.expanduser("~/Documents")
        if os.path.isdir(docs_dir):
            dialog.set_current_folder(docs_dir)

        for cat_name, patterns in SUPPORTED_CONTEXT_CATEGORIES:
            ff = Gtk.FileFilter()
            ff.set_name(cat_name)
            for pat in patterns:
                ff.add_pattern(pat)
                ff.add_pattern(pat.upper())
            dialog.add_filter(ff)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All Files (*)")
        filter_all.add_pattern("*")
        dialog.add_filter(filter_all)

        resp = dialog.run()
        if resp == Gtk.ResponseType.ACCEPT:
            filenames = dialog.get_filenames()
            dialog.destroy()
            for fp in filenames:
                self.add_context_file(fp)
        else:
            dialog.destroy()

    def add_context_file(self, file_path: str) -> Optional[ContextAttachment]:
        """Extracts content from file_path and adds it to active context attachments."""
        try:
            att = ContextFileLoader.load_file(file_path)
            # Replace if already loaded, otherwise append
            self.context_attachments = [
                a for a in self.context_attachments if a.file_path != file_path
            ]
            self.context_attachments.append(att)
            self._refresh_context_tray()
            self._add_system_message(
                f"📎 Loaded into Context: **{att.filename}** (`{att.file_type}` — {att.summary_label})\n"
                f"Click the file chip below the chat to preview extracted content, or type a prompt to query it."
            )
            return att
        except Exception as e:
            self._add_system_message(f"⚠️ Could not load file into context: {e}")
            return None

    def remove_context_file(self, file_path: str):
        self.context_attachments = [
            a for a in self.context_attachments if a.file_path != file_path
        ]
        self._refresh_context_tray()

    def clear_context_files(self):
        self.context_attachments.clear()
        self._refresh_context_tray()

    def _refresh_context_tray(self):
        for child in self.context_chips_box.get_children():
            self.context_chips_box.remove(child)

        if not self.context_attachments:
            self.context_tray_box.hide()
            return

        for att in self.context_attachments:
            chip = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            chip.get_style_context().add_class("context-chip")

            type_badge = Gtk.Label(label=att.file_type)
            type_badge.get_style_context().add_class("format-badge")
            chip.pack_start(type_badge, False, False, 0)

            preview_btn = Gtk.Button(label=f"{att.filename} ({att.summary_label})")
            preview_btn.set_relief(Gtk.ReliefStyle.NONE)
            preview_btn.set_tooltip_text(f"Click to preview extracted context from {att.filename}")
            preview_btn.connect("clicked", lambda _b, a=att: self._show_context_preview_dialog(a))
            chip.pack_start(preview_btn, False, False, 0)

            rm_btn = Gtk.Button(label="✕")
            rm_btn.set_relief(Gtk.ReliefStyle.NONE)
            rm_btn.set_tooltip_text(f"Remove {att.filename} from context")
            rm_btn.connect("clicked", lambda _b, p=att.file_path: self.remove_context_file(p))
            chip.pack_start(rm_btn, False, False, 0)

            self.context_chips_box.pack_start(chip, False, False, 0)

        self.context_tray_box.show_all()

    def _show_context_preview_dialog(self, att: ContextAttachment):
        dialog = Gtk.Dialog(
            title=f"Context Preview — {att.filename} ({att.file_type})",
            transient_for=self,
            flags=0,
        )
        dialog.add_buttons(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)
        dialog.set_default_size(640, 440)

        area = dialog.get_content_area()
        area.set_spacing(8)
        area.set_margin_start(12)
        area.set_margin_end(12)
        area.set_margin_top(12)
        area.set_margin_bottom(12)

        info_lbl = Gtk.Label(
            label=f"File: {att.file_path}  |  Type: {att.file_type}  |  {att.summary_label}",
            xalign=0,
        )
        info_lbl.get_style_context().add_class("chat-meta")
        area.pack_start(info_lbl, False, False, 0)

        tv = Gtk.TextView()
        tv.set_editable(False)
        tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        tv.get_buffer().set_text(att.extracted_text)

        sc = Gtk.ScrolledWindow()
        sc.set_vexpand(True)
        sc.add(tv)
        area.pack_start(sc, True, True, 0)

        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def _on_input_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_Return and not (event.state & Gdk.ModifierType.SHIFT_MASK):
            self.on_send()
            return True
        return False

    def on_send(self):
        buf = self.input_entry.get_buffer()
        start, end = buf.get_bounds()
        text = buf.get_text(start, end, True).strip()
        if not text and not self.context_attachments:
            return
        if not text and self.context_attachments:
            text = "Please analyze and summarize the attached context file(s)."

        buf.set_text("")
        if self.context_attachments:
            attached_names = ", ".join(a.filename for a in self.context_attachments)
            display_user_msg = f"📎 [Context: {attached_names}]\n{text}"
        else:
            display_user_msg = text
        self._add_user_message(display_user_msg)

        sys_buf = self.sys_prompt_entry.get_buffer()
        s_start, s_end = sys_buf.get_bounds()
        sys_prompt = sys_buf.get_text(s_start, s_end, True).strip()

        assist_bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        assist_bubble.get_style_context().add_class("chat-bubble-assistant")

        model_title = self.meta_labels["Model"].get_text()
        meta = Gtk.Label(label=f"Assistant ({model_title if model_title != '—' else 'Local GGUF'})", xalign=0)
        meta.get_style_context().add_class("chat-meta")

        content_label = Gtk.Label(label="⏳ Waiting for GPU inference...", xalign=0)
        content_label.set_line_wrap(True)
        content_label.set_selectable(True)

        assist_bubble.pack_start(meta, False, False, 0)
        assist_bubble.pack_start(content_label, False, False, 0)
        self.chat_messages_box.pack_start(assist_bubble, False, False, 0)
        self.chat_messages_box.show_all()
        self._scroll_to_bottom()

        ngl = int(self.gpu_layers_scale.get_value())
        history_snapshot = list(self.chat_history)
        context_block = "\n\n".join(a.format_for_prompt() for a in self.context_attachments)
        image_urls = [a.image_data_url for a in self.context_attachments if a.image_data_url]
        token_count = [0]
        response_text = []

        def on_status(status_msg):
            GLib.idle_add(self.status_gpu_label.set_text, status_msg)

        def on_token(token):
            response_text.append(token)
            token_count[0] += 1
            GLib.idle_add(content_label.set_text, "".join(response_text))
            GLib.idle_add(self.status_tokens_label.set_text, f"Tokens: {token_count[0]}")
            GLib.idle_add(self._scroll_to_bottom)

        def on_done(full_reply):
            hist_user = f"{context_block}\n\n{text}" if context_block else text
            self.chat_history.append({"role": "user", "content": hist_user})
            self.chat_history.append({"role": "assistant", "content": full_reply})
            GLib.idle_add(self._scroll_to_bottom)

        def on_error(err):
            msg = (
                f"{err}\n\n"
                f"ℹ️ **Inference Setup**:\n"
                f"Select a `.gguf` model from the **Model Files Browser** on the left (or click **Open**),\n"
                f"and GGUFpad will automatically launch `llama-server` on your RTX 3060."
            )
            GLib.idle_add(content_label.set_text, msg)
            GLib.idle_add(self._scroll_to_bottom)

        self.engine.stream_completion(
            prompt=text,
            system_prompt=sys_prompt,
            history=history_snapshot,
            context_block=context_block,
            image_data_urls=image_urls,
            temperature=0.7,
            model_path=self.current_model_path,
            n_gpu_layers=ngl,
            on_status=on_status,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def clear_chat(self):
        self.chat_history.clear()
        for child in self.chat_messages_box.get_children():
            self.chat_messages_box.remove(child)
        self.status_tokens_label.set_text("Tokens: 0")
        self._add_system_message("Session history cleared. Ready for new prompt.")

    def _scroll_to_bottom(self):
        adj = self.chat_scroll.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())


class GGUFpadApp(Gtk.Application):
    def __init__(self):
        super().__init__(
            application_id="org.bluechile.ggufpad",
            flags=Gio.ApplicationFlags.HANDLES_OPEN | Gio.ApplicationFlags.NON_UNIQUE
        )
        self.window = None

    def do_activate(self):
        if not self.window:
            self.window = GGUFpadWindow(self)
        self.window.present()

    def do_open(self, files, hint):
        if not self.window:
            self.window = GGUFpadWindow(self)
        if files:
            path = files[0].get_path()
            if path and os.path.isfile(path):
                self.window.load_model(path)
        self.window.present()


def main():
    app = GGUFpadApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
