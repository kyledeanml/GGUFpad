#!/usr/bin/env python3
"""
GGUFpad - Local GGUF LLM Studio & Workstation
Styled after PythonPad: dark neon theme (#0b0f19), headerbar with linked controls,
sidebar inspector, interactive prompt/inference console, and bottom statusbar.
Designed for Linux Mint Cinnamon.
"""

import sys
import os
import threading
import json
import time

sys.dont_write_bytecode = True
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
gi.require_version('Pango', '1.0')

from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib, Gio
from gguf_parser import GGUFReader
from gguf_engine import GGUFEngine


class GGUFpadWindow(Gtk.ApplicationWindow):
    """Main window for GGUFpad, matching PythonPad's aesthetics and controls."""

    APP_NAME = "GGUFpad"
    APP_VERSION = "2.0"

    def __init__(self, app, initial_file=None):
        super().__init__(application=app, title=self.APP_NAME)
        self.set_default_size(1120, 760)
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

        self._apply_dark_neon_theme()
        self._build_header_bar()
        self._build_main_area()
        self._build_status_bar()

        if initial_file and os.path.isfile(initial_file):
            self.load_model(initial_file)

        self.show_all()

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

        # File actions: Open GGUF, Clear Chat
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        Gtk.StyleContext.add_class(btn_box.get_style_context(), "linked")

        self.btn_open = Gtk.Button.new_from_icon_name("document-open-symbolic", Gtk.IconSize.BUTTON)
        self.btn_open.set_tooltip_text("Open GGUF Model (Ctrl+O)")
        self.btn_open.connect("clicked", lambda w: self.on_open_file())
        btn_box.pack_start(self.btn_open, False, False, 0)

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

        # Right control group: Sidebar toggle & Refresh
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
        self.paned.set_position(340)
        self.add(self.paned)

        # Left Column: Model Inspector & Parameters Sidebar
        self.sidebar_scroll = Gtk.ScrolledWindow()
        self.sidebar_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.sidebar_scroll.get_style_context().add_class("sidebar-panel")

        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        sidebar_box.set_margin_start(14)
        sidebar_box.set_margin_end(14)
        sidebar_box.set_margin_top(14)
        sidebar_box.set_margin_bottom(14)
        self.sidebar_scroll.add(sidebar_box)
        self.paned.pack1(self.sidebar_scroll, False, False)

        # Model Info Card
        lbl_meta = Gtk.Label(label="GGUF MODEL SPECIFICATION", xalign=0)
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
            row.pack_start(k_lbl, True, True, 0)
            row.pack_start(v_lbl, False, False, 0)
            self.meta_card.pack_start(row, False, False, 0)
            self.meta_labels[key] = v_lbl
        sidebar_box.pack_start(self.meta_card, False, False, 0)

        # Hardware & Offload Tuning
        lbl_tuning = Gtk.Label(label="INFERENCE & GPU TUNING", xalign=0)
        lbl_tuning.get_style_context().add_class("section-header")
        sidebar_box.pack_start(lbl_tuning, False, False, 0)

        tune_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        tune_card.get_style_context().add_class("meta-card")

        # GPU Offload Slider
        self.gpu_layers_label = Gtk.Label(label="GPU Layers (-ngl): 33 (Max RTX 3060)", xalign=0)
        self.gpu_layers_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 80, 1)
        self.gpu_layers_scale.set_value(33)
        self.gpu_layers_scale.connect("value-changed", lambda s: self.gpu_layers_label.set_text(f"GPU Layers (-ngl): {int(s.get_value())}"))
        tune_card.pack_start(self.gpu_layers_label, False, False, 0)
        tune_card.pack_start(self.gpu_layers_scale, False, False, 0)

        # Temperature Slider
        self.temp_label = Gtk.Label(label="Temperature: 0.70", xalign=0)
        self.temp_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.5, 0.05)
        self.temp_scale.set_value(0.70)
        self.temp_scale.connect("value-changed", lambda s: self.temp_label.set_text(f"Temperature: {s.get_value():.2f}"))
        tune_card.pack_start(self.temp_label, False, False, 0)
        tune_card.pack_start(self.temp_scale, False, False, 0)

        sidebar_box.pack_start(tune_card, False, False, 0)

        # System Role Template
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
        sys_scroll.set_min_content_height(75)
        sys_scroll.add(self.sys_prompt_entry)
        sidebar_box.pack_start(sys_scroll, False, False, 0)

        # Right Column: Chat Workspace
        workspace_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.paned.pack2(workspace_box, True, False)

        # Message Scroll Area
        self.chat_scroll = Gtk.ScrolledWindow()
        self.chat_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
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
            "• Click **Open (Ctrl+O)** to inspect any `.gguf` model architecture, quantization, and parameters.\n"
            "• Hardware acceleration is connected to your **NVIDIA GeForce RTX 3060 (12GB VRAM)**.\n"
            "• Type your prompt in the editor below and press **Enter** (or click Send)."
        )

        # Bottom Input Bar
        input_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        input_container.set_margin_start(16)
        input_container.set_margin_end(16)
        input_container.set_margin_bottom(10)
        input_container.set_margin_top(6)

        input_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
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

        self.paned.get_parent() or None
        # Add to window vertical layout
        # Re-pack with a main vertical box
        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.remove(self.paned)
        main_vbox.pack_start(self.paned, True, True, 0)
        main_vbox.pack_start(self.statusbar, False, False, 0)
        self.add(main_vbox)

    def _on_toggle_sidebar(self, btn):
        if btn.get_active():
            self.sidebar_scroll.show()
            self.paned.set_position(340)
        else:
            self.sidebar_scroll.hide()

    def _on_template_changed(self, combo):
        name = combo.get_active_text()
        if name in self.sys_templates:
            self.sys_prompt_entry.get_buffer().set_text(self.sys_templates[name])

    def on_open_file(self):
        dialog = Gtk.FileChooserDialog(
            title="Open GGUF Model",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT
        )

        filter_gguf = Gtk.FileFilter()
        filter_gguf.set_name("GGUF Models (*.gguf)")
        filter_gguf.add_pattern("*.gguf")
        dialog.add_filter(filter_gguf)

        filter_all = Gtk.FileFilter()
        filter_all.set_name("All Files")
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
                f"Loaded GGUF Model: **{summary['Filename']}**\n\n"
                f"• **Architecture**: {summary['Architecture']}\n"
                f"• **Parameters**: {summary['Parameters']}\n"
                f"• **Quantization**: {summary['Quantization']}\n"
                f"• **Context Window**: {summary['Context Length']} tokens\n"
                f"• **Transformer Blocks**: {blocks} (Configured for RTX 3060 offloading)"
            )
        except Exception as e:
            self._add_system_message(f"⚠️ Error reading GGUF file: {str(e)}")

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

    def _on_input_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_Return and not (event.state & Gdk.ModifierType.SHIFT_MASK):
            self.on_send()
            return True
        return False

    def on_send(self):
        buf = self.input_entry.get_buffer()
        start, end = buf.get_bounds()
        text = buf.get_text(start, end, True).strip()
        if not text:
            return

        buf.set_text("")
        self._add_user_message(text)

        sys_buf = self.sys_prompt_entry.get_buffer()
        s_start, s_end = sys_buf.get_bounds()
        sys_prompt = sys_buf.get_text(s_start, s_end, True).strip()
        temp = self.temp_scale.get_value()

        assist_bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        assist_bubble.get_style_context().add_class("chat-bubble-assistant")

        model_title = self.meta_labels["Model"].get_text()
        meta = Gtk.Label(label=f"Assistant ({model_title if model_title != '—' else 'Local GGUF'})", xalign=0)
        meta.get_style_context().add_class("chat-meta")

        content_label = Gtk.Label(label="", xalign=0)
        content_label.set_line_wrap(True)
        content_label.set_selectable(True)

        assist_bubble.pack_start(meta, False, False, 0)
        assist_bubble.pack_start(content_label, False, False, 0)
        self.chat_messages_box.pack_start(assist_bubble, False, False, 0)
        self.chat_messages_box.show_all()
        self._scroll_to_bottom()

        token_count = [0]
        response_text = []

        def on_token(token):
            response_text.append(token)
            token_count[0] += 1
            GLib.idle_add(content_label.set_text, "".join(response_text))
            GLib.idle_add(self.status_tokens_label.set_text, f"Tokens: {token_count[0]}")
            GLib.idle_add(self._scroll_to_bottom)

        def on_done():
            GLib.idle_add(self._scroll_to_bottom)

        def on_error(err):
            msg = (
                f"{err}\n\n"
                f"ℹ️ **Inference Setup**:\n"
                f"To stream completions directly, run:\n"
                f"   `llama-server -m <model>.gguf -ngl {int(self.gpu_layers_scale.get_value())} --port 8080`\n"
                f"or connect with Ollama / any OpenAI-compatible GGUF endpoint."
            )
            GLib.idle_add(content_label.set_text, msg)
            GLib.idle_add(self._scroll_to_bottom)

        self.engine.stream_completion(
            prompt=text,
            system_prompt=sys_prompt,
            temperature=temp,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def clear_chat(self):
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
