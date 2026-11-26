"""Team 7 desktop capture UI. Run with Python; recording starts only on request."""
import argparse
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from capture.camera_core import CaptureGroup, discover

BASE = Path(__file__).resolve().parent.parent


def configure_display():
    # Avoid Windows bitmap scaling placing the controls beyond the laptop display.
    if os.name == 'nt':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    cv2.setNumThreads(2)


class RecorderApp:
    def __init__(self, root, auto_preview=False):
        self.root = root
        self.group = CaptureGroup()
        self.events = queue.Queue()
        self.devices, self.combos, self.canvases, self.card_status = {}, [], [], []
        self.photos, self.last_drawn = {}, {}
        self.rec = None
        self.busy = False
        self.disconnecting = self.closing = False
        self.auto_preview = auto_preview
        root.title('Team 7 · 멀티 카메라 녹화')
        width = min(1180, root.winfo_screenwidth()-60)
        height = min(840, root.winfo_screenheight()-100)
        root.geometry(f'{width}x{height}+20+20')
        root.minsize(min(900, width), min(620, height))
        root.configure(bg='#101827')
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#101827')
        style.configure('TLabel', background='#101827', foreground='#e6edf5', font=('맑은 고딕', 10))
        style.configure('TButton', font=('맑은 고딕', 10), padding=(12, 7))
        style.configure('Title.TLabel', font=('맑은 고딕', 21, 'bold'))
        style.configure('Hint.TLabel', foreground='#99adc7', font=('맑은 고딕', 9))
        style.configure('Card.TFrame', background='#1b283b')
        style.configure('Card.TLabel', background='#1b283b', foreground='#cbd8e8')
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='멀티 카메라 스튜디오', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='TEAM 7  /  최대 4대 미리보기 · 전체 녹화 · 프레임 시각 기록',
                  style='Hint.TLabel').pack(anchor='w', pady=(3, 14))
        bar = ttk.Frame(outer)
        bar.pack(fill='x')
        self.scan_button = ttk.Button(bar, text='카메라 새로 찾기', command=self.scan)
        self.scan_button.pack(side='left')
        self.backend = tk.StringVar(value='DirectShow')
        self.backend_combo = ttk.Combobox(bar, textvariable=self.backend, width=12,
                                        values=['DirectShow', 'MSMF'], state='readonly')
        self.backend_combo.pack(side='left', padx=7)
        self.backend_combo.bind('<<ComboboxSelected>>', lambda _: self.scan())
        self.resolution = tk.StringVar(value='1280x720')
        self.res_combo = ttk.Combobox(bar, textvariable=self.resolution, width=12,
                                     values=['640x480', '1280x720', '1920x1080'], state='readonly')
        self.res_combo.pack(side='left', padx=(12, 4))
        self.fps = tk.StringVar(value='30')
        self.fps_combo = ttk.Combobox(bar, textvariable=self.fps, width=4,
                                     values=['15', '30'], state='readonly')
        self.fps_combo.pack(side='left')
        ttk.Label(bar, text=' FPS').pack(side='left')
        self.connect_button = ttk.Button(bar, text='선택 카메라 켜기', command=self.toggle_connection)
        self.connect_button.pack(side='right')
        grid = ttk.Frame(outer)
        grid.pack(fill='both', expand=True, pady=12)
        for i in range(2):
            grid.columnconfigure(i, weight=1, uniform='camera')
            grid.rowconfigure(i, weight=1, uniform='camera')
        for i in range(4):
            card = ttk.Frame(grid, padding=10, style='Card.TFrame')
            card.grid(row=i//2, column=i%2, sticky='nsew', padx=4, pady=4)
            ttk.Label(card, text=f'CAMERA {i+1:02d}', style='Card.TLabel').pack(anchor='w')
            combo = ttk.Combobox(card, state='readonly', values=['사용 안 함'])
            combo.set('사용 안 함')
            combo.pack(fill='x', pady=5)
            self.combos.append(combo)
            canvas = tk.Canvas(card, bg='#0a111e', highlightthickness=0, width=360, height=140)
            canvas.pack(fill='both', expand=True)
            self.canvases.append(canvas)
            status = ttk.Label(card, text='카메라를 선택하세요', style='Card.TLabel')
            canvas.pack_forget()
            status.pack(side='bottom', anchor='w', pady=(5, 0))
            canvas.pack(fill='both', expand=True)
            self.card_status.append(status)
        location = ttk.Frame(outer)
        location.pack(fill='x')
        self.output = tk.StringVar(value=str(BASE / 'recordings'))
        ttk.Label(location, text='저장 폴더 ').pack(side='left')
        self.path_entry = ttk.Entry(location, textvariable=self.output)
        self.path_entry.pack(side='left', fill='x', expand=True)
        self.browse_button = ttk.Button(location, text='변경', command=self.choose_folder)
        self.browse_button.pack(side='left', padx=5)
        ttk.Button(location, text='폴더 열기', command=self.open_folder).pack(side='left')
        notes = ttk.Frame(outer)
        notes.pack(fill='x', pady=7)
        ttk.Label(notes, text='촬영 메모 ').pack(side='left')
        self.note = tk.StringVar()
        self.note_entry = ttk.Entry(notes, textvariable=self.note)
        self.note_entry.pack(side='left', fill='x', expand=True)
        controls = ttk.Frame(outer)
        controls.pack(fill='x', pady=(4, 5))
        self.record_button = tk.Button(controls, text='●  전체 녹화 시작', command=self.toggle_record,
                                      bg='#dd435c', fg='white', activebackground='#ef5b73',
                                      activeforeground='white', disabledforeground='#a9a9a9',
                                      font=('맑은 고딕', 14, 'bold'), relief='flat', padx=22, pady=9)
        self.record_button.pack(side='left')
        self.timer = ttk.Label(controls, text='00:00', font=('Consolas', 24, 'bold'))
        self.timer.pack(side='left', padx=18)
        self.status = tk.StringVar(value='카메라 목록을 확인하고 있습니다.')
        status_label = ttk.Label(outer, textvariable=self.status, wraplength=width-50)
        hint_label = ttk.Label(outer, style='Hint.TLabel', text='영상은 카메라별 AVI로 저장됩니다. USB 카메라의 정확한 시점 정렬에는 공통 점멸 신호와 CSV 시각을 함께 사용하세요.',
                               wraplength=width-50)
        # Reserve bottom controls before assigning the remaining space to previews.
        # This keeps the record button visible on smaller/high-DPI laptop screens.
        for widget in [grid, location, notes, controls]:
            widget.pack_forget()
        hint_label.pack(side='bottom', fill='x')
        status_label.pack(side='bottom', fill='x', pady=5)
        controls.pack(side='bottom', fill='x', pady=(4, 5))
        notes.pack(side='bottom', fill='x', pady=7)
        location.pack(side='bottom', fill='x')
        grid.pack(fill='both', expand=True, pady=8)
        root.after(100, self.scan)
        root.after(50, self.tick)

    def task(self, kind, fn):
        self.busy = True
        def work():
            try:
                self.events.put((kind, fn(), None))
            except Exception as exc:
                self.events.put((kind, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def scan(self):
        if self.busy or self.group.workers:
            return
        self.status.set('연결된 카메라를 찾는 중…')
        backend = cv2.CAP_DSHOW if self.backend.get() == 'DirectShow' else cv2.CAP_MSMF
        self.task('scan', lambda: discover(backend))

    def toggle_connection(self):
        if self.busy or self.rec or self.disconnecting:
            return
        if self.group.workers:
            self.group.disconnect()
            self.disconnecting = True
            self.status.set('카메라 연결을 해제하는 중…')
            return
        selected = {i+1: self.devices[c.get()] for i, c in enumerate(self.combos) if c.get() in self.devices}
        try:
            size = tuple(map(int, self.resolution.get().split('x')))
            self.group.connect(selected, size, int(self.fps.get()))
            self.photos.clear()
            self.last_drawn.clear()
            self.status.set('카메라 연결 중… 모든 선택 화면이 준비되면 녹화할 수 있습니다.')
        except Exception as exc:
            messagebox.showerror('카메라 연결', str(exc), parent=self.root)

    def toggle_record(self):
        if self.busy:
            return
        if self.rec:
            if not self.rec.stop_event.is_set():
                self.group.stop_recording()
                self.status.set('녹화 종료 · 남은 프레임과 파일 정보를 저장하는 중…')
            return
        directory, fps, note = self.output.get(), int(self.fps.get()), self.note.get()
        self.status.set('카메라별 녹화 파일을 준비하는 중…')
        self.task('record', lambda: self.group.start_recording(directory, fps, note))

    def choose_folder(self):
        path = filedialog.askdirectory(initialdir=self.output.get(), parent=self.root)
        if path:
            self.output.set(path)

    def open_folder(self):
        try:
            path = Path(self.output.get())
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror('폴더 열기', str(exc), parent=self.root)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.status.set('녹화 파일을 마무리하고 카메라 연결을 해제합니다…')

    def handle_events(self):
        while True:
            try:
                kind, result, error = self.events.get_nowait()
            except queue.Empty:
                return
            self.busy = False
            if error:
                self.status.set(error)
                if not self.closing:
                    messagebox.showerror('작업 실패', error, parent=self.root)
                continue
            if kind == 'scan':
                self.devices = {f"[{d['index']}] {d['name']}": d for d in result}
                values = ['사용 안 함'] + list(self.devices)
                # Prefer the external cameras present on this laptop. Device identity
                # is still preserved in session.json; indices may change after reconnect.
                preferred = [k for k, d in self.devices.items()
                             if not any(s in d['name'].lower() for s in ('ativ', 'integrated', 'built-in', 'virtual', 'obs'))]
                if not preferred:
                    preferred = list(self.devices)
                for i, combo in enumerate(self.combos):
                    combo.configure(values=values)
                    combo.set(preferred[i] if i < min(4, len(preferred)) else '사용 안 함')
                self.status.set(f'{len(result)}대 발견 · 사용할 카메라를 선택하고 [선택 카메라 켜기]를 누르세요.')
                if self.auto_preview and result and not self.closing:
                    self.auto_preview = False
                    self.toggle_connection()
            elif kind == 'record':
                self.rec = result
                self.status.set(f'녹화 중 · {result.folder.name}')

    def tick(self):
        self.handle_events()
        now = time.perf_counter_ns()
        if self.rec:
            elapsed = ((self.rec.stop_ns or now)-self.rec.start_ns)/1e9
            self.timer.configure(text=f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d}')
            if not self.rec.stop_event.is_set():
                bad = [(s, w) for s, w in self.group.workers.items()
                       if w.snapshot()[5] or now-w.snapshot()[2] > 5_000_000_000]
                if self.rec.error or bad:
                    if bad:
                        self.rec.error = self.rec.error or f'카메라 {bad[0][0]} 영상 수신 중단'
                    self.group.stop_recording()
            if self.rec.done.is_set():
                rec = self.rec
                self.group.stop_recording()
                self.rec = None
                drops = sum(rec.dropped.values())
                self.status.set(f"{'오류로 종료: '+rec.error if rec.error else '저장 완료'} · "
                                f'{sum(rec.counts.values())} 프레임 · 저장 대기열 누락 {drops} · {rec.folder.name}')
                if rec.error and not self.closing:
                    messagebox.showerror('녹화 오류', rec.error, parent=self.root)
        if self.closing and not self.busy:
            if self.group.recording:
                self.group.stop_recording()
            self.group.disconnect()
            if self.group.clear_stopped() and self.rec is None:
                self.root.destroy()
                return
        if self.disconnecting and self.group.clear_stopped():
            self.disconnecting = False
            self.status.set('카메라 연결을 해제했습니다.')
        for i, canvas in enumerate(self.canvases):
            worker = self.group.workers.get(i+1)
            cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
            if worker:
                frame, seq, stamp, rate, state, error = worker.snapshot()
                stalled = stamp and now-stamp > 2_000_000_000
                frames = (f' · 저장 {self.rec.counts.get(i+1, 0)} / 누락 {self.rec.dropped.get(i+1, 0)}'
                          if self.rec else '')
                size = worker.actual.get('size_px', ['-', '-'])
                text = error or f'{"수신 지연" if stalled else state} · {size[0]}×{size[1]} · {rate:.1f} fps{frames}'
                self.card_status[i].configure(text=text)
                key = seq, cw, ch
                if frame is not None and self.last_drawn.get(i) != key:
                    scale = min(cw/frame.shape[1], ch/frame.shape[0], 1)
                    small = cv2.resize(frame, (max(1, int(frame.shape[1]*scale)), max(1, int(frame.shape[0]*scale))))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    picture = Image.fromarray(rgb)
                    picture.thumbnail((cw, ch), Image.Resampling.BILINEAR)
                    self.photos[i] = ImageTk.PhotoImage(picture)
                    canvas.delete('all')
                    canvas.create_image(cw//2, ch//2, image=self.photos[i])
                    if stalled:
                        canvas.create_text(cw//2, 20, text='수신 지연 — 마지막 프레임', fill='#ffbb66')
                    self.last_drawn[i] = key
                elif frame is None:
                    canvas.delete('all')
                    canvas.create_text(cw//2, ch//2, text='연결 확인 중…' if not error else '연결 실패', fill='#8da6c4')
            else:
                canvas.delete('all')
                canvas.create_text(cw//2, ch//2, text=f'CAMERA {i+1:02d}\n사용할 카메라를 선택하세요',
                                   fill='#647c9a', justify='center', font=('맑은 고딕', 12))
                self.card_status[i].configure(text='대기')
        connected = bool(self.group.workers)
        blocked = self.busy or self.closing or self.disconnecting
        for widget in [self.scan_button, self.backend_combo, self.res_combo, self.fps_combo, *self.combos]:
            widget.configure(state='disabled' if connected or blocked else ('readonly' if isinstance(widget, ttk.Combobox) else 'normal'))
        self.connect_button.configure(text='카메라 끄기' if connected else '선택 카메라 켜기',
                                      state='disabled' if blocked or self.rec else 'normal')
        self.record_button.configure(text='■  전체 녹화 종료' if self.rec else '●  전체 녹화 시작',
                                     bg='#354968' if self.rec else '#dd435c',
                                     state='disabled' if blocked or (self.rec and self.rec.stop_event.is_set())
                                     or (not self.rec and not self.group.ready()) else 'normal')
        for widget in [self.path_entry, self.browse_button, self.note_entry]:
            widget.configure(state='disabled' if self.rec or blocked else 'normal')
        self.root.after(80, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='Open selected cameras after discovery; do not record.')
    args = parser.parse_args()
    configure_display()
    root = tk.Tk()
    RecorderApp(root, args.preview)
    root.mainloop()


if __name__ == '__main__':
    main()
