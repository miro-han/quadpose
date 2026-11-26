"""Interactive box/positive/negative prompts for official SAM 2 video tracking."""
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from perception.segmentation.sam_session import SamSession, ROOT, visualize


class SegmentApp:
    def __init__(self, root):
        self.root, self.engine = root, SamSession()
        self.events = queue.Queue()
        self.busy = self.closing = False
        self.frame = self.first_frame = self.mask = self.initial_mask = None
        self.box, self.points, self.labels = None, [], []
        self.loaded_spec = self.prepared_spec = None
        self.viewing_result = False
        self.drag_start = None
        self.mapping = None
        self.change = 0
        self.accepted_change = -1
        self.widgets = []
        root.title('Team 7 · 물체 지정과 SAM 2 추적')
        root.geometry(f'{min(1120,root.winfo_screenwidth()-60)}x{min(830,root.winfo_screenheight()-100)}+30+30')
        root.minsize(800, 600)
        root.protocol('WM_DELETE_WINDOW', self.close)
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='물체 지정 · 마스크 확인 · 추적', font=('맑은 고딕', 19, 'bold')).pack(anchor='w')
        ttk.Label(outer, text='① 상자를 드래그 → ② 마스크 확인 → ③ 포함/제외 클릭 후 재확인 → ④ 추적 시작').pack(anchor='w', pady=5)
        files = ttk.Frame(outer)
        files.pack(fill='x', pady=4)
        sample = ROOT/'data'/'sample_videos'/'WIN_20261124_20_05_34_Pro.mp4'
        if not sample.exists():
            sample = Path.home()/'Pictures'/'Camera Roll'/sample.name
        self.path = tk.StringVar(value=str(sample))
        ttk.Entry(files, textvariable=self.path, state='readonly').pack(side='left', fill='x', expand=True)
        self.button(files, '영상 선택', self.browse).pack(side='left', padx=5)
        settings = ttk.Frame(outer)
        settings.pack(fill='x', pady=4)
        self.start, self.length = tk.StringVar(value='0'), tk.StringVar(value='30')
        for text, variable, upper in [('시작 프레임', self.start, 1000000), ('처리 길이(프레임)', self.length, 120)]:
            ttk.Label(settings, text=text).pack(side='left', padx=4)
            spin = ttk.Spinbox(settings, from_=0 if variable is self.start else 1, to=upper, textvariable=variable, width=7)
            spin.pack(side='left')
            self.widgets.append(spin)
        self.button(settings, '선택 구간 불러오기', self.load_frame).pack(side='left', padx=8)
        self.range_info = tk.StringVar(value='CPU 모드: 우선 30프레임 이하로 시험하세요.')
        ttk.Label(outer, textvariable=self.range_info).pack(anchor='w', pady=3)
        modes = ttk.Frame(outer)
        modes.pack(fill='x', pady=4)
        self.mode = tk.StringVar(value='box')
        for text, value in [('박스 그리기', 'box'), ('포함 클릭 (+)', 'positive'), ('제외 클릭 (-)', 'negative')]:
            radio = ttk.Radiobutton(modes, text=text, variable=self.mode, value=value)
            radio.pack(side='left', padx=5)
            self.widgets.append(radio)
        self.button(modes, '지정 초기화', self.reset_prompts).pack(side='left', padx=8)
        self.display_edges = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes, text='마스크로 선택한 에지 보기', variable=self.display_edges,
                        command=self.render).pack(side='right')
        footer = ttk.Frame(outer)
        footer.pack(side='bottom', fill='x')
        row = ttk.Frame(footer)
        row.pack(fill='x', pady=8)
        self.preview_button = self.button(row, '마스크 확인 / 수정 적용', self.preview)
        self.preview_button.pack(side='left', padx=4)
        self.track_button = self.button(row, '확인한 마스크로 추적 시작', self.track)
        self.track_button.pack(side='left', padx=4)
        self.cancel_button = ttk.Button(row, text='처리 중단', command=self.cancel)
        self.cancel_button.pack(side='left', padx=4)
        ttk.Button(row, text='결과 폴더 열기', command=self.open_results).pack(side='right')
        self.status = tk.StringVar(value='상자가 보이는 구간을 불러온 뒤 박스를 그리세요.')
        ttk.Label(footer, textvariable=self.status, wraplength=1000).pack(anchor='w')
        ttk.Label(footer, text='마스크는 모델 예측입니다. 손에 가려진 경계와 실제 물체 모서리는 다를 수 있습니다.',
                  foreground='#965a12').pack(anchor='w', pady=5)
        self.canvas = tk.Canvas(outer, bg='#101827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, pady=6)
        self.canvas.bind('<Configure>', lambda _: self.render())
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        root.after(200, self.load_frame)
        root.after(100, self.tick)

    def button(self, parent, text, command):
        widget = ttk.Button(parent, text=text, command=command)
        self.widgets.append(widget)
        return widget

    def browse(self):
        name = filedialog.askopenfilename(filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('All', '*.*')])
        if name:
            self.path.set(name)
            self.start.set('0')
            self.load_frame()

    def load_frame(self):
        if self.busy:
            return
        cap = None
        try:
            start, length = int(self.start.get()), int(self.length.get())
            if start < 0 or not 1 <= length <= 120:
                raise ValueError('시작은 0 이상, 길이는 1~120프레임을 입력하세요.')
            cap = cv2.VideoCapture(self.path.get())
            total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or start >= total:
                raise ValueError('영상 경로와 시작 프레임을 확인하세요.')
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('선택 프레임을 읽지 못했습니다.')
            self.loaded_spec = (self.path.get(), start, min(length, total-start))
            self.first_frame = frame
            self.reset_prompts()
            self.range_info.set(f'전체 {total}프레임 / {fps:.2f} FPS · 현재 구간 {start}~{start+self.loaded_spec[2]-1}')
            self.status.set('상자 전체를 둘러싸는 박스를 드래그하세요.')
        except Exception as exc:
            messagebox.showerror('영상 불러오기', str(exc), parent=self.root)
        finally:
            if cap is not None:
                cap.release()

    def reset_prompts(self):
        if self.busy:
            return
        self.box, self.points, self.labels = None, [], []
        self.mask = self.initial_mask = None
        self.frame = self.first_frame
        self.change += 1
        self.accepted_change = -1
        self.viewing_result = False
        self.render()

    def point(self, event):
        if self.mapping is None:
            return None
        scale, ox, oy, w, h = self.mapping
        x, y = (event.x-ox)/scale, (event.y-oy)/scale
        if 0 <= x < w and 0 <= y < h:
            return [float(x), float(y)]
        return None

    def press(self, event):
        if self.busy or self.viewing_result:
            return
        point = self.point(event)
        if point is None:
            return
        if self.mode.get() == 'box':
            self.drag_start = point
        elif self.box is not None:
            self.points.append(point)
            self.labels.append(1 if self.mode.get() == 'positive' else 0)
            self.change += 1
            self.render()
            self.status.set('클릭을 추가했습니다. [마스크 확인 / 수정 적용]을 누르세요.')

    def drag(self, event):
        point = self.point(event)
        if self.drag_start is not None and point is not None:
            self.draw_box(self.drag_start + point, tag='drag')

    def release(self, event):
        point, begin = self.point(event), self.drag_start
        self.drag_start = None
        self.canvas.delete('drag')
        if begin is None or point is None or self.busy:
            return
        box = [min(begin[0], point[0]), min(begin[1], point[1]), max(begin[0], point[0]), max(begin[1], point[1])]
        if box[2]-box[0] < 5 or box[3]-box[1] < 5:
            return
        self.box, self.points, self.labels = box, [], []
        self.mask = self.initial_mask = None
        self.change += 1
        self.status.set('박스를 지정했습니다. [마스크 확인 / 수정 적용]을 누르세요.')
        self.render()

    def draw_box(self, box, tag='box'):
        if self.mapping is None:
            return
        scale, ox, oy, _, _ = self.mapping
        self.canvas.delete(tag)
        self.canvas.create_rectangle(box[0]*scale+ox, box[1]*scale+oy, box[2]*scale+ox,
                                     box[3]*scale+oy, outline='#48baff', width=2, tags=tag)

    def render(self):
        if self.frame is None:
            return
        frame = self.frame
        if self.mask is not None:
            overlay, edges = visualize(frame, self.mask)
            if self.display_edges.get():
                frame = frame.copy()
                frame[edges > 0] = (0, 255, 0)
            else:
                frame = overlay
        h, w = frame.shape[:2]
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        scale = min(cw/w, ch/h)
        size = (max(1, int(w*scale)), max(1, int(h*scale)))
        small = cv2.resize(frame, size)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        ox, oy = (cw-size[0])/2, (ch-size[1])/2
        self.mapping = scale, ox, oy, w, h
        self.canvas.delete('all')
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo)
        if not self.viewing_result:
            if self.box:
                self.draw_box(self.box)
            for (x, y), label in zip(self.points, self.labels):
                x, y = x*scale+ox, y*scale+oy
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='#31e078' if label else '#ff4e68', outline='white')

    def task(self, kind, callback):
        self.busy = True
        self.engine.stop.clear()
        def work():
            try:
                self.events.put((kind, callback(), None))
            except Exception as exc:
                self.events.put((kind, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def notify(self, text):
        self.events.put(('status', text, None))

    def preview(self):
        if self.busy or self.box is None or self.loaded_spec is None:
            return
        box, points, labels = self.box.copy(), self.points.copy(), self.labels.copy()
        self.preview_change = self.change
        self.frame, self.mask, self.viewing_result = self.first_frame, self.initial_mask, False
        spec = self.loaded_spec
        def work():
            if self.prepared_spec != spec or self.engine.state is None or self.engine.metadata.get('written_frames', 0):
                self.engine.prepare(*spec, self.notify)
                self.prepared_spec = spec
            if self.engine.stop.is_set():
                raise RuntimeError('처리를 중단했습니다.')
            self.notify('지정한 대상의 마스크를 계산 중…')
            return self.engine.preview(box, points, labels)
        self.task('preview', work)

    def track(self):
        if self.busy or self.accepted_change != self.change:
            return
        if self.engine.metadata.get('written_frames', 0):
            messagebox.showinfo('새 추적', '같은 결과를 덮어쓰지 않도록 [선택 구간 불러오기] 후 다시 지정하세요.')
            self.prepared_spec = None
            return
        self.task('track', lambda: self.engine.track(self.notify,
                    lambda frame, mask, index: self.events.put(('frame', (frame, mask, index), None))))

    def cancel(self):
        self.engine.stop.set()
        self.status.set('현재 모델 계산이 끝난 뒤 중단합니다…')

    def open_results(self):
        folder = self.engine.folder or ROOT/'outputs'/'segmentation'
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def close(self):
        self.closing = True
        self.cancel()

    def tick(self):
        while True:
            try:
                kind, result, error = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'status':
                self.status.set(result)
                continue
            if kind == 'frame':
                self.frame, self.mask, _ = result
                self.viewing_result = True
                self.render()
                continue
            self.busy = False
            if error:
                self.accepted_change = -1
                self.status.set(error)
                if not self.closing:
                    messagebox.showerror('처리 오류', error, parent=self.root)
            elif kind == 'preview':
                self.frame = self.first_frame
                self.mask = self.initial_mask = result
                self.accepted_change = self.preview_change
                self.viewing_result = False
                self.status.set('마스크를 확인하세요. 수정하려면 포함/제외 클릭 후 다시 적용하고, 맞으면 추적을 시작하세요.')
                self.render()
            elif kind == 'track':
                self.frame, self.mask, self.viewing_result = self.first_frame, self.initial_mask, False
                self.status.set(f'{self.engine.metadata["status"]} · {self.engine.metadata["written_frames"]}프레임 저장 · {result.name}')
                self.render()
        if self.closing and not self.busy:
            self.root.destroy()
            return
        for widget in self.widgets:
            widget.configure(state='disabled' if self.busy or self.closing else 'normal')
        if self.box is None:
            self.preview_button.configure(state='disabled')
        if self.accepted_change != self.change or self.initial_mask is None or not self.initial_mask.any():
            self.track_button.configure(state='disabled')
        self.cancel_button.configure(state='normal' if self.busy else 'disabled')
        self.root.after(100, self.tick)


def main():
    if os.name == 'nt':
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            pass
    cv2.setNumThreads(2)
    root = tk.Tk()
    SegmentApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
