"""Recording integrity checks. --hardware exercises UI buttons with live cameras."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from capture.camera_core import Recording, write_json

BASE = Path(__file__).resolve().parent.parent
QA = BASE / 'outputs' / 'camera' / 'recorder_checks'


def verify_session(folder, expected=None):
    meta = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed', meta
    report = {}
    for slot, n in meta['written_frames'].items():
        with (folder / f'camera_{slot}_timestamps.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        cap = cv2.VideoCapture(str(folder / f'camera_{slot}.avi'))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert [frame.shape[1], frame.shape[0]] == meta['cameras'][slot]['actual']['size_px']
            decoded += 1
        cap.release()
        assert decoded == len(rows) == n and n > 0, (slot, decoded, len(rows), n)
        assert [int(r['video_frame_index']) for r in rows] == list(range(n))
        stamps = [int(r['host_monotonic_ns']) for r in rows]
        assert all(a < b for a, b in zip(stamps, stamps[1:]))
        assert stamps[0] >= meta['start_monotonic_ns']
        assert stamps[-1] <= meta['stop_monotonic_ns']
        if expected is not None:
            assert n == expected
        report[slot] = dict(decoded_frames=decoded, csv_rows=len(rows),
                            first_offset_ms=(stamps[0]-meta['start_monotonic_ns'])/1e6,
                            last_offset_ms=(stamps[-1]-meta['start_monotonic_ns'])/1e6,
                            measured_fps=(n-1)*1e9/(stamps[-1]-stamps[0]) if n > 1 else 0,
                            queue_dropped=meta['queue_dropped_frames'][slot])
    return report


def synthetic_check():
    cameras = {i: dict(actual=dict(size_px=[160, 120])) for i in range(1, 5)}
    rec = Recording(QA / 'synthetic', cameras, 30, 'Four synthetic sources; not a camera performance test', queue_size=100)
    rec.begin()
    frame = np.full((120, 160, 3), 100, np.uint8)
    rec.push(1, 0, rec.start_ns-1, time.time_ns(), frame)  # reject pre-start frames
    for j in range(12):
        for slot in cameras:
            image = frame.copy()
            cv2.putText(image, f'{slot}/{j}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            rec.push(slot, j+1, time.perf_counter_ns(), time.time_ns(), image)
    rec.finish()
    rec.push(1, 100, time.perf_counter_ns(), time.time_ns(), frame)  # reject post-stop frames
    assert rec.done.wait(15), 'Writer did not finish'
    report = verify_session(rec.folder, expected=12)
    # A second session must be isolated and may contain different camera slots.
    second = Recording(QA / 'synthetic', {2: cameras[2]}, 15, 'Repeat recording test')
    second.begin()
    second.push(2, 500, time.perf_counter_ns(), time.time_ns(), frame)
    second.finish()
    assert second.done.wait(15)
    verify_session(second.folder, expected=1)
    assert second.folder != rec.folder
    print('SYNTHETIC PASS: four streams, timestamps, start/stop boundaries, repeated session', flush=True)
    return report


def hardware_ui_check():
    import tkinter as tk
    from PIL import ImageGrab
    from capture.camera_recorder import RecorderApp, configure_display
    configure_display()
    root = tk.Tk()
    app = RecorderApp(root)
    app.output.set(str(QA / 'hardware'))
    app.note.set('UI verification: two external cameras, short recording')
    state = dict(stage='discover', deadline=time.monotonic()+60, failures=[], session=None)

    def poll():
        try:
            if time.monotonic() > state['deadline']:
                raise TimeoutError(f"Timeout in {state['stage']}")
            stage = state['stage']
            if stage == 'discover' and app.devices and not app.busy:
                choices = [k for k, d in app.devices.items() if d['name'] == 'Web Camera']
                assert len(choices) >= 2, 'Two external Web Camera devices required'
                for i, combo in enumerate(app.combos):
                    combo.set(choices[i] if i < 2 else '사용 안 함')
                app.connect_button.invoke()
                state['stage'] = 'connecting'
            elif stage == 'connecting' and app.group.ready():
                assert len(app.group.workers) == 2
                state['stage'] = 'warmup'
                state['until'] = time.monotonic()+2
            elif stage == 'warmup' and time.monotonic() >= state['until']:
                app.record_button.invoke()
                state['stage'] = 'starting'
            elif stage == 'starting' and app.rec:
                state['session'] = app.rec.folder
                state['stage'] = 'recording'
                state['until'] = time.monotonic()+4
            elif stage == 'recording' and time.monotonic() >= state['until']:
                root.update_idletasks()
                root.lift()
                x, y = root.winfo_rootx(), root.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x+root.winfo_width(), y+root.winfo_height())).save(QA / 'ui_live_check.png')
                app.record_button.invoke()
                state['stage'] = 'saving'
            elif stage == 'saving' and app.rec is None:
                assert state['session'] is not None
                report = verify_session(state['session'])
                write_json(QA / 'hardware_report.json', dict(session=str(state['session']), cameras=report))
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
                print('HARDWARE UI PASS: connect, preview, record button, stop, decode + CSV', flush=True)
                app.close()
                return
        except Exception as exc:
            state['failures'].append(str(exc))
            print('HARDWARE FAIL:', exc, flush=True)
            app.close()
            return
        root.after(200, poll)

    root.after(500, poll)
    root.mainloop()
    if state['failures']:
        raise RuntimeError('; '.join(state['failures']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    synthetic_check()
    if args.hardware:
        hardware_ui_check()
