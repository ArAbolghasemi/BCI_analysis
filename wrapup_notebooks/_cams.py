"""Behaviour cameras, locked to the neural clock.

The rig puts a **1 Hz TTL** into two places at once, and that is the whole synchronisation:

* **WaveSurfer ``AI1: sync pulse``**, sampled at 1 kHz alongside ``AI7: y-galvo`` --
  so the TTL and the two-photon frame clock sit on one ADC clock;
* **bit 31 of column 0 of each camera's timestamp CSV**, latched into the metadata of every
  camera frame as it is acquired.

Both see exactly the same 9620 transitions on ``mouse_0403L/20230725``, so transition *k* in
the CSV *is* transition *k* on AI1, and interpolating between those anchors carries any camera
frame onto the WaveSurfer clock with the clock drift already in it. Two-photon frame times
come from the y-galvo sawtooth resets on the same clock, which lands every camera frame on a
2P frame index -- the index the cursor, the trials and the neural data all use.

Verified three ways when this was written (2026-11-04), all in ``02_learning_and_remapping``:

1. ``AI1`` sampled at the galvo frame times reproduces the packet's own
   ``data_ws['sync_pulse']`` at **lag 0, r = 0.999555** over all 144,000 frames, which pins
   the frame index with no offset;
2. cam1 and cam3 put their sync transitions on the **same frame index**, never more than one
   frame apart;
3. reward-triggered motion energy on cam3 sits at **0.92x** control before reward and
   **1.62x** after, peaking at **+1.03 s** -- licking, locked to reward, and independent of
   the electronics.

**Do not align on the cameras' ISO timestamps across machines.** The camera host clock on
20230725 is wrong by **63.85 min**. Those timestamps are good only *between* cameras, which is
exactly how ``cam4`` -- the one camera whose status word never toggles -- is bridged here.

Nothing in this module draws anything; the layout and the render loop stay in the notebook.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__all__ = ["find_day", "load_sync", "CamSync", "FrameFeed", "panel_shape", "probe_size",
           "seek_is_exact", "ffmpeg_exe"]

# The galvo reset is a single huge negative step, but it straddles two samples about a quarter
# of the time. Detect generously and merge, or a strict threshold silently drops ~3000 frames
# and the whole alignment rots from there.
GALVO_STEP = -4000.0
GALVO_MERGE = 5
SYNC_BIT = 31


def ffmpeg_exe() -> str:
    """The ffmpeg that ships with imageio-ffmpeg (the conda one segfaults on these AVIs)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------------------
# finding the files
# --------------------------------------------------------------------------------------
def find_day(raw_root, conv_root, mouse: str, day: str, cams=(1, 2, 3, 4)):
    """Locate the WaveSurfer file, the timestamp CSVs and the converted mp4s for one day.

    The CSVs are read from the **raw** tree: the converted tree's copy of ``cam1`` was
    truncated to 0 bytes by a bug in the conversion notebook, and because that copy is made
    with an exclusive create whose ``FileExistsError`` is swallowed, re-running will skip it
    rather than repair it.
    """
    raw = Path(raw_root) / mouse / day
    conv = Path(conv_root) / mouse / day / "cam"
    ws = sorted((raw / "wavesurfer_data").glob("exp_*.h5"))
    if not ws:
        raise FileNotFoundError(f"no exp_*.h5 in {raw / 'wavesurfer_data'}")
    out = SimpleNamespace(ws=ws[0], mouse=mouse, day=day, cams=tuple(cams), csv={}, mp4={})
    for c in cams:
        cs = glob.glob(str(raw / "cam" / f"cam{c}" / "times_*.csv"))
        mp = glob.glob(str(conv / f"cam{c}" / "*.mp4"))
        if not cs or not mp:
            raise FileNotFoundError(f"cam{c}: csv={cs} mp4={mp}")
        out.csv[c], out.mp4[c] = Path(cs[0]), Path(mp[0])
    return out


# --------------------------------------------------------------------------------------
# the two clocks
# --------------------------------------------------------------------------------------
def _ws_channels(h5_path, want=("sync pulse", "y-galvo")):
    """Pull named analog channels out of a WaveSurfer file without holding all seven."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f["header/AIChannelNames"][()]]
        fs = float(np.asarray(f["header/Acquisition/SampleRate"][()]).ravel()[0])
        clk = np.asarray(f["header/ClockAtRunStart"][()]).ravel()
        A = f["sweep_0001"]["analogScans"]
        n = A.shape[1]
        idx = {}
        for w in want:
            hit = [i for i, nm in enumerate(names) if w.lower() in nm.lower()]
            if not hit:
                raise KeyError(f"no WaveSurfer channel matching {w!r} in {names}")
            idx[w] = hit[0]
        out = {w: np.empty(n, np.int16) for w in want}
        for s in range(0, n, 1_000_000):          # chunked: the file is ~70 MB of int16
            blk = A[:, s:s + 1_000_000]
            for w, i in idx.items():
                out[w][s:s + blk.shape[1]] = blk[i]
    return {w: v.astype(np.float32) for w, v in out.items()}, fs, clk, names


def _edges_interp(x, thr):
    """Threshold crossings with linear interpolation, in samples (so sub-sample precision)."""
    hi = x > thr
    k = np.flatnonzero(np.diff(hi.astype(np.int8)) != 0)
    a, b = x[k], x[k + 1]
    return k + (thr - a) / (b - a)


def _galvo_frames(g):
    """Two-photon frame starts = sawtooth resets, merged across the two-sample ones."""
    c = np.flatnonzero(np.diff(g) < GALVO_STEP)
    if c.size == 0:
        raise ValueError("no galvo resets found -- wrong channel?")
    keep = np.concatenate([[True], np.diff(c) > GALVO_MERGE])
    return c[keep] + 1


def _csv_status(csv_path):
    """Column 0 of the timestamp CSV: the camera's status word, one per acquired frame."""
    import pandas as pd

    return pd.read_csv(csv_path, header=None, usecols=[0], names=["s"],
                       dtype={"s": np.int64})["s"].to_numpy()


def _csv_host_seconds(csv_path):
    """Column 4, the host wall clock. Only good BETWEEN cameras -- see the module docstring."""
    import pandas as pd

    t = pd.read_csv(csv_path, header=None, usecols=[4], names=["t"],
                    dtype={"t": str})["t"].to_numpy()
    return t.astype("datetime64[ns]").astype("int64") / 1e9


class CamSync:
    """Camera frames and 2P frames on one clock.

    ``frame_t`` holds every two-photon frame's time in WaveSurfer seconds. ``cam_frame(c, t)``
    turns a time into a (fractional) frame index of camera ``c``; ``cam_of_2p(c, i)`` goes
    straight from a 2P frame index. ``anchors`` keeps the per-camera fit so the notebook can
    print and check it rather than take it on trust.
    """

    def __init__(self, frame_t, sync_t, anchors, meta, sync_at_frame=None):
        self.frame_t = frame_t          # (n_2p,)   WaveSurfer seconds
        self.sync_t = sync_t            # (n_edge,) WaveSurfer seconds
        self.anchors = anchors          # cam -> (edge_t, cam_frame_at_edge)
        self.meta = meta
        # AI1 sampled at each 2P frame. Kept so the notebook can hold it against the packet's
        # own data_ws['sync_pulse'] -- if those two agree at lag 0 the frame index is pinned.
        self.sync_at_frame = sync_at_frame

    # -- the two directions ------------------------------------------------------------
    def cam_frame(self, cam, t):
        at, af = self.anchors[cam]
        t = np.asarray(t, float)
        f = np.interp(t, at, af)
        r = self.rate(cam)                       # straight-line outside the TTL's own span,
        f = np.where(t < at[0], af[0] + (t - at[0]) * r, f)       # i.e. the first and last
        f = np.where(t > at[-1], af[-1] + (t - at[-1]) * r, f)    # ~10 s of the recording
        return f

    def cam_of_2p(self, cam, i):
        return self.cam_frame(cam, self.frame_t[np.asarray(i, int)])

    def rate(self, cam):
        at, af = self.anchors[cam]
        return float((af[-1] - af[0]) / (at[-1] - at[0]))

    def n_2p(self):
        return int(self.frame_t.size)

    # -- what the notebook prints ------------------------------------------------------
    def table(self):
        rows = []
        for c in sorted(self.anchors):
            m = self.meta[c]
            at, af = self.anchors[c]
            rows.append(dict(cam=c, frames=m["n_frames"], transitions=m["n_edges"],
                             rate_hz=self.rate(c),
                             # when this camera's frame 0 happened, on the WaveSurfer clock
                             t0_ws=float(at[0] - af[0] / self.rate(c)),
                             residual_ms=m.get("residual_ms", np.nan), how=m["how"]))
        return rows


def load_sync(files, cache=None, recompute=False):
    """Build the camera/2P frame map for one day. ``cache`` is an .npz path, written once.

    Every camera whose status word toggles is anchored directly on the 1 Hz TTL. A camera
    whose status word never changes -- ``cam4`` on 20230725 -- is carried across on the host
    clock from the first camera that does, which is sound because they share one machine, so
    that clock's absolute error cancels.
    """
    if cache is not None and Path(cache).exists() and not recompute:
        z = np.load(cache, allow_pickle=True)
        return CamSync(z["frame_t"], z["sync_t"],
                       {int(k.split("_")[1]): (z[k], z[k.replace("at_", "af_")])
                        for k in z.files if k.startswith("at_")},
                       z["meta"].item(), z["sync_at_frame"])

    ch, fs, clk, names = _ws_channels(files.ws)
    sync, galvo = ch["sync pulse"], ch["y-galvo"]
    frames = _galvo_frames(galvo)
    frame_t = frames / fs
    sync_at_frame = sync[np.clip(frames, 0, sync.size - 1)].astype(np.float32)
    thr = float((sync.max() + sync.min()) / 2.0)
    sync_t = _edges_interp(sync, thr) / fs

    anchors, meta, ref = {}, {}, None
    host = {}
    for c in files.cams:
        s = _csv_status(files.csv[c])
        bit = (s >> SYNC_BIT) & 1
        e = np.flatnonzero(np.diff(bit) != 0) + 1
        meta[c] = dict(n_frames=int(s.size), n_edges=int(e.size))
        if e.size == sync_t.size:
            anchors[c] = (sync_t.copy(), e.astype(float))
            # how well does one straight line fit? (a large residual means a dropped TTL)
            A, B = np.polyfit(e.astype(float), sync_t, 1)
            meta[c]["residual_ms"] = float(np.std(sync_t - (A * e + B)) * 1e3)
            meta[c]["how"] = "1 Hz TTL"
            if ref is None:
                ref = c
        else:
            meta[c]["how"] = "pending"
    if ref is None:
        raise RuntimeError("no camera carries the sync TTL -- cannot align this day")

    # the stragglers ride the host clock across from `ref`
    for c in files.cams:
        if c in anchors:
            continue
        if ref not in host:
            host[ref] = _csv_host_seconds(files.csv[ref])
        host[c] = _csv_host_seconds(files.csv[c])
        at, af = anchors[ref]
        h_at_edges = np.interp(af, np.arange(host[ref].size, dtype=float), host[ref])
        af_c = np.interp(h_at_edges, host[c], np.arange(host[c].size, dtype=float))
        anchors[c] = (at.copy(), af_c)
        meta[c]["how"] = f"host clock via cam{ref}"
        # the honest error here is how far the NEAREST WHOLE frame of this camera sits from
        # the host time we were aiming at -- anything finer is lost to its own frame grid
        near = np.clip(np.round(af_c).astype(int), 0, host[c].size - 1)
        meta[c]["residual_ms"] = float(np.std(host[c][near] - h_at_edges) * 1e3)

    cs = CamSync(frame_t, sync_t, anchors, meta, sync_at_frame)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d = dict(frame_t=frame_t, sync_t=sync_t, meta=np.array(meta, dtype=object),
                 sync_at_frame=sync_at_frame)
        for c, (at, af) in anchors.items():
            d[f"at_{c}"], d[f"af_{c}"] = at, af
        np.savez_compressed(cache, **d)
    return cs


# --------------------------------------------------------------------------------------
# getting the pixels out, without melting the machine
# --------------------------------------------------------------------------------------
def probe_size(mp4):
    """(width, height) of an mp4, from ffmpeg's own banner."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(mp4)],
                       capture_output=True, text=True, errors="ignore")
    m = re.search(r"Video:.*?,\s(\d+)x(\d+)", p.stderr)
    if not m:
        raise RuntimeError(f"could not read the frame size of {mp4}")
    return int(m.group(1)), int(m.group(2))


def _geom(mp4, width, crop):
    """(out_w, out_h, crop_filter). One place, so a layout's idea of the frame shape and
    ffmpeg's can never drift apart."""
    sw, sh = probe_size(mp4)
    cropf = ""
    if crop is not None:                     # (x0, x1, y0, y1) as fractions of the frame
        x0, x1, y0, y1 = crop
        cx, cy = int(sw * x0) // 2 * 2, int(sh * y0) // 2 * 2
        cw, ch = int(sw * (x1 - x0)) // 2 * 2, int(sh * (y1 - y0)) // 2 * 2
        cropf, sw, sh = f"crop={cw}:{ch}:{cx}:{cy},", cw, ch
    w = int(width)
    return w, int(round(sh * w / sw / 2.0)) * 2, cropf


def panel_shape(mp4, width, crop=None):
    """(w, h) a FrameFeed would hand back, without opening one -- so a layout can be sized
    from the real frame shapes instead of a guess."""
    w, h, _ = _geom(mp4, width, crop)
    return w, h


class FrameFeed:
    """Hand back the frame nearest each requested index, from ONE sequential ffmpeg pass.

    The requested indices only ever go forwards, so this never seeks and never buffers: one
    decoded frame is alive at a time, whatever the length of the movie. ffmpeg does the
    decoding, the downscaling and -- through ``stride`` -- most of the throwing away, so what
    crosses the pipe is a small fraction of the source.

    ``stride`` is the only accuracy knob: frames are taken every ``stride`` source frames, so
    the worst error is ``stride/2`` camera frames (4.2 ms at stride 1, 120 Hz). Keep it well
    under the output frame interval and it costs nothing visible.
    """

    def __init__(self, mp4, first, last, width, stride=2, fps_nominal=120.0, gray=True,
                 crop=None):
        self.mp4, self.stride, self.fps = Path(mp4), int(max(stride, 1)), float(fps_nominal)
        self.w, self.h, cropf = _geom(mp4, width, crop)
        self.gray = bool(gray)
        self.bpp = 1 if gray else 3
        self.first = int(max(first, 0))
        self.last = int(last)
        self._n = None                      # source index of the frame in hand
        self._buf = None
        vf = (f"select='not(mod(n\\,{self.stride}))'," if self.stride > 1 else "")
        vf += cropf + f"scale={self.w}:{self.h},format={'gray' if gray else 'rgb24'}"
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-ss", f"{self.first / self.fps:.6f}",
             "-i", str(self.mp4),
             "-frames:v", str(int((self.last - self.first) // self.stride) + 8),
             "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.w * self.h * self.bpp * 4)
        self._k = 0

    def _advance(self):
        nbytes = self.w * self.h * self.bpp
        raw = self.proc.stdout.read(nbytes)
        if raw is None or len(raw) < nbytes:
            return False
        a = np.frombuffer(raw, np.uint8)
        self._buf = a.reshape(self.h, self.w) if self.gray else a.reshape(self.h, self.w, 3)
        self._n = self.first + self._k * self.stride
        self._k += 1
        return True

    def at(self, idx):
        """The decoded frame nearest source index ``idx``. ``idx`` must not go backwards."""
        idx = int(idx)
        if self._buf is None and not self._advance():
            return None
        while self._n + self.stride <= idx:          # walk forward, one frame in memory
            if not self._advance():
                break
        return self._buf

    def close(self):
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def seek_is_exact(mp4, probe_frame=4000, width=96, fps_nominal=120.0):
    """Does ``-ss`` land on the frame we asked for? Compares a seek against a longer seek
    that walks up to the same frame. Returns (ok, max_abs_difference)."""
    a = FrameFeed(mp4, probe_frame, probe_frame + 2, width, stride=1, fps_nominal=fps_nominal)
    b = FrameFeed(mp4, probe_frame - 30, probe_frame + 2, width, stride=1,
                  fps_nominal=fps_nominal)
    try:
        fa = a.at(probe_frame)
        fb = b.at(probe_frame)
        if fa is None or fb is None:
            return False, np.inf
        d = float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())
        return d == 0.0, d
    finally:
        a.close()
        b.close()
