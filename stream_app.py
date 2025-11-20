from matplotlib.pyplot import step


from werkzeug.utils import secure_filename
import pandas as pd
import onnxruntime as ort
import uuid

import eventlet

import RPi.GPIO as GPIO
from queue import Queue

import atexit

import logging


import os, json, time, threading, queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from libcamera import Transform 
import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, Response, jsonify, render_template, request

from flask_socketio import SocketIO

import collections, collections.abc
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSet = collections.abc.MutableSet
collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence

from dronekit import connect, VehicleMode, LocationGlobalRelative
import math

from pymavlink import mavutil
from threading import Event

# Limitează paralelismul BLAS/OpenCV (evită thrashing pe CPU mic)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
try:
    cv2.setNumThreads(5)
except Exception:
    pass
import logging

logging.basicConfig(
    level=logging.INFO,  # <- era INFO
    format='[%(levelname)s] (%(threadName)s) %(message)s'
)
# (opțional) dacă vrei și Flask/Werkzeug în DEBUG:
logging.getLogger('werkzeug').setLevel(logging.INFO)

 
# ========= Config =========

USE_PICAM       = True   # True pe Raspberry Pi (PiCamera2), False pentru USB cam
CAM_INDEX       = 0
#W, H            = 640, 480
W, H            = 320, 240
JPEG_QUALITY    = 70
MJPEG_HZ        = 30

POSE_IMGSZ       = 352
POSE_FPS         = 5.0
KP_OK_CONF       = 0.50
KP_MIN_RATIO     = 0.50   # relaxat pt. demo (0.80 în producție)
MIN_DRAW_CONF    = 0.20   # pentru overlay să vezi scheletul

WINDOW_SEC       = 4.0
HOP_SEC          = 0.5

DEFAULT_TAU_ON   = 0.60
DEFAULT_TAU_OFF  = 0.55
DEFAULT_COOLDOWN = 7.0

# ========= Paths (independente de CWD) =========
# stream_app.py este în /home/dariuc/Triton → vrem BASE_DIR = /home/dariuc/Triton
BASE_DIR   = Path(__file__).resolve().parent
APP_DIR    = BASE_DIR
MODELS_DIR = BASE_DIR / "models"

POSE_WEIGHTS    = str(MODELS_DIR / "yolo11n-pose.pt")
XGB_PIPELINE    = str(MODELS_DIR / "xgb_pipeline.joblib")
THRESHOLDS_JSON = str(MODELS_DIR / "threshold.json")
FEATURE_SCHEMA  = str(MODELS_DIR / "features_schema.json")
CALIBRATOR_PATH = str(MODELS_DIR / "calibrator.joblib")




# ========= App/State =========
# Acum folosim template_folder explicit către Triton_pose/templates
app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
socketio = SocketIO(app, async_mode="threading")


raw_lock   = threading.Lock()
jpeg_lock  = threading.Lock()
pose_lock  = threading.Lock()
latest_raw  = None
latest_jpeg = None
pose_jpeg   = None

q_kp   = queue.Queue(maxsize=256)
q_norm = queue.Queue(maxsize=256)
q_feat = queue.Queue(maxsize=64)

last_label = "N/A"
last_proba = 0.0
cooldown_until = 0.0
consec_on  = 0
consec_off = 0
pose_thread_started = False

tau_on = DEFAULT_TAU_ON
tau_off = DEFAULT_TAU_OFF
cooldown_s = DEFAULT_COOLDOWN

# ========= Praguri / Calibrator =========
try:
    if os.path.exists(THRESHOLDS_JSON):
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            th = json.load(f)
        if "tau_on" in th:
            tau_on = float(th["tau_on"])
            tau_off = float(th.get("tau_off", max(0.5, tau_on - 0.05)))
            cooldown_s = float(th.get("cooldown_s", DEFAULT_COOLDOWN))
        elif "threshold" in th:
            tau_on = float(th["threshold"])  # din OOF/Platt
            tau_off = max(0.5, tau_on - 0.05)
            cooldown_s = DEFAULT_COOLDOWN
except Exception as e:
    logging.debug("WARN thresholds load:", e)

CALIBRATOR = None
try:
    if os.path.exists(CALIBRATOR_PATH):
        CALIBRATOR = joblib.load(CALIBRATOR_PATH)
except Exception as e:
    logging.error("WARN calibrator load:", e)
    CALIBRATOR = None

# ========= YOLO / Torch =========
from ultralytics import YOLO
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    HALF   = (DEVICE != "cpu")
except Exception:
    DEVICE, HALF = "cpu", False

pose_model = YOLO(POSE_WEIGHTS); logging.debug("YOLO loaded:", POSE_WEIGHTS)
XGB        = joblib.load(XGB_PIPELINE); logging.debug("XGB loaded:", XGB_PIPELINE)
logging.debug(f"thresholds: tau_on={tau_on:.3f} tau_off={tau_off:.3f} cooldown={cooldown_s:.1f}s")

# ========= Feature schema tolerantă =========
FEATURE_LIST = None
try:
    with open(FEATURE_SCHEMA, "r", encoding="utf-8") as f:
        _schema = json.load(f)
    feats = _schema.get("feature_names")
    if feats is None:
        cols = _schema.get("columns", [])
        drop = {"video_id", "t_start", "t_end", "label"}
        feats = [c for c in cols if c not in drop]
    if not feats:
        raise ValueError("no feature names in schema")
    FEATURE_LIST = list(feats)
    logging.debug("feature_schema OK, features:", len(FEATURE_LIST))
except Exception as e:
    logging.error("feature_schema missing/unreadable:", e)
    FEATURE_LIST = None

# ========= KP idx =========
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16

@dataclass
class KPPacket:
    t: float
    frame: np.ndarray
    xy: np.ndarray
    conf: np.ndarray

@dataclass
class NormPacket:
    t: float
    xy_norm: np.ndarray
    conf: np.ndarray

@dataclass
class FeatPacket:
    t_start: float
    t_end: float
    feats: dict


###########################################




def _gps_get(s, key, default=None):
    if isinstance(s, dict):
        return s.get(key, default)
    return getattr(s, key, default)



GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)
servo2 = GPIO.PWM(12, 50)
servo1.start(7.5)
servo2.start(7.5)
time.sleep(0.3)
servo1.ChangeDutyCycle(0)
servo2.ChangeDutyCycle(0)

right_stream_type = "daria"


detection_thread = None
stop_detection_event = Event()

detection_liv_thread = None
stop_detection_liv_event = Event()

segmnetation_thread= None
stop_segmentation_event = Event()

pose_thread= None
stop_pose_event = Event()


stop_takeoff_event = Event()

smart_stream_mode = False



pose_triggered = False

streaming = False
frame_lock = threading.Lock()
output_lock = threading.Lock()
mar_lock = threading.Lock()
seg_lock = threading.Lock()
pose_lock = threading.Lock()


label_map = {0: "inot", 1: "inec"}




output_frame_pose = None
output_lock_pose = threading.Lock()

frame_buffer = None
output_frame = None
yolo_output_frame = None
detected_flag = False
popup_sent = False
last_detection_time = 0


event_location = None  # Fixed: Initialize event_location
mar_output_frame = None
seg_output_frame = None

pose_triggered = False






# ========= Utils =========
def cleanup():
    try: servo1.stop()
    except: pass
    try: servo2.stop()
    except: pass

    try: GPIO.cleanup()
    except: pass

atexit.register(cleanup)
# Pornește un thread daemon care rulează funcția specificată.
# Este folosit pentru a executa procese paralele (ex: camere, inferență, streaming) fără a bloca aplicația.
def start_thread(func, name="WorkerThread"):
    t = threading.Thread(target=func, name=name, daemon=True)
    t.start()
    return t
# Activează și apoi resetează două servomotoare pentru a executa aruncarea colacului.
def activate_servos():
    logging.info("Activare servomotoare")
    servo1.ChangeDutyCycle(12.5)
    servo2.ChangeDutyCycle(2.5)
    time.sleep(0.3)
    servo1.ChangeDutyCycle(0)
    servo2.ChangeDutyCycle(0)
    time.sleep(2)
    servo1.ChangeDutyCycle(7.5)
    servo2.ChangeDutyCycle(7.5)
    time.sleep(0.3)
    servo1.ChangeDutyCycle(0)
    servo2.ChangeDutyCycle(0)


def blank_jpeg():
    img = np.zeros((H,W,3), np.uint8)
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])[1].tobytes()

def extract_xy_conf(res):
    kpts = getattr(res, "keypoints", None)
    if kpts is None or getattr(kpts, "xy", None) is None or len(kpts.xy) == 0:
        return None, None
    xy = kpts.xy[0]
    try: xy = xy.detach().cpu().numpy()
    except: xy = np.asarray(xy)
    xy = xy[:, :2].astype(np.float32)

    if getattr(kpts, "conf", None) is not None and len(kpts.conf) > 0:
        c = kpts.conf[0]
        try: conf = c.detach().cpu().numpy().astype(np.float32)
        except: conf = np.asarray(c, dtype=np.float32)
    else:
        conf = np.ones(xy.shape[0], dtype=np.float32)

    if xy.shape[0] < 17:
        pad_xy = np.full((17 - xy.shape[0], 2), -1.0, np.float32)
        xy = np.vstack([xy, pad_xy])
        pad_cf = np.zeros(17 - conf.shape[0], np.float32)
        conf = np.concatenate([conf, pad_cf])
    elif xy.shape[0] > 17:
        xy = xy[:17]; conf = conf[:17]
    return xy, conf

def frame_quality_ok(conf, kp_ok_conf=KP_OK_CONF, min_ratio=KP_MIN_RATIO):
    return float((conf >= kp_ok_conf).mean()) >= min_ratio

def normalize_TS(xy):
    xy = xy.copy()
    valid = (xy[:,0]>=0)&(xy[:,1]>=0)
    if valid[L_HIP] and valid[R_HIP]:
        pelvis = (xy[L_HIP] + xy[R_HIP]) / 2.0
    elif valid[L_HIP]:
        pelvis = xy[L_HIP]
    elif valid[R_HIP]:
        pelvis = xy[R_HIP]
    else:
        pelvis = np.array([0.0,0.0], np.float32)
    xy[valid] -= pelvis
    scale = 1.0
    if valid[L_SH] and valid[R_SH]:
        d = np.linalg.norm(xy[L_SH] - xy[R_SH])
        if d > 1e-6: scale = d
    xy[valid] /= (scale + 1e-6)
    return xy

def draw_pose_overlay(frame, xy, conf, min_conf=MIN_DRAW_CONF):
    CONN = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for i,(x,y) in enumerate(xy):
        if conf[i] >= min_conf and x>=0 and y>=0:
            cv2.circle(frame,(int(x),int(y)),3,(0,255,0),-1)
    for i,j in CONN:
        if conf[i] >= min_conf and conf[j] >= min_conf:
            x1,y1 = xy[i]; x2,y2 = xy[j]
            if x1>=0 and y1>=0 and x2>=0 and y2>=0:
                cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
    return frame

def pick_best_xy(res_obj, draw_conf=MIN_DRAW_CONF):
    best = None
    best_count = -1
    seq = res_obj if isinstance(res_obj, (list, tuple)) else [res_obj]
    for r in seq:
        xy0, cf0 = extract_xy_conf(r)
        if xy0 is None or cf0 is None:
            continue
        count = int((cf0 >= draw_conf).sum())
        if count > best_count:
            best = (xy0, cf0)
            best_count = count
    return best

def features_on_window(xs):
    X = np.asarray(xs, dtype=np.float32)
    if X.shape[0] < 2: return None

    vL = np.linalg.norm(np.diff(X[:, L_WR, :], axis=0), axis=1)
    vR = np.linalg.norm(np.diff(X[:, R_WR, :], axis=0), axis=1)
    v_wrist_L_mean = float(np.nanmean(vL))
    v_wrist_R_mean = float(np.nanmean(vR))
    v_wrist_L_var  = float(np.nanvar (vL))
    v_wrist_R_var  = float(np.nanvar (vR))

    sh_y = (X[:, L_SH, 1] + X[:, R_SH, 1]) / 2.0
    frac_hands_above_shoulders = float(((X[:, L_WR, 1] < sh_y) & (X[:, R_WR, 1] < sh_y)).mean())

    armL = np.linalg.norm(X[:, L_SH, :] - X[:, L_WR, :], axis=1)
    armR = np.linalg.norm(X[:, R_SH, :] - X[:, R_WR, :], axis=1)
    arm_len_mean = float(np.nanmean(np.concatenate([armL, armR])))
    arm_asym     = float(abs(np.nanmean(armL) - np.nanmean(armR)))

    trunk = ((X[:, L_SH, :] + X[:, R_SH, :]) / 2.0) - ((X[:, L_HIP, :] + X[:, R_HIP, :]) / 2.0)
    ang = np.arctan2(trunk[:,0], -trunk[:,1])
    trunk_angle_std = float(np.nanstd(ang))

    pelvis = ((X[:, L_HIP, :] + X[:, R_HIP, :]) / 2.0)
    step = np.linalg.norm(np.diff(pelvis, axis=0), axis=1)
    pelvis_disp_mean = float(np.nanmean(step))
    pelvis_disp_var  = float(np.nanvar (step))

    return {
        "v_wrist_L_mean": v_wrist_L_mean,
        "v_wrist_R_mean": v_wrist_R_mean,
        "v_wrist_L_var":  v_wrist_L_var,
        "v_wrist_R_var":  v_wrist_R_var,
        "frac_hands_above_shoulders": frac_hands_above_shoulders,
        "arm_len_mean": arm_len_mean,
        "arm_asym": arm_asym,
        "trunk_angle_std": trunk_angle_std,
        "pelvis_disp_mean": pelvis_disp_mean,
        "pelvis_disp_var":  pelvis_disp_var,
    }




    return buffer.tobytes()

# === GPS SIMULATOR ===
USE_SIMULATOR = False

class GPSValue:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
class BaseGPSProvider:
    def get_location(self):
        raise NotImplementedError()
    def close(self):
        pass        

class MockGPSProvider:
    def __init__(self):
        self.coordinates = deque([
            (44.4391 + i * 0.0001, 26.0961 + i * 0.0001, 80.0) for i in range(20)
        ])

    def get_location(self):
        try:
            lat, lon, alt = self.coordinates[0]
            self.coordinates.rotate(-1)
            logging.debug(f"[MOCK GPS] Coordonată returnată: lat={lat}, lon={lon}, alt={alt}")
            return GPSValue(lat, lon, alt)
        except Exception as e:
            logging.exception("[MOCK GPS] Eroare la generarea coordonatei")
            return GPSValue(None, None, None)

# === DroneKit setup ===


class DroneKitGPSProvider(BaseGPSProvider):
    def __init__(self, connection_string='/dev/ttyUSB0', baud_rate=57600, bypass=False):
        self.bypass = bypass
        self.vehicle = None
        self.location = GPSValue(None, None, None)
        self.connected = False
        self.connection_string = connection_string
        self.baud_rate = baud_rate

        self.command_queue = Queue()
        self.current_state = "IDLE"
        self.current_command = None
        self.state_machine_thread = start_thread(self._run_state_machine, "DroneStateMachine")

    def _connect_drone(self):
        try:
            print("[DroneKitGPSProvider] Conectare la dronă...")
            self.vehicle = connect(self.connection_string, baud=self.baud_rate, wait_ready=False)
            time.sleep(1)
            self.vehicle.add_attribute_listener('location.global_frame', self.gps_callback)
            time.sleep(1)
            self.connected = True
            logging.info("[DroneKitGPSProvider] Conectare la Pixhawk completă")
        except Exception as e:
            logging.error(f"[DroneKitGPSProvider] Eroare la conectare: {e}")
            self.connected = False

    def gps_callback(self, self_ref, attr_name, value):
        try:
            self.location = GPSValue(value.lat, value.lon, value.alt)
            logging.debug(f"[GPS] lat={value.lat}, lon={value.lon}, alt={value.alt}")
        except Exception as e:
            logging.exception("[GPS] Eroare în gps_callback")
    def get_location(self):
        return self.location
    def ensure_connection(self):
        if not self.connected or self.vehicle is None:
            raise Exception("Drone not connected")
        return True

    def arm_and_takeoff(self, target_altitude, vehicle_mode):
        global stop_takeoff_event
        try:
            self.ensure_connection()
        except:
            return "[DroneKit] Drone not connected"
        stop_takeoff_event.clear()
        print("[DroneKit] Armare..........in mod ", vehicle_mode)
        self.vehicle.mode = VehicleMode(vehicle_mode)

        if not self.wait_until_ready():
            return "[DroneKit] Nu e armabilă. Ieșire."

        print(f"[DroneKit] Decolare la {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)

        while not stop_takeoff_event.is_set():
            alt = self.vehicle.location.global_relative_frame.alt
            print("Altitudine (față de nivelul mării):", self.vehicle.location.global_frame.alt)
            print("Altitudine relativă (față de decolare):", alt)
            print("Altitudine target_altitude:", target_altitude * 0.95)
            if alt >= target_altitude * 0.95:
                print("[DroneKit] Altitudine atinsă.")
                break
            time.sleep(1)

        return "Drone Takeoff"

    def wait_until_ready(self, timeout=30):

        if not self.ensure_connection():
            return False
        # print("[INFO] Trimit comanda de armare forțată (MAV_CMD_COMPONENT_ARM_DISARM)...")
        # self.vehicle._master.mav.command_long_send(
        #     self.vehicle._master.target_system,
        #     self.vehicle._master.target_component,
        #     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        #     0,          # confirmation
        #     1,          # param1: 1=arm, 0=disarm
        #     21196,      # param2: magic code pentru override
        #     0, 0, 0, 0, 0
        # )
        self.vehicle.armed = True

        print("..............[DroneKit] Așteptăm ca drona să fie armabilă...")
        start = time.time()
        while not self.vehicle.armed:

            print("..  -> Mode:", self.vehicle.mode.name)
            print("..  -> Is armable:", self.vehicle.is_armable)
            print("..  -> ARMED:", self.vehicle.armed)
            print("..  -> EKF OK:", self.vehicle.ekf_ok)
            print("..  -> GPS fix:", self.vehicle.gps_0.fix_type)
            print("..  -> Sateliți:", self.vehicle.gps_0.satellites_visible)
            print("..  -> Sistem:", self.vehicle.system_status.state)
            print("Altitudine (față de nivelul mării):", self.vehicle.location.global_frame.alt)
            print("Altitudine relativă (față de decolare):", self.vehicle.location.global_relative_frame.alt)

            if time.time() - start > timeout:
                print(".....[DroneKit] Timeout atins. Nu e armabilă.")
                return False
            time.sleep(1)

        print("[DroneKit] Drona este gata.")
        return True

    def land_drone(self):
        global stop_takeoff_event

        try:
            self.ensure_connection()
        except:
            return "[DroneKit] Drone not connected"

        print("[DroneKit] Aterizare..bbbbbbbbb.")
        stop_takeoff_event.set()
        print("[DroneKit] Aterizare...")
        self.vehicle.mode = VehicleMode("LAND")

        while self.vehicle.armed:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f"[LAND] Altitudine: {alt:.2f} m")

            if alt is not None and alt < 0.1:
                print("[LAND] Altitudine mică → dezarmez")
                self.vehicle._master.mav.command_long_send(
                    self.vehicle._master.target_system,
                    self.vehicle._master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 21196, 0, 0, 0, 0, 0
                )
                break
            time.sleep(1)

        print("[DroneKit] Aterizare completă.")
        return "Drone Landing"
# Adaugă o comandă în coada de execuție pentru dronă.
# Comenzile pot fi: takeoff, land, orbit, auto_search etc.
# Vor fi procesate de state machine în ordinea primirii.

    def enqueue_command(self, command, args=None):
        self.command_queue.put((command, args or {}))

# Rulează un state machine pentru dronă care procesează comenzile primite din coadă (`command_queue`).
# Fiecare comandă este tratată secvențial: takeoff → zboară → revino → aterizează etc.
# Se ocupă de logica de zbor autonom și controlează tranziția între stările dronei (IDLE, TAKING_OFF, IN_AIR, LANDING).

    def _run_state_machine(self):
        # Pas 0: Conectare obligatorie
        self._connect_drone()
        if not self.connected:
            print("[STATE] Nu se poate porni state machine: drona nu este conectată.")
            return

        while True:
            command, args = self.command_queue.get()
            self.current_command = command

            if command == "land":
                self.current_state = "LANDING"
                stop_takeoff_event.set()
                self.land_drone()
                self.current_state = "IDLE"
                continue

            if command == "takeoff":
                self.current_state = "TAKING_OFF"
                alt = args.get("altitude", 2)
                mode = args.get("mode", "GUIDED")
                self.arm_and_takeoff(alt, mode)
                
                self.current_state = "IN_AIR"
                continue

            if self.current_state != "IN_AIR":
                print(f"[STATE] Ignor comanda '{command}'  pentru ca drona nu a decolat.")
                continue

            if command == "auto_search":
                self._handle_auto_search(args)
            elif command == "goto_and_return":
                self._handle_goto_and_return(args)
            elif command == "orbit":
                self._handle_orbit(args)

            self.current_command = None


# Returnează un snapshot cu starea curentă a dronei: poziție, altitudine, mod, stare armare și conexiune.
    def get_status(self):
        return {
            "state": self.current_state,
            "command": self.current_command,
            "alt": self.vehicle.location.global_relative_frame.alt if self.vehicle else None,
            "lat": self.location.lat,
            "lon": self.location.lon,
            "mode": self.vehicle.mode.name if self.vehicle else None,
            "armed": self.vehicle.armed if self.vehicle else None,
            "connected": self.connected
        }

    def close(self):
        if self.vehicle:
            print("[DroneKit] Închidere conexiune cu drona...")
            self.vehicle.close()
    def set_roi(self,location):
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_SET_ROI,
            0,
            0, 0, 0, 0,
            location.lat, location.lon, location.alt
        )
        self.vehicle.send_mavlink(msg)

    def send_ned_velocity(self, velocity_x, velocity_y, velocity_z, duration):

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            velocity_x, velocity_y, velocity_z,
            0, 0, 0,
            0, 0
        )
        for _ in range(duration):
            self.vehicle.send_mavlink(msg)
            time.sleep(1)

    #----Comenzi drona----
    
    # Trimite drona să orbiteze în jurul unei locații date (ex: locația unei detecții).
    # Se mișcă în cerc pe baza calculelor NED (velocity în North-East-Down).
    def _handle_orbit(self, args):

        center_location = args.get("location")
        radius = args.get("radius", 5)
        velocity = args.get("speed", 1.0)
        duration = args.get("duration", 20)
        if center_location and radius > 0 and velocity > 0 and duration > 0:

            print("[DRONA] Încep orbitarea în jurul punctului...")
            self.set_roi(center_location)

            angle = 0
            step_time = 1
            steps = int(duration / step_time)

            for _ in range(steps):
                vx = -velocity * math.sin(angle)
                vy = velocity * math.cos(angle)
                self.send_ned_velocity(vx, vy, 0, 1)
                angle += (velocity / radius) * step_time
                time.sleep(0.1)

            print("[DRONA] Orbită completă sau întreruptă.")

    # Trimite drona la o locație dată, așteaptă, apoi se întoarce la punctul de decolare.
    def _handle_goto_and_return(self, args):


        target_location = args.get("location")
        speed = args.get("speed", 4)
        if target_location and speed:

   
            home_location = self.vehicle.location.global_relative_frame
            self.vehicle.groundspeed = speed

            print("[DRONA] Zbor către punctul țintă...")
            self.vehicle.simple_goto(target_location)

            while self.get_distance_metres(self.vehicle.location.global_relative_frame, target_location) > 2:
                print("[DRONA] Apropiere de punctul țintă...")
                time.sleep(1)

            print("[DRONA] Ajuns la destinație. Aștept 3 secunde...")
            time.sleep(3)

            print("[DRONA] Revenire la punctul inițial...")
            self.vehicle.simple_goto(home_location)
            while self.get_distance_metres(self.vehicle.location.global_relative_frame, home_location) > 2:
                print("[DRONA] Apropiere de poziția inițială...")
                time.sleep(1)

            print("[DRONA] Revenit la poziția inițială.")


    def get_distance_metres(self, aLocation1, aLocation2):

        dlat = aLocation2.lat - aLocation1.lat
        dlon = aLocation2.lon - aLocation1.lon
        return math.sqrt((dlat*dlat) + (dlon*dlon)) * 1.113195e5

    # Execută o căutare automată pe o zonă definită (serpuita), oprindu-se dacă se detectează un pericol.
    # În caz de detecție, zboară acolo, orbitează pentru asigurarea alarmei, activează servomotorul, apoi revine la bază.

    def _handle_auto_search(self, args):
        global detected_flag, event_location

        area_size = args.get("area_size", 5)
        step = args.get("step", 1)
        height = args.get("height", 2)
        speed = args.get("speed", 4)
        if area_size and step and height and speed:
            print("[DRONA] Execut misiune...")
            home = self.vehicle.location.global_relative_frame
            self.vehicle.groundspeed = speed
            print("[AUTO] Locatie de start salvată.")

            # dx = np.linspace(0, area_size, int(area_size / step))
            # dy = np.linspace(0, area_size, int(area_size / step))

            dx = np.arange(0, area_size + step, step)
            dy = np.arange(0, area_size + step, step)
            
            print("[AUTO] Încep serpuirea...")

            for i, y in enumerate(dy):
                for x in (dx if i % 2 == 0 else reversed(dx)):
                    if detected_flag and event_location:
                        print("[AUTO] Detecție activată! Mă duc la locația salvată.")
                        self.vehicle.simple_goto(event_location)
                        time.sleep(5)
                        
                        self._handle_orbit({"location": event_location, "radius": 3, "speed": 1.0, "duration": 20})
                        
     
                        print("[AUTO] Activez servomotorul!")
                        activate_servos()
                        print("[AUTO] Revin la punctul inițial...")
                        self.vehicle.simple_goto(home)
                        while self.get_distance_metres(self.vehicle.location.global_relative_frame, home) > 2:
                            time.sleep(1)
                        print("[AUTO] Misiune completă.")
                        return
                    
                    new_location = LocationGlobalRelative(
                        home.lat + (y / 111111),  # ~1 grad lat ≈ 111km
                        home.lon + (x / (111111 * math.cos(math.radians(home.lat)))),
                        height
                    )
                    self.vehicle.simple_goto(new_location)
                    while self.get_distance_metres(self.vehicle.location.global_relative_frame, new_location) > 2:
                        time.sleep(1)

            print("[AUTO] Misiune completă fără detecție.")
            self.vehicle.simple_goto(home)
            while self.get_distance_metres(self.vehicle.location.global_relative_frame, home) > 2:
                time.sleep(1)
#########################################################################
#########################################################################
# Initialize GPS provider
gps_provider = MockGPSProvider() if USE_SIMULATOR else DroneKitGPSProvider(bypass=False)
##########################################################################
##########################################################################




# Capturează frame-uri de la cameră și le salvează împreună cu datele GPS curente într-un buffer global.
# Rulează constant și este sursa de date pentru toate celelalte threaduri de detecție sau streaming.

def camera_thread():
    global frame_buffer,latest_raw, latest_jpeg


    logging.debug("Firul principal (camera) a pornit.")

    use_picam = False
    picam = None

    if USE_PICAM:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            # cfg = picam.create_video_configuration(main={"size": (W, H), "format": "RGB888"})
            cfg = picam.create_video_configuration(
                main={"format": "RGB888", "size": (W, H)},
            transform=Transform(hflip=True, vflip=True) )
            picam.configure(cfg)
            picam.start()
            use_picam = True
            logging.debug("Using PiCamera2 at", (W, H))
        except Exception as e:
            logging.error("PiCamera2 not available, fallback to USB:", e)

    if not use_picam:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logging.debug("[ERR] USB camera cannot open."); return
        lologging.debugg("Using USB camera at", (W, H))

    next_jpeg_at = 0.0
    period = 1.0 / MJPEG_HZ
   

    while True:
        if use_picam:
            frame_rgb = picam.capture_array()
            # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame =frame_rgb
        else:
            ret, frame = cap.read()
            if not ret:
                cap.grab(); ret, frame = cap.retrieve()
                if not ret:
                    time.sleep(0.01); continue

        with raw_lock:
            latest_raw = frame
        gps_snapshot = {
                "lat": 47.6499,
                "lon": 26.2490,
                "alt": 1.4,
                "timestamp": time.time(),
                }

        gps = gps_provider.get_location()
        lat = _gps_get(gps_snapshot, "lat")
        lon = _gps_get(gps_snapshot, "lon")
        alt = _gps_get(gps_snapshot, "alt")

        lat_s = f"{lat:.6f}" if isinstance(lat, (int, float)) else "—"
        lon_s = f"{lon:.6f}" if isinstance(lon, (int, float)) else "—"
        alt_s = f"{alt:.1f}"  if isinstance(alt, (int, float))  else "—"

        txt = f"Lat: {lat_s} Lon: {lon_s} Alt: {alt_s}"
        cv2.putText(frame, txt, (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


        with frame_lock:
            frame_buffer = {"image": frame, "gps": gps_snapshot}   

        now = time.time()
        if now >= next_jpeg_at:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    latest_jpeg = buf.tobytes()
            next_jpeg_at = now + period
            
     
        else:
            time.sleep(0.1)

      
       

# Rulează modelul YOLOv11 pe frame-urile capturate pentru a detecta obiecte, inclusiv „om_la_inec”.
# În caz de detecție, salvează poziția GPS și trimite informații prin WebSocket.
# Marchează pe imagine offsetul și distanța față de centrul camerei.
def yolo_function_thread():
    
    global frame_buffer, yolo_output_frame, detected_flag, popup_sent, last_detection_time,  event_location
    cam_x, cam_y = 320, 240
   # cam_x, cam_y = W,H
    PIXELS_PER_CM = 10
    object_present = False
    logging.info("start YOLO.")  
    model = YOLO("/home/dariuc/yolo-stream/models/my_model.pt")
    logging.info("Firul yolo_function_thread este activ...................")  
    
    yoloTimestamp=0.0
    
    while not stop_detection_event.is_set():
    
     

          
        with frame_lock:
            data = frame_buffer.copy() if frame_buffer is not None else None
        if data is None:
            time.sleep(0.01)
            logging.debug("data is None")
            continue
            
        frame = data["image"]
        gps_info = data["gps"]

        if gps_info['timestamp']==yoloTimestamp:
            time.sleep(0.01)
            logging.debug("skip time")
            continue
        yoloTimestamp=gps_info['timestamp']

        logging.debug("BEFORE yoloTimestamp=",time.time())

        results = model(frame, verbose=False)
        logging.debug("AFTER RUN MODEL yoloTimestamp=",time.time())


        annotated = frame.copy()
        names = results[0].names
        class_ids = results[0].boxes.cls.tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if gps_info["lat"] and gps_info["lon"]:
            gps_text = f"Lat: {gps_info['lat']:.6f} Lon: {gps_info['lon']:.6f} Alt: {gps_info['alt']:.1f}"
            timestamp_text = f"Timp: {time.strftime('%H:%M:%S', time.localtime(gps_info['timestamp']))}"
            cv2.putText(annotated, gps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(annotated, timestamp_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
        current_detection = False
        for i, cls_id in enumerate(class_ids):
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(cls_id)]
            color = (0, 255, 0) if label == "om_la_inec" else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if label == "om_la_inec":
                current_detection = True
                if not object_present:
                    # Save event location when first detected
                    if gps_info["lat"] and gps_info["lon"]:
                        event_location = LocationGlobalRelative(
                            gps_info["lat"], 
                            gps_info["lon"], 
                            gps_info["alt"]
                        )
                        logging.info(f"[DETECTION] Event location saved: {event_location}")
                    
                    detected_flag = True
                    popup_sent = True
                    last_detection_time = time.time()
                    object_present = True
                    
                obj_x = (x1 + x2) // 2
                obj_y = (y1 + y2) // 2
                dx_cm = (obj_x - cam_x) / PIXELS_PER_CM
                dy_cm = (obj_y - cam_y) / PIXELS_PER_CM
                dist_cm = (dx_cm**2 + dy_cm**2)**0.5
                cv2.line(annotated, (cam_x, cam_y), (obj_x, obj_y), (0, 0, 255), 2)
                cv2.circle(annotated, (cam_x, cam_y), 5, (255, 0, 0), -1)
                cv2.circle(annotated, (obj_x, obj_y), 5, (0, 255, 0), -1)
                offset_text = f"x:{dx_cm:.1f}cm | y:{dy_cm:.1f}cm"
                dist_text = f"Dist: {dist_cm:.1f}cm"
                cv2.putText(annotated, offset_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(annotated, dist_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                
        if not current_detection:
            detected_flag = False
            popup_sent = False
            object_present = False
        
        obiecte_detectate = []
        nivel_detectat = None

        for i, cls_id in enumerate(class_ids):
            label = names[int(cls_id)]

            if label in ["Meduze", "Rechin", "person", "rip_current"]:
                obiecte_detectate.append(label)

            if label == "lvl_mic":
                nivel_detectat = "mic"
            elif label == "lvl_mediu":
                nivel_detectat = "mediu"
            elif label == "lvl_adanc":
                nivel_detectat = "adânc"


        socketio.emit("detection_update", {
            "obiecte": obiecte_detectate,
            "nivel": nivel_detectat
        })     
        with output_lock:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            yolo_output_frame = cv2.imencode('.jpg', annotated, encode_params)[1].tobytes()
    
        time.sleep(0.05)
        



# Rulează modelul YOLO pentru detecția de forme de viață: meduze, rechini și oameni.
# Dacă este activat modul smart, schimbă automat streamul cu modul de clasificare înec (`xgb`) când detectează o persoană.
# În orice caz, trimite etichetele detectate prin WebSocket și le desenează în streamul `mar_feed`.

def livings_inference_thread(video=None):
    global mar_output_frame, frame_buffer
    obiecte_detectate = []

    logging.info("Firul livings_inference_thread rulează...")
    model = YOLO("models/livings.pt")
    try:
        model.fuse()
    except Exception:
        pass

    LIVINGS_IMG_SZ = 320   # sau 288
    LIVINGS_FPS    = 3.0
    last_t = 0.0

    while not stop_detection_liv_event.is_set():
        if not streaming:
            time.sleep(0.1)
            continue

        # limitare FPS
        now = time.monotonic()
        if now - last_t < 1.0 / LIVINGS_FPS:
            time.sleep(0.005)
            continue
        last_t = now

        obiecte_detectate.clear()

        # preluare frame
        with frame_lock:
            data = frame_buffer.copy() if frame_buffer is not None else None
        if data is None:
            time.sleep(0.01)
            continue

        frame = np.ascontiguousarray(data["image"])
        annotated = frame.copy()   # <-- IMPORTANT: mereu definit!

        # inferență pe ndarray: stream=False
        try:
            results = model.predict(
                source=frame,
                imgsz=LIVINGS_IMG_SZ,
                conf=0.40,
                device=DEVICE,
                half=(DEVICE != "cpu"),
                stream=False,
                verbose=False
            )
        except Exception as e:
            logging.exception("[LIVINGS] Eroare la predict: %s", e)
            # totuși trimitem ultimul annotated (frame simplu) ca să nu cadă feed-ul
            with mar_lock:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                mar_output_frame = cv2.imencode('.jpg', annotated, encode_params)[1].tobytes()
            time.sleep(0.01)
            continue

        # results e listă; procesează primul element dacă există
        if results:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])

                    if 0 <= cls_id < 3:
                        name = ["Meduze", "Rechin", "person"][cls_id]
                    else:
                        logging.debug("[LIVINGS] cls_id invalid: %s", cls_id)
                        continue

                    obiecte_detectate.append(name)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # smart switch (dacă îl folosești)
        if smart_stream_mode and results:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 2:  # person
                        print("[SMART SWITCH] Detected person → switching to raw + xgb")
                        stop_detection_liv_event.set()
                        stop_segmentation_event.set()
                        socketio.emit("stream_config_update", {"left": "raw", "right": "xgb"})
                        global pose_thread, stop_pose_event
                        if pose_thread is None or not pose_thread.is_alive():
                            stop_pose_event.clear()
                            pose_thread = start_thread(pose_xgb_inference_thread, "PoseXGBDetection")
                        return

        socketio.emit("detection_update", {"obiecte": obiecte_detectate})

        with mar_lock:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            mar_output_frame = cv2.imencode('.jpg', annotated, encode_params)[1].tobytes()

        time.sleep(0.01)
# Rulează modelul YOLOv11 de segmentare semantică pentru a colora zonele din apă pe baza adâncimii sau a curenților de rupere.
# Suportă clase ca: „lvl_mic”, „lvl_mediu”, „lvl_adanc”, „rip_current”.
# Fiecare mască este desenată peste imaginea originală și este transmisă prin streamul `seg_feed`.
# Trimite și eticheta corespunzătoare prin WebSocket la frontend.

def segmentation_inference_thread(video=None):
    global seg_output_frame,frame_buffer

    logging.info("Firul segmentation_inference_thread rulează...")
    model = YOLO("models/yolo11n-seg-custom.pt")  # <- model YOLOv11n SEGMENTARE

    # Culori personalizate pentru fiecare clasă
    color_map = {
        "lvl_mic": (255, 255, 0),
        "lvl_mediu": (0, 255, 0),
        "lvl_adanc": (0, 0, 255),
        "rip_current": (0, 165, 255)
    }

    while not stop_segmentation_event.is_set():
        if not streaming:
            time.sleep(0.1)
            continue

        with frame_lock:
            data = frame_buffer.copy() if frame_buffer is not None else None
        if data is None:
            time.sleep(0.05)
            continue

        frame = data["image"]
        gps_info = data["gps"]
        annotated = frame.copy()
        nivel_detectat = None

        # rulează modelul YOLOv11n pe frame
        results = model.predict(source=frame, conf=0.5, stream=False)

        for r in results:
            masks = r.masks
            names = r.names

            if masks is not None and masks.data is not None:
                for i, mask in enumerate(masks.data):
                    if i >= len(r.boxes.cls):
                        continue  # siguranță la indexare

                    cls_id = int(r.boxes.cls[i].item())
                    label = names[cls_id]
                    color = color_map.get(label, (0, 255, 255))

                    mask_np = mask.cpu().numpy()
                    annotated[mask_np > 0.5] = color

                    # Afișează eticheta în centrul măștii
                    ys, xs = np.where(mask_np > 0.5)
                    if len(xs) > 0 and len(ys) > 0:
                        cx, cy = int(np.mean(xs)), int(np.mean(ys))
                        cv2.putText(annotated, label, (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

                    # Interpretare nivel
                    if label in ["lvl_mic", "lvl_mediu", "lvl_adanc"]:
                        nivel_detectat = label.replace("lvl_", "")
                    elif label == "rip_current":
                        nivel_detectat = "rip_current"

        socketio.emit("detection_update", {"nivel": nivel_detectat})

        with seg_lock:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            seg_output_frame = cv2.imencode('.jpg', annotated, encode_params)[1].tobytes()
            #seg_output_frame = cv2.imencode('.jpg', annotated)[1].tobytes()

        time.sleep(0.05)

# Aplică YOLO Pose Estimation pe fiecare frame și extrage 34 de coordonate per frame (x, y).
# Păstrează un buffer de 30 frame-uri consecutive, le transformă într-un vector [1020] și face clasificare cu XGBoost.
# Dacă este detectat „inec”, trimite alertă prin WebSocket și marchează frame-ul.
# Dacă nu apare niciun „inec” timp de 10 secunde în modul smart, revine automat la streamul inițial `mar + seg`.
def pose_xgb_inference_thread(video=None, model=None):


    global smart_stream_mode,pose_thread_started
    global stop_pose_event, stop_detection_liv_event, stop_segmentation_event
    global pose_lock
    global streaming, socketio
    global detection_liv_thread, segmentation_thread

    global livings_inference_thread, segmentation_inference_thread
    global start_thread

    last_t = 0.0 
    last_inec_time = time.time()
    pose_thread_started = True






    while not stop_pose_event.is_set():
        if not streaming:
            time.sleep(0.1)
            continue
        
        if not pose_thread_started:
            time.sleep(0.02); continue
        logging.debug("pose_thread_started=%s",pose_thread_started)
        now = time.monotonic()
        if now - last_t < 1.0/POSE_FPS:
            time.sleep(0.001); continue
        last_t = now

        with raw_lock:
            frame = None if latest_raw is None else latest_raw.copy()
        if frame is None:
            time.sleep(0.005); continue

        res = pose_model.predict(frame, imgsz=POSE_IMGSZ, device=DEVICE, half=HALF, verbose=False)
        best = pick_best_xy(res, MIN_DRAW_CONF)

        if best is not None:
            xy, conf = best
            overlay = frame.copy()
            draw_pose_overlay(overlay, xy, conf, min_conf=MIN_DRAW_CONF)

            vis_ratio = float((conf >= KP_OK_CONF).mean())
            cv2.putText(overlay, f"vis={vis_ratio:.2f}  {last_label} p={last_proba:.2f}", (10,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,0,255) if last_label=='INEC' else (0,255,0), 2)

            ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with pose_lock:
                    global pose_jpeg
                    pose_jpeg = buf.tobytes()

            pushed = False
            if frame_quality_ok(conf):
                pkt = KPPacket(t=now, frame=frame, xy=xy, conf=conf)
                try:
                    q_kp.put(pkt, timeout=0.01)
                    pushed = True
                except queue.Full:
                    pass
            if pushed:
                logging.debug(f"pose->kp   vis={vis_ratio:.2f}  q_kp={q_kp.qsize()}")
        else:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with pose_lock:
                    pose_jpeg = buf.tobytes()


        #############################
        # # smart switch (după ce am consumat batchul curent)
        # if smart_stream_mode and (time.time() - last_inec_time) > 10:
        #     print("[SMART SWITCH] No inec in 10s → switching back to mar + seg")
        #     stop_pose_event.set()
        #     stop_detection_liv_event.clear()
        #     stop_segmentation_event.clear()
        #     start_thread(livings_inference_thread, "LivingsThread")
        #     start_thread(segmentation_inference_thread, "SegmentationThread")
        #     socketio.emit("stream_config_update", {"left": "mar", "right": "seg"})
        #     return




def normalizer_thread():
    global pose_thread_started
    while True:
        if not pose_thread_started:
            time.sleep(0.02); continue
        try:
            pkt: KPPacket = q_kp.get(timeout=0.2)
        except queue.Empty:
            continue
        xy_norm = normalize_TS(pkt.xy)
        npkt = NormPacket(t=pkt.t, xy_norm=xy_norm, conf=pkt.conf)
        try:
            q_norm.put(npkt, timeout=0.01)
            logging.debug(f"kp->norm    q_norm={q_norm.qsize()}")
        except queue.Full:
            pass

def feature_extractor_thread():
    global pose_thread_started
    win = deque()
    next_eval_at = 0.0
    while True:
        if not pose_thread_started:
            time.sleep(0.02); continue
        try:
            npkt: NormPacket = q_norm.get(timeout=0.3)
        except queue.Empty:
            continue
        now = npkt.t
        win.append((now, npkt.xy_norm))
        t_min = now - WINDOW_SEC - 1e-6
        while win and win[0][0] < t_min:
            win.popleft()

        if now < next_eval_at:
            continue
        next_eval_at = now + HOP_SEC

        xs = [xy for (t, xy) in win if t >= now - WINDOW_SEC - 1e-6]
        if len(xs) < 4:
            continue
        feats = features_on_window(xs)
        if feats is None:
            continue
        fpkt = FeatPacket(t_start=now - WINDOW_SEC, t_end=now, feats=feats)
        try:
            q_feat.put(fpkt, timeout=0.01)
            logging.debug(f"norm->feat  window={len(xs)}  q_feat={q_feat.qsize()}")
        except queue.Full:
            pass

def classifier_thread():
    global last_label, last_proba, cooldown_until, consec_on, consec_off,pose_thread_started

    feature_order = FEATURE_LIST
    if feature_order is None and hasattr(XGB, "feature_names_in_"):
        feature_order = list(XGB.feature_names_in_)

    while True:
       # logging.debug("classifier_thread->run:%s",pose_thread_started)
        if not pose_thread_started:
            time.sleep(0.02);
            continue
        #logging.debug("classifier_thread->run")
        try:
            fpkt: FeatPacket = q_feat.get(timeout=0.3)
        except queue.Empty:
            continue

        feats = fpkt.feats
        cols = feature_order if feature_order is not None else list(feats.keys())
        row = {c: feats.get(c, np.nan) for c in cols}
        X = pd.DataFrame([row], columns=cols)

        if hasattr(XGB, "predict_proba"):
            p = XGB.predict_proba(X)[0]
            classes = getattr(XGB, "classes_", [0,1])
            pos_idx = list(classes).index(1) if 1 in classes else 1
            proba = float(p[pos_idx])
        else:
            yhat = XGB.predict(X)[0]
            proba = float(yhat) if isinstance(yhat,(int,float,np.floating)) else 0.0

        if CALIBRATOR is not None:
            try:
                proba = float(CALIBRATOR.predict_proba(np.array([[proba]]))[:,1][0])
            except Exception:
                pass

        now = time.monotonic()
        state_changed = False

        if now < cooldown_until:
            last_proba = proba

            # lat = gps_info.get('lat'); lon = gps_info.get('lon')
            # lat_str = f"{lat:.6f}" if lat is not None else "necunoscut"
            # lon_str = f"{lon:.6f}" if lon is not None else "necunoscut"
            lat_str =  "necunoscut"
            lon_str =  "necunoscut"

            socketio.emit("detection_update", {
                "eveniment": last_label + "consec_on:"+str(consec_on) + " ,consec_off:" + str(consec_off),
                "nivel": last_label,
                "proba": round(float(proba), 3),
                "timestamp": float(now),
                "gps": {"lat": lat_str, "lon": lon_str}
            }) 
            logging.debug(f"feat->cls  (cooldown) p={proba:.3f} label={last_label}")
            continue

        if proba >= tau_on:
            consec_on += 1;  consec_off = 0
        elif proba <= tau_off:
            consec_off += 1; consec_on = 0

        new_label = last_label
        if last_label != "INEC" and consec_on >= 2:
            new_label = "INEC"
            consec_on = consec_off = 0
            cooldown_until = now + cooldown_s
            state_changed = True
        elif last_label != "INOT" and consec_off >= 2:
            new_label = "INOT"
            consec_on = consec_off = 0
            state_changed = True

        last_proba = proba
        if state_changed:
            last_label = new_label

            # lat = gps_info.get('lat'); lon = gps_info.get('lon')
            # lat_str = f"{lat:.6f}" if lat is not None else "necunoscut"
            # lon_str = f"{lon:.6f}" if lon is not None else "necunoscut"
            lat_str =  "necunoscut"
            lon_str =  "necunoscut"

            socketio.emit("detection_update", {
                "eveniment": last_label + "consec_on:"+str(consec_on) + " ,consec_off:" + str(consec_off),
                "nivel": last_label,
                "proba": round(float(proba), 3),
                "timestamp": float(now),
                "gps": {"lat": lat_str, "lon": lon_str}
            })    

 
        logging.debug(f"feat->cls   p={proba:.3f}  label={last_label}  on={consec_on} off={consec_off}")


        


# === Flask Routes ===
@app.route("/")
def index():
    return render_template("index.html")
# Trimite streamul video brut de la cameră către browser, tip MJPEG.
@app.route("/video_feed")
def video_feed():

    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with jpeg_lock:
                frame = latest_jpeg if latest_jpeg is not None else blank_jpeg()
            yield boundary + frame + b"\r\n"
            time.sleep(1.0 / MJPEG_HZ)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/yolo_feed")
def yolo_feed():
    def generate():
        global yolo_output_frame
        while True:
            logging.info(f"[FEED] yolo_feed..")
            with output_lock:
                frame = yolo_output_frame if yolo_output_frame is not None else blank_jpeg()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / MJPEG_HZ)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/yolo_feed_snapshot")
def yolo_feed_snapshot():
    global yolo_output_frame
    logging.info(f"[FEED] yolo_feed_snapshot..")
    with output_lock:
        frame = yolo_output_frame if yolo_output_frame is not None else blank_jpeg()
    return Response(frame, mimetype='image/jpeg')



@app.route("/mar_feed")
def mar_feed():
    def generate():
        while True:
            logging.info(f"[FEED] mar_feed_..")
            with mar_lock:
                frame = mar_output_frame or blank_jpeg()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / MJPEG_HZ)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/seg_feed")
def seg_feed():
    def generate():
        while True:
            with seg_lock:
                frame = seg_output_frame or blank_jpeg()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / MJPEG_HZ)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/xgb_feed")
def xgb_feed():
    def generate():
        while True:
            with pose_lock:
                frame = pose_jpeg or blank_jpeg()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / MJPEG_HZ)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/set_right_stream", methods=["POST"])
def set_right_stream():
    global right_stream_type,detection_thread,detection_liv_thread,stop_detection_event,stop_detection_liv_event,pose_thread,stop_pose_event,segmnetation_thread,stop_segmentation_event
    data = request.json
    selected = data.get("type")
    logging.info(f"[FLASK] Set right stream type: {selected}")
    if selected in ["yolo", "seg", "mar", "xgb"]:
        right_stream_type = selected
        logging.info(f"[FLASK] right_stream_type: {right_stream_type}")
        if selected == "yolo":
            #oprire alte threaduri
            stop_detection_liv_event.set()
            stop_segmentation_event.set()
            stop_pose_event.set()
            pose_thread_started=False
             #pornire thread
            if detection_thread is None or not detection_thread.is_alive():
                stop_detection_event.clear()
                detection_thread =start_thread(yolo_function_thread, "DetectionThread 1")
            #repornire thread
            if detection_thread and detection_thread.is_alive():
                stop_detection_event.set()
                detection_thread.join()  # așteaptă să se termine curentul thread
                stop_detection_event.clear()
                detection_thread =start_thread(yolo_function_thread, "DetectionThread 2")    

        elif selected == "seg":
            #oprire alte threaduri
            stop_detection_event.set()
            stop_detection_liv_event.set()
            stop_pose_event.set()
            pose_thread_started=False
            #pornire thread
            if segmnetation_thread is None or not segmnetation_thread.is_alive():
                stop_segmentation_event.clear()
                segmnetation_thread =start_thread(segmentation_inference_thread, "SegmentationDetection")
            #repornire thread
            if segmnetation_thread and segmnetation_thread.is_alive():
                stop_segmentation_event.set()
                segmnetation_thread.join()  # așteaptă să se termine curentul thread
                stop_segmentation_event.clear()
                segmnetation_thread =start_thread(segmentation_inference_thread, "SegmentationDetection") 

        elif selected == "mar":
            #oprire alte threaduri
            stop_detection_event.set()
            stop_segmentation_event.set()
            stop_pose_event.set()
            pose_thread_started=False
            logging.info("MMMMMMMMMAAAAAAARRRRRRRR")
            #pornire thread
            if detection_liv_thread is None or not detection_liv_thread.is_alive():
                stop_detection_liv_event.clear()
                detection_liv_thread =start_thread(livings_inference_thread, "LivingsDetection")
            #repornire thread
            if detection_liv_thread and detection_liv_thread.is_alive():
                stop_detection_liv_event.set()
                detection_liv_thread.join()  # așteaptă să se termine curentul thread
                stop_detection_liv_event.clear()
                detection_liv_thread =start_thread(livings_inference_thread, "LivingsDetection") 
        elif selected == "xgb":
            #oprire alte threaduri
            stop_detection_event.set()
            stop_detection_liv_event.set()
            stop_segmentation_event.set()
            #pornire thread
            if pose_thread is None or not pose_thread.is_alive():
                stop_pose_event.clear()
                pose_thread =start_thread(pose_xgb_inference_thread, "PoseXGBDetection")
            #repornire thread
            # if pose_thread and pose_thread.is_alive():
            #     stop_pose_event.set()
        #  pose_thread_started=False
            #     pose_thread.join()  # așteaptă să se termine curentul thread
            #     stop_pose_event.clear()
            #     pose_thread =start_thread(pose_xgb_inference_thread, "PoseXGBDetection") 
        else:
            logging.error(f"[FLASK] Invalid stream type: {selected}")



        return jsonify({"status": "ok", "current": right_stream_type})
    return jsonify({"status": "invalid"})
    
# Trimite streamul ales (YOLO, segmentare etc.) către interfață.
# Se schimbă în funcție de `right_stream_type`.
@app.route("/right_feed")
def right_feed():
    def generate():
        global right_stream_type,detection_thread,detection_liv_thread 

        while True:
            if right_stream_type == "yolo":
                with output_lock:
                    frame = yolo_output_frame or blank_jpeg()
            elif right_stream_type == "seg":

                with seg_lock:
                    frame = seg_output_frame or blank_jpeg()
            elif right_stream_type == "mar":

                with mar_lock:
                    frame = mar_output_frame or blank_jpeg()
            elif right_stream_type == "xgb":
                with pose_lock:
                    frame = pose_jpeg or blank_jpeg()
            else:
                frame = blank_jpeg()

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
# Pornește threadurile `seg` + `mar`, și pornește clasificarea înec (`xgb`) doar când se activează `pose_triggered`.
# Garantează că inferența complexă începe doar dacă a fost activată explicit.
@app.route("/start_official")
def start_official():
    global pose_thread_started
    start_thread(livings_inference_thread, "LivingsDetection")
    start_thread(segmentation_inference_thread, "SegmentationDetection")

    def watch_pose_trigger():
        global pose_triggered, pose_thread_started
        while not pose_thread_started:
            if pose_triggered:
                
                start_thread(pose_xgb_inference_thread, "PoseXGBDetection")
                break
            time.sleep(0.5)

    start_thread(watch_pose_trigger, "Watcher_XGB")
    return jsonify({"status": "official detection started"})

@app.route("/start_stream")
def start_stream():
    global streaming
    streaming = True
    return jsonify({"status": "started"})

@app.route("/stop_stream")
def stop_stream():
    global streaming
    streaming = False
    stop_detection_event.set()
    stop_detection_liv_event.set()
    return jsonify({"status": "stopped"})

@app.route("/detection_status")
def detection_status():
    return jsonify({"detected": popup_sent})
    
# Activează manual servomotoarele printr-un request GET.
@app.route("/misca")
def activate():
    activate_servos()
    return "Servomotor activat"
  





@app.route("/start_seg")
def start_seg():
    start_thread(lambda: segmentation_inference_thread(), "SegThread")
    return jsonify({"status": "segmentation started"})

@app.route("/start_livings")
def start_livings():
    start_thread(lambda: livings_inference_thread(), "LivingsThread")
    return jsonify({"status": "livings started"})

@app.route("/start_pose_xgb")
def start_pose_xgb():
    start_thread(lambda: pose_xgb_inference_thread(), "XGBThread")
    return jsonify({"status": "xgb started"})

# Activează modul „smart stream” – dacă detectează „person” în `livings` și comută automat la `pose_xgb`.
# Când nu se mai detectează înec, revine la `mar + seg`.        
@app.route("/start_smart_mode")
def start_smart_mode():
    global smart_stream_mode
    smart_stream_mode = True
    socketio.emit("stream_config_update", {"left": "mar", "right": "seg"})

    stop_detection_event.set()
    stop_pose_event.set()
    pose_thread_started=False

    stop_detection_liv_event.clear()
    stop_segmentation_event.clear()

    start_thread(segmentation_inference_thread, "SegmentationThread")
    start_thread(livings_inference_thread, "LivingsThread")

    return jsonify({"status": "smart stream mode activated"})


@socketio.on('drone_command')
def handle_drone_command(data):
    global event_location
    action = data.get('action')
    print(f"[WS] Comandă primită: {action}")

    if action == 'takeoff':
        gps_provider.enqueue_command("takeoff", {"altitude": 1, "mode": "GUIDED"})
    elif action == 'land':
       gps_provider.enqueue_command("land")
    elif action == 'goto_and_return':
       gps_provider.enqueue_command("goto_and_return", {"location": event_location, "speed": 4})
    elif action == 'orbit':
        gps_provider.enqueue_command("orbit", {"location": event_location, "radius": 5, "speed": 1.0, "duration": 20})
    elif action == 'auto_search':
        gps_provider.enqueue_command("auto_search", {"area_size": 5, "step": 1, "height": 2, "speed": 4})
    elif action == 'mission':
          gps_provider.enqueue_command("takeoff", {"altitude": 2, "mode": "GUIDED"})
          gps_provider.enqueue_command("auto_search", {"area_size": 5, "step": 1, "height": 2, "speed": 4})
          gps_provider.enqueue_command("land")      
    else:
        print(f"[WS] Comandă necunoscută: {action}")
        
@socketio.on('joystick_command')
def handle_joystick(data):
    x = data.get('x', 0)
    y = data.get('y', 0)
    z = data.get('z', 0)
    yaw = data.get('yaw', 0)

    print(f"[JOYSTICK] x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw:.2f}")
    try:
        gps_provider.ensure_connection()
        vehicle = gps_provider.vehicle

        # Aici poți adapta comenzile – ex:
        send_ned_velocity(x, y, z, 1, vehicle)
    except Exception as e:
        print(f"[JOYSTICK ERROR] {e}")
 # Trimite periodic (la fiecare secundă) prin WebSocket statusul actual al dronei: stare, mod, baterie, poziție, etc.       
def status_broadcast_loop():
    while True:
        try:
            if gps_provider.connected and gps_provider.vehicle:
                socketio.emit('drone_status', {
                    "connected": True,
                    "battery": {
                        "voltage": gps_provider.vehicle.battery.voltage,
                        "current": gps_provider.vehicle.battery.current
                    },
                    "armed": gps_provider.vehicle.armed,
                    "mode": gps_provider.vehicle.mode.name,
                    "location": {
                        "lat": gps_provider.vehicle.location.global_frame.lat,
                        "lon": gps_provider.vehicle.location.global_frame.lon
                    },
                    "event_location": {
                        "lat": event_location.lat if event_location else None,
                        "lon": event_location.lon if event_location else None
                    },
                    "altitude_global": gps_provider.vehicle.location.global_frame.alt if gps_provider.vehicle.location else None,
                    "altitude_relative": gps_provider.vehicle.location.global_relative_frame.alt if gps_provider.vehicle.location else None
                })
        except Exception as e:
            print(f"[STATUS ERROR] {e}")
        time.sleep(1)

start_thread(camera_thread, "CameraThread")
start_thread(status_broadcast_loop, "StatusBroadcast")

start_thread(normalizer_thread, "Normalizer")
start_thread(feature_extractor_thread, "FeatExtractor")
start_thread(classifier_thread, "Classifier")

if __name__ == "__main__":


    logging.info("Pornire server Flask + SocketIO")
    socketio.run(app, host="0.0.0.0", port=5000)
