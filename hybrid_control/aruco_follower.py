#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle
import numpy as np
from numpy_ringbuffer import RingBuffer
import pyaudio
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Utilidades del pipeline online (tuyas)
from functions_online import OverlapHandler, load_model_layers, open_mic, get_sample

# ==============================================================
#  TESTS DE STRETCH (tus funciones pegadas aquí)
# ==============================================================

import stretch_body.robot

def test_basic_connection():
    """Test 1: Basic robot connection and status"""
    print("=== Test 1: Basic Connection ===")
    r = stretch_body.robot.Robot()
    did_startup = r.startup()
    print(f'Robot connected to hardware: {did_startup}')
    if not did_startup:
        print("Failed to connect to robot hardware!")
        return False
    is_homed = r.is_homed()
    print(f'Robot is homed: {is_homed}')
    print(f'Base: {r.base}')
    print(f'Arm: {r.arm}')
    print(f'Lift: {r.lift}')
    print(f'End of arm joints: {r.end_of_arm.joints}')
    print(f'Head joints: {r.head.joints}')
    r.stop()
    print("Robot connection closed.")
    return True

def test_joint_status():
    """Test 2: Check current joint positions and home if needed"""
    print("\n=== Test 2: Joint Status and Homing ===")
    r = stretch_body.robot.Robot()
    if not r.startup():
        print("Failed to connect!")
        return False

    print(f"Base position (x, y, theta): ({r.base.status['x']:.3f}, {r.base.status['y']:.3f}, {r.base.status['theta']:.3f})")
    print(f"Arm position: {r.arm.status['pos']:.3f} meters")
    print(f"Lift position: {r.lift.status['pos']:.3f} meters")

    for joint_name in r.end_of_arm.joints:
        joint = r.end_of_arm.get_joint(joint_name)
        if joint_name != 'stretch_gripper':
            print(f"{joint_name} position: {joint.status['pos']:.3f} radians")
        else:
            print(f"{joint_name} position: {joint.status['pos']:.1f}")

    for joint_name in r.head.joints:
        joint = r.head.get_joint(joint_name)
        print(f"{joint_name} position: {joint.status['pos']:.3f} radians")

    if not r.is_homed():
        print("\nRobot is not homed. Homing now...")
        # Para ejecución automática SIN prompt:
        # input("Press Enter to continue with homing, or Ctrl+C to cancel...")
        r.home()
        print("Homing complete!")
    else:
        print("\nRobot is already homed.")

    r.stop()
    return True

def test_simple_movements():
    """Test 3: Simple movement commands"""
    print("\n=== Test 3: Simple Movements ===")
    r = stretch_body.robot.Robot()
    if not r.startup():
        print("Failed to connect!")
        return False

    if not r.is_homed():
        print("Robot must be homed first! Homing now…")
        r.home()

    print("Starting simple movement test...")

    # 1) Mover brazo y lift 10 cm
    print("1. Moving arm out 10cm and lift up 10cm...")
    current_arm = r.arm.status['pos']
    current_lift = r.lift.status['pos']
    r.arm.move_to(current_arm + 0.1)
    r.lift.move_to(current_lift + 0.1)
    r.push_command()
    r.wait_command()
    print("   Arm and lift movement complete!")

    # 2) Gripper
    print("2. Opening gripper...")
    r.end_of_arm.move_to('stretch_gripper', 50)  # medio abierto
    r.wait_command()
    print("   Gripper opened!")

    # 3) Cabeza
    print("3. Moving head to look around...")
    r.head.get_joint('head_pan').move_to(0.5)
    r.head.get_joint('head_tilt').move_to(-0.3)
    r.wait_command()
    print("   Head movement complete!")

    # 4) Regresar a posiciones iniciales
    print("4. Returning to starting positions...")
    r.arm.move_to(current_arm)
    r.lift.move_to(current_lift)
    r.push_command()
    r.wait_command()

    r.end_of_arm.move_to('stretch_gripper', 0)
    r.head.get_joint('head_pan').move_to(0.0)
    r.head.get_joint('head_tilt').move_to(0.0)
    r.wait_command()
    print("   Returned to starting positions!")

    r.stop()
    print("Simple movement test complete!")
    return True


# ==============================================================
#  KWS ONLINE + TRIGGER “SHEILA” → ejecutar Test 3
# ==============================================================

parser = argparse.ArgumentParser()
parser.add_argument("-d","--datafolder",default='data2',type=str)
parser.add_argument("-r","--resultfolder",default='',type=str)
parser.add_argument("--frames_per_stft",default=7,type=int)
parser.add_argument("--samples_in_window",default=30,type=int)
args = parser.parse_args()

CHANNELS = 1
FORMAT = pyaudio.paInt16

# Trigger config
KEYWORD_NAME   = "sheila"
MIN_CONF       = 0.60
N_CONSEC       = 2
COOLDOWN_S     = 10.0   # subí el cooldown para que el test 3 termine tranquilo
RUN_TEST3_ONCE = True   # si True, no vuelve a ejecutarlo en esta sesión tras correr una vez

def main():
    # Forzar TF a CPU si no hay GPU
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # ----- Carga de parámetros y modelo -----
    with open(os.path.join(args.resultfolder,"parameters.pickle"), 'rb') as f:
        train_params = pickle.load(f)
    with open(os.path.join(args.datafolder,"parameters.pickle"), 'rb') as f:
        dataset_params = pickle.load(f)

    keywords = dataset_params['keywords']
    sampling_rate = dataset_params['sampling_rate']

    if KEYWORD_NAME not in keywords:
        raise ValueError(f"La palabra '{KEYWORD_NAME}' no está en keywords: {keywords}")

    kw_idx = 1 + keywords.index(KEYWORD_NAME)  # 0=(unknown), 1..K=keywords, K+1=(null)

    frame_length = train_params['frame_length']
    frame_step   = train_params['frame_step']
    fft_length   = train_params['fft_length']
    lower_freq   = train_params['lower_freq']
    upper_freq   = train_params['upper_freq']
    n_mel_bins   = train_params['n_mel_bins']
    n_mfcc_bins  = train_params['n_mfcc_bins']

    stft_size    = frame_length + (args.frames_per_stft-1) * frame_step
    stream_chunk = stft_size + frame_step - frame_length

    print('STFT size:', stft_size)
    print('Samples per stream:', stream_chunk)

    init_bn, conv1, conv2, conv3, rec_layers, num_kwd = load_model_layers(
        os.path.join(args.resultfolder,'model.weights.h5'),
        args.datafolder,
        model_num=train_params['model_num'],
        feat_dim=13
    )

    specgram_feats = fft_length//2+1
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        n_mel_bins, specgram_feats, sampling_rate, lower_freq, upper_freq)

    overlap = OverlapHandler(stft_size, frame_length-frame_step)

    # Buffer para GUI
    num_outputs = num_kwd + 2
    pred_data = RingBuffer(capacity=args.samples_in_window, dtype=(np.float32, num_outputs))
    for _ in range(args.samples_in_window):
        pred_data.append(np.zeros(num_outputs, dtype=np.float32))

    # GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Audio
    stream, pa = open_mic(FORMAT, CHANNELS, sampling_rate, stream_chunk)

    # Primer bloque
    overlap.insert(get_sample(stream, pa, stream_chunk))
    spectro = tf.signal.stft(overlap.get(), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    mel_spectro = tf.tensordot(tf.abs(spectro), mel_mat, 1)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectro + 1e-8))[:,:n_mfcc_bins]

    for i in range(args.frames_per_stft):
        conv1.insert(init_bn(mfcc[i].numpy()))
        if conv1.pool_out:
            conv2.insert(conv1.get())
            if conv2.pool_out:
                conv3.insert(conv2.get())
                if conv3.pool_out:
                    soft = rec_layers(
                        np.expand_dims(np.expand_dims(conv3.get(), axis=0), axis=0),
                        training=False)
                    pred_data.append(soft[0,0,:])

    im = plt.imshow(np.array(pred_data).transpose(), aspect='auto', interpolation="none", cmap='Reds')
    ax.yaxis.tick_right()
    ax.set_yticks(list(range(num_outputs)))
    ax.set_yticklabels(['(unknown)'] + keywords + ['(null)'])
    plt.xlabel('Output frames')
    plt.title('Keyword spotting - output probabilities')
    plt.subplots_adjust(bottom=0.12, top=0.9, left=0.05, right=0.85)

    # ----- Estado del trigger -----
    last_trigger_ts = 0.0
    over_thresh_count = 0
    test3_ran = False
    test_running = False  # evitar reentrancia

    def run_test3_once():
        nonlocal test3_ran, test_running
        if test_running:
            print("[INFO] Test 3 ya en ejecución; ignorando trigger adicional.")
            return
        if RUN_TEST3_ONCE and test3_ran:
            print("[INFO] Test 3 ya fue ejecutado; ignorando nuevos triggers.")
            return
        test_running = True
        try:
            print("[ACTION] Ejecutando Test 3: Simple Movements…")
            ok = test_simple_movements()
            print(f"[ACTION] Test 3 completado → ok={ok}")
            test3_ran = True
        except Exception as e:
            print(f"[ERR] Test 3 falló: {e}")
        finally:
            test_running = False

    # ----- Loop de actualización -----
    def update_fig(_n):
        nonlocal last_trigger_ts, over_thresh_count
        overlap.insert(get_sample(stream, pa, stream_chunk))
        spectro = tf.signal.stft(overlap.get(), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
        mel_spectro = tf.tensordot(tf.abs(spectro), mel_mat, 1)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectro + 1e-8))[:,:n_mfcc_bins]

        for i in range(args.frames_per_stft):
            conv1.insert(init_bn(mfcc[i].numpy()))
            if conv1.pool_out:
                conv2.insert(conv1.get())
                if conv2.pool_out:
                    conv3.insert(conv2.get())
                    if conv3.pool_out:
                        soft = rec_layers(
                            np.expand_dims(np.expand_dims(conv3.get(), axis=0), axis=0),
                            training=False)
                        pred_data.append(soft[0,0,:])

                        # Trigger “sheila”
                        p_kw = float(soft[0,0,kw_idx].numpy())
                        now = time.time()
                        if p_kw >= MIN_CONF:
                            over_thresh_count += 1
                        else:
                            over_thresh_count = 0

                        if (over_thresh_count >= N_CONSEC) and ((now - last_trigger_ts) >= COOLDOWN_S):
                            print(f"[TRIGGER] '{KEYWORD_NAME}' conf≈{p_kw:.2f} → ejecutar Test 3")
                            run_test3_once()
                            last_trigger_ts = now
                            over_thresh_count = 0

        im.set_array(np.array(pred_data).transpose())
        return im,

    anim = animation.FuncAnimation(fig, update_fig, blit=False, interval=50)

    try:
        plt.show()
    except Exception:
        print("Plot Closed")

    # Limpieza audio
    stream.stop_stream(); stream.close(); pa.terminate()
    print("Program Terminated")

if __name__ == "__main__":
    main()
