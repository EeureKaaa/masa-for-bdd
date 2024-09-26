import cv2
import numpy as np
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from time import time, sleep
import torch.multiprocessing as mp
import torch
print("imported")
# Initialize a mask buffer (Queue) with a limited capacity
def initialize_mask_buffer(buffer_size):
    return Queue(maxsize=buffer_size)

# Setup logging
def initialize_logger(gpu_id):
    logger = logging.getLogger(f'GPU_{gpu_id}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - GPU %(gpu_id)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(lambda record: setattr(record, 'gpu_id', gpu_id) or True)
        logger.addHandler(stream_handler)
    
    return logger

def initialize_detector(config_file, checkpoint_file, gpu_id):
    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        masa_model = init_masa(config_file, checkpoint_file, device=device)
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
        detector = {'model': masa_model, 'test_pipeline': masa_test_pipeline}  
        logger = initialize_logger(gpu_id)
        logger.info(f'Initialized detector on GPU {gpu_id}')
        return detector, logger
    except Exception as e:
        logger = initialize_logger(gpu_id)
        logger.error(f'Failed to initialize detector on GPU {gpu_id}: {e}')
        raise e

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def draw_bboxes(frame, bboxes, frame_id, scores):
    # Create a 640x640x3 NumPy array with random values
    image =frame
    
    # Create 6 random bounding boxes

    # Ensure x_min < x_max and y_min < y_max
    # bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 10)
    bboxes = np.clip(bboxes, 0, 639)

    # Plot the image and bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    n_bboxs = []
    # Add bounding boxes
    for bbox, scores in zip(bboxes, scores):
        if scores < 0.2:
            continue
        x_min, y_min, x_max, y_max = bbox
        # n_bboxs.append([x_min, y_min, x_max, y_max])
        print(bbox)
        width = x_max - x_min
        height = y_max - y_min
        rect = Rectangle((x_min, y_min), width, height, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

    plt.axis('off')
    plt.title('640x640x3 Random Image with 6 Bounding Boxes')
    plt.show()

    print("Bounding Boxes (x_min, y_min, x_max, y_max):")
    print(frame_id, n_bboxs)

    # Save the image with bounding boxes
    plt.savefig(f'/home/wxh/data_wxh/hhz/masa/image_with_bboxes_frame{frame_id}.png', bbox_inches='tight', pad_inches=0)
    print("\nImage with bounding boxes saved as 'image_with_bboxes.png'")

def detect_objects(detector, frame, frame_idx, video_len, score_thr=0.3):
    try:
        frame = (np.transpose(frame,(1,2,0))*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        masa_model = detector['model']
        masa_test_pipeline = detector['test_pipeline']
        result = inference_masa(masa_model, frame,
                                frame_id=frame_idx,
                                video_len=video_len,
                                test_pipeline=masa_test_pipeline,
                                text_prompt="object",
                                detector_type='mmdet')
        if isinstance(result, tuple):
            result = result[0]
        bboxes = result[0].pred_track_instances.bboxes.cpu().numpy()
        scores = result[0].pred_track_instances.scores.cpu().numpy()
        labels = result[0].pred_track_instances.instances_id.cpu().numpy()
        
        # draw_bboxes(frame, bboxes, frame_idx, scores)
        

        detected_objects = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if score >= score_thr:
                x1, y1, x2, y2 = bbox.astype(int)
                if x1==x2 or y1==y2 or abs(x1-x2)/abs(y1-y2) > 7 or abs(y1-y2)/abs(x1-x2) > 7:
                    continue
                detected_objects.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(score),
                    'class_id': int(label)
                })
        return detected_objects
    except Exception as e:
        print(e)
        return []

def process_video(video, video_idx, output_folder, detector, logger, mask_buffer, score_thr=0.2, target_width=640):
    mask_folder = os.path.join(output_folder, 'mask')
    image_folder = os.path.join(output_folder, 'image')
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    total_frames = video.shape[0]
    for frame_count in range(video.shape[0]):
        frame_name = f'frame{frame_count:06d}'
        mask_path = os.path.join(mask_folder, f'{frame_name}.npz')
        image_path = os.path.join(image_folder, f'{frame_name}.png')
        if os.path.exists(mask_path):
            continue
        
        frame = video[frame_count].numpy()
        
        # # Resize frame while maintaining aspect ratio
        # pp = frame.shape[1] / target_width
        # target_height = int(frame.shape[0] / pp)
        resized_frame = frame

        # Detect objects
        detected_objects = detect_objects(detector, resized_frame, frame_count, video.shape[0], score_thr)

        # Create mask
        masks = {}
        masks['image_size'] = resized_frame.shape
        for idx, obj in enumerate(detected_objects, start=1):
            # mask = np.zeros(resized_frame.shape[:2], dtype=np.int32)
            x1, y1, x2, y2 = obj['bbox']
            score = obj['score']
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[2]-1), min(y2, frame.shape[1]-1)
            # mask[y1:y2, x1:x2] = 1
            masks[obj['class_id']] = (x1, y1, x2, y2, score)

        try:
            # if not os.path.exists(mask_path):
            np.savez_compressed(mask_path, masks)
            cv2.imwrite(image_path, (np.transpose(frame,(1,2,0))*255).astype(np.uint8))
                # Add mask and video path to buffer (blocks if buffer is full)
                # frame_idx = str(video_idx).rjust(12, '0') + '_' + str(frame_count).rjust(3, '0')
                # mask_buffer.put((video_idx, frame_count, masks, mask_path))

        except Exception as e:
            logger.error(f'Error saving files for frame {frame_count}: {e}')

        if frame_count % 100 == 0:
            logger.info(f'Processed frame {frame_count}/{total_frames}')

def worker_gpu(gpu_id, videos, config_file, checkpoint_file, output_dir, score_thr, target_width, mask_buffer, processed_videos, dataset):
    detector, logger = initialize_detector(config_file, checkpoint_file, gpu_id)
    for video_idx in videos:
        if video_idx < len(dataset):
            video_folder_name = f'video{dataset.get_video_path(video_idx).split("/")[-2]}'
            output_folder = os.path.join(output_dir, video_folder_name)
            os.makedirs(output_folder, exist_ok=True)
            video = dataset.get_video(video_idx)
            process_video(video, video_idx, output_folder, detector, logger, mask_buffer, score_thr, target_width)

            # Increment processed videos safely using a lock
            # with processed_videos_lock:
            #     processed_videos[0] += 1
            with processed_videos.get_lock():
                processed_videos.value += 1
        else:
            logger.warning(f'Invalid video index {video_idx}, skipping.')

def distribute_videos_among_gpus(video_indices, num_gpu, num_threads_per_gpu):
    gpu_assignment = [[[] for _ in range(num_threads_per_gpu)] for _ in range(num_gpu)]
    for i, video_idx in enumerate(video_indices):
        gpu_index = i // num_threads_per_gpu % num_gpu
        thread_index = i // num_gpu % num_threads_per_gpu
        gpu_assignment[gpu_index][thread_index].append(video_idx)
    return gpu_assignment

def monitor_progress(processed_videos, total_videos, start_time, interval=5):
    while True:
        sleep(interval)
        # count = processed_videos[0]
        with processed_videos.get_lock():
            count = processed_videos.value
        elapsed = time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        remaining = total_videos - count
        est_remaining = remaining / rate if rate > 0 else float('inf')

        print(f'Processed {count}/{total_videos} videos. '
              f'Elapsed: {elapsed:.2f}s, '
              f'Rate: {rate:.2f} videos/s, '
              f'Estimated remaining time: {est_remaining:.2f}s')

        if count >= total_videos:
            break

def masking(args, mask_buffer, sample_list, video_dataset, phase='train'):
    global video_dir
    global dataset
    
    dataset = video_dataset
    
    video_dir = args.video_dir
    output_dir = os.path.join(args.output_dir, phase)
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    score_thr = args.score_thr
    target_width = args.target_width
    num_threads_per_gpu = args.num_sub_process_per_gpu

    total_videos = len(sample_list)
    if total_videos == 0:
        print('No videos found in the specified video directory.')
        exit(1)

    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus == 0:
        print('No GPUs available. Exiting.')
        exit(1)
    print(f'Number of available GPUs: {num_available_gpus}')

    distributed_videos = distribute_videos_among_gpus(sample_list, num_available_gpus, num_threads_per_gpu)

    processed_videos = mp.Value('i', 0)

    # Initialize mask buffer with a limit of 1000 masks
    # mask_buffer = initialize_mask_buffer(buffer_size=1000)

    start_time = time()

    processes = []
    
    # print(f'Starting processing on {num_available_gpus} GPUs with {num_threads_per_gpu} threads per GPU.')
    # print('Distributed video size: ', np.array(distributed_videos).shape)
    
    for gpu_id in range(num_available_gpus):
        videos_for_gpu = distributed_videos[gpu_id]
        for sub_process_id in range(num_threads_per_gpu):
            if len(videos_for_gpu[sub_process_id]) == 0:
                continue
            p = mp.Process(
                target=worker_gpu,
                args=(gpu_id, videos_for_gpu[sub_process_id], config_file, checkpoint_file, output_dir, score_thr, target_width, mask_buffer, processed_videos,video_dataset)
            )
            p.start()
            processes.append(p)
            print(f'Started processing on GPU {gpu_id} at sub_process {sub_process_id} with {len(videos_for_gpu[sub_process_id])} video(s).')

    monitor = mp.Process(
        target=monitor_progress,
        args=(processed_videos, total_videos, start_time)
    )
    monitor.start()

    for p in processes:
        p.join()
    monitor.join()
    
    # processed_videos = [0]  # Using a list to make it mutable
    # processed_videos_lock = threading.Lock()

    # start_time = time()

    # # Use ThreadPoolExecutor for multi-threaded execution
    # with ThreadPoolExecutor(max_workers=num_available_gpus * num_threads_per_gpu) as executor:
    #     futures = []
    #     for gpu_id in range(num_available_gpus):
    #         videos_for_gpu = distributed_videos[gpu_id]
    #         for thread_id in range(num_threads_per_gpu):
    #             if len(videos_for_gpu[thread_id]) == 0:
    #                 continue
    #             future = executor.submit(worker_gpu, gpu_id, videos_for_gpu[thread_id], config_file, checkpoint_file, output_dir, score_thr, target_width, mask_buffer, processed_videos_lock, processed_videos, video_dataset)
    #             futures.append(future)

    #     # Wait for all threads to complete
    #     for future in as_completed(futures):
    #         future.result()

    elapsed_total = time() - start_time
    print(f'All videos processed in {elapsed_total:.2f} seconds.')


