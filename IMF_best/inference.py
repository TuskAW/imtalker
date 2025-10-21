import argparse
import os
import torch
from tqdm import tqdm
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch
import torch.nn.functional as F
from torch import nn
import os
import cv2
import numpy as np
import face_alignment
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from networks.model_adaptive import IMFModel


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.input_size = opt.input_size

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def process_img(self, img) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
        mult = 360. / img.shape[0]
        resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult, 
                                 interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized_img)
        bboxes = [
            (int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score)
            for (x1, y1, x2, y2, score) in bboxes if score > 0.95
        ]
        if len(bboxes) == 0:
            print("[WARN] No face detected, fallback to center crop")
            return cv2.resize(img, (self.input_size, self.input_size))
    
        x1, y1, x2, y2, _ = bboxes[0]
        bsy, bsx = int((y2 - y1) / 2), int((x2 - x1) / 2)
        my, mx = int((y1 + y2) / 2), int((x1 + x2) / 2)
        bs = int(max(bsy, bsx) * 1.3)
    
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        crop_img = cv2.resize(crop_img, (self.input_size, self.input_size),
                              interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return Image.fromarray(crop_img)

    def default_img_loader(self, path) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 1)
    vid = vid.clamp(0, 1).cpu().numpy()
    vid = (vid * 255).astype(np.uint8)
    T, H, W, C = vid.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    # write frame by frame
    for frame in vid:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved successfully to {save_path}")


class Demo(nn.Module):
    def __init__(self, args, gen):
        super(Demo, self).__init__()
        self.args = args
        print('==> loading model')
        self.gen = gen.to("cuda")
        self.gen.eval()

        print('==> preparing save path')
        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)

        # image and video processor
        self.processor = DataProcessor(args)

    @torch.no_grad()
    def process_single(self, source_path, driving_path):
        """Single file processing"""
        print(f"==> Processing single file: {source_path}, {driving_path}")
        source_img = self.processor.default_img_loader(source_path)
        #if self.args.crop:
        #    source_img = self.processor.process_img(source_img)
        source_img = self.processor.transform(source_img).unsqueeze(0).to("cuda")

        cap = cv2.VideoCapture(driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if self.args.fps is None else self.args.fps
        driving_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.args.crop:
                frame = self.processor.process_img(frame)
            frame = self.processor.transform(frame).unsqueeze(0).to("cuda")
            driving_frames.append(frame)
        cap.release()

        print(f"==> Total frames: {len(driving_frames)}")

        vid_target_recon = []
        for frame in tqdm(driving_frames, desc="Inferencing"):
            out = self.gen(frame, source_img)
            vid_target_recon.append(out)

        vid_target_recon = torch.cat(vid_target_recon, dim=0)
        save_name = f"{Path(source_path).stem}_{Path(driving_path).stem}.mp4"
        save_video(vid_target_recon, os.path.join(self.save_path, save_name), fps)

    @torch.no_grad()
    def process_batch(self, root_dir):
        """Batch processing: each subfolder contains 1 image and 1 video"""
        subdirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        for sub in subdirs:
            img_files = [f for f in os.listdir(sub) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            vid_files = [f for f in os.listdir(sub) if f.lower().endswith((".mp4", ".avi", ".mov"))]
            if not img_files or not vid_files:
                print(f"[WARN] {sub} missing image or video, skipping")
                continue
            img_path = os.path.join(sub, img_files[0])
            vid_path = os.path.join(sub, vid_files[0])
            self.process_single(img_path, vid_path)

    def run(self):
        """Run mode selection: single file or batch"""
        if hasattr(self.args, "source_path") and hasattr(self.args, "driving_path") \
            and self.args.source_path and self.args.driving_path:
            self.process_single(self.args.source_path, self.args.driving_path)
        elif hasattr(self.args, "data_dir") and self.args.data_dir:
            self.process_batch(self.args.data_dir)
        else:
            raise ValueError("Please provide --source_path + --driving_path OR --data_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input/output modes
    parser.add_argument("--source_path", type=str, default=None, 
                        help="Path to a single source image (single file mode)")
    parser.add_argument("--driving_path", type=str, default=None, 
                        help="Path to a single driving video (single file mode)")
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Batch mode: directory with subfolders, each containing one image and one video")
    parser.add_argument("--save_path", type=str, default="./results", 
                        help="Directory to save the results")

    # Model
    parser.add_argument("--imf_path", type=str, required=True,
                        help="Path to pretrained IMF model checkpoint")
    parser.add_argument("--input_size", type=int, default=256, 
                        help="Input resolution for images and video frames")
    parser.add_argument('--swin_res_threshold', type=int, default=128, 
                        help='Resolution threshold to switch to Swin Attention.')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads.')
    parser.add_argument('--window_size', type=int, default=8, 
                        help='Window size for Swin Attention.')




    # Inference options
    parser.add_argument("--fps", type=int, default=None,
                        help="Output video fps, None = keep driving video original fps")
    parser.add_argument("--crop", action="store_true", 
                        help="Crop faces from input image/video before inference")

    args = parser.parse_args()

    # Load model
    ae = IMFModel(args)
    ae_state_dict = torch.load(args.imf_path, map_location="cpu")["state_dict"]
    ae_state_dict = {k.replace("gen.", ""): v for k, v in ae_state_dict.items() if k.startswith("gen.")}
    missing_gen, unexpected_gen = ae.load_state_dict(ae_state_dict, strict=False)

    # Run demo
    demo = Demo(args, ae)
    demo.run()


# Example usage:

# Single file mode
# python demo.py --source_path ./examples/source.jpg --driving_path ./examples/driving.mp4 \
#                --imf_path ./exps/checkpoints/last.ckpt --save_folder ./results --crop

# Batch mode
# python demo.py --data_dir ./dataset --imf_path ./exps/checkpoints/last.ckpt --save_folder ./results

