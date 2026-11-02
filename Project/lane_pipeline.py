import numpy as np
import cv2 
import matplotlib.pyplot as plt

X_RANGE = (-7.0, 7.0)
Z_RANGE = (7.0, 20.0)
PX_PER_M = 30
GAUSS_KSIZE = 5

def read_calib(path):
    calib = {}
    with open(path) as f:
        for row in f:
            if ":" in row:
                name, values = row.split(":", 1)
                calib[name.strip()] = np.array(values.split(), dtype=float)
    return calib

def compute_trapezoid(calib, x_range=X_RANGE, z_range=Z_RANGE):
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    T = calib["Tr_cam_to_road"].reshape(3, 4)
    R, t = T[:, :3], T[:, 3]
 
    def show(x, z):
        p_road = np.array([x, 0.0, z])           
        p_cam = R.T @ (p_road - t)               
        p_cam_rect = R0_rect @ p_cam                
        u, v, w = P2 @ np.append(p_cam_rect, 1.0)   
        return [u / w, v / w]                      
 
    return np.float32([
        show(x_range[0], z_range[0]),          
        show(x_range[1], z_range[0]),          
        show(x_range[1], z_range[1]),          
        show(x_range[0], z_range[1]),          
    ])

def apply_roi(img, trapezoid):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid.astype(np.int32)], 255)   
    return cv2.bitwise_and(img, img, mask=mask)            

def brid_eye_view(img, trapezoid, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    width = int((x_range[1] - x_range[0]) * px_per_m)
    height = int((z_range[1] - z_range[0]) * px_per_m)
    trapezoid_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])  
    M = cv2.getPerspectiveTransform(trapezoid, trapezoid_dst)      
    M_inv = cv2.getPerspectiveTransform(trapezoid_dst, trapezoid)  
    bew = cv2.warpPerspective(img, M, (width, height)) 
    return bew, M, M_inv

# ---------------- Nuova pipeline ispirata al GitHub ----------------
def canny_bev_pipeline(img_bev):
    # 1. Conversione a scala di grigi
    gray = cv2.cvtColor(img_bev, cv2.COLOR_RGB2GRAY)
    
    # 2. Miglioramento contrasto con CLAHE (come nel GitHub)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Blur Gaussiano
    blur = cv2.GaussianBlur(enhanced, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    
    # 4. Canny Edge Detector
    canny_out = cv2.Canny(blur, 50, 150)
    
    # 5. Filtraggio pendenze verticali con Sobel X (evita elementi trasversali)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.uint8(np.absolute(sobelx))
    _, vert_mask = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
    
    edges_filtered = cv2.bitwise_and(canny_out, vert_mask)
    return edges_filtered

def lane_pipeline(path, img):
    calib = read_calib(path)
    trapezoid = compute_trapezoid(calib)
    img_roi = apply_roi(img, trapezoid)
    img_bird_eye, M, M_inv = brid_eye_view(img_roi, trapezoid)
    
    img_thresh = canny_bev_pipeline(img_bird_eye)
    
    print_images(img, img_bird_eye, img_thresh, trapezoid)
    return M_inv, img_thresh

def print_images(img, img_bird_eye, img_thresh, trapezoid):
    vis = img.copy()
    cv2.polylines(vis, [trapezoid.astype("int32")], True, (255, 0, 0), 3)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(vis);          ax[0].set_title("ROI Originale")
    ax[1].imshow(img_bird_eye); ax[1].set_title("Bird-eye")
    ax[2].imshow(img_thresh, cmap="gray"); ax[2].set_title("CLAHE + Canny BEV")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()