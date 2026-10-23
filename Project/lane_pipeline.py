import numpy as np
import cv2 
import matplotlib.pyplot as plt

X_RANGE = (-7.0, 7.0)              # larghezza della strada in metri
Z_RANGE = (7.0, 20.0)              # profondità della strada in metri                                  s
PX_PER_M = 30                      # risoluzione della bird-eye view (1 px = 1/30 m)
GAUSS_KSIZE = 5                    # coefficiente del filtro gaussiano, per ridurre il rumore
K_threshold = 3.0
PERCENTILE = 97    # tengo il 3% dei pixel con gradiente più forte
MIN_GRAD = 0.0       # soglia minima assoluta (da tarare, vedi sotto)                   # coefficiente per la soglia trasformazione binaria: media + k * deviazione standard

#legge da file calibrazione
def read_calib(path):
    calib = {}
    with open(path) as f:
        for row in f:
            if ":" in row:
                name, values = row.split(":", 1)
                calib[name.strip()] = np.array(values.split(), dtype=float)
    return calib

"""
    Nei file di KITTI Road ci sono i dati della camera (P2) e la sua posizione
    rispetto alla strada (Tr_cam_to_road), cioe' le informazioni su campo visivo,
    posizione e orientazione.
"""
def compute_trapezoid(calib, x_range=X_RANGE, z_range=Z_RANGE):
    P2 = calib["P2"].reshape(3, 4)                 # proiezione camera 2 (f, cx, cy, ...)
    R0_rect = calib["R0_rect"].reshape(3, 3)       # camera non rettificata -> camera rettificata
    T = calib["Tr_cam_to_road"].reshape(3, 4)      # trasformazione camera -> strada: [R | t]
    R, t = T[:, :3], T[:, 3] #rotazione e traslazione

    def show(x, z):
        p_road = np.array([x, 0.0, z])           
        p_cam = R.T @ (p_road - t)               # esprime il punto nel sistema camera
        p_cam_rect = R0_rect @ p_cam                # coordinate della camera rettificata
        u, v, w = P2 @ np.append(p_cam_rect, 1.0)   # sono le coordinate del punto proiettato nell'immagine
        return [u / w, v / w]                      # ottengo posizione in pixel

    return np.float32([
        show(x_range[0], z_range[0]),          
        show(x_range[1], z_range[0]),          
        show(x_range[1], z_range[1]),          
        show(x_range[0], z_range[1]),          
    ])

#Scarta tutto cio' che sta fuori dal trapezio.
def apply_roi(img, trapezoid):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid.astype(np.int32)], 255)   
    return cv2.bitwise_and(img, img, mask=mask)            

def brid_eye_view(img, trapezoid, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    # dimensioni della bird-eye view in pixel
    width = int((x_range[1] - x_range[0]) * px_per_m)
    height = int((z_range[1] - z_range[0]) * px_per_m)
    trapezoid_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])  
    M = cv2.getPerspectiveTransform(trapezoid, trapezoid_dst)      # matrice di trasformazione
    M_inv = cv2.getPerspectiveTransform(trapezoid_dst, trapezoid)  # matrice inversa
    bew =  cv2.warpPerspective(img, M, (width, height)) 
    return bew, M, M_inv

#image segmentation 

def gaussian_filter(img, ksize=GAUSS_KSIZE): 
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, (ksize, ksize), 0)

def gradient_filter(img): 
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 
    return np.sqrt(gx**2 + gy**2)               

def threshold(img, percentile=PERCENTILE, min_grad=MIN_GRAD, block=60): 
    out = np.zeros(img.shape, np.uint8)
    for c in range(0, img.shape[1], block):
        blk = img[:, c:c + block]
        thr = max(np.percentile(blk, percentile), min_grad)
        out[:, c:c + block] = (blk > thr).astype(np.uint8) * 255
    return out

def lane_pipeline(path, img, out_path):
    calib = read_calib(path)
    trapezoid = compute_trapezoid(calib)
    img_roi = apply_roi(img, trapezoid)
    img_bird_eye, M, M_inv = brid_eye_view(img_roi, trapezoid)
    img_filtered = gaussian_filter(img_bird_eye)
    img_norm = normalise(img_filtered)
    valid = valid_bev_mask(img.shape, trapezoid)
    img_grad = gradient_filter(img_filtered) * valid
    img_thresh = threshold(img_grad)
    print_images(img, img_bird_eye, img_filtered, img_grad, img_thresh, trapezoid, out_path)
    #print("soglia percentile:", np.percentile(img_grad, PERCENTILE))
    
    return M_inv, img_thresh, img_filtered

def print_images(img,img_bird_eye, img_filtered, img_grad, img_thresh, trapezoid, out_path):
    vis = img.copy()
    cv2.polylines(vis, [trapezoid.astype("int32")], True, (255, 0, 0), 3)
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    ax = ax.ravel()
    ax[0].imshow(vis);                        ax[0].set_title("ROI")
    ax[1].imshow(img_bird_eye);               ax[1].set_title("Bird-eye")
    ax[2].imshow(img_filtered);  ax[2].set_title("Gaussiano")
    ax[3].imshow(img_grad);      ax[3].set_title("Gradiente")
    ax[4].imshow(img_thresh);    ax[4].set_title("Soglia")
    for a in ax:
        a.axis("off")
    #plt.show()
    # save the image in the folder debug_images
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def normalise(img):
    f = img.astype(np.float32) + 1.0          # +1 avoids division by zero
    bg = cv2.GaussianBlur(f, (0, 0), 15)      # sigma 15 px, about 0.5 m at 30 px/m
    r = f / bg                                # ~1 where the brightness is local average
    return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



# debug function: print all candidates on both BEV and line reconstruction on black background
def valid_bev_mask(img_shape, trapezoid, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    ones = np.full(img_shape[:2], 255, np.uint8)
    roi = apply_roi(ones, trapezoid)
    bev, _, _ = brid_eye_view(roi, trapezoid, x_range, z_range, px_per_m)
    return cv2.erode((bev > 0).astype(np.uint8), np.ones((9, 9), np.uint8))