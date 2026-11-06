#!/usr/bin/env python3
"""
calibrate.py : calibration d'une caméra avec un damier (OpenCV).
"""

import argparse
import glob
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Calibration de caméra (damier)")
    p.add_argument("--images", default="images", help="dossier des photos")
    p.add_argument("--cols", type=int, default=9, help="coins intérieurs (horizontal)")
    p.add_argument("--rows", type=int, default=6, help="coins intérieurs (vertical)")
    p.add_argument("--square", type=float, default=25.0,
                   help="taille réelle d'un carré en mm (à mesurer avec une règle)")
    p.add_argument("--out-corners", default="output/corners", help="images avec coins")
    p.add_argument("--out-results", default="results", help="dossier des résultats")
    p.add_argument("--max-dim", type=int, default=1600,
                   help="réduit l'image pour la détection (accélère). 0 = pas de réduction")
    p.add_argument("--show", action="store_true",
                   help="affiche chaque détection avec cv2.imshow (serveur X requis)")
    return p.parse_args()


def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(set(files))


def find_corners(gray, pattern_size, max_dim):
    """Détecte les coins du damier. Retourne (found, corners) en pleine résolution."""
    h, w = gray.shape
    scale = 1.0
    small = gray
    if max_dim and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, pattern_size, flags)
    if not found:
        return False, None

    # Retour à la pleine résolution, puis affinement sub-pixel
    corners = corners / scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
    return True, corners


def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)

    os.makedirs(args.out_corners, exist_ok=True)
    os.makedirs(args.out_results, exist_ok=True)

    # 1. Points 3D du damier (Z = 0, le damier est plan), en mm
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square

    files = list_images(args.images)
    if not files:
        raise SystemExit(f"Aucune image trouvée dans '{args.images}'")
    print(f"{len(files)} images trouvées. Damier : {args.cols}x{args.rows} coins, "
          f"carré = {args.square} mm\n")

    objpoints, imgpoints, used, rejected = [], [], [], []
    image_size = None  # (largeur, hauteur)

    # 2. Détection des coins
    for path in files:
        name = os.path.basename(path)
        img = cv2.imread(path)  # applique l'orientation EXIF
        if img is None:
            print(f"[ERREUR] lecture impossible : {name}")
            rejected.append(name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])

        if image_size is None:
            image_size = size
        elif size != image_size:
            print(f"[IGNORÉE] {name} : taille {size} différente de {image_size}")
            rejected.append(name)
            continue

        found, corners = find_corners(gray, pattern_size, args.max_dim)
        if not found:
            print(f"[NON DÉTECTÉ] {name}")
            rejected.append(name)
            continue

        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(name)
        print(f"[OK] {name}")

        # 3. Sauvegarde de l'image avec les coins dessinés
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        cv2.imwrite(os.path.join(args.out_corners, f"corners_{name}"), vis)

        if args.show:
            cv2.imshow("coins", cv2.resize(vis, None, fx=0.3, fy=0.3))
            cv2.waitKey(300)

    if args.show:
        cv2.destroyAllWindows()

    print(f"\nImages utilisées : {len(used)} / {len(files)}")
    if len(used) < 10:
        print("ATTENTION : moins de 10 images valides. Prenez plus de photos variées.")
    if len(used) < 3:
        raise SystemExit("Pas assez d'images valides pour calibrer.")

    # 4. Calibration
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Erreur de reprojection par image (aperçu ; détaillé à l'étape 6)
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        errors.append(err)
    errors = np.array(errors)

    print("\n===== RÉSULTATS =====")
    print(f"Taille image : {image_size[0]} x {image_size[1]} px")
    print(f"RMS (calibrateCamera) : {rms:.4f} px")
    print(f"Erreur de reprojection moyenne : {errors.mean():.4f} px")
    print("\nMatrice intrinsèque K :")
    print(np.array2string(K, precision=3, suppress_small=True))
    print(f"\nfx = {K[0, 0]:.2f} px   fy = {K[1, 1]:.2f} px")
    print(f"cx = {K[0, 2]:.2f} px   cy = {K[1, 2]:.2f} px")
    print("\nDistorsion (k1, k2, p1, p2, k3) :")
    print(dist.ravel())
    print("\nErreur par image :")
    for n, e in zip(used, errors):
        print(f"  {n:30s} {e:.4f} px")

    # Sauvegarde
    out_file = os.path.join(args.out_results, "calib.npz")
    np.savez(
        out_file,
        K=K,
        dist=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        rms=rms,
        errors=errors,
        image_size=np.array(image_size),
        pattern_size=np.array(pattern_size),
        square_size=args.square,
        used_images=np.array(used),
        objpoints=np.array(objpoints),
        imgpoints=np.array(imgpoints),
    )
    print(f"\nRésultats sauvegardés dans {out_file}")
    print(f"Images avec coins dans {args.out_corners}/")
    if rejected:
        print(f"Images rejetées ({len(rejected)}) : {', '.join(rejected)}")


if __name__ == "__main__":
    main()