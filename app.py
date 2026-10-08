import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Panorama Generator", layout="wide")

st.title("Image Registration and Panorama Generation")

left_file = st.file_uploader(
    "Upload Left Image",
    type=["jpg", "jpeg", "png"]
)

right_file = st.file_uploader(
    "Upload Right Image",
    type=["jpg", "jpeg", "png"]
)

if left_file and right_file:

    left_img = np.array(Image.open(left_file).convert("RGB"))
    right_img = np.array(Image.open(right_file).convert("RGB"))

    col1, col2 = st.columns(2)

    with col1:
        st.image(left_img, caption="Left Image")

    with col2:
        st.image(right_img, caption="Right Image")

    if st.button("Generate Panorama"):

        try:

            gray1 = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

            sift = cv2.SIFT_create()

            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)

            matcher = cv2.BFMatcher()

            matches = matcher.knnMatch(des1, des2, k=2)

            good_matches = []

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 10:
                st.error("Not enough matching features found.")
                st.stop()

            pts1 = np.float32(
                [kp1[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            pts2 = np.float32(
                [kp2[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                pts2,
                pts1,
                cv2.RANSAC,
                5.0
            )

            h1, w1 = left_img.shape[:2]
            h2, w2 = right_img.shape[:2]

            corners2 = np.float32([
                [0, 0],
                [0, h2],
                [w2, h2],
                [w2, 0]
            ]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(
                corners2,
                H
            )

            corners1 = np.float32([
                [0, 0],
                [0, h1],
                [w1, h1],
                [w1, 0]
            ]).reshape(-1, 1, 2)

            all_corners = np.concatenate(
                (corners1, transformed_corners),
                axis=0
            )

            [xmin, ymin] = np.int32(
                all_corners.min(axis=0).ravel() - 0.5
            )

            [xmax, ymax] = np.int32(
                all_corners.max(axis=0).ravel() + 0.5
            )

            translation = [-xmin, -ymin]

            H_translation = np.array([
                [1, 0, translation[0]],
                [0, 1, translation[1]],
                [0, 0, 1]
            ])

            panorama = cv2.warpPerspective(
                right_img,
                H_translation.dot(H),
                (xmax - xmin, ymax - ymin)
            )

            panorama[
                translation[1]:h1 + translation[1],
                translation[0]:w1 + translation[0]
            ] = left_img

            gray_pan = cv2.cvtColor(
                panorama,
                cv2.COLOR_RGB2GRAY
            )

            coords = cv2.findNonZero(gray_pan)

            x, y, w, h = cv2.boundingRect(coords)

            panorama = panorama[y:y+h, x:x+w]

            st.success("Panorama Generated")

            st.image(
                panorama,
                caption="Generated Panorama",
                use_container_width=True
            )

        except Exception as e:
            st.error(str(e))