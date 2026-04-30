import cv2
import numpy as np
import os

def segment_and_unroll_iris(image_path, x_P=2.5, x_I=1.2):

    # Konwersja do odcieni szarości
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error! Path not found")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Próg binaryzacji ze średniej 
    P = np.mean(gray) 

    # Progi dla źrenicy i tęczówki
    P_P = P / x_P
    P_I = P / x_I

    # Detekcja źrenicy
    _, thresh_pupil = cv2.threshold(gray, P_P, 255, cv2.THRESH_BINARY_INV)

    # Operacje morfologiczne otwarcia i zamknięcia
    kernel = np.ones((7, 7), np.uint8)
    morph_pupil = cv2.morphologyEx(thresh_pupil, cv2.MORPH_OPEN, kernel) 
    morph_pupil = cv2.morphologyEx(morph_pupil, cv2.MORPH_CLOSE, kernel) 

    # Detekcja tęczówki
    _, thresh_iris = cv2.threshold(gray, P_I, 255, cv2.THRESH_BINARY_INV)

    # Operacje morfologiczne dla tęczówki
    kernel_open = np.ones((17, 17), np.uint8) 
    kernel_close = np.ones((17, 17), np.uint8)
    morph_iris = cv2.morphologyEx(thresh_iris, cv2.MORPH_OPEN, kernel_open)
    morph_iris = cv2.morphologyEx(morph_iris, cv2.MORPH_CLOSE, kernel_close)

    # Wycznaczanie promieni i środka

    # Projekcje dla źrenicy
    proj_x = np.sum(morph_pupil, axis=0)
    proj_y = np.sum(morph_pupil, axis=1)

    # Filtr dla wykrytych obszarów
    max_x_indices = np.where(proj_x > 0)[0]
    max_y_indices = np.where(proj_y > 0)[0]

    # Uogólnione szukanie środka. Średni punkt z wykrytych projekcji
    c_x = int((max_x_indices[0] + max_x_indices[-1]) / 2)
    c_y = int((max_y_indices[0] + max_y_indices[-1]) / 2)

    # Sztywny próg dla źrenicy
    non_zero_x = np.where(proj_x > 0)[0]
    # Wyznaczanie promienia źrenicy
    r_pupil = int((non_zero_x[-1] - non_zero_x[0]) / 2)
    
    # Robimy projekcję tylko na uciętym fragmencie oka
    strip_margin = 50
    y_start = max(0, c_y - strip_margin)
    y_end = min(h, c_y + strip_margin)

    iris_strip = morph_iris[y_start:y_end, :]

    # Wybieramy tylko te projekcje, które są proporcjonajlnie mniejsze od maksimum.
    proj_x_iris = np.sum(iris_strip, axis=0)
    non_zero_x_iris = np.where(proj_x_iris > 0)[0]
    
    # Wyznaczanie promienia tęczówki
    r_iris = int((non_zero_x_iris[-1] - non_zero_x_iris[0]) / 2)
    print(r_iris, r_pupil)
    # Rozwinięcie tęczówki do prostokąta

    # Zmiana na współrzędne biegunowe
    theta_res = 360  
    r_res = r_iris - r_pupil 
    
    unrolled_iris = np.zeros((r_res, theta_res), dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, theta_res)

    for i in range(r_res):
        r = r_pupil + i
        for j in range(theta_res):
            theta = angles[j]
            
            # Wzór na współrzędne kartezjańskie
            x = int(c_x + r * np.cos(theta))
            y = int(c_y + r * np.sin(theta))
            
            # Sprawdzenie granic obrazu
            if 0 <= x < w and 0 <= y < h:
                unrolled_iris[i, j] = gray[y, x]

    # Zwrócenie wyników do ewentualnej wizualizacji/analizy
    results = {
        'original_gray': gray,
        'center': (c_x, c_y),
        'r_pupil': r_pupil,
        'r_iris': r_iris,
        'unrolled_iris': unrolled_iris
    }
    
    return results

if __name__ == "__main__":
    input_folder = "images"
    output_folder = "results"
    
    if not os.path.exists(input_folder):
        print(f"Błąd: Nie znaleziono folderu '{input_folder}'.")
    else:
        valid_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        # Zapis plików dla wszysktich zdjęć
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                
                print(f"Przetwarzanie pliku: {filename}...")
                
                try:
                    res = segment_and_unroll_iris(image_path, x_P=3.5, x_I=1.6)
                    
                    # Konwersja kolorów
                    img_debug = cv2.cvtColor(res['original_gray'], cv2.COLOR_GRAY2BGR)
                    c_x, c_y = res['center']
                    
                    # Rysowanie granic i środka
                    cv2.circle(img_debug, (c_x, c_y), res['r_pupil'], (0, 0, 255), 2)
                    cv2.circle(img_debug, (c_x, c_y), res['r_iris'], (0, 0, 255), 2)
                    cv2.circle(img_debug, (c_x, c_y), 2, (0, 0, 255), 3)

                    seg_filename = f"{base_name}_segmentacja.jpg"
                    unroll_filename = f"{base_name}_rozwinieta.jpg"
                    
                    path_seg = os.path.join(output_folder, seg_filename)
                    path_unroll = os.path.join(output_folder, unroll_filename)

                    cv2.imwrite(path_seg, img_debug)
                    cv2.imwrite(path_unroll, res['unrolled_iris'])
                    
                    print(f"Zapisano: {seg_filename} oraz {unroll_filename}")
                    print(f"  Promień źrenicy: {res['r_pupil']}, Promień tęczówki: {res['r_iris']}")
                    
                except Exception as e:
                    print(f"Wystąpił błąd podczas przetwarzania {filename}: {e}")

        print("\nZakończono przetwarzanie wszystkich plików.")